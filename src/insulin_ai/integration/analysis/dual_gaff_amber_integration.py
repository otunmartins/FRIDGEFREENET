#!/usr/bin/env python3
"""
Dual GAFF+AMBER MD Integration
==============================

This module provides MD integration using the proven dual approach:
1. GAFF for polymer parameterization (DirectPolymerBuilder)
2. AMBER for insulin simulation (simple_insulin_simulation.py approach)
3. Combined properly without CYS/CYX template generator issues

Based on the successful dual_gaff_amber_md_simulation.py script.
"""

import os
import sys
import time
import threading
import uuid
import tempfile
import shutil
import importlib.resources
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

# Add the src directory to Python path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from openmm.app import (
        PDBFile, Modeller, ForceField, Simulation,
        StateDataReporter, PDBReporter,
        NoCutoff, HBonds,
    )
    from openmm import BrownianIntegrator, Platform, unit, LangevinMiddleIntegrator
    from openmm.app import Topology
    from openmmforcefields.generators import GAFFTemplateGenerator
    from openff.toolkit.topology import Molecule
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError as e:
    OPENMM_AVAILABLE = False
    print(f"⚠️ OpenMM or related packages not available: {e}")

try:
    from insulin_ai.utils.direct_polymer_builder import DirectPolymerBuilder
    POLYMER_BUILDER_AVAILABLE = True
except ImportError:
    POLYMER_BUILDER_AVAILABLE = False
    print("⚠️ DirectPolymerBuilder not available")

try:
    from insulin_ai.integration.analysis.simple_working_md_simulator import SimpleWorkingMDSimulator
    SIMPLE_SIMULATOR_AVAILABLE = True
except ImportError:
    SIMPLE_SIMULATOR_AVAILABLE = False
    print("⚠️ SimpleWorkingMDSimulator not available")


class DualGaffAmberIntegration:
    """
    Dual GAFF+AMBER MD Integration for insulin-polymer composite systems.
    
    This class implements the successful dual approach:
    - Uses DirectPolymerBuilder for polymer creation
    - Uses GAFF for polymer parameterization
    - Uses AMBER for insulin parameterization (native CYX support)
    - Combines systems properly with spatial separation
    """
    
    def __init__(self, output_dir: str = "dual_gaff_amber_simulations"):
        """Initialize the dual GAFF+AMBER integration"""
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Also ensure the automated simulations directory exists, as it's scanned by the UI
        self.automated_simulations_dir = Path("automated_simulations")
        self.automated_simulations_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.polymer_builder = DirectPolymerBuilder() if POLYMER_BUILDER_AVAILABLE else None
        
        # Keep a reference to the simple simulator for its helper methods if needed
        # but the main logic will be self-contained here.
        self.simulator = SimpleWorkingMDSimulator() if SIMPLE_SIMULATOR_AVAILABLE else None
        
        # Simulation state
        self.current_simulation = None
        self.simulation_running = False
        self.simulation_thread = None
        
        # Check dependencies
        self.dependencies_ok = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        
        missing = []
        if not OPENMM_AVAILABLE:
            missing.append("OpenMM")
        if not POLYMER_BUILDER_AVAILABLE:
            missing.append("DirectPolymerBuilder")
        if not SIMPLE_SIMULATOR_AVAILABLE:
            missing.append("SimpleWorkingMDSimulator")
            
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            return False
        
        print("✅ All dependencies available for dual GAFF+AMBER approach")
        return True
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get status of all dependencies"""
        
        return {
            'openmm': {
                'available': OPENMM_AVAILABLE,
                'description': 'OpenMM molecular dynamics engine'
            },
            'polymer_builder': {
                'available': POLYMER_BUILDER_AVAILABLE,
                'description': 'DirectPolymerBuilder for polymer creation'
            },
            'simple_simulator': {
                'available': SIMPLE_SIMULATOR_AVAILABLE,
                'description': 'SimpleWorkingMDSimulator for MD'
            },
            'overall': {
                'available': self.dependencies_ok,
                'description': 'All systems operational for dual GAFF+AMBER'
            }
        }
    
    def find_insulin_file(self, custom_insulin_pdb: str = None) -> str:
        """Find suitable insulin PDB file"""
        
        if custom_insulin_pdb and os.path.exists(custom_insulin_pdb):
            return custom_insulin_pdb
        
        # Look for insulin files in standard locations
        insulin_candidates = [
            "src/insulin_ai/integration/data/insulin/insulin_default.pdb",
            "src/insulin_ai/integration/data/insulin/human_insulin_1mso.pdb"
        ]
        
        for candidate in insulin_candidates:
            if os.path.exists(candidate):
                return candidate
        
        raise ValueError("No insulin PDB file found. Please provide insulin_pdb parameter.")
    
    def fix_insulin_pdb_residues(self, input_pdb_path: str, output_pdb_path: str, log_callback):
        """
        Fix insulin PDB file to use CYX for disulfide-bonded cysteines.
        
        This creates a corrected PDB file that AMBER can properly handle.
        """
        
        log_callback("🔧 Fixing insulin PDB residue names for AMBER compatibility...")
        
        try:
            with open(input_pdb_path, 'r') as f:
                pdb_lines = f.readlines()
            
            fixed_lines = []
            cys_count = 0
            
            for line in pdb_lines:
                if line.startswith(('ATOM', 'HETATM')) and 'CYS' in line:
                    # Replace CYS with CYX in the residue name field (columns 18-20)
                    if line[17:20].strip() == 'CYS':
                        line = line[:17] + 'CYX' + line[20:]
                        cys_count += 1
                
                fixed_lines.append(line)
            
            # Write corrected PDB
            with open(output_pdb_path, 'w') as f:
                f.writelines(fixed_lines)
            
            if cys_count > 0:
                log_callback(f"   ✅ Fixed {cys_count} CYS → CYX atom records")
                log_callback(f"   📁 Corrected PDB saved: {output_pdb_path}")
            else:
                log_callback("   ✅ No CYS residues found - insulin already properly formatted")
            
            return output_pdb_path
            
        except Exception as e:
            log_callback(f"   ⚠️ PDB fixing failed: {e}")
            log_callback(f"   🔄 Using original PDB: {input_pdb_path}")
            return input_pdb_path
    
    def extract_polymer_info_from_path(self, simulation_input_file: str) -> Tuple[str, str]:
        """
        Extract polymer PSMILES and prepare for simulation.
        
        Args:
            simulation_input_file: Path to simulation input file or directory
            
        Returns:
            Tuple of (psmiles, expected_polymer_pdb_path)
        """
        
        # Handle different input types
        input_path = Path(simulation_input_file)
        
        if input_path.is_dir():
            # Directory containing polymer files
            psmiles_file = input_path / "psmiles.txt"
            if psmiles_file.exists():
                with open(psmiles_file, 'r') as f:
                    psmiles = f.read().strip()
                    
                # Look for polymer PDB
                polymer_pdb_candidates = list(input_path.glob("*polymer*.pdb"))
                expected_pdb = polymer_pdb_candidates[0] if polymer_pdb_candidates else None
                
                return psmiles, str(expected_pdb) if expected_pdb else None
        
        elif input_path.suffix == '.txt':
            # Direct PSMILES file
            with open(input_path, 'r') as f:
                psmiles = f.read().strip()
            return psmiles, None
        
        else:
            # Assume it's a PSMILES string if it contains certain characters
            if any(char in str(input_path) for char in ['[*]', '=', '#']):
                return str(input_path), None
        
        raise ValueError(f"Unable to extract polymer information from: {simulation_input_file}")
    
    def run_md_simulation_async(self, 
                              pdb_file: str,
                              temperature: float = 310.0,
                              equilibration_steps: int = 10000,  # 20 ps
                              production_steps: int = 50000,     # 100 ps
                              save_interval: int = 500,
                              output_prefix: str = None,
                              output_callback: Optional[Callable] = None,
                              manual_polymer_dir: str = None,
                              **kwargs) -> str:
        """
        Run dual GAFF+AMBER simulation asynchronously.
        
        Args:
            pdb_file: Path to polymer input file/directory or PSMILES
            temperature: Simulation temperature (K)
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            save_interval: Save trajectory every N steps
            output_prefix: Prefix for output files
            output_callback: Callback function for output messages
            manual_polymer_dir: Manual polymer directory
            
        Returns:
            Simulation ID
        """
        
        if not self.dependencies_ok:
            raise RuntimeError("Dependencies not available for dual GAFF+AMBER simulation")
        
        # Generate simulation ID
        simulation_id = output_prefix or f"dual_gaff_amber_{uuid.uuid4().hex[:8]}"
        
        # Store simulation parameters
        self.current_simulation = {
            'id': simulation_id,
            'pdb_file': pdb_file,
            'temperature': temperature,
            'equilibration_steps': equilibration_steps,
            'production_steps': production_steps,
            'save_interval': save_interval,
            'manual_polymer_dir': manual_polymer_dir,
            'status': 'starting',
            'start_time': time.time(),
            'approach': 'dual_gaff_amber'
        }
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(
            target=self._run_dual_simulation_thread,
            args=(simulation_id, pdb_file, temperature, equilibration_steps, 
                  production_steps, save_interval, output_callback, manual_polymer_dir),
            kwargs=kwargs
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        self.simulation_running = True
        
        return simulation_id
    
    def _run_dual_simulation_thread(self, 
                                  simulation_id: str, 
                                  pdb_file: str,
                                  temperature: float, 
                                  equilibration_steps: int,
                                  production_steps: int, 
                                  save_interval: int,
                                  output_callback: Optional[Callable],
                                  manual_polymer_dir: str = None,
                                  **kwargs):
        """
        Run the dual GAFF+AMBER simulation in a separate thread.
        This has been refactored for simplicity and robustness.
        """
        
        def log_callback(message: str):
            """Helper to send log messages through callback"""
            if output_callback:
                try:
                    output_callback(message)
                except Exception as e:
                    print(f"[CALLBACK_ERROR] {e}: {message}")
            else:
                print(message)
        
        try:
            self.current_simulation['status'] = 'running'
            
            output_dir = self.output_dir / simulation_id
            output_dir.mkdir(exist_ok=True)

            with tempfile.TemporaryDirectory() as temp_dir:
                log_callback("🚀 DUAL GAFF+AMBER MD SIMULATION (REFACTORED V2)")
                log_callback("=" * 80)
                log_callback(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                log_callback(f"📁 Output directory: {output_dir}")

                # --- Step 1: Create Polymer from PSMILES ---
                log_callback("\n🔗 STEP 1: Creating Polymer Structure from PSMILES")
                log_callback("-" * 40)
                psmiles = pdb_file # Assume pdb_file is the PSMILES string
                log_callback(f"🧬 Using PSMILES: {psmiles}")

                polymer_chain_length = kwargs.get('polymer_chain_length', 15)
                
                # This method returns a dictionary containing the polymer_smiles
                polymer_result = self.polymer_builder.build_polymer_chain(
                    psmiles_str=psmiles,
                    chain_length=polymer_chain_length,
                    output_dir=str(output_dir / "polymer_build"),
                )
                
                if not polymer_result or not polymer_result.get('success'):
                    raise RuntimeError(f"Failed to build polymer: {polymer_result.get('error', 'Unknown error')}")

                polymer_smiles = polymer_result['polymer_smiles']
                log_callback(f"✅ Polymer SMILES generated: {polymer_smiles}")

                # --- Step 2: Generate Polymer Topology and Coordinates from SMILES ---
                log_callback("\n🔗 STEP 2: Generating Polymer 3D Structure from SMILES")
                log_callback("-" * 40)
                polymer_molecule = Molecule.from_smiles(polymer_smiles, allow_undefined_stereo=True)
                log_callback("✅ OpenFF Molecule created from SMILES.")
                
                log_callback("🧮 Pre-computing Gasteiger partial charges for the polymer...")
                polymer_molecule.assign_partial_charges("gasteiger")
                log_callback("✅ Gasteiger charges computed and assigned.")
                
                log_callback("🔄 Generating 3D conformer for the polymer...")
                polymer_molecule.generate_conformers(n_conformers=1)
                polymer_topology = polymer_molecule.to_topology().to_openmm()
                
                # CORRECTED: Convert OpenFF Quantity to OpenMM Quantity.
                # The .magnitude attribute gives the raw numpy array, which we then correctly
                # associate with OpenMM's angstrom unit.
                polymer_positions = polymer_molecule.conformers[0].magnitude * unit.angstrom
                log_callback("✅ Polymer OpenMM Topology and Positions created from SMILES.")

                # --- Step 3: Prepare Insulin Structure ---
                log_callback("\n🧬 STEP 3: Preparing Insulin Structure")
                log_callback("-" * 40)
                
                try:
                    # Use importlib.resources for robust path handling in an installed package
                    with importlib.resources.path('insulin_ai.integration.data.insulin', 'output.pdb') as insulin_pdb_path:
                        insulin_pdb_path_str = str(insulin_pdb_path)
                        
                    # The PDB file needs to be in the CWD for Modeller, so we copy it to a temporary file
                    temp_pdb_path = os.path.join(temp_dir, "insulin_for_modeller.pdb")
                    shutil.copy(insulin_pdb_path_str, temp_pdb_path)
                    
                    log_callback(f"🧬 Loading and cleaning insulin from: {insulin_pdb_path_str}")
                    fixer = PDBFixer(filename=temp_pdb_path)
                    fixer.findMissingResidues()
                    fixer.findMissingAtoms()
                    fixer.addMissingAtoms()
                    fixer.addMissingHydrogens(7.4)
                    log_callback("   ✅ Insulin structure cleaned with PDBFixer.")

                except (ModuleNotFoundError, FileNotFoundError):
                    log_callback("❌ CRITICAL: Could not locate the insulin PDB file within the package.")
                    raise

                modeller = Modeller(fixer.topology, fixer.positions)
                log_callback(f"   ✅ Loaded {modeller.topology.getNumResidues()} residues.")

                # --- Step 4: Creating Composite System ---
                log_callback("\n🔗 STEP 4: Creating Composite System")
                log_callback("----------------------------------------")
                
                num_polymer_chains = kwargs.get('num_polymer_chains', 1)
                log_callback(f"🧬 Adding {num_polymer_chains} polymer chain(s) to the modeller.")

                # Get the size of the insulin molecule to create a reasonable packing box
                insulin_positions = np.array([v.value_in_unit(unit.nanometer) for v in modeller.positions])
                
                for i in range(num_polymer_chains):
                    # Create a copy of the polymer positions to modify
                    new_polymer_positions = np.copy(polymer_positions.value_in_unit(unit.nanometer))
                    
                    # Get current system positions to find a clash-free spot
                    current_positions = np.array([v.value_in_unit(unit.nanometer) for v in modeller.positions])
                    
                    # Find a new position using random walk
                    # Start from the insulin positions
                    start_pos = insulin_positions[0]  # Use first insulin position as starting point

                    # Initialize polymer position with random rotation
                    rotation_matrix = np.random.rand(3, 3)
                    q, r = np.linalg.qr(rotation_matrix)
                    rotated_positions = np.dot(new_polymer_positions, q)
                    
                    # Perform random walk until we find a clash-free position
                    current_pos = start_pos
                    step_size = 0.5  # 3 Å steps
                    max_steps = 5000  # Maximum steps to prevent infinite loops
                    
                    for step in range(max_steps):
                        # Random direction for the step
                        direction = np.random.randn(3)
                        direction = direction / np.linalg.norm(direction)  # Normalize
                        
                        # Take step
                        current_pos = current_pos + direction * step_size
                        
                        # Apply translation to rotated polymer
                        translated_positions = rotated_positions + current_pos
                        
                        # Check for clashes with existing atoms
                        min_dist = np.min(np.linalg.norm(current_positions[:, np.newaxis, :] - translated_positions[np.newaxis, :, :], axis=2))
                        
                        if min_dist > 0.5:  # If minimum distance is > 3 Å, we accept the position
                            log_callback(f"   ✅ Found clash-free position for polymer {i+1} after {step+1} random walk steps.")
                            break
                    else:
                        log_callback(f"   ⚠️ Could not find a clash-free position for polymer {i+1} after {max_steps} random walk steps.")

                    # Add the transformed polymer to the modeller
                    modeller.add(polymer_topology, translated_positions * unit.nanometer)
                    log_callback(f"  ✅ Added polymer chain {i+1} of {num_polymer_chains} with random walk positioning.")

                composite_pdb_path = output_dir / "composite_system.pdb"
                with open(composite_pdb_path, 'w') as f:
                    PDBFile.writeFile(modeller.topology, modeller.positions, f)
                log_callback(f"✅ Composite system PDB saved: {composite_pdb_path}")
                log_callback(f"   Total atoms: {modeller.topology.getNumAtoms()}")

                # --- Step 5: Set up Dual Force Field and System ---
                log_callback("\n⚙️ STEP 5: Creating Dual Force Field System")
                log_callback("-" * 40)

                log_callback("🔗 Creating GAFF template generator for the polymer...")
                # Use the SAME polymer_molecule object to ensure a perfect match.
                gaff_template_generator = GAFFTemplateGenerator(molecules=[polymer_molecule])
                log_callback("✅ GAFF template generator created.")

                log_callback("🧬 Applying AMBER ff14SB for protein and implicit solvent...")
                forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                forcefield.registerTemplateGenerator(gaff_template_generator.generator)
                log_callback("✅ AMBER force field loaded and GAFF generator registered.")
                
                log_callback("🔧 Creating OpenMM System...")
                system = forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=NoCutoff,
                    constraints=HBonds
                )
                log_callback(f"✅ System created successfully with {system.getNumParticles()} particles.")

                # --- Step 6: Run MD Simulation ---
                log_callback(f"\n🏃 STEP 6: Running MD Simulation")
                log_callback("-" * 40)
                
                integrator = LangevinMiddleIntegrator(
                    temperature * unit.kelvin,
                    1.0 / unit.picosecond,
                    2.0 * unit.femtoseconds
                )

                # Explicitly select OpenCL platform if available, otherwise fall back to CPU
                platform_names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
                if 'OpenCL' in platform_names:
                    platform_name = 'OpenCL'
                else:
                    log_callback("⚠️ OpenCL platform not found, falling back to CPU.")
                    platform_name = 'CPU'
                
                platform = Platform.getPlatformByName(platform_name)
                log_callback(f"🖥️ Using platform: {platform.getName()}")

                simulation = Simulation(modeller.topology, system, integrator, platform)
                simulation.context.setPositions(modeller.positions)

                log_callback("💫 Minimizing energy...")
                simulation.minimizeEnergy()
                log_callback("✅ Energy minimization complete.")
                # End of Selection
                initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                initial_energy_kj_mol = initial_energy.value_in_unit(unit.kilojoules_per_mole)
                log_callback(f"   Initial energy: {initial_energy_kj_mol:.2f} kJ/mol")
                self.current_simulation['initial_energy'] = initial_energy_kj_mol  # Store initial energy

                log_file = output_dir / "simulation.log"
                trajectory_file = output_dir / "trajectory.pdb"
                
                # Add StateDataReporter to report simulation stats every 100 ps
                # Timestep is 2 fs, so 100 ps = 50,000 steps
                report_interval = 50000 
                simulation.reporters.append(StateDataReporter(
                    str(log_file), report_interval, step=True, time=True, 
                    potentialEnergy=True, temperature=True, speed=True,
                    separator='\t'
                ))
                
                simulation.reporters.append(PDBReporter(str(trajectory_file), save_interval))
                log_callback(f"📊 Reporters configured (log, trajectory).")

                log_callback(f"🔄 Running equilibration ({equilibration_steps} steps)...")
                simulation.step(equilibration_steps)
                
                log_callback(f"🏃 Running production ({production_steps} steps)...")
                simulation.step(production_steps)
                
                final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                final_energy_kj_mol = final_energy.value_in_unit(unit.kilojoules_per_mole)
                log_callback(f"✅ Simulation finished!")
                log_callback(f"   Final energy: {final_energy_kj_mol:.2f} kJ/mol")

                # --- Step 6: Finalize ---
                self.current_simulation['status'] = 'completed'
                self.current_simulation['final_energy'] = final_energy_kj_mol  # Store the actual final energy!
                
                # Store output files for post-processing
                output_files = []
                if trajectory_file.exists():
                    output_files.append(str(trajectory_file))
                if log_file.exists():
                    output_files.append(str(log_file))
                self.current_simulation['output_files'] = output_files
                
                log_callback(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")

        except Exception as e:
            error_msg = f"❌ Dual GAFF+AMBER simulation failed: {str(e)}"
            log_callback(f"\n{error_msg}")
            
            self.current_simulation['status'] = 'failed'
            self.current_simulation['error'] = str(e)
            
            import traceback
            traceback.print_exc()
            
        finally:
            self.simulation_running = False
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status in format expected by UI"""
        
        if self.current_simulation is None:
            return {
                'status': 'no_simulation',
                'simulation_running': False,
                'simulation_info': None
            }
        
        # Return status in the format the UI expects
        return {
            'simulation_running': self.current_simulation.get('status') == 'running' if self.current_simulation else False,
            'simulation_info': self.current_simulation.copy() if self.current_simulation else {}
        }
    
    def wait_for_simulation_completion(self, simulation_id: str, 
                                     output_callback: Optional[Callable] = None,
                                     timeout_minutes: int = 60) -> Dict[str, Any]:
        """Wait for simulation to complete and return results.
        
        Args:
            simulation_id: ID of the simulation to wait for
            output_callback: Optional callback for status messages
            timeout_minutes: Maximum time to wait in minutes
            
        Returns:
            Dict with success status and results
        """
        import time
        
        if output_callback is None:
            output_callback = lambda msg: None
        
        if not self.current_simulation or self.current_simulation.get('id') != simulation_id:
            return {
                'success': False,
                'error': f'No simulation found with ID: {simulation_id}'
            }
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        output_callback(f"⏳ Waiting for simulation {simulation_id} to complete...")
        
        # Wait for simulation to complete
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                return {
                    'success': False,
                    'error': f'Simulation timeout after {timeout_minutes} minutes'
                }
            
            # Check simulation status
            if not self.simulation_running and self.current_simulation:
                status = self.current_simulation.get('status', 'unknown')
                
                if status == 'completed':
                    output_callback(f"✅ Simulation {simulation_id} completed successfully!")
                    return {
                        'success': True,
                        'results': {
                            'simulation_id': simulation_id,
                            'status': status,
                            'total_time_s': elapsed,
                            'final_energy': self.current_simulation.get('final_energy'),
                            'frames_saved': self.current_simulation.get('frames_saved'),
                            'simulation_info': self.current_simulation.copy()
                        }
                    }
                elif status == 'failed':
                    error_msg = self.current_simulation.get('error', 'Unknown error')
                    output_callback(f"❌ Simulation {simulation_id} failed: {error_msg}")
                    return {
                        'success': False,
                        'error': f'Simulation failed: {error_msg}'
                    }
            
            # Wait and check again
            time.sleep(5)  # Check every 5 seconds
            
            # Provide periodic status updates
            if int(elapsed) % 30 == 0 and elapsed > 0:  # Every 30 seconds
                remaining = max(0, timeout_seconds - elapsed)
                output_callback(f"⏳ Still waiting... {remaining/60:.1f} minutes remaining")
    
    def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation results in the format expected by active learning."""
        
        if not self.current_simulation or self.current_simulation.get('id') != simulation_id:
            return {
                'success': False,
                'error': f'No simulation found with ID: {simulation_id}'
            }
        
        # Check if simulation is completed
        status = self.current_simulation.get('status', 'unknown')
        
        if status == 'completed':
            # Return results in the format expected by post-processing
            return {
                'success': True,
                'simulation_id': simulation_id,
                'initial_energy': self.current_simulation.get('initial_energy'),
                'final_energy': self.current_simulation.get('final_energy'),
                'frames_saved': self.current_simulation.get('frames_saved'),
                'output_files': self.current_simulation.get('output_files', []),
                'simulation_info': self.current_simulation.copy()
            }
        elif status == 'failed':
            return {
                'success': False,
                'error': self.current_simulation.get('error', 'Simulation failed')
            }
        else:
            return {
                'success': False,
                'error': f'Simulation not completed yet. Status: {status}'
            }

    def is_simulation_running(self) -> bool:
        """Check if a simulation is currently running"""
        return self.simulation_running and self.current_simulation and self.current_simulation.get('status') == 'running'
    
    def get_automated_simulation_candidates(self, base_dir: str = "automated_simulations") -> List[Dict[str, Any]]:
        """
        Get list of candidates generated by automation pipeline for UI compatibility
        
        Args:
            base_dir: Base directory where automated simulations are stored
            
        Returns:
            List of candidate info dictionaries
        """
        candidates = []
        
        try:
            from pathlib import Path
            from datetime import datetime
            
            base_path = Path(base_dir)
            
            if not base_path.exists():
                print(f"⚠️ Automated simulations directory not found: {base_dir}")
                return []
            
            # Look for session directories
            for session_dir in base_path.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith('session_'):
                    session_id = session_dir.name
                    
                    # Look for session-level automation_results.json
                    results_file = session_dir / "automation_results.json"
                    
                    if results_file.exists():
                        try:
                            import json
                            with open(results_file, 'r') as f:
                                results = json.load(f)
                            
                            # Extract candidates from polymer_boxes section
                            polymer_boxes = results.get('polymer_boxes', [])
                            
                            for polymer_box in polymer_boxes:
                                if polymer_box.get('success'):
                                    candidate_id = polymer_box.get('candidate_id', 'unknown')
                                    candidate_dir_name = f"candidate_{candidate_id}"
                                    candidate_dir_path = session_dir / candidate_dir_name
                                    
                                    candidate_info = {
                                        'candidate_id': candidate_id,
                                        'session_id': session_id,
                                        'candidate_dir': str(candidate_dir_path),
                                        'name': f"disk_candidate_{candidate_id}",
                                        'psmiles': polymer_box.get('psmiles', ''),
                                        'smiles': polymer_box.get('polymer_smiles', ''),
                                        'timestamp': polymer_box.get('timestamp', ''),
                                        'source': 'dual_gaff_amber_automation',
                                        'status': 'ready_for_simulation',
                                        'ready_for_md': True,  # CRITICAL: Add this flag for UI filtering
                                        'polymer_pdb': polymer_box.get('polymer_pdb', ''),
                                        'parameters': polymer_box.get('parameters', {})
                                    }
                                    
                                    candidates.append(candidate_info)
                                    print(f"✅ Found automation candidate: {candidate_info['name']} ({candidate_info['psmiles'][:30]}...)")
                                    
                        except Exception as e:
                            print(f"⚠️ Error loading session results from {results_file}: {e}")
                            continue
            
            print(f"🔍 Found {len(candidates)} automation candidates for dual GAFF+AMBER")
            return candidates
            
        except Exception as e:
            print(f"❌ Error scanning for automation candidates: {e}")
            return []
    
    def stop_simulation(self) -> bool:
        """Stop the current simulation"""
        try:
            if self.simulation_thread and self.simulation_thread.is_alive():
                # Note: Python threads can't be forcibly stopped, but we can mark it as stopped
                if self.current_simulation:
                    self.current_simulation['status'] = 'stopped'
                print("🛑 Simulation stop requested")
                return True
            return False
        except Exception as e:
            print(f"❌ Error stopping simulation: {e}")
            return False
    
    def get_available_simulations(self) -> List[Dict[str, Any]]:
        """Get list of available completed simulations"""
        simulations = []
        
        try:
            # Look for simulation directories in the output directory
            if not self.output_dir.exists():
                return []
            
            for sim_dir in self.output_dir.iterdir():
                if sim_dir.is_dir():
                    # Look for trajectory file to confirm it's a completed simulation
                    trajectory_file = sim_dir / 'trajectory.pdb'
                    log_file = sim_dir / 'simulation.log'
                    
                    if trajectory_file.exists():
                        # Extract simulation info
                        sim_info = {
                            'id': sim_dir.name,
                            'timestamp': datetime.fromtimestamp(sim_dir.stat().st_mtime).isoformat(),
                            'input_file': 'dual_gaff_amber_system',
                            'total_atoms': self._estimate_atoms_from_trajectory(trajectory_file),
                            'performance': 1.0,  # Default value
                            'success': True,
                            'force_field': 'Dual GAFF+AMBER'
                        }
                        
                        # Try to get more info from log file if available
                        if log_file.exists():
                            log_info = self._parse_simulation_log(log_file)
                            sim_info.update(log_info)
                        
                        simulations.append(sim_info)
            
            # Sort by timestamp (newest first)
            simulations.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            print(f"Error getting available simulations: {e}")
        
        return simulations
    
    def get_simulation_files(self, simulation_id: str) -> Dict[str, Any]:
        """Get files for a specific simulation"""
        try:
            sim_dir = self.output_dir / simulation_id
            
            if not sim_dir.exists():
                return {'success': False, 'error': f'Simulation directory not found: {sim_dir}'}
            
            # Look for simulation files
            files = {}
            
            # Common dual GAFF+AMBER output files
            file_patterns = {
                'trajectory.pdb': 'trajectory',
                'simulation.log': 'log',
                'final_system.pdb': 'final_structure',
                'equilibration.pdb': 'equilibration',
                'production.pdb': 'production'
            }
            
            for filename, file_type in file_patterns.items():
                file_path = sim_dir / filename
                if file_path.exists():
                    files[file_type] = str(file_path)
            
            # Also scan for any additional PDB or log files
            for file_path in sim_dir.glob('*.pdb'):
                if file_path.name not in file_patterns:
                    files[f'pdb_{file_path.stem}'] = str(file_path)
            
            for file_path in sim_dir.glob('*.log'):
                if file_path.name not in file_patterns:
                    files[f'log_{file_path.stem}'] = str(file_path)
            
            return {
                'success': True,
                'files': files,
                'simulation_dir': str(sim_dir)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _estimate_atoms_from_trajectory(self, trajectory_file: Path) -> int:
        """Estimate number of atoms from trajectory file"""
        try:
            with open(trajectory_file, 'r') as f:
                lines = f.readlines()
            
            atom_count = 0
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    atom_count += 1
                elif line.startswith('ENDMDL'):
                    break  # Only count first frame
            
            return atom_count
        except:
            return 0
    
    def _parse_simulation_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse simulation log file for additional info"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            info = {}
            
            # Try to extract performance info
            lines = content.split('\n')
            for line in lines:
                if 'performance' in line.lower() or 'ns/day' in line.lower():
                    try:
                        # Try to extract numerical value
                        import re
                        numbers = re.findall(r'\d+\.?\d*', line)
                        if numbers:
                            info['performance'] = float(numbers[0])
                    except:
                        pass
            
            return info
        except:
            return {} 