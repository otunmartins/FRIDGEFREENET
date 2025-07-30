#!/usr/bin/env python3
"""
Simple Working MD Simulator
===========================

This module implements the EXACT approach from the working openmm_test.py script:
1. RDKit PDB → SMILES conversion
2. OpenFF SMILES → Molecule + Gasteiger charges  
3. GAFF Template generator
4. Force field: amber/protein.ff14SB.xml + implicit/gbn2.xml
5. Load pre-processed composite system
6. Run simulation with implicit solvent

This WORKS - unlike the over-engineered complex system.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime

import numpy as np

# RDKit imports
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available")

# OpenFF imports
try:
    from openff.toolkit import Molecule
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    print("⚠️  OpenFF toolkit not available")

# OpenMM imports
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, ForceField, Simulation
    from openmm.app import StateDataReporter, PDBReporter
    from openmm import LangevinIntegrator, Platform
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("⚠️  OpenMM not available")

# OpenMMForceFields imports
try:
    from openmmforcefields.generators import GAFFTemplateGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False
    print("⚠️  openmmforcefields not available")


class CallbackStateReporter:
    """Custom state reporter that sends output through callback instead of console"""
    
    def __init__(self, callback: Callable, reportInterval: int):
        self._callback = callback
        self._reportInterval = reportInterval
        self._hasInitialized = False
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needsEnergy = True
        
    def describeNextReport(self, simulation):
        steps_left = simulation.currentStep % self._reportInterval
        steps = self._reportInterval - steps_left
        return (steps, False, False, False, True, None)
        
    def report(self, simulation, state):
        if not self._hasInitialized:
            self._hasInitialized = True
            
        step = simulation.currentStep
        time_ps = state.getTime().value_in_unit(unit.picosecond)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        
        # Proper temperature calculation using degrees of freedom
        num_dof = 3 * simulation.system.getNumParticles() - simulation.system.getNumConstraints()
        if num_dof > 0:
            # Convert kJ/mol to J/mol (×1000) and use R = 8.314 J/(mol·K)
            temp = (2 * ke * 1000) / (num_dof * 8.314)
        else:
            temp = 0.0
        
        if self._callback:
            try:
                message = f"📊 Step {step:8d} | Time: {time_ps:8.1f} ps | PE: {pe:10.1f} kJ/mol | T: {temp:6.1f} K"
                self._callback(message)
            except Exception as e:
                # Fallback: print to console if callback fails
                print(f"[CALLBACK_ERROR] {e}: {message}")
        else:
            # Fallback: print to console if no callback
            message = f"📊 Step {step:8d} | Time: {time_ps:8.1f} ps | PE: {pe:10.1f} kJ/mol | T: {temp:6.1f} K"
            print(message)


class SimpleWorkingMDSimulator:
    """
    Simple MD simulator that uses the EXACT working approach.
    
    Based on the successful openmm_test.py pattern:
    - RDKit for SMILES generation
    - GAFF for small molecule parameterization  
    - Implicit solvent (GBn2)
    - No complex fallbacks or over-engineering
    """
    
    def __init__(self, output_dir: str = "simple_md_simulations"):
        """Initialize the simple working simulator"""
        
        # Check required dependencies
        missing_deps = []
        if not RDKIT_AVAILABLE:
            missing_deps.append("rdkit")
        if not OPENFF_AVAILABLE:
            missing_deps.append("openff-toolkit")
        if not OPENMM_AVAILABLE:
            missing_deps.append("openmm")
        if not OPENMMFORCEFIELDS_AVAILABLE:
            missing_deps.append("openmmforcefields")
            
        if missing_deps:
            raise ImportError(f"Missing required dependencies: {missing_deps}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get best platform
        self.platform = self._get_best_platform()
        
        print(f"🚀 Simple Working MD Simulator initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Platform: {self.platform.getName()}")
    
    def _get_best_platform(self) -> mm.Platform:
        """Get the best available platform"""
        platform_names = [mm.Platform.getPlatform(i).getName() 
                          for i in range(mm.Platform.getNumPlatforms())]
        
        print("🔍 Available OpenMM platforms:")
        for name in platform_names:
            print(f"   • {name}")
        
        for preferred in ['CUDA', 'OpenCL', 'CPU', 'Reference']:
            if preferred in platform_names:
                platform = mm.Platform.getPlatformByName(preferred)
                print(f"✅ Selected platform: {preferred}")
                return platform
        
        raise RuntimeError("No suitable OpenMM platform found")
    
    def create_polymer_force_field(self, polymer_pdb_path: str) -> GAFFTemplateGenerator:
        """
        Create GAFF template generator for polymer using EXACT working approach.
        
        This follows the working script exactly:
        1. RDKit: Load PDB → Generate SMILES
        2. OpenFF: SMILES → Molecule 
        3. Assign Gasteiger charges
        4. Create GAFF template generator
        """
        print(f"\n🧪 Creating polymer force field from: {polymer_pdb_path}")
        
        # Step 1: RDKit PDB → SMILES (EXACT working approach)
        print("📋 Step 1: RDKit PDB → SMILES conversion")
        try:
            rdkit_mol = Chem.MolFromPDBFile(polymer_pdb_path)
            if rdkit_mol is None:
                raise ValueError(f"RDKit failed to read PDB: {polymer_pdb_path}")
            
            smiles = Chem.MolToSmiles(rdkit_mol)
            print(f"✅ Generated SMILES: {smiles}")
            
        except Exception as e:
            raise RuntimeError(f"RDKit SMILES generation failed: {e}")
        
        # Step 2: OpenFF SMILES → Molecule (EXACT working approach)
        print("📋 Step 2: OpenFF SMILES → Molecule")
        try:
            molecule = Molecule.from_smiles(smiles)
            print(f"✅ OpenFF molecule created: {molecule.n_atoms} atoms")
            
        except Exception as e:
            raise RuntimeError(f"OpenFF molecule creation failed: {e}")
        
        # Step 3: Assign Gasteiger charges (EXACT working approach)
        print("📋 Step 3: Assigning Gasteiger charges")
        try:
            molecule.assign_partial_charges("gasteiger")
            print(f"✅ Gasteiger charges assigned")
            
        except Exception as e:
            raise RuntimeError(f"Gasteiger charge assignment failed: {e}")
        
        # Step 4: Create GAFF template generator (EXACT working approach)
        print("📋 Step 4: Creating GAFF template generator")
        try:
            gaff = GAFFTemplateGenerator(molecules=molecule)
            print(f"✅ GAFF template generator created")
            return gaff
            
        except Exception as e:
            raise RuntimeError(f"GAFF template generator creation failed: {e}")
    
    def create_force_field(self, gaff_generator: GAFFTemplateGenerator) -> ForceField:
        """
        Create force field using EXACT working approach.
        
        Force field combination from working script:
        - amber/protein.ff14SB.xml (protein)
        - implicit/gbn2.xml (implicit solvent)
        """
        print(f"\n🔧 Creating force field (EXACT working approach)")
        
        try:
            # EXACT force field from working script
            forcefield = ForceField(
                "amber/protein.ff14SB.xml",  # Protein force field
                "implicit/gbn2.xml",         # Generalized Born implicit solvent
            )
            
            # Register GAFF template generator
            forcefield.registerTemplateGenerator(gaff_generator.generator)
            
            print(f"✅ Force field created with GAFF template generator")
            return forcefield
            
        except Exception as e:
            raise RuntimeError(f"Force field creation failed: {e}")
    
    def create_system(self, forcefield: ForceField, topology, 
                     temperature: float = 310.0) -> mm.System:
        """
        Create OpenMM system using EXACT working approach.
        
        System parameters from working script:
        - nonbondedMethod=app.NoCutoff (no PBC)
        - solventDielectric=78.5 (water)
        - soluteDielectric=1.0 (protein/solute)
        - constraints=app.HBonds
        - rigidWater=False (not applicable)
        - removeCMMotion=True
        """
        print(f"\n⚙️  Creating system (EXACT working approach)")
        
        # Check for UNL residues and report them
        unl_residues = []
        for residue in topology.residues():
            if residue.name == 'UNL':
                unl_residues.append(residue)
        
        if unl_residues:
            print(f"⚠️  Found {len(unl_residues)} UNL residues - these should be handled by GAFF")
            print(f"   • The GAFF template generator should parameterize these automatically")
        
        try:
            # EXACT system creation from working script
            system = forcefield.createSystem(
                topology,
                nonbondedMethod=app.NoCutoff,    # No cutoff for implicit solvent
                solventDielectric=78.5,          # Water dielectric constant
                soluteDielectric=1.0,            # Protein/solute dielectric
                constraints=app.HBonds,          # Constrain H-bonds
                rigidWater=False,                # Not applicable for implicit
                removeCMMotion=True              # Remove center of mass motion
            )
            
            print(f"✅ System created with implicit solvent")
            print(f"   • Nonbonded method: NoCutoff")
            print(f"   • Solvent dielectric: 78.5")
            print(f"   • Solute dielectric: 1.0")
            print(f"   • Constraints: HBonds")
            
            return system
            
        except Exception as e:
            print(f"❌ System creation failed: {e}")
            
            # If UNL residues are the problem, provide helpful error message
            if unl_residues and "UNL" in str(e):
                print(f"\n💡 SOLUTION: The UNL residues need to be properly mapped to the polymer molecule.")
                print(f"   This suggests the polymer SMILES doesn't exactly match the UNL structure in the PDB.")
                print(f"   In the working script, this was handled by using a compatible composite PDB.")
                
                # Try to find a compatible composite file
                composite_dir = Path("automated_simulations")
                print(f"\n🔍 Looking for alternative composite files...")
                for pdb_file in composite_dir.rglob("*_preprocessed.pdb"):
                    if "insulin" in str(pdb_file).lower():
                        print(f"   • Found: {pdb_file}")
                        
            raise RuntimeError(f"System creation failed: {e}")
    
    def run_simulation(self, 
                      polymer_pdb_path: str,
                      composite_pdb_path: str,
                      temperature: float = 310.0,
                      equilibration_steps: int = 125000,   # Quick Test: 250 ps with 2 fs timestep (user requested quick mode)
                      production_steps: int = 500000,      # Quick Test: 1 ns with 2 fs timestep (was 2500000 = 5 ns)
                      save_interval: int = 1000,
                      output_prefix: str = None,
                      output_callback: Optional[Callable] = None,
                      stop_condition_check: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run MD simulation using EXACT working approach.
        
        This follows the working script step by step:
        1. Create polymer force field from polymer PDB
        2. Create complete force field 
        3. Load pre-processed composite system
        4. Create system with implicit solvent
        5. Set up simulation with Langevin integrator
        6. Energy minimization
        7. MD simulation with reporting
        """
        
        # Generate output prefix
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"simple_md_{timestamp}"
        
        # Create output directory for this simulation
        sim_output_dir = self.output_dir / output_prefix
        sim_output_dir.mkdir(exist_ok=True)
        
        def log_message(msg: str):
            """Helper to log messages (callback handled by CallbackStateReporter)"""
            print(msg)
        
        log_message(f"\n{'='*80}")
        log_message(f"🚀 SIMPLE WORKING MD SIMULATION STARTING")
        log_message(f"{'='*80}")
        log_message(f"📁 Output directory: {sim_output_dir}")
        log_message(f"🧪 Polymer PDB: {polymer_pdb_path}")
        log_message(f"🧬 Composite PDB: {composite_pdb_path}")
        log_message(f"🌡️  Temperature: {temperature} K")
        log_message(f"🔄 Equilibration steps: {equilibration_steps} ({equilibration_steps * 2 / 1000:.1f} ps)")
        log_message(f"🏃 Production steps: {production_steps} ({production_steps * 2 / 1000000:.1f} ns)")
        log_message(f"💾 Save interval: {save_interval} steps ({save_interval * 2 / 1000:.1f} ps)")
        
        start_time = time.time()
        
        try:
            # Step 1: Create polymer force field (EXACT working approach)
            log_message(f"\n📋 STEP 1: Creating polymer force field")
            gaff_generator = self.create_polymer_force_field(polymer_pdb_path)
            
            # Step 2: Create complete force field (EXACT working approach)
            log_message(f"\n📋 STEP 2: Creating complete force field")
            forcefield = self.create_force_field(gaff_generator)
            
            # Step 3: Load pre-processed composite system (EXACT working approach)
            log_message(f"\n📋 STEP 3: Loading pre-processed composite system")
            pdbfile = PDBFile(composite_pdb_path)
            log_message(f"✅ Loaded composite system: {pdbfile.topology.getNumAtoms()} atoms")
            
            # Step 4: Create system (EXACT working approach)
            log_message(f"\n📋 STEP 4: Creating OpenMM system")
            system = self.create_system(forcefield, pdbfile.topology, temperature)
            
            # Step 5: Set up simulation (EXACT working approach)
            log_message(f"\n📋 STEP 5: Setting up simulation")
            
            # EXACT integrator from working script
            integrator = LangevinIntegrator(
                temperature * unit.kelvin,      # Temperature
                1.0 / unit.picosecond,         # Friction coefficient  
                2.0 * unit.femtosecond         # Timestep
            )
            
            # Create simulation
            simulation = Simulation(pdbfile.topology, system, integrator, self.platform)
            simulation.context.setPositions(pdbfile.positions)
            
            log_message(f"✅ Simulation created")
            log_message(f"   • Integrator: Langevin")
            log_message(f"   • Temperature: {temperature} K")
            log_message(f"   • Friction: 1.0 ps⁻¹")
            log_message(f"   • Timestep: 2.0 fs")
            log_message(f"   • Platform: {self.platform.getName()}")
            
            # Step 6: Energy minimization (EXACT working approach)
            log_message(f"\n📋 STEP 6: Energy minimization")
            simulation.minimizeEnergy(maxIterations=1000)
            
            state = simulation.context.getState(getEnergy=True)
            minimized_energy = state.getPotentialEnergy()
            log_message(f"✅ Energy minimization completed")
            log_message(f"   • Minimized energy: {minimized_energy}")
            
            # Step 7: Set up reporters (EXACT working approach)
            log_message(f"\n📋 STEP 7: Setting up reporters")
            
            # Calculate reporting intervals
            # With 2 fs timestep: 100 ps = 50,000 steps
            timestep_fs = 2.0  # femtoseconds
            report_interval_ps = 100.0  # picoseconds  
            report_interval_steps = int(report_interval_ps * 1000 / timestep_fs)  # 50,000 steps
            
            log_message(f"⏱️  Reporting every {report_interval_steps} steps ({report_interval_ps} ps)")
            
            # PDB trajectory reporter
            trajectory_file = str(sim_output_dir / f"{output_prefix}_trajectory.pdb")
            pdb_reporter = PDBReporter(trajectory_file, save_interval)
            simulation.reporters.append(pdb_reporter)
            
            # State data reporter (to file)
            log_file = str(sim_output_dir / f"{output_prefix}_log.txt")
            state_reporter = StateDataReporter(
                log_file,
                report_interval_steps,  # Report every 100 ps
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                separator='\t'
            )
            simulation.reporters.append(state_reporter)
            
            # Custom callback reporter for app interface (instead of console reporter)
            if output_callback:
                log_message(f"🔧 Setting up callback reporter with callback: {type(output_callback)}")
                callback_reporter = CallbackStateReporter(output_callback, report_interval_steps)
                simulation.reporters.append(callback_reporter)
                log_message(f"✅ Callback reporter configured for app interface")
                
                # Test the callback immediately
                try:
                    test_message = "🧪 TEST: Callback reporter is working correctly"
                    output_callback(test_message)
                    log_message(f"✅ Callback test successful")
                except Exception as e:
                    log_message(f"❌ Callback test failed: {e}")
            else:
                log_message(f"⚠️ No output_callback provided, using console reporter")
                # Console reporter only if no callback
                console_reporter = StateDataReporter(
                    None,  # Output to console
                    report_interval_steps,   # Report every 100 ps
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    temperature=True,
                    separator='\t'
                )
                simulation.reporters.append(console_reporter)
            
            log_message(f"✅ Reporters configured")
            log_message(f"   • Trajectory: {trajectory_file}")
            log_message(f"   • Log file: {log_file}")
            log_message(f"   • Output frequency: Every {report_interval_ps} ps")
            
            # Step 8: Run equilibration (if requested)
            if equilibration_steps > 0:
                log_message(f"\n📋 STEP 8: Equilibration simulation")
                log_message(f"🔄 Running {equilibration_steps} equilibration steps...")
                
                eq_start = time.time()
                
                # Run equilibration in chunks to allow for stopping
                chunk_size = 10000  # Larger chunks, check stop condition every 10,000 steps (20 ps)
                steps_completed = 0
                
                while steps_completed < equilibration_steps:
                    # Check if we should stop
                    if stop_condition_check and stop_condition_check():
                        log_message(f"🛑 Equilibration stopped by user at step {steps_completed}")
                        return {
                            'success': False,
                            'message': 'Simulation stopped by user during equilibration',
                            'steps_completed': steps_completed,
                            'phase': 'equilibration'
                        }
                    
                    # Calculate steps for this chunk
                    steps_this_chunk = min(chunk_size, equilibration_steps - steps_completed)
                    simulation.step(steps_this_chunk)
                    steps_completed += steps_this_chunk
                    
                    # Less frequent progress updates during equilibration
                    if steps_completed % (chunk_size * 5) == 0 or steps_completed >= equilibration_steps:
                        progress = (steps_completed / equilibration_steps) * 100
                        elapsed = time.time() - eq_start
                        time_ps = steps_completed * timestep_fs / 1000
                        log_message(f"   Equilibration progress: {steps_completed}/{equilibration_steps} ({progress:.1f}%) - {time_ps:.1f} ps")
                
                eq_time = time.time() - eq_start
                log_message(f"✅ Equilibration completed in {eq_time:.1f} seconds")
            
            # Step 9: Run production simulation (EXACT working approach)
            log_message(f"\n📋 STEP 9: Production simulation")
            log_message(f"🔄 Running {production_steps} production steps...")
            
            prod_start = time.time()
            
            # Run production in chunks to allow for stopping
            chunk_size = 10000  # Larger chunks, check stop condition every 10,000 steps (20 ps)
            steps_completed = 0
            
            while steps_completed < production_steps:
                # Check if we should stop
                if stop_condition_check and stop_condition_check():
                    log_message(f"🛑 Production stopped by user at step {steps_completed}")
                    # Get current state before stopping
                    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
                    final_pe = final_state.getPotentialEnergy()
                    
                    total_time = time.time() - start_time
                    
                    return {
                        'success': False,
                        'message': 'Simulation stopped by user during production',
                        'steps_completed': steps_completed,
                        'phase': 'production',
                        'final_energy': final_pe,
                        'total_time': total_time,
                        'trajectory_file': trajectory_file
                    }
                
                # Calculate steps for this chunk
                steps_this_chunk = min(chunk_size, production_steps - steps_completed)
                simulation.step(steps_this_chunk)
                steps_completed += steps_this_chunk
                
                # Less frequent progress updates during production  
                if steps_completed % (chunk_size * 5) == 0 or steps_completed >= production_steps:
                    progress = (steps_completed / production_steps) * 100
                    elapsed = time.time() - prod_start
                    time_ps = steps_completed * timestep_fs / 1000
                    # Show both production progress and total simulation progress
                    total_simulation_steps = equilibration_steps + steps_completed
                    total_progress = (total_simulation_steps / (equilibration_steps + production_steps)) * 100
                    log_message(f"   Production: {steps_completed}/{production_steps} ({progress:.1f}%) - Total: {total_simulation_steps}/{equilibration_steps + production_steps} ({total_progress:.1f}%) - {time_ps:.1f} ps")
            
            prod_time = time.time() - prod_start
            
            # Final state
            final_state = simulation.context.getState(getEnergy=True, getPositions=True)
            final_pe = final_state.getPotentialEnergy()
            final_positions = final_state.getPositions()
            
            total_time = time.time() - start_time
            
            log_message(f"✅ Production simulation completed in {prod_time:.1f} seconds")
            log_message(f"   • Final potential energy: {final_pe}")
            
            # Simulation results
            results = {
                'success': True,
                'output_prefix': output_prefix,
                'output_directory': str(sim_output_dir),
                'trajectory_file': trajectory_file,
                'log_file': log_file,
                'final_energy': final_pe,
                'total_time': total_time,
                'equilibration_time': eq_time if equilibration_steps > 0 else 0.0,
                'production_time': prod_time,
                'equilibration_steps': equilibration_steps,
                'production_steps': production_steps,
                'platform': self.platform.getName(),
                'temperature': temperature,
                'approach_used': 'simple_working_approach'
            }
            
            log_message(f"\n{'='*80}")
            log_message(f"🎉 SIMULATION COMPLETED SUCCESSFULLY!")
            log_message(f"{'='*80}")
            log_message(f"📊 Results:")
            log_message(f"   • Total time: {total_time:.1f} seconds")
            log_message(f"   • Final energy: {final_pe}")
            log_message(f"   • Trajectory: {trajectory_file}")
            log_message(f"   • Log file: {log_file}")
            log_message(f"   • Approach: Simple Working Method")
            
            return results
            
        except Exception as e:
            error_msg = f"❌ Simulation failed: {str(e)}"
            log_message(error_msg)
            
            # Return error results
            return {
                'success': False,
                'error': str(e),
                'output_prefix': output_prefix,
                'output_directory': str(sim_output_dir),
                'total_time': time.time() - start_time,
                'approach_used': 'simple_working_approach'
            }


def test_simple_simulator():
    """Test the simple simulator with working files"""
    
    print("🧪 Testing Simple Working MD Simulator")
    
    # Test files from the working example
    polymer_pdb = "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb"
    composite_pdb = "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/insulin_polymer_composite_001_e36e15_preprocessed.pdb"
    
    if not os.path.exists(polymer_pdb):
        print(f"❌ Polymer PDB not found: {polymer_pdb}")
        return
    
    if not os.path.exists(composite_pdb):
        print(f"❌ Composite PDB not found: {composite_pdb}")
        return
    
    try:
        simulator = SimpleWorkingMDSimulator("test_simple_md")
        
        results = simulator.run_simulation(
            polymer_pdb_path=polymer_pdb,
            composite_pdb_path=composite_pdb,
            temperature=310.0,
            equilibration_steps=5000,   # Short test
            production_steps=10000,     # Short test
            save_interval=1000
        )
        
        if results['success']:
            print(f"🎉 Test SUCCESSFUL!")
            print(f"📁 Results in: {results['output_directory']}")
        else:
            print(f"❌ Test FAILED: {results['error']}")
            
    except Exception as e:
        print(f"❌ Test FAILED with exception: {e}")


if __name__ == "__main__":
    test_simple_simulator() 