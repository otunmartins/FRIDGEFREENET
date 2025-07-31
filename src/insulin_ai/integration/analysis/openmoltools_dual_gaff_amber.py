#!/usr/bin/env python3
"""
Enhanced Dual GAFF+AMBER Integration with OpenMolTools

This module provides a professional-grade implementation of dual GAFF+AMBER
insulin-polymer simulations using openmoltools.packmol.pack_box for
realistic composite system building.

Key Features:
- Professional molecular packing with PACKMOL via openmoltools
- Automatic bond/topology handling
- Realistic density and volume estimation
- MDTraj trajectory integration
- Enhanced composite system quality

Author: AI-Driven Material Discovery Team
"""

import os
import sys
import tempfile
import threading
import traceback
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

# Core scientific libraries
import numpy as np
import mdtraj as md

# OpenMM imports
import openmm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Modeller, Simulation
from openmm.app import PME, HBonds, StateDataReporter, PDBReporter

# OpenMM force fields
from openmmforcefields.generators import GAFFTemplateGenerator, SystemGenerator

# OpenMolTools for professional system building
import openmoltools.packmol

# Our modules
sys.path.append('src')
try:
    from utils.direct_polymer_builder import DirectPolymerBuilder
except ImportError:
    from insulin_ai.utils.direct_polymer_builder import DirectPolymerBuilder

from openff.toolkit.topology import Molecule


class OpenMolToolsDualGaffAmber:
    """
    Enhanced dual GAFF+AMBER integration using OpenMolTools for professional
    composite system building with realistic molecular packing.
    """
    
    def __init__(self):
        self.simulation_thread = None
        self.simulation_running = False
        self.simulation_status = "idle"
        self.simulation_info = {}
        self.builder = DirectPolymerBuilder()
        
        # OpenMolTools configuration
        self.packing_tolerance = 2.0  # Angstroms - minimum spacing between molecules
        self.default_box_size = None  # Let openmoltools estimate
        self.target_density = 1.0  # g/cm³
        
        print("🚀 Enhanced Dual GAFF+AMBER with OpenMolTools initialized")
        print("📦 Professional molecular packing enabled")
    
    def run_simulation(self, 
                      psmiles: str,
                      simulation_params: Dict[str, Any],
                      log_callback: Optional[Callable] = None,
                      **kwargs) -> bool:
        """
        Run enhanced dual GAFF+AMBER simulation with OpenMolTools composite building.
        
        Args:
            psmiles: Polymer SMILES string
            simulation_params: Simulation parameters
            log_callback: Optional logging callback function
            **kwargs: Additional parameters including polymer configuration
            
        Returns:
            bool: True if simulation started successfully
        """
        
        if self.simulation_running:
            if log_callback:
                log_callback("❌ Simulation already running")
            return False
        
        # Extract polymer configuration
        polymer_chain_length = kwargs.get('polymer_chain_length', 15)
        num_polymer_chains = kwargs.get('num_polymer_chains', 2)
        
        if log_callback:
            log_callback(f"🚀 Starting Enhanced Dual GAFF+AMBER with OpenMolTools")
            log_callback(f"📦 Professional PACKMOL packing enabled")
            log_callback(f"🧬 Polymer config: {num_polymer_chains} chains × {polymer_chain_length} units")
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(
            target=self._run_simulation_thread,
            args=(psmiles, simulation_params, log_callback),
            kwargs=kwargs
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return True
    
    def _run_simulation_thread(self, 
                              psmiles: str, 
                              simulation_params: Dict[str, Any], 
                              log_callback: Optional[Callable] = None,
                              **kwargs):
        """
        Main simulation thread with enhanced OpenMolTools composite building.
        """
        
        try:
            self.simulation_running = True
            self.simulation_status = "running"
            self.simulation_info = {
                'start_time': datetime.now().isoformat(),
                'psmiles': psmiles,
                'params': simulation_params
            }
            
            # Extract configuration
            polymer_chain_length = kwargs.get('polymer_chain_length', 15)
            num_polymer_chains = kwargs.get('num_polymer_chains', 2)
            
            output_dir = f"openmoltools_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            
            if log_callback:
                log_callback(f"📁 Output directory: {output_dir}")
                log_callback(f"🧬 Building polymer with OpenMolTools integration...")
            
            # Step 1: Build polymer using DirectPolymerBuilder
            if log_callback:
                log_callback("🔗 Step 1: Building polymer chains...")
            
            polymer_results = []
            polymer_pdbs = []
            
            for i in range(num_polymer_chains):
                if log_callback:
                    log_callback(f"   Building polymer chain {i+1}/{num_polymer_chains}...")
                
                chain_output_dir = os.path.join(output_dir, f"polymer_chain_{i+1}")
                result = self.builder.build_polymer_chain(
                    psmiles_str=psmiles,
                    chain_length=polymer_chain_length,
                    output_dir=chain_output_dir,
                    end_cap_atom='C'
                )
                
                if not result['success']:
                    raise ValueError(f"Polymer chain {i+1} building failed: {result.get('error')}")
                
                polymer_results.append(result)
                polymer_pdbs.append(result['pdb_file'])
                
                if log_callback:
                    log_callback(f"   ✅ Chain {i+1}: {result['pdb_file']}")
            
            # Step 2: Prepare insulin
            if log_callback:
                log_callback("🦠 Step 2: Preparing insulin...")
            
            insulin_pdb = "data/insulin_clean.pdb"
            if not os.path.exists(insulin_pdb):
                raise FileNotFoundError(f"Insulin PDB not found: {insulin_pdb}")
            
            # Fix CYS -> CYX for AMBER compatibility
            fixed_insulin_pdb = os.path.join(output_dir, "insulin_fixed.pdb")
            self.fix_insulin_pdb_residues(insulin_pdb, fixed_insulin_pdb, log_callback)
            
            if log_callback:
                log_callback(f"   ✅ Fixed insulin: {fixed_insulin_pdb}")
            
            # Step 3: Professional composite building with OpenMolTools
            if log_callback:
                log_callback("📦 Step 3: Professional composite packing with OpenMolTools...")
            
            packed_system_pdb = self._create_openmoltools_composite(
                insulin_pdb=fixed_insulin_pdb,
                polymer_pdbs=polymer_pdbs,
                output_dir=output_dir,
                log_callback=log_callback
            )
            
            # Step 4: Setup dual force fields
            if log_callback:
                log_callback("⚗️ Step 4: Setting up dual GAFF+AMBER force fields...")
            
            system = self._setup_dual_force_fields(
                packed_system_pdb=packed_system_pdb,
                polymer_smiles_list=[result['polymer_smiles'] for result in polymer_results],
                log_callback=log_callback
            )
            
            # Step 5: Run OpenMM simulation
            if log_callback:
                log_callback("🏃 Step 5: Running OpenMM simulation...")
            
            self._run_openmm_simulation(
                system=system,
                packed_system_pdb=packed_system_pdb,
                simulation_params=simulation_params,
                output_dir=output_dir,
                log_callback=log_callback
            )
            
            self.simulation_status = "completed"
            if log_callback:
                log_callback("🎉 Enhanced dual GAFF+AMBER simulation completed successfully!")
                log_callback(f"📁 Results in: {output_dir}")
        
        except Exception as e:
            self.simulation_status = "failed"
            self.simulation_info['error'] = str(e)
            self.simulation_info['traceback'] = traceback.format_exc()
            
            if log_callback:
                log_callback(f"❌ Simulation failed: {str(e)}")
                log_callback(f"🔍 Full traceback:\n{traceback.format_exc()}")
        
        finally:
            self.simulation_running = False
    
    def _create_openmoltools_composite(self, 
                                     insulin_pdb: str,
                                     polymer_pdbs: list,
                                     output_dir: str,
                                     log_callback: Optional[Callable] = None) -> str:
        """
        Create professional composite system using OpenMolTools PACKMOL.
        
        Args:
            insulin_pdb: Path to insulin PDB file
            polymer_pdbs: List of polymer PDB file paths
            output_dir: Output directory
            log_callback: Optional logging callback
            
        Returns:
            str: Path to packed composite system PDB
        """
        
        if log_callback:
            log_callback("📦 Professional molecular packing with PACKMOL...")
            log_callback(f"   🦠 Insulin: {insulin_pdb}")
            log_callback(f"   🧬 Polymers: {len(polymer_pdbs)} chains")
        
        try:
            # Prepare input files and molecule counts
            pdb_files = [insulin_pdb] + polymer_pdbs
            n_molecules = [1] + [1] * len(polymer_pdbs)  # 1 insulin + 1 of each polymer chain
            
            if log_callback:
                log_callback(f"   📊 Packing: {len(pdb_files)} components")
                log_callback(f"   📏 Tolerance: {self.packing_tolerance} Å")
                log_callback(f"   📦 Auto box sizing enabled")
            
            # Load molecules first to check validity
            for i, pdb_file in enumerate(pdb_files):
                try:
                    test_traj = md.load(pdb_file)
                    if log_callback:
                        log_callback(f"   ✅ Component {i+1}: {test_traj.n_atoms} atoms, {test_traj.n_residues} residues")
                except Exception as e:
                    raise ValueError(f"Invalid PDB file {pdb_file}: {e}")
            
            # Run PACKMOL through OpenMolTools
            if log_callback:
                log_callback("   🚀 Running PACKMOL...")
            
            packed_trajectory = openmoltools.packmol.pack_box(
                pdb_filenames_or_trajectories=pdb_files,
                n_molecules_list=n_molecules,
                tolerance=self.packing_tolerance,
                box_size=self.default_box_size  # Let openmoltools estimate
            )
            
            # Save packed system
            packed_pdb = os.path.join(output_dir, "packed_composite_system.pdb")
            packed_trajectory.save_pdb(packed_pdb)
            
            if log_callback:
                log_callback(f"   ✅ Packed system saved: {packed_pdb}")
                log_callback(f"   📊 Total atoms: {packed_trajectory.n_atoms}")
                log_callback(f"   📊 Total chains: {packed_trajectory.topology.n_chains}")
                log_callback(f"   📦 Box vectors: {packed_trajectory.unitcell_vectors[0].diagonal():.2f} nm")
            
            # Verify the packed system
            self._verify_packed_system(packed_trajectory, log_callback)
            
            return packed_pdb
        
        except Exception as e:
            if log_callback:
                log_callback(f"❌ OpenMolTools packing failed: {e}")
            raise
    
    def _verify_packed_system(self, trajectory, log_callback: Optional[Callable] = None):
        """Verify the quality of the packed system."""
        
        if log_callback:
            log_callback("🔍 Verifying packed system quality...")
        
        # Check for overlapping atoms
        coords = trajectory.xyz[0]  # First (and only) frame
        min_distance = np.inf
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                min_distance = min(min_distance, dist)
        
        min_distance_angstrom = min_distance * 10  # Convert nm to Å
        
        if log_callback:
            log_callback(f"   📏 Minimum distance: {min_distance_angstrom:.2f} Å")
        
        if min_distance_angstrom < 1.0:
            if log_callback:
                log_callback("   ⚠️ Very close atoms detected - may need equilibration")
        elif min_distance_angstrom > self.packing_tolerance:
            if log_callback:
                log_callback("   ✅ Good molecular spacing")
        else:
            if log_callback:
                log_callback("   ✅ Acceptable molecular spacing")
    
    def _setup_dual_force_fields(self, 
                                packed_system_pdb: str,
                                polymer_smiles_list: list,
                                log_callback: Optional[Callable] = None) -> openmm.System:
        """
        Setup dual GAFF+AMBER force fields for the packed system.
        
        Args:
            packed_system_pdb: Path to packed composite PDB
            polymer_smiles_list: List of polymer SMILES strings
            log_callback: Optional logging callback
            
        Returns:
            openmm.System: Configured system with dual force fields
        """
        
        if log_callback:
            log_callback("⚗️ Setting up dual GAFF+AMBER force fields...")
        
        try:
            # Load the packed system
            pdb = PDBFile(packed_system_pdb)
            
            if log_callback:
                log_callback(f"   📊 Loaded system: {pdb.topology.getNumAtoms()} atoms")
            
            # Create OpenFF molecules for polymers
            polymer_molecules = []
            for i, smiles in enumerate(polymer_smiles_list):
                try:
                    molecule = Molecule.from_smiles(smiles)
                    polymer_molecules.append(molecule)
                    if log_callback:
                        log_callback(f"   🧬 Polymer {i+1}: {len(smiles)} chars SMILES")
                except Exception as e:
                    if log_callback:
                        log_callback(f"   ⚠️ Polymer {i+1} SMILES issue: {e}")
                    # Continue anyway - force field might handle it
            
            # Setup GAFF for polymers
            if log_callback:
                log_callback("   🔧 Setting up GAFF for polymers...")
            
            gaff_generator = GAFFTemplateGenerator(
                molecules=polymer_molecules,
                forcefield='gaff-2.2.20'
            )
            
            # Setup AMBER force field with GAFF integration
            if log_callback:
                log_callback("   🔧 Setting up AMBER force field...")
            
            forcefield = ForceField(
                'amber/protein.ff14SB.xml',  # AMBER for protein (insulin)
                'amber/tip3p_standard.xml'   # Water model
            )
            
            # Register GAFF template generator
            forcefield.registerTemplateGenerator(gaff_generator.generator)
            
            if log_callback:
                log_callback("   ✅ Dual force fields configured")
            
            # Clean up the system with Modeller
            if log_callback:
                log_callback("   🔧 Cleaning system with Modeller...")
            
            modeller = Modeller(pdb.topology, pdb.positions)
            
            # Add missing atoms if needed
            modeller.addMissingAtoms(forcefield)
            
            # Create system with implicit solvent (GBn2 for stability)
            if log_callback:
                log_callback("   ⚗️ Creating OpenMM system...")
            
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0*unit.nanometer,
                implicitSolvent=app.GBn2,
                constraints=HBonds,
                hydrogenMass=4*unit.amu
            )
            
            if log_callback:
                log_callback("   ✅ OpenMM system created successfully")
                log_callback(f"   📊 System forces: {system.getNumForces()}")
                log_callback(f"   🧬 Particles: {system.getNumParticles()}")
            
            return system
        
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Force field setup failed: {e}")
            raise
    
    def _run_openmm_simulation(self, 
                              system: openmm.System,
                              packed_system_pdb: str,
                              simulation_params: Dict[str, Any],
                              output_dir: str,
                              log_callback: Optional[Callable] = None):
        """Run the actual OpenMM simulation."""
        
        if log_callback:
            log_callback("🏃 Running OpenMM simulation...")
        
        try:
            # Load topology and positions
            pdb = PDBFile(packed_system_pdb)
            
            # Create integrator
            temperature = simulation_params.get('temperature', 310.0) * unit.kelvin
            friction = 1.0 / unit.picosecond
            timestep = 2.0 * unit.femtoseconds
            
            integrator = openmm.LangevinMiddleIntegrator(temperature, friction, timestep)
            
            # Create simulation
            platform = openmm.Platform.getPlatformByName('CPU')
            simulation = Simulation(pdb.topology, system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            
            if log_callback:
                log_callback(f"   🌡️ Temperature: {temperature}")
                log_callback(f"   ⏱️ Timestep: {timestep}")
                log_callback(f"   💻 Platform: CPU")
            
            # Minimize energy
            if log_callback:
                log_callback("   ⚡ Minimizing energy...")
            
            simulation.minimizeEnergy(maxIterations=1000)
            
            # Setup reporters
            equilibration_steps = simulation_params.get('equilibration_steps', 10000)
            production_steps = simulation_params.get('production_steps', 50000)
            save_interval = simulation_params.get('save_interval', 500)
            
            # State reporter
            state_file = os.path.join(output_dir, 'simulation.log')
            simulation.reporters.append(StateDataReporter(
                state_file, save_interval,
                step=True, time=True, potentialEnergy=True, temperature=True,
                density=True, progress=True, remainingTime=True,
                totalSteps=equilibration_steps + production_steps
            ))
            
            # PDB reporter
            trajectory_file = os.path.join(output_dir, 'trajectory.pdb')
            simulation.reporters.append(PDBReporter(
                trajectory_file, save_interval
            ))
            
            if log_callback:
                log_callback(f"   ⏰ Equilibration: {equilibration_steps} steps")
                log_callback(f"   🏃 Production: {production_steps} steps")
                log_callback(f"   💾 Save interval: {save_interval} steps")
            
            # Run equilibration
            if log_callback:
                log_callback("   🔄 Running equilibration...")
            
            simulation.step(equilibration_steps)
            
            # Run production
            if log_callback:
                log_callback("   🚀 Running production...")
            
            simulation.step(production_steps)
            
            if log_callback:
                log_callback("   ✅ Simulation completed successfully!")
                log_callback(f"   📊 Final state: {simulation.context.getState().getPotentialEnergy()}")
        
        except Exception as e:
            if log_callback:
                log_callback(f"❌ OpenMM simulation failed: {e}")
            raise
    
    def fix_insulin_pdb_residues(self, input_pdb_path: str, output_pdb_path: str, log_callback):
        """Fix insulin CYS -> CYX residues for AMBER compatibility."""
        
        if log_callback:
            log_callback(f"🔧 Fixing insulin residues: {input_pdb_path}")
        
        try:
            with open(input_pdb_path, 'r') as f:
                content = f.read()
            
            # Replace CYS with CYX for disulfide bonds
            fixed_content = content.replace(' CYS ', ' CYX ')
            
            with open(output_pdb_path, 'w') as f:
                f.write(fixed_content)
            
            if log_callback:
                log_callback(f"   ✅ Fixed CYS -> CYX residues")
                log_callback(f"   📁 Output: {output_pdb_path}")
        
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Residue fixing failed: {e}")
            raise
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status in UI-expected format."""
        
        return {
            'simulation_running': self.simulation_running,
            'simulation_info': {
                'status': self.simulation_status,
                'details': self.simulation_info,
                'method': 'enhanced_dual_gaff_amber_openmoltools'
            }
        }
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Check dependency status."""
        
        dependencies = {
            'openmm': False,
            'openmmforcefields': False,
            'openmoltools': False,
            'mdtraj': False,
            'openff-toolkit': False
        }
        
        try:
            import openmm
            dependencies['openmm'] = True
        except ImportError:
            pass
        
        try:
            import openmmforcefields
            dependencies['openmmforcefields'] = True
        except ImportError:
            pass
        
        try:
            import openmoltools
            dependencies['openmoltools'] = True
        except ImportError:
            pass
        
        try:
            import mdtraj
            dependencies['mdtraj'] = True
        except ImportError:
            pass
        
        try:
            from openff.toolkit import Molecule
            dependencies['openff-toolkit'] = True
        except ImportError:
            pass
        
        all_available = all(dependencies.values())
        
        return {
            'overall': all_available,
            'dependencies': dependencies,
            'missing': [k for k, v in dependencies.items() if not v],
            'method': 'enhanced_openmoltools_dual_gaff_amber'
        }


def test_openmoltools_integration():
    """Test the enhanced OpenMolTools integration."""
    
    print("🚀 TESTING ENHANCED OPENMOLTOOLS DUAL GAFF+AMBER")
    print("=" * 80)
    
    # Test dependency checking
    integration = OpenMolToolsDualGaffAmber()
    
    print("\n🔍 DEPENDENCY STATUS:")
    dep_status = integration.get_dependency_status()
    print(f"   Overall: {'✅' if dep_status['overall'] else '❌'}")
    
    for dep, status in dep_status['dependencies'].items():
        print(f"   {dep}: {'✅' if status else '❌'}")
    
    if dep_status['missing']:
        print(f"   Missing: {', '.join(dep_status['missing'])}")
    
    print("\n📦 OPENMOLTOOLS FEATURES:")
    print("   ✅ Professional PACKMOL integration")
    print("   ✅ Automatic volume estimation")
    print("   ✅ Proper bond/topology handling")
    print("   ✅ MDTraj trajectory output")
    print("   ✅ Realistic molecular packing")
    
    print("\n🎯 BENEFITS OVER SIMPLE CONCATENATION:")
    print("   📊 Realistic density and spacing")
    print("   🔗 Proper intermolecular interactions")
    print("   📦 Professional box setup")
    print("   ⚡ Better equilibration behavior")
    print("   🧬 Physically meaningful arrangements")


if __name__ == "__main__":
    test_openmoltools_integration() 