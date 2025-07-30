#!/usr/bin/env python3
"""
OpenMM MD Simulator for Insulin-Polymer Systems
Using Amber force field for biomolecular simulations
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np
import pandas as pd
from datetime import datetime

# OpenMM imports
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.app import HBonds, NoCutoff, PME, Simulation
    from openmm.app import StateDataReporter, DCDReporter, PDBReporter
    from openmm.app import CharmmParameterSet, CharmmPsfFile
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("⚠️  OpenMM not available. Install with: conda install -c conda-forge openmm")

# Additional imports for structure processing
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️  RDKit not available. Install with: conda install -c conda-forge rdkit")

class OpenMMInsulinSimulator:
    """OpenMM-based MD simulator for insulin-polymer systems"""
    
    def __init__(self, output_dir: str = "openmm_simulations"):
        """Initialize the OpenMM simulator
        
        Args:
            output_dir: Directory for simulation outputs
        """
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is not available. Install with: conda install -c conda-forge openmm")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Simulation parameters
        self.default_params = {
            'temperature': 310.0,  # Body temperature in K
            'pressure': 1.0,       # Atmospheric pressure in bar
            'timestep': 2.0,       # fs
            'friction': 1.0,       # ps^-1
            'equilibration_steps': 50000,   # 100 ps
            'production_steps': 250000,     # 500 ps
            'cutoff': 1.0,         # nm
            'switch_distance': 0.9, # nm
            'constraint_tolerance': 1e-6,
            'platform': 'CUDA'     # CUDA, OpenCL, or CPU
        }
        
        # Available force fields
        self.force_fields = {
            'amber99sb': 'amber99sb.xml',
            'amber14sb': 'amber14sb.xml', 
            'amber99sbildn': 'amber99sbildn.xml',
            'charmm36': 'charmm36.xml',
            'charmm36m': 'charmm36m.xml'
        }
        
        self.water_models = {
            'tip3p': 'tip3p.xml',
            'tip4pew': 'tip4pew.xml',
            'spce': 'spce.xml'
        }
        
        # Initialize platform
        self.platform = self._get_best_platform()
        
        print(f"🚀 OpenMM Insulin Simulator initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Platform: {self.platform.getName()}")
        
    def _get_best_platform(self) -> mm.Platform:
        """Get the best available platform"""
        platform_names = [mm.Platform.getPlatform(i).getName() 
                          for i in range(mm.Platform.getNumPlatforms())]
        
        # Priority order: CUDA > OpenCL > CPU
        if 'CUDA' in platform_names:
            platform = mm.Platform.getPlatformByName('CUDA')
            print("🚀 Using CUDA platform for GPU acceleration")
        elif 'OpenCL' in platform_names:
            platform = mm.Platform.getPlatformByName('OpenCL')
            print("⚡ Using OpenCL platform")
        else:
            platform = mm.Platform.getPlatformByName('CPU')
            print("🖥️  Using CPU platform")
        
        return platform
    
    def prepare_insulin_polymer_system(self, 
                                     insulin_pdb: str,
                                     polymer_pdb: str,
                                     force_field: str = 'amber99sb',
                                     water_model: str = 'tip3p',
                                     box_padding: float = 1.0,
                                     ionic_strength: float = 0.15) -> Tuple[app.Topology, mm.System, np.ndarray]:
        """Prepare insulin-polymer system for simulation
        
        Args:
            insulin_pdb: Path to insulin PDB file
            polymer_pdb: Path to polymer PDB file  
            force_field: Force field to use
            water_model: Water model to use
            box_padding: Box padding in nm
            ionic_strength: Ionic strength in M
            
        Returns:
            Tuple of (topology, system, positions)
        """
        print(f"🧬 Preparing insulin-polymer system...")
        print(f"   💊 Insulin: {insulin_pdb}")
        print(f"   🧪 Polymer: {polymer_pdb}")
        print(f"   🔬 Force field: {force_field}")
        print(f"   💧 Water model: {water_model}")
        
        # Load structures
        insulin_pdb_obj = PDBFile(insulin_pdb)
        polymer_pdb_obj = PDBFile(polymer_pdb)
        
        # Create modeller and add structures
        modeller = Modeller(insulin_pdb_obj.topology, insulin_pdb_obj.positions)
        modeller.add(polymer_pdb_obj.topology, polymer_pdb_obj.positions)
        
        # Load force field
        if force_field not in self.force_fields:
            raise ValueError(f"Unsupported force field: {force_field}")
        
        if water_model not in self.water_models:
            raise ValueError(f"Unsupported water model: {water_model}")
        
        forcefield = ForceField(self.force_fields[force_field], self.water_models[water_model])
        
        # Add solvent
        print(f"💧 Adding solvent with {box_padding} nm padding...")
        modeller.addSolvent(forcefield, 
                           padding=box_padding*unit.nanometer,
                           ionicStrength=ionic_strength*unit.molar)
        
        # Create system
        print(f"⚙️  Creating system...")
        system = forcefield.createSystem(modeller.topology,
                                       nonbondedMethod=PME,
                                       nonbondedCutoff=self.default_params['cutoff']*unit.nanometer,
                                       constraints=HBonds,
                                       constraintTolerance=self.default_params['constraint_tolerance'])
        
        print(f"✅ System prepared:")
        print(f"   🔬 Total atoms: {system.getNumParticles()}")
        print(f"   🧪 Force groups: {system.getNumForces()}")
        
        return modeller.topology, system, modeller.positions
    
    def create_simulation(self, 
                         topology: app.Topology,
                         system: mm.System,
                         positions: np.ndarray,
                         temperature: float = None,
                         pressure: float = None,
                         timestep: float = None) -> Simulation:
        """Create OpenMM simulation object
        
        Args:
            topology: System topology
            system: System object
            positions: Initial positions
            temperature: Temperature in K
            pressure: Pressure in bar
            timestep: Timestep in fs
            
        Returns:
            OpenMM Simulation object
        """
        # Use defaults if not specified
        temp = temperature or self.default_params['temperature']
        press = pressure or self.default_params['pressure']
        dt = timestep or self.default_params['timestep']
        
        print(f"🔧 Creating simulation...")
        print(f"   🌡️  Temperature: {temp} K")
        print(f"   💨 Pressure: {press} bar")
        print(f"   ⏱️  Timestep: {dt} fs")
        
        # Create integrator
        integrator = mm.LangevinMiddleIntegrator(temp*unit.kelvin,
                                               self.default_params['friction']/unit.picosecond,
                                               dt*unit.femtosecond)
        
        # Add barostat for NPT ensemble
        system.addForce(mm.MonteCarloBarostat(press*unit.bar, temp*unit.kelvin))
        
        # Create simulation
        simulation = Simulation(topology, system, integrator, self.platform)
        simulation.context.setPositions(positions)
        
        return simulation
    
    def minimize_energy(self, simulation: Simulation, max_iterations: int = 1000) -> Dict[str, Any]:
        """Minimize energy of the system
        
        Args:
            simulation: OpenMM simulation object
            max_iterations: Maximum minimization iterations
            
        Returns:
            Dictionary with minimization results
        """
        print(f"⚡ Minimizing energy (max {max_iterations} iterations)...")
        
        # Get initial energy
        initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        # Minimize
        start_time = time.time()
        simulation.minimizeEnergy(maxIterations=max_iterations)
        minimize_time = time.time() - start_time
        
        # Get final energy
        final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        
        results = {
            'initial_energy': initial_energy.value_in_unit(unit.kilojoule_per_mole),
            'final_energy': final_energy.value_in_unit(unit.kilojoule_per_mole),
            'energy_change': (final_energy - initial_energy).value_in_unit(unit.kilojoule_per_mole),
            'minimize_time': minimize_time,
            'iterations': max_iterations
        }
        
        print(f"✅ Energy minimization completed:")
        print(f"   🔋 Initial energy: {results['initial_energy']:.1f} kJ/mol")
        print(f"   🔋 Final energy: {results['final_energy']:.1f} kJ/mol")
        print(f"   📉 Energy change: {results['energy_change']:.1f} kJ/mol")
        print(f"   ⏱️  Time: {minimize_time:.2f} seconds")
        
        return results
    
    def run_equilibration(self, 
                         simulation: Simulation,
                         steps: int = None,
                         output_prefix: str = "equilibration") -> Dict[str, Any]:
        """Run equilibration phase
        
        Args:
            simulation: OpenMM simulation object
            steps: Number of equilibration steps
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with equilibration results
        """
        eq_steps = steps or self.default_params['equilibration_steps']
        timestep = self.default_params['timestep']
        
        print(f"🔄 Running equilibration...")
        print(f"   📊 Steps: {eq_steps}")
        print(f"   ⏱️  Time: {eq_steps * timestep / 1000:.1f} ps")
        
        # Setup reporters
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(exist_ok=True)
        
        # State data reporter (energy, temperature, etc.)
        state_file = output_dir / "equilibration_state.csv"
        simulation.reporters.append(StateDataReporter(str(state_file), 
                                                     reportInterval=1000,
                                                     step=True, time=True,
                                                     potentialEnergy=True,
                                                     kineticEnergy=True,
                                                     totalEnergy=True,
                                                     temperature=True,
                                                     volume=True,
                                                     density=True))
        
        # Run equilibration
        start_time = time.time()
        simulation.step(eq_steps)
        eq_time = time.time() - start_time
        
        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy()
        final_positions = state.getPositions()
        
        results = {
            'steps': eq_steps,
            'simulation_time_ps': eq_steps * timestep / 1000,
            'wall_time': eq_time,
            'final_energy': final_energy.value_in_unit(unit.kilojoule_per_mole),
            'performance_ns_per_day': (eq_steps * timestep * 86400) / (eq_time * 1e6),
            'state_file': str(state_file)
        }
        
        print(f"✅ Equilibration completed:")
        print(f"   🔋 Final energy: {results['final_energy']:.1f} kJ/mol")
        print(f"   ⏱️  Wall time: {eq_time:.2f} seconds")
        print(f"   🚀 Performance: {results['performance_ns_per_day']:.2f} ns/day")
        
        return results
    
    def run_production(self, 
                      simulation: Simulation,
                      steps: int = None,
                      output_prefix: str = "production",
                      save_interval: int = 1000) -> Dict[str, Any]:
        """Run production MD simulation
        
        Args:
            simulation: OpenMM simulation object
            steps: Number of production steps
            output_prefix: Prefix for output files
            save_interval: Interval for saving frames
            
        Returns:
            Dictionary with production results
        """
        prod_steps = steps or self.default_params['production_steps']
        timestep = self.default_params['timestep']
        
        print(f"🏃 Running production simulation...")
        print(f"   📊 Steps: {prod_steps}")
        print(f"   ⏱️  Time: {prod_steps * timestep / 1000:.1f} ps")
        print(f"   💾 Save interval: {save_interval} steps")
        
        # Setup output directory
        output_dir = self.output_dir / output_prefix
        output_dir.mkdir(exist_ok=True)
        
        # Clear previous reporters
        simulation.reporters.clear()
        
        # State data reporter
        state_file = output_dir / "production_state.csv"
        simulation.reporters.append(StateDataReporter(str(state_file),
                                                     reportInterval=save_interval,
                                                     step=True, time=True,
                                                     potentialEnergy=True,
                                                     kineticEnergy=True,
                                                     totalEnergy=True,
                                                     temperature=True,
                                                     volume=True,
                                                     density=True))
        
        # Trajectory reporter (DCD format)
        trajectory_file = output_dir / "production_trajectory.dcd"
        simulation.reporters.append(DCDReporter(str(trajectory_file), save_interval))
        
        # PDB reporter for final structure
        final_pdb = output_dir / "production_final.pdb"
        simulation.reporters.append(PDBReporter(str(final_pdb), prod_steps))
        
        # Run production
        start_time = time.time()
        
        # Progress tracking
        progress_interval = max(1, prod_steps // 10)
        
        for step in range(0, prod_steps, progress_interval):
            remaining_steps = min(progress_interval, prod_steps - step)
            simulation.step(remaining_steps)
            
            current_step = step + remaining_steps
            progress = (current_step / prod_steps) * 100
            elapsed = time.time() - start_time
            
            if current_step < prod_steps:
                eta = elapsed * (prod_steps / current_step - 1)
                print(f"   📊 Progress: {progress:.1f}% ({current_step}/{prod_steps}) - ETA: {eta:.1f}s")
        
        prod_time = time.time() - start_time
        
        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy()
        
        results = {
            'steps': prod_steps,
            'simulation_time_ps': prod_steps * timestep / 1000,
            'wall_time': prod_time,
            'final_energy': final_energy.value_in_unit(unit.kilojoule_per_mole),
            'performance_ns_per_day': (prod_steps * timestep * 86400) / (prod_time * 1e6),
            'state_file': str(state_file),
            'trajectory_file': str(trajectory_file),
            'final_pdb': str(final_pdb)
        }
        
        print(f"✅ Production simulation completed:")
        print(f"   🔋 Final energy: {results['final_energy']:.1f} kJ/mol")
        print(f"   ⏱️  Wall time: {prod_time:.2f} seconds ({prod_time/60:.1f} minutes)")
        print(f"   🚀 Performance: {results['performance_ns_per_day']:.2f} ns/day")
        print(f"   📁 Trajectory: {trajectory_file}")
        
        return results
    
    def analyze_trajectory(self, trajectory_file: str, topology_file: str) -> Dict[str, Any]:
        """Analyze MD trajectory
        
        Args:
            trajectory_file: Path to trajectory file
            topology_file: Path to topology file
            
        Returns:
            Dictionary with analysis results
        """
        print(f"📊 Analyzing trajectory: {trajectory_file}")
        
        try:
            # Use MDTraj for analysis if available
            import mdtraj as md
            
            # Load trajectory
            traj = md.load(trajectory_file, top=topology_file)
            
            # Basic analysis
            results = {
                'num_frames': len(traj),
                'num_atoms': traj.n_atoms,
                'num_residues': traj.n_residues,
                'time_ps': traj.time[-1] if len(traj) > 0 else 0,
                'box_dimensions': traj.unitcell_lengths[0] if len(traj) > 0 else None,
                'analysis_available': True
            }
            
            # RMSD calculation (if trajectory has multiple frames)
            if len(traj) > 1:
                rmsd = md.rmsd(traj, traj[0]) * 10  # Convert to Angstroms
                results['rmsd_mean'] = np.mean(rmsd)
                results['rmsd_std'] = np.std(rmsd)
                results['rmsd_final'] = rmsd[-1]
            
            # Radius of gyration
            rg = md.compute_rg(traj) * 10  # Convert to Angstroms
            results['rg_mean'] = np.mean(rg)
            results['rg_std'] = np.std(rg)
            
            print(f"✅ Trajectory analysis completed:")
            print(f"   📊 Frames: {results['num_frames']}")
            print(f"   🔬 Atoms: {results['num_atoms']}")
            print(f"   ⏱️  Time: {results['time_ps']:.1f} ps")
            if 'rmsd_mean' in results:
                print(f"   📏 RMSD: {results['rmsd_mean']:.2f} ± {results['rmsd_std']:.2f} Å")
            print(f"   🎯 Radius of gyration: {results['rg_mean']:.2f} ± {results['rg_std']:.2f} Å")
            
        except ImportError:
            print("⚠️  MDTraj not available. Install with: conda install -c conda-forge mdtraj")
            results = {
                'analysis_available': False,
                'message': 'MDTraj not available for detailed analysis'
            }
        
        return results
    
    def run_complete_simulation(self, 
                              insulin_pdb: str,
                              polymer_pdb: str = None,
                              force_field: str = 'amber99sb',
                              water_model: str = 'tip3p',
                              temperature: float = 310.0,
                              equilibration_steps: int = 50000,
                              production_steps: int = 250000,
                              output_prefix: str = None) -> Dict[str, Any]:
        """Run complete MD simulation workflow
        
        Args:
            insulin_pdb: Path to insulin PDB file
            polymer_pdb: Path to polymer PDB file (optional)
            force_field: Force field to use
            water_model: Water model to use
            temperature: Temperature in K
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with complete simulation results
        """
        if output_prefix is None:
            output_prefix = f"insulin_sim_{uuid.uuid4().hex[:8]}"
        
        print(f"🚀 Starting complete insulin MD simulation...")
        print(f"   📊 Simulation ID: {output_prefix}")
        print(f"   🧬 Insulin: {insulin_pdb}")
        if polymer_pdb:
            print(f"   🧪 Polymer: {polymer_pdb}")
        print(f"   🌡️  Temperature: {temperature} K")
        print(f"   🔄 Equilibration: {equilibration_steps} steps")
        print(f"   🏃 Production: {production_steps} steps")
        
        start_time = time.time()
        
        # Prepare system
        if polymer_pdb:
            topology, system, positions = self.prepare_insulin_polymer_system(
                insulin_pdb, polymer_pdb, force_field, water_model
            )
        else:
            # Insulin only
            pdb = PDBFile(insulin_pdb)
            forcefield = ForceField(self.force_fields[force_field], self.water_models[water_model])
            
            modeller = Modeller(pdb.topology, pdb.positions)
            modeller.addSolvent(forcefield, padding=1.0*unit.nanometer)
            
            topology = modeller.topology
            system = forcefield.createSystem(topology,
                                           nonbondedMethod=PME,
                                           nonbondedCutoff=1.0*unit.nanometer,
                                           constraints=HBonds)
            positions = modeller.positions
        
        # Create simulation
        simulation = self.create_simulation(topology, system, positions, temperature)
        
        # Energy minimization
        minimize_results = self.minimize_energy(simulation)
        
        # Equilibration
        eq_results = self.run_equilibration(simulation, equilibration_steps, 
                                          f"{output_prefix}_equilibration")
        
        # Production
        prod_results = self.run_production(simulation, production_steps,
                                         f"{output_prefix}_production")
        
        # Analysis
        analysis_results = self.analyze_trajectory(
            prod_results['trajectory_file'],
            prod_results['final_pdb']
        )
        
        total_time = time.time() - start_time
        
        # Compile results
        complete_results = {
            'simulation_id': output_prefix,
            'total_wall_time': total_time,
            'insulin_pdb': insulin_pdb,
            'polymer_pdb': polymer_pdb,
            'force_field': force_field,
            'water_model': water_model,
            'temperature': temperature,
            'platform': self.platform.getName(),
            'minimization': minimize_results,
            'equilibration': eq_results,
            'production': prod_results,
            'analysis': analysis_results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🎉 Complete simulation finished!")
        print(f"   ⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   📊 Performance: {prod_results['performance_ns_per_day']:.2f} ns/day")
        print(f"   📁 Output directory: {self.output_dir / output_prefix}")
        
        return complete_results

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python openmm_md_simulator.py <insulin_pdb> [polymer_pdb] [options]")
        print("Example: python openmm_md_simulator.py insulin.pdb polymer.pdb")
        print("Example: python openmm_md_simulator.py insulin.pdb")
        sys.exit(1)
    
    insulin_pdb = sys.argv[1]
    polymer_pdb = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create simulator
    simulator = OpenMMInsulinSimulator()
    
    # Run simulation
    results = simulator.run_complete_simulation(
        insulin_pdb=insulin_pdb,
        polymer_pdb=polymer_pdb,
        temperature=310.0,  # Body temperature
        equilibration_steps=50000,  # 100 ps
        production_steps=250000     # 500 ps
    )
    
    print(f"\n📊 Simulation Summary:")
    print(f"   🆔 ID: {results['simulation_id']}")
    print(f"   ⏱️  Total time: {results['total_wall_time']:.2f} seconds")
    print(f"   🚀 Performance: {results['production']['performance_ns_per_day']:.2f} ns/day")
    print(f"   📁 Results saved to: {results['production']['final_pdb']}")

if __name__ == "__main__":
    main() 