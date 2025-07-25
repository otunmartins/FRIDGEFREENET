#!/usr/bin/env python3
"""
Enhanced OpenMM MD Simulator with Comprehensive Reporting
For Insulin-Polymer Systems with detailed analysis and monitoring
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# OpenMM imports
try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.app import HBonds, NoCutoff, PME, Simulation
    from openmm.app import StateDataReporter, DCDReporter, PDBReporter
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False
    print("⚠️  OpenMM not available. Install with: conda install -c conda-forge openmm")

# Analysis imports
try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("⚠️  MDTraj not available. Install with: conda install -c conda-forge mdtraj")

class EnhancedStateReporter:
    """Enhanced state reporter with real-time analysis"""
    
    def __init__(self, file, reportInterval, 
                 step=True, report_time=True, potentialEnergy=True, kineticEnergy=True,
                 totalEnergy=True, temperature=True, volume=True, density=True,
                 speed=True, systemSize=None):
        self._file = open(file, 'w')
        self._reportInterval = reportInterval
        self._step = step
        self._time = report_time
        self._potentialEnergy = potentialEnergy
        self._kineticEnergy = kineticEnergy
        self._totalEnergy = totalEnergy
        self._temperature = temperature
        self._volume = volume
        self._density = density
        self._speed = speed
        self._systemSize = systemSize
        
        self._energies = []
        self._temperatures = []
        self._volumes = []
        self._times = []
        self._lastReportTime = time.time()
        self._stepCount = 0
        
        # Write header
        headers = []
        if step: headers.append("Step")
        if report_time: headers.append("Time (ps)")
        if potentialEnergy: headers.append("Potential Energy (kJ/mol)")
        if kineticEnergy: headers.append("Kinetic Energy (kJ/mol)")
        if totalEnergy: headers.append("Total Energy (kJ/mol)")
        if temperature: headers.append("Temperature (K)")
        if volume: headers.append("Volume (nm³)")
        if density: headers.append("Density (g/mL)")
        if speed: headers.append("Speed (ns/day)")
        
        self._file.write('#' + '\t'.join(headers) + '\n')
        self._file.flush()
    
    def report(self, simulation, state):
        """Generate detailed report"""
        self._stepCount += 1
        
        # Get state information
        step = simulation.context.getStepCount()
        
        if step % self._reportInterval == 0:
            current_time = time.time()
            
            values = []
            
            if self._step:
                values.append(str(step))
            
            if self._time:
                sim_time = state.getTime().value_in_unit(unit.picosecond)
                values.append(f"{sim_time:.2f}")
                self._times.append(sim_time)
            
            if self._potentialEnergy:
                pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                values.append(f"{pe:.2f}")
                self._energies.append(pe)
            
            if self._kineticEnergy:
                ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                values.append(f"{ke:.2f}")
            
            if self._totalEnergy:
                total_e = (state.getPotentialEnergy() + state.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole)
                values.append(f"{total_e:.2f}")
            
            if self._temperature:
                # Use a more robust temperature calculation
                ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                # Convert to Kelvin using kT = (3/2) * N * kB * T
                # ke = (3/2) * N * kB * T, so T = (2 * ke) / (3 * N * kB)
                # where kB = 8.314 J/(mol·K) = 0.008314 kJ/(mol·K)
                kB = 0.008314  # kJ/(mol·K)
                num_atoms = len(state.getPositions())
                temp = (2 * ke) / (3 * num_atoms * kB)
                values.append(f"{temp:.2f}")
                self._temperatures.append(temp)
            
            if self._volume:
                box_vectors = state.getPeriodicBoxVectors()
                volume = (box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]).value_in_unit(unit.nanometer**3)
                values.append(f"{volume:.3f}")
                self._volumes.append(volume)
            
            if self._density:
                if self._systemSize and len(self._volumes) > 0:
                    # Estimate density (rough calculation)
                    density = self._systemSize / (self._volumes[-1] * 0.6022)  # Rough conversion
                    values.append(f"{density:.3f}")
                else:
                    values.append("N/A")
            
            if self._speed:
                elapsed = current_time - self._lastReportTime
                if elapsed > 0 and step > 0:
                    timestep = simulation.integrator.getStepSize().value_in_unit(unit.femtosecond)
                    steps_per_second = self._reportInterval / elapsed
                    ns_per_day = (steps_per_second * timestep * 86400) / 1e6
                    values.append(f"{ns_per_day:.2f}")
                else:
                    values.append("0.00")
                
                self._lastReportTime = current_time
            
            # Write to file
            self._file.write('\t'.join(values) + '\n')
            self._file.flush()
            
            # Real-time console output
            if step % (self._reportInterval * 10) == 0:
                print(f"📊 Step {step:6d}: PE={self._energies[-1] if self._energies else 'N/A':8.1f} kJ/mol, "
                      f"T={self._temperatures[-1] if self._temperatures else 'N/A':6.1f} K, "
                      f"V={self._volumes[-1] if self._volumes else 'N/A':6.2f} nm³")
    
    def get_statistics(self):
        """Get simulation statistics"""
        stats = {}
        
        if self._energies:
            stats['energy'] = {
                'mean': np.mean(self._energies),
                'std': np.std(self._energies),
                'min': np.min(self._energies),
                'max': np.max(self._energies),
                'final': self._energies[-1],
                'drift': self._energies[-1] - self._energies[0] if len(self._energies) > 1 else 0
            }
        
        if self._temperatures:
            stats['temperature'] = {
                'mean': np.mean(self._temperatures),
                'std': np.std(self._temperatures),
                'target': 310.0,  # Assuming body temperature
                'deviation': abs(np.mean(self._temperatures) - 310.0)
            }
        
        if self._volumes:
            stats['volume'] = {
                'mean': np.mean(self._volumes),
                'std': np.std(self._volumes),
                'change_percent': ((self._volumes[-1] - self._volumes[0]) / self._volumes[0] * 100) if len(self._volumes) > 1 else 0
            }
        
        return stats
    
    def describeNextReport(self, simulation):
        """Describe the next report"""
        steps_until_report = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps_until_report, True, True, True, True, None)
    
    def close(self):
        """Close the reporter"""
        self._file.close()

class EnhancedOpenMMSimulator:
    """Enhanced OpenMM simulator with comprehensive analysis"""
    
    def __init__(self, output_dir: str = "enhanced_openmm_simulations"):
        """Initialize the enhanced simulator"""
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is not available")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced parameters
        self.default_params = {
            'temperature': 310.0,  # Body temperature
            'pressure': 1.0,       # bar
            'timestep': 2.0,       # fs
            'friction': 1.0,       # ps^-1
            'equilibration_steps': 50000,   # 100 ps
            'production_steps': 250000,     # 500 ps
            'cutoff': 1.0,         # nm
            'constraint_tolerance': 1e-6,
            'platform': 'CUDA'
        }
        
        # Force fields and water models
        self.force_fields = {
            'amber99sb': 'amber99sb.xml',
            'amber14sb': 'amber14sb.xml', 
            'amber99sbildn': 'amber99sbildn.xml',
            'charmm36': 'charmm36.xml'
        }
        
        self.water_models = {
            'tip3p': 'tip3p.xml',
            'tip4pew': 'tip4pew.xml',
            'spce': 'spce.xml'
        }
        
        # Initialize platform
        self.platform = self._get_best_platform()
        
        # Analysis storage
        self.simulation_data = {}
        
        print(f"🚀 Enhanced OpenMM Simulator initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🖥️  Platform: {self.platform.getName()}")
    
    def _get_best_platform(self) -> mm.Platform:
        """Get the best available platform with detailed info"""
        platform_names = [mm.Platform.getPlatform(i).getName() 
                          for i in range(mm.Platform.getNumPlatforms())]
        
        print("🔍 Available OpenMM platforms:")
        for name in platform_names:
            platform = mm.Platform.getPlatformByName(name)
            print(f"   • {name}")
            if name == 'CUDA':
                try:
                    device_count = platform.getPropertyDefaultValue('DeviceIndex')
                    print(f"     CUDA devices available")
                except:
                    print(f"     CUDA device detection failed")
            elif name == 'OpenCL':
                try:
                    print(f"     OpenCL devices available")
                except:
                    print(f"     OpenCL device detection failed")
        
        # Priority: CUDA > OpenCL > CPU > Reference
        for preferred in ['CUDA', 'OpenCL', 'CPU', 'Reference']:
            if preferred in platform_names:
                platform = mm.Platform.getPlatformByName(preferred)
                print(f"✅ Selected platform: {preferred}")
                return platform
        
        raise RuntimeError("No suitable OpenMM platform found")
    
    def prepare_system(self, pdb_file: str, 
                      force_field: str = 'amber99sb',
                      water_model: str = 'tip3p',
                      box_padding: float = 1.0,
                      ionic_strength: float = 0.15) -> Tuple[app.Topology, mm.System, np.ndarray, Dict]:
        """Prepare system with detailed reporting"""
        
        print(f"\n🧬 System Preparation Report")
        print(f"=" * 50)
        print(f"📁 Input PDB: {pdb_file}")
        print(f"🔬 Force field: {force_field}")
        print(f"💧 Water model: {water_model}")
        print(f"📦 Box padding: {box_padding} nm")
        print(f"🧪 Ionic strength: {ionic_strength} M")
        
        # Load and analyze initial structure
        pdb = PDBFile(pdb_file)
        
        # Count atoms and residues
        initial_atoms = len(list(pdb.topology.atoms()))
        initial_residues = len(list(pdb.topology.residues()))
        initial_chains = len(list(pdb.topology.chains()))
        
        print(f"\n📊 Initial Structure Analysis:")
        print(f"   🔬 Atoms: {initial_atoms}")
        print(f"   🧪 Residues: {initial_residues}")
        print(f"   🔗 Chains: {initial_chains}")
        
        # List residue types
        residue_types = {}
        for residue in pdb.topology.residues():
            res_name = residue.name
            residue_types[res_name] = residue_types.get(res_name, 0) + 1
        
        print(f"   📋 Residue composition:")
        for res_name, count in sorted(residue_types.items()):
            print(f"      {res_name}: {count}")
        
        # Create modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Load force field
        print(f"\n⚙️  Loading force field and water model...")
        forcefield = ForceField(self.force_fields[force_field], self.water_models[water_model])
        
        # Add missing hydrogens
        print(f"🔧 Adding missing hydrogen atoms...")
        initial_atoms_before_h = len(list(modeller.topology.atoms()))
        modeller.addHydrogens(forcefield)
        atoms_after_h = len(list(modeller.topology.atoms()))
        print(f"   ➕ Added {atoms_after_h - initial_atoms_before_h} hydrogen atoms")
        
        # Add solvent
        print(f"\n💧 Solvating system...")
        print(f"   📦 Adding water box with {box_padding} nm padding...")
        
        modeller.addSolvent(forcefield, 
                           padding=box_padding*unit.nanometer,
                           ionicStrength=ionic_strength*unit.molar)
        
        # Final system analysis
        final_atoms = len(list(modeller.topology.atoms()))
        final_residues = len(list(modeller.topology.residues()))
        water_molecules = final_residues - initial_residues
        
        print(f"\n✅ Final Solvated System:")
        print(f"   🔬 Total atoms: {final_atoms}")
        print(f"   🧪 Total residues: {final_residues}")
        print(f"   💧 Water molecules: {water_molecules}")
        print(f"   📈 System size increase: {final_atoms/initial_atoms:.1f}x")
        
        # Estimate box dimensions
        positions = modeller.positions
        coords = np.array([[pos.x, pos.y, pos.z] for pos in positions]) * 10  # Convert to Angstroms
        box_size = np.max(coords, axis=0) - np.min(coords, axis=0)
        print(f"   📏 Estimated box size: {box_size[0]:.1f} × {box_size[1]:.1f} × {box_size[2]:.1f} Å")
        
        # Create system
        print(f"\n🔧 Creating molecular system...")
        system = forcefield.createSystem(modeller.topology,
                                       nonbondedMethod=PME,
                                       nonbondedCutoff=self.default_params['cutoff']*unit.nanometer,
                                       constraints=HBonds)
        
        # Analyze forces
        print(f"⚡ Force field analysis:")
        force_names = []
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            force_name = force.__class__.__name__
            force_names.append(force_name)
            print(f"   • {force_name}")
        
        # System statistics
        system_info = {
            'initial_atoms': initial_atoms,
            'final_atoms': final_atoms,
            'water_molecules': water_molecules,
            'box_size': box_size.tolist(),
            'residue_types': residue_types,
            'force_names': force_names,
            'force_field': force_field,
            'water_model': water_model
        }
        
        print(f"\n🎉 System preparation completed!")
        
        return modeller.topology, system, modeller.positions, system_info
    
    def run_enhanced_simulation(self, 
                              pdb_file: str,
                              force_field: str = 'amber99sb',
                              temperature: float = 310.0,
                              equilibration_steps: int = 50000,
                              production_steps: int = 250000,
                              save_interval: int = 1000,
                              analysis_interval: int = 10000,
                              output_prefix: str = None) -> Dict[str, Any]:
        """Run simulation with comprehensive analysis and reporting"""
        
        if output_prefix is None:
            output_prefix = f"enhanced_sim_{uuid.uuid4().hex[:8]}"
        
        sim_dir = self.output_dir / output_prefix
        sim_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 Enhanced MD Simulation")
        print(f"=" * 50)
        print(f"🆔 Simulation ID: {output_prefix}")
        print(f"📁 Output directory: {sim_dir}")
        print(f"🌡️  Temperature: {temperature} K")
        print(f"🔄 Equilibration: {equilibration_steps} steps ({equilibration_steps * 2 / 1000:.1f} ps)")
        print(f"🏃 Production: {production_steps} steps ({production_steps * 2 / 1000:.1f} ps)")
        print(f"💾 Save interval: {save_interval} steps")
        
        start_time = time.time()
        
        # 1. System preparation
        topology, system, positions, system_info = self.prepare_system(pdb_file, force_field)
        
        # 2. Create simulation
        print(f"\n🔧 Creating simulation context...")
        integrator = mm.LangevinMiddleIntegrator(temperature*unit.kelvin,
                                               self.default_params['friction']/unit.picosecond,
                                               self.default_params['timestep']*unit.femtosecond)
        integrator.setConstraintTolerance(self.default_params['constraint_tolerance'])
        
        # Add barostat
        system.addForce(mm.MonteCarloBarostat(1.0*unit.bar, temperature*unit.kelvin))
        
        try:
            simulation = Simulation(topology, system, integrator, self.platform)
            simulation.context.setPositions(positions)
            print(f"✅ Simulation context created successfully on {self.platform.getName()}")
        except Exception as e:
            print(f"❌ Failed to create simulation context: {e}")
            return {'success': False, 'error': str(e)}
        
        # 3. Energy minimization
        print(f"\n⚡ Energy minimization...")
        initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"   🔋 Initial energy: {initial_energy.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
        
        minimize_start = time.time()
        simulation.minimizeEnergy(maxIterations=1000)
        minimize_time = time.time() - minimize_start
        
        final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        energy_change = final_energy - initial_energy
        print(f"   🔋 Final energy: {final_energy.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
        print(f"   📉 Energy change: {energy_change.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
        print(f"   ⏱️  Minimization time: {minimize_time:.2f} seconds")
        
        # 4. Equilibration
        print(f"\n🔄 Equilibration phase...")
        eq_dir = sim_dir / "equilibration"
        eq_dir.mkdir(exist_ok=True)
        
        # Setup enhanced reporters
        eq_state_file = eq_dir / "equilibration_detailed.csv"
        eq_reporter = EnhancedStateReporter(str(eq_state_file), 
                                          reportInterval=save_interval//10,
                                          systemSize=system_info['final_atoms'])
        
        # Clear any existing reporters
        simulation.reporters.clear()
        
        # Add reporters
        simulation.reporters.append(eq_reporter)
        simulation.reporters.append(DCDReporter(str(eq_dir / "equilibration.dcd"), save_interval))
        
        # Run equilibration
        eq_start = time.time()
        print(f"   📊 Running {equilibration_steps} equilibration steps...")
        
        for step in range(0, equilibration_steps, analysis_interval):
            remaining = min(analysis_interval, equilibration_steps - step)
            simulation.step(remaining)
            
            # Progress report
            progress = (step + remaining) / equilibration_steps * 100
            elapsed = time.time() - eq_start
            if step > 0:
                eta = elapsed * (equilibration_steps / (step + remaining) - 1)
                print(f"   📈 Equilibration progress: {progress:.1f}% - ETA: {eta:.0f}s")
        
        eq_time = time.time() - eq_start
        eq_stats = eq_reporter.get_statistics()
        eq_reporter.close()
        
        print(f"   ✅ Equilibration completed in {eq_time:.2f} seconds")
        if 'temperature' in eq_stats:
            print(f"   🌡️  Average temperature: {eq_stats['temperature']['mean']:.1f} ± {eq_stats['temperature']['std']:.1f} K")
        if 'energy' in eq_stats:
            print(f"   🔋 Energy drift: {eq_stats['energy']['drift']:.1f} kJ/mol")
        
        # 5. Production simulation
        print(f"\n🏃 Production phase...")
        prod_dir = sim_dir / "production"
        prod_dir.mkdir(exist_ok=True)
        
        # Setup production reporters
        prod_state_file = prod_dir / "production_detailed.csv"
        prod_reporter = EnhancedStateReporter(str(prod_state_file),
                                            reportInterval=save_interval,
                                            systemSize=system_info['final_atoms'])
        
        simulation.reporters.clear()
        simulation.reporters.append(prod_reporter)
        simulation.reporters.append(DCDReporter(str(prod_dir / "production.dcd"), save_interval))
        simulation.reporters.append(PDBReporter(str(prod_dir / "frames.pdb"), save_interval * 10))
        
        # Run production
        prod_start = time.time()
        print(f"   📊 Running {production_steps} production steps...")
        
        for step in range(0, production_steps, analysis_interval):
            remaining = min(analysis_interval, production_steps - step)
            simulation.step(remaining)
            
            # Progress and performance monitoring
            progress = (step + remaining) / production_steps * 100
            elapsed = time.time() - prod_start
            
            if step > 0:
                eta = elapsed * (production_steps / (step + remaining) - 1)
                steps_per_second = (step + remaining) / elapsed
                ns_per_day = (steps_per_second * 2.0 * 86400) / 1e6  # 2 fs timestep
                
                print(f"   📈 Production progress: {progress:.1f}% - "
                      f"Performance: {ns_per_day:.2f} ns/day - ETA: {eta:.0f}s")
        
        prod_time = time.time() - prod_start
        prod_stats = prod_reporter.get_statistics()
        prod_reporter.close()
        
        # Save final structure
        final_state = simulation.context.getState(getPositions=True, getEnergy=True)
        with open(str(prod_dir / "final_structure.pdb"), 'w') as f:
            PDBFile.writeFile(topology, final_state.getPositions(), f)
        
        print(f"   ✅ Production completed in {prod_time:.2f} seconds")
        
        # 6. Performance analysis
        total_time = time.time() - start_time
        total_steps = equilibration_steps + production_steps
        steps_per_second = total_steps / total_time
        ns_per_day = (steps_per_second * 2.0 * 86400) / 1e6
        
        print(f"\n📊 Performance Summary:")
        print(f"   ⏱️  Total wall time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"   🚀 Average performance: {ns_per_day:.2f} ns/day")
        print(f"   📈 Steps per second: {steps_per_second:.1f}")
        
        # 7. Comprehensive analysis
        print(f"\n🔬 Post-simulation analysis...")
        analysis_results = self.analyze_simulation(
            trajectory_file=str(prod_dir / "production.dcd"),
            topology_file=str(prod_dir / "final_structure.pdb"),
            state_file=str(prod_state_file),
            output_dir=sim_dir
        )
        
        # 8. Generate summary report
        simulation_results = {
            'simulation_id': output_prefix,
            'timestamp': datetime.now().isoformat(),
            'input_file': pdb_file,
            'system_info': system_info,
            'parameters': {
                'force_field': force_field,
                'temperature': temperature,
                'equilibration_steps': equilibration_steps,
                'production_steps': production_steps,
                'platform': self.platform.getName()
            },
            'timing': {
                'minimization': minimize_time,
                'equilibration': eq_time,
                'production': prod_time,
                'total': total_time
            },
            'performance': {
                'steps_per_second': steps_per_second,
                'ns_per_day': ns_per_day
            },
            'equilibration_stats': eq_stats,
            'production_stats': prod_stats,
            'analysis': analysis_results,
            'files': {
                'equilibration_dcd': str(eq_dir / "equilibration.dcd"),
                'production_dcd': str(prod_dir / "production.dcd"),
                'final_pdb': str(prod_dir / "final_structure.pdb"),
                'equilibration_data': str(eq_state_file),
                'production_data': str(prod_state_file)
            },
            'success': True
        }
        
        # Save comprehensive report
        report_file = sim_dir / "simulation_report.json"
        with open(report_file, 'w') as f:
            json.dump(simulation_results, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive report saved: {report_file}")
        
        return simulation_results
    
    def analyze_simulation(self, trajectory_file: str, topology_file: str, 
                         state_file: str, output_dir: Path) -> Dict[str, Any]:
        """Comprehensive trajectory and state analysis"""
        
        analysis_results = {}
        
        try:
            # 1. State data analysis
            print(f"   📊 Analyzing state data...")
            state_df = pd.read_csv(state_file, comment='#', delimiter='\t')
            
            # Energy analysis
            if 'Potential Energy (kJ/mol)' in state_df.columns:
                pe_col = 'Potential Energy (kJ/mol)'
                analysis_results['energy_analysis'] = {
                    'mean_potential': float(state_df[pe_col].mean()),
                    'std_potential': float(state_df[pe_col].std()),
                    'min_potential': float(state_df[pe_col].min()),
                    'max_potential': float(state_df[pe_col].max()),
                    'energy_drift': float(state_df[pe_col].iloc[-1] - state_df[pe_col].iloc[0]),
                    'convergence_window': int(len(state_df) * 0.1)  # Last 10% for convergence
                }
                
                # Energy convergence analysis
                convergence_start = int(len(state_df) * 0.9)
                convergence_data = state_df[pe_col].iloc[convergence_start:]
                analysis_results['energy_analysis']['converged_mean'] = float(convergence_data.mean())
                analysis_results['energy_analysis']['converged_std'] = float(convergence_data.std())
            
            # Temperature analysis
            if 'Temperature (K)' in state_df.columns:
                temp_col = 'Temperature (K)'
                analysis_results['temperature_analysis'] = {
                    'mean_temperature': float(state_df[temp_col].mean()),
                    'std_temperature': float(state_df[temp_col].std()),
                    'target_temperature': 310.0,
                    'temperature_deviation': float(abs(state_df[temp_col].mean() - 310.0))
                }
            
            # Volume analysis (if available)
            if 'Volume (nm³)' in state_df.columns:
                vol_col = 'Volume (nm³)'
                analysis_results['volume_analysis'] = {
                    'mean_volume': float(state_df[vol_col].mean()),
                    'volume_change_percent': float((state_df[vol_col].iloc[-1] - state_df[vol_col].iloc[0]) / state_df[vol_col].iloc[0] * 100)
                }
            
            # 2. Generate plots
            print(f"   📈 Generating analysis plots...")
            self.generate_analysis_plots(state_df, output_dir)
            
            # 3. MDTraj analysis (if available)
            if MDTRAJ_AVAILABLE and os.path.exists(trajectory_file):
                print(f"   🧬 Analyzing trajectory structure...")
                try:
                    traj = md.load(trajectory_file, top=topology_file)
                    
                    analysis_results['structural_analysis'] = {
                        'num_frames': len(traj),
                        'num_atoms': traj.n_atoms,
                        'num_residues': traj.n_residues,
                        'simulation_time_ps': float(traj.time[-1]) if len(traj) > 0 else 0
                    }
                    
                    # RMSD analysis
                    if len(traj) > 1:
                        rmsd = md.rmsd(traj, traj[0]) * 10  # Convert to Angstroms
                        analysis_results['structural_analysis']['rmsd'] = {
                            'mean': float(np.mean(rmsd)),
                            'std': float(np.std(rmsd)),
                            'final': float(rmsd[-1]),
                            'max': float(np.max(rmsd))
                        }
                    
                    # Radius of gyration
                    rg = md.compute_rg(traj) * 10  # Convert to Angstroms
                    analysis_results['structural_analysis']['radius_of_gyration'] = {
                        'mean': float(np.mean(rg)),
                        'std': float(np.std(rg)),
                        'change_percent': float((rg[-1] - rg[0]) / rg[0] * 100) if rg[0] != 0 else 0
                    }
                    
                except Exception as e:
                    print(f"     ⚠️  MDTraj analysis failed: {e}")
                    analysis_results['mdtraj_error'] = str(e)
            
            print(f"   ✅ Analysis completed successfully")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            analysis_results['analysis_error'] = str(e)
        
        return analysis_results
    
    def generate_analysis_plots(self, state_df: pd.DataFrame, output_dir: Path):
        """Generate comprehensive analysis plots"""
        
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MD Simulation Analysis', fontsize=16, fontweight='bold')
            
            # Energy plot
            if 'Potential Energy (kJ/mol)' in state_df.columns:
                axes[0, 0].plot(state_df.index, state_df['Potential Energy (kJ/mol)'], 'b-', alpha=0.7)
                axes[0, 0].set_title('Potential Energy vs Time')
                axes[0, 0].set_ylabel('Energy (kJ/mol)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Temperature plot
            if 'Temperature (K)' in state_df.columns:
                axes[0, 1].plot(state_df.index, state_df['Temperature (K)'], 'r-', alpha=0.7)
                axes[0, 1].axhline(y=310, color='k', linestyle='--', label='Target (310 K)')
                axes[0, 1].set_title('Temperature vs Time')
                axes[0, 1].set_ylabel('Temperature (K)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Volume plot
            if 'Volume (nm³)' in state_df.columns:
                axes[1, 0].plot(state_df.index, state_df['Volume (nm³)'], 'g-', alpha=0.7)
                axes[1, 0].set_title('System Volume vs Time')
                axes[1, 0].set_ylabel('Volume (nm³)')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Performance plot
            if 'Speed (ns/day)' in state_df.columns:
                axes[1, 1].plot(state_df.index, state_df['Speed (ns/day)'], 'm-', alpha=0.7)
                axes[1, 1].set_title('Simulation Performance')
                axes[1, 1].set_ylabel('Speed (ns/day)')
                axes[1, 1].set_xlabel('Frame')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / "analysis_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     📊 Analysis plots saved: {plot_file}")
            
        except Exception as e:
            print(f"     ⚠️  Plot generation failed: {e}")

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python openmm_md_enhanced.py <pdb_file> [options]")
        print("Example: python openmm_md_enhanced.py insulin.pdb")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    # Create enhanced simulator
    simulator = EnhancedOpenMMSimulator()
    
    # Run enhanced simulation
    results = simulator.run_enhanced_simulation(
        pdb_file=pdb_file,
        temperature=310.0,
        equilibration_steps=25000,  # Shorter for testing
        production_steps=50000,     # Shorter for testing
        save_interval=500
    )
    
    if results['success']:
        print(f"\n🎉 Enhanced simulation completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   🆔 Simulation ID: {results['simulation_id']}")
        print(f"   ⏱️  Total time: {results['timing']['total']:.2f} seconds")
        print(f"   🚀 Performance: {results['performance']['ns_per_day']:.2f} ns/day")
        print(f"   🔬 System size: {results['system_info']['final_atoms']} atoms")
        
        if 'energy_analysis' in results['analysis']:
            energy = results['analysis']['energy_analysis']
            print(f"   🔋 Final energy: {energy['converged_mean']:.1f} ± {energy['converged_std']:.1f} kJ/mol")
        
        if 'structural_analysis' in results['analysis']:
            struct = results['analysis']['structural_analysis']
            print(f"   📏 Final RMSD: {struct['rmsd']['final']:.2f} Å")
    else:
        print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 