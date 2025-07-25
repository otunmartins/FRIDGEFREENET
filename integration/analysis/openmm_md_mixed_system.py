#!/usr/bin/env python3
"""
OpenMM MD Simulator for Mixed Insulin-Polymer Systems
Handles unknown residues and complex mixed systems
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
    print("⚠️  MDTraj not available for advanced analysis")

class MixedSystemReporter:
    """State reporter optimized for mixed systems"""
    
    def __init__(self, file, reportInterval):
        self._file = open(file, 'w')
        self._reportInterval = reportInterval
        self._energies = []
        self._temperatures = []
        self._volumes = []
        self._lastReportTime = time.time()
        
        # Write header
        self._file.write('#Step\tTime(ps)\tPE(kJ/mol)\tKE(kJ/mol)\tTotal(kJ/mol)\tTemp(K)\tVolume(nm³)\tSpeed(ns/day)\n')
        self._file.flush()
    
    def describeNextReport(self, simulation):
        """Describe the next report"""
        steps_until_report = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps_until_report, True, True, True, True, None)
    
    def report(self, simulation, state):
        """Generate report for mixed systems"""
        step = simulation.context.getStepCount()
        
        if step % self._reportInterval == 0:
            current_time = time.time()
            
            # Get basic state information
            sim_time = state.getTime().value_in_unit(unit.picosecond)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            total_e = pe + ke
            
            # Calculate temperature safely
            kB = 0.008314  # kJ/(mol·K)
            num_atoms = len(state.getPositions())
            temp = (2 * ke) / (3 * num_atoms * kB) if num_atoms > 0 else 0
            
            # Calculate volume if periodic
            try:
                box_vectors = state.getPeriodicBoxVectors()
                volume = (box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]).value_in_unit(unit.nanometer**3)
            except:
                volume = 0.0
            
            # Calculate performance
            elapsed = current_time - self._lastReportTime
            if elapsed > 0 and step > 0:
                timestep = simulation.integrator.getStepSize().value_in_unit(unit.femtosecond)
                steps_per_second = self._reportInterval / elapsed
                ns_per_day = (steps_per_second * timestep * 86400) / 1e6
            else:
                ns_per_day = 0.0
            
            # Store data
            self._energies.append(pe)
            self._temperatures.append(temp)
            self._volumes.append(volume)
            self._lastReportTime = current_time
            
            # Write to file
            self._file.write(f"{step}\t{sim_time:.2f}\t{pe:.2f}\t{ke:.2f}\t{total_e:.2f}\t{temp:.2f}\t{volume:.3f}\t{ns_per_day:.2f}\n")
            self._file.flush()
            
            # Console output
            if step % (self._reportInterval * 5) == 0:
                print(f"📊 Step {step:6d}: PE={pe:8.1f} kJ/mol, T={temp:6.1f} K, V={volume:6.2f} nm³, Speed={ns_per_day:.1f} ns/day")
    
    def get_statistics(self):
        """Get simulation statistics"""
        stats = {}
        
        if self._energies:
            stats['energy'] = {
                'mean': np.mean(self._energies),
                'std': np.std(self._energies),
                'final': self._energies[-1],
                'drift': self._energies[-1] - self._energies[0] if len(self._energies) > 1 else 0
            }
        
        if self._temperatures:
            stats['temperature'] = {
                'mean': np.mean(self._temperatures),
                'std': np.std(self._temperatures),
                'target': 310.0,
                'deviation': abs(np.mean(self._temperatures) - 310.0)
            }
        
        return stats
    
    def close(self):
        """Close the reporter"""
        self._file.close()

class MixedSystemSimulator:
    """OpenMM simulator optimized for insulin-polymer mixed systems"""
    
    def __init__(self, output_dir: str = "mixed_system_simulations"):
        """Initialize the mixed system simulator"""
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is not available")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get best platform
        self.platform = self._get_best_platform()
        
        print(f"🚀 Mixed System OpenMM Simulator initialized")
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
    
    def analyze_pdb_composition(self, pdb_file: str) -> Dict[str, Any]:
        """Analyze PDB file composition"""
        print(f"\n🔬 Analyzing PDB composition: {pdb_file}")
        
        pdb = PDBFile(pdb_file)
        
        # Count components
        total_atoms = len(list(pdb.topology.atoms()))
        total_residues = len(list(pdb.topology.residues()))
        total_chains = len(list(pdb.topology.chains()))
        
        # Analyze residue types
        residue_types = {}
        protein_residues = 0
        unknown_residues = 0
        water_residues = 0
        
        for residue in pdb.topology.residues():
            res_name = residue.name
            residue_types[res_name] = residue_types.get(res_name, 0) + 1
            
            if res_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                           'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                           'THR', 'TRP', 'TYR', 'VAL']:
                protein_residues += 1
            elif res_name in ['HOH', 'WAT']:
                water_residues += 1
            elif res_name == 'UNL':
                unknown_residues += 1
        
        composition = {
            'total_atoms': total_atoms,
            'total_residues': total_residues,
            'total_chains': total_chains,
            'residue_types': residue_types,
            'protein_residues': protein_residues,
            'unknown_residues': unknown_residues,
            'water_residues': water_residues,
            'topology': pdb.topology,
            'positions': pdb.positions
        }
        
        print(f"   🔬 Total atoms: {total_atoms}")
        print(f"   🧪 Total residues: {total_residues}")
        print(f"   🔗 Chains: {total_chains}")
        print(f"   🧬 Protein residues: {protein_residues}")
        print(f"   ❓ Unknown residues (UNL): {unknown_residues}")
        print(f"   💧 Water residues: {water_residues}")
        
        return composition
    
    def create_mixed_system(self, composition: Dict[str, Any], 
                          temperature: float = 310.0) -> Tuple[mm.System, app.Topology, np.ndarray]:
        """Create a system for mixed insulin-polymer systems"""
        
        print(f"\n🔧 Creating mixed system...")
        
        topology = composition['topology']
        positions = composition['positions']
        
        # Create system with minimal force field for mixed systems
        system = mm.System()
        
        # Add particles (atoms)
        for atom in topology.atoms():
            # Use carbon mass for unknown atoms, proper masses for known ones
            if atom.element is not None:
                mass = atom.element.mass
            else:
                mass = 12.0 * unit.amu  # Carbon mass for unknown atoms
            system.addParticle(mass)
        
        print(f"   ➕ Added {system.getNumParticles()} particles")
        
        # Add basic harmonic bond forces for stability
        bond_force = mm.HarmonicBondForce()
        
        # Add bonds with reasonable parameters
        bond_count = 0
        for bond in topology.bonds():
            atom1_idx = bond[0].index
            atom2_idx = bond[1].index
            
            # Use reasonable bond parameters
            bond_length = 1.5 * unit.angstrom  # Reasonable default
            bond_strength = 1000.0 * unit.kilojoule_per_mole / (unit.angstrom**2)
            
            bond_force.addBond(atom1_idx, atom2_idx, bond_length, bond_strength)
            bond_count += 1
        
        if bond_count > 0:
            system.addForce(bond_force)
            print(f"   🔗 Added {bond_count} harmonic bonds")
        
        # Add Lennard-Jones interactions
        lj_force = mm.NonbondedForce()
        
        for atom in topology.atoms():
            # Use reasonable LJ parameters
            sigma = 3.5 * unit.angstrom
            epsilon = 0.5 * unit.kilojoule_per_mole
            charge = 0.0 * unit.elementary_charge  # Neutral for simplicity
            
            lj_force.addParticle(charge, sigma, epsilon)
        
        # Set cutoff for non-bonded interactions
        lj_force.setNonbondedMethod(mm.NonbondedForce.CutoffNonPeriodic)
        lj_force.setCutoffDistance(1.0 * unit.nanometer)
        
        system.addForce(lj_force)
        print(f"   ⚡ Added non-bonded interactions")
        
        # Add center of mass motion removal
        system.addForce(mm.CMMotionRemover())
        print(f"   🌀 Added CM motion remover")
        
        print(f"   ✅ Mixed system created with {system.getNumForces()} forces")
        
        return system, topology, positions
    
    def run_mixed_simulation(self, pdb_file: str,
                           temperature: float = 310.0,
                           equilibration_steps: int = 10000,  # Shorter for mixed systems
                           production_steps: int = 25000,    # Shorter for mixed systems
                           save_interval: int = 500,
                           output_prefix: str = None) -> Dict[str, Any]:
        """Run simulation on mixed insulin-polymer system"""
        
        if output_prefix is None:
            output_prefix = f"mixed_sim_{uuid.uuid4().hex[:8]}"
        
        sim_dir = self.output_dir / output_prefix
        sim_dir.mkdir(exist_ok=True)
        
        print(f"\n🚀 Mixed System MD Simulation")
        print(f"=" * 50)
        print(f"🆔 Simulation ID: {output_prefix}")
        print(f"📁 Output directory: {sim_dir}")
        print(f"🌡️  Temperature: {temperature} K")
        print(f"🔄 Equilibration: {equilibration_steps} steps ({equilibration_steps * 2 / 1000:.1f} ps)")
        print(f"🏃 Production: {production_steps} steps ({production_steps * 2 / 1000:.1f} ps)")
        
        start_time = time.time()
        
        try:
            # 1. Analyze composition
            composition = self.analyze_pdb_composition(pdb_file)
            
            # 2. Create mixed system
            system, topology, positions = self.create_mixed_system(composition, temperature)
            
            # 3. Create simulation
            print(f"\n🔧 Creating simulation context...")
            integrator = mm.LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,  # friction
                2.0 * unit.femtosecond  # timestep
            )
            
            simulation = Simulation(topology, system, integrator, self.platform)
            simulation.context.setPositions(positions)
            
            print(f"✅ Simulation context created on {self.platform.getName()}")
            
            # 4. Energy minimization
            print(f"\n⚡ Energy minimization...")
            initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            print(f"   🔋 Initial energy: {initial_energy.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
            
            minimize_start = time.time()
            simulation.minimizeEnergy(maxIterations=500)  # Fewer iterations for mixed systems
            minimize_time = time.time() - minimize_start
            
            final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            energy_change = final_energy - initial_energy
            print(f"   🔋 Final energy: {final_energy.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
            print(f"   📉 Energy change: {energy_change.value_in_unit(unit.kilojoule_per_mole):.1f} kJ/mol")
            print(f"   ⏱️  Minimization time: {minimize_time:.2f} seconds")
            
            # 5. Equilibration
            print(f"\n🔄 Equilibration phase...")
            eq_dir = sim_dir / "equilibration"
            eq_dir.mkdir(exist_ok=True)
            
            eq_reporter = MixedSystemReporter(str(eq_dir / "equilibration.csv"), save_interval)
            
            simulation.reporters.clear()
            simulation.reporters.append(eq_reporter)
            simulation.reporters.append(DCDReporter(str(eq_dir / "equilibration.dcd"), save_interval))
            
            eq_start = time.time()
            simulation.step(equilibration_steps)
            eq_time = time.time() - eq_start
            
            eq_stats = eq_reporter.get_statistics()
            eq_reporter.close()
            
            print(f"   ✅ Equilibration completed in {eq_time:.2f} seconds")
            if 'temperature' in eq_stats:
                print(f"   🌡️  Average temperature: {eq_stats['temperature']['mean']:.1f} ± {eq_stats['temperature']['std']:.1f} K")
            
            # 6. Production
            print(f"\n🏃 Production phase...")
            prod_dir = sim_dir / "production"
            prod_dir.mkdir(exist_ok=True)
            
            prod_reporter = MixedSystemReporter(str(prod_dir / "production.csv"), save_interval)
            
            simulation.reporters.clear()
            simulation.reporters.append(prod_reporter)
            simulation.reporters.append(DCDReporter(str(prod_dir / "production.dcd"), save_interval))
            simulation.reporters.append(PDBReporter(str(prod_dir / "frames.pdb"), save_interval * 5))
            
            prod_start = time.time()
            simulation.step(production_steps)
            prod_time = time.time() - prod_start
            
            prod_stats = prod_reporter.get_statistics()
            prod_reporter.close()
            
            # Save final structure
            final_state = simulation.context.getState(getPositions=True, getEnergy=True)
            with open(str(prod_dir / "final_structure.pdb"), 'w') as f:
                PDBFile.writeFile(topology, final_state.getPositions(), f)
            
            print(f"   ✅ Production completed in {prod_time:.2f} seconds")
            
            # 7. Performance summary
            total_time = time.time() - start_time
            total_steps = equilibration_steps + production_steps
            steps_per_second = total_steps / total_time
            ns_per_day = (steps_per_second * 2.0 * 86400) / 1e6
            
            print(f"\n📊 Simulation Summary:")
            print(f"   ⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            print(f"   🚀 Performance: {ns_per_day:.2f} ns/day")
            print(f"   🔬 System size: {composition['total_atoms']} atoms")
            print(f"   🧬 Protein residues: {composition['protein_residues']}")
            print(f"   ❓ Unknown residues: {composition['unknown_residues']}")
            
            # 8. Generate plots
            self.generate_analysis_plots(prod_dir / "production.csv", sim_dir)
            
            # 9. Comprehensive results
            results = {
                'simulation_id': output_prefix,
                'timestamp': datetime.now().isoformat(),
                'input_file': pdb_file,
                'system_composition': {
                    'total_atoms': composition['total_atoms'],
                    'protein_residues': composition['protein_residues'],
                    'unknown_residues': composition['unknown_residues'],
                    'water_residues': composition['water_residues']
                },
                'parameters': {
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
                'files': {
                    'equilibration_dcd': str(eq_dir / "equilibration.dcd"),
                    'production_dcd': str(prod_dir / "production.dcd"),
                    'final_pdb': str(prod_dir / "final_structure.pdb"),
                    'production_data': str(prod_dir / "production.csv")
                },
                'success': True
            }
            
            # Save results
            with open(sim_dir / "simulation_report.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"📄 Report saved: {sim_dir / 'simulation_report.json'}")
            
            return results
            
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'simulation_id': output_prefix
            }
    
    def generate_analysis_plots(self, data_file: Path, output_dir: Path):
        """Generate analysis plots for mixed systems"""
        try:
            print(f"   📈 Generating analysis plots...")
            
            # Read data
            df = pd.read_csv(data_file, delimiter='\t', comment='#')
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Mixed System MD Analysis', fontsize=14, fontweight='bold')
            
            # Energy plot
            if 'PE(kJ/mol)' in df.columns:
                axes[0, 0].plot(df['Step'], df['PE(kJ/mol)'], 'b-', alpha=0.7)
                axes[0, 0].set_title('Potential Energy')
                axes[0, 0].set_ylabel('Energy (kJ/mol)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Temperature plot
            if 'Temp(K)' in df.columns:
                axes[0, 1].plot(df['Step'], df['Temp(K)'], 'r-', alpha=0.7)
                axes[0, 1].axhline(y=310, color='k', linestyle='--', label='Target')
                axes[0, 1].set_title('Temperature')
                axes[0, 1].set_ylabel('Temperature (K)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Volume plot
            if 'Volume(nm³)' in df.columns:
                axes[1, 0].plot(df['Step'], df['Volume(nm³)'], 'g-', alpha=0.7)
                axes[1, 0].set_title('Volume')
                axes[1, 0].set_ylabel('Volume (nm³)')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Performance plot
            if 'Speed(ns/day)' in df.columns:
                axes[1, 1].plot(df['Step'], df['Speed(ns/day)'], 'm-', alpha=0.7)
                axes[1, 1].set_title('Performance')
                axes[1, 1].set_ylabel('Speed (ns/day)')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = output_dir / "analysis_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"     📊 Plots saved: {plot_file}")
            
        except Exception as e:
            print(f"     ⚠️  Plot generation failed: {e}")

def main():
    """Main function for mixed system simulation"""
    if len(sys.argv) < 2:
        print("Usage: python openmm_md_mixed_system.py <pdb_file>")
        print("Example: python openmm_md_mixed_system.py insulin_embedded_09e0ce87.pdb")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    # Create mixed system simulator
    simulator = MixedSystemSimulator()
    
    # Run simulation
    results = simulator.run_mixed_simulation(
        pdb_file=pdb_file,
        temperature=310.0,
        equilibration_steps=5000,   # Shorter for testing
        production_steps=15000,     # Shorter for testing
        save_interval=250
    )
    
    if results['success']:
        print(f"\n🎉 Mixed system simulation completed successfully!")
        print(f"📊 Final Summary:")
        print(f"   🆔 ID: {results['simulation_id']}")
        print(f"   ⏱️  Time: {results['timing']['total']:.1f} seconds")
        print(f"   🚀 Performance: {results['performance']['ns_per_day']:.1f} ns/day")
        print(f"   🔬 System: {results['system_composition']['total_atoms']} atoms")
        print(f"   🧬 Protein: {results['system_composition']['protein_residues']} residues")
        print(f"   ❓ Polymer: {results['system_composition']['unknown_residues']} residues")
    else:
        print(f"❌ Simulation failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main() 