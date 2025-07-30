#!/usr/bin/env python3
"""
Fix for Polymer Simulation Issue

The main problem is that the template generator can't match the OpenFF molecule 
to the UNL residue in the topology. This happens because:

1. The OpenFF molecule has 207 atoms (with explicit hydrogens)
2. The PDB topology has 177 atoms (as read from PDB)
3. Template generator matching fails

Solution: Use the proper OpenMM forcefields workflow with topology-based molecule creation.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, Modeller, ForceField
    from openmm.app import HBonds, Simulation
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from pdbfixer import PDBFixer
    PDBFIXER_AVAILABLE = True
except ImportError:
    PDBFIXER_AVAILABLE = False

try:
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator, SystemGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False

try:
    from openff.toolkit import Molecule, Topology as OFFTopology
    OPENFF_TOOLKIT_AVAILABLE = True
except ImportError:
    OPENFF_TOOLKIT_AVAILABLE = False

class PolymerSimulationFixer:
    """
    Fixes polymer simulation issues by properly handling UNL residue parameterization
    """
    
    def __init__(self):
        """Initialize the polymer simulation fixer"""
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check that all required dependencies are available"""
        missing = []
        if not OPENMM_AVAILABLE:
            missing.append("openmm")
        if not PDBFIXER_AVAILABLE:
            missing.append("pdbfixer")
        if not OPENMMFORCEFIELDS_AVAILABLE:
            missing.append("openmmforcefields")
        if not OPENFF_TOOLKIT_AVAILABLE:
            missing.append("openff-toolkit")
        
        if missing:
            raise ImportError(f"Missing required dependencies: {missing}")
        
        print("✅ All dependencies available")
    
    def fix_pdb_preprocessing(self, pdb_file: str, output_path: str = None) -> str:
        """
        Preprocess PDB file to ensure proper connectivity and hydrogens
        
        Args:
            pdb_file: Input PDB file path
            output_path: Output path (optional)
            
        Returns:
            Path to the fixed PDB file
        """
        print(f"🔧 Preprocessing PDB file: {pdb_file}")
        
        if output_path is None:
            output_path = str(Path(pdb_file).with_suffix('.fixed.pdb'))
        
        # Initialize PDBFixer
        fixer = PDBFixer(filename=pdb_file)
        
        print(f"   📊 Initial structure: {len(list(fixer.topology.atoms()))} atoms")
        
        # Find and fix missing residues/atoms
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        
        # Add hydrogens at physiological pH
        fixer.addMissingHydrogens(7.4)
        
        print(f"   📊 Final structure: {len(list(fixer.topology.atoms()))} atoms")
        
        # Save fixed structure
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        print(f"   ✅ Fixed PDB saved: {output_path}")
        return output_path
    
    def create_molecule_from_topology(self, topology, positions) -> Optional[Molecule]:
        """
        Create OpenFF Molecule directly from OpenMM topology and positions
        
        This approach ensures the molecule matches the topology exactly.
        
        Args:
            topology: OpenMM topology
            positions: Atom positions
            
        Returns:
            OpenFF Molecule object or None if creation fails
        """
        print(f"   🧪 Creating OpenFF molecule from topology...")
        
        try:
            # Convert OpenMM topology to OpenFF topology
            off_topology = OFFTopology.from_openmm(topology, unique_molecules=[])
            
            if len(off_topology.molecules) == 0:
                print(f"   ⚠️  No molecules found in topology")
                return None
            
            # Take the first molecule (should be our polymer)
            molecule = off_topology.molecules[0]
            
            print(f"   ✅ Created molecule: {molecule.n_atoms} atoms, {molecule.n_bonds} bonds")
            print(f"   📊 Formula: {molecule.hill_formula}")
            
            return molecule
            
        except Exception as e:
            print(f"   ❌ Failed to create molecule from topology: {e}")
            return None
    
    def create_system_with_proper_template_generator(self, 
                                                   pdb_file: str,
                                                   temperature: float = 310.0,
                                                   force_field: str = "gaff-2.11") -> Tuple[mm.System, app.Topology, List]:
        """
        Create OpenMM system using proper template generator workflow
        
        Args:
            pdb_file: Path to PDB file
            temperature: Simulation temperature
            force_field: Small molecule force field to use
            
        Returns:
            Tuple of (system, topology, positions)
        """
        print(f"🚀 Creating system with proper template generator workflow")
        
        # Step 1: Load and preprocess PDB
        print(f"   📁 Loading PDB: {pdb_file}")
        pdb = PDBFile(pdb_file)
        
        # Check for UNL residues
        unl_residues = [res for res in pdb.topology.residues() if res.name == 'UNL']
        print(f"   📊 Found {len(unl_residues)} UNL residues")
        
        if not unl_residues:
            print(f"   ⚠️  No UNL residues found - using standard force field")
            forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
            system = forcefield.createSystem(pdb.topology)
            return system, pdb.topology, pdb.positions
        
        # Step 2: Create molecule from topology for UNL residues
        print(f"   🧪 Creating molecules for UNL residues...")
        
        molecules = []
        for i, residue in enumerate(unl_residues):
            print(f"      Processing UNL residue {i+1}/{len(unl_residues)}")
            
            # Create a sub-topology with just this residue
            modeller = Modeller(pdb.topology, pdb.positions)
            
            # Extract atoms for this residue
            residue_atoms = list(residue.atoms())
            
            # Create molecule from the residue topology
            try:
                # Method 1: Use SystemGenerator with explicit topology
                molecule = self.create_molecule_from_topology(pdb.topology, pdb.positions)
                if molecule:
                    molecules.append(molecule)
                    print(f"         ✅ Created molecule: {molecule.n_atoms} atoms")
                    break  # For now, just handle the first molecule
                
            except Exception as e:
                print(f"         ❌ Failed to create molecule: {e}")
                continue
        
        if not molecules:
            raise ValueError("Failed to create any molecules from UNL residues")
        
        # Step 3: Use SystemGenerator with pre-created molecules
        print(f"   ⚡ Creating system with SystemGenerator...")
        
        try:
            # Use SystemGenerator for proper molecule handling
            system_generator = SystemGenerator(
                forcefields=['amber/protein.ff14SB.xml', 'implicit/gbn2.xml'],
                small_molecule_forcefield=force_field,
                molecules=molecules,  # Pre-register our molecules
                cache='polymer_cache.json'
            )
            
            print(f"      🎯 Creating system from topology...")
            system = system_generator.create_system(pdb.topology)
            
            print(f"   ✅ System created successfully!")
            print(f"      🧪 Forces: {system.getNumForces()}")
            print(f"      ⚛️  Particles: {system.getNumParticles()}")
            
            return system, pdb.topology, pdb.positions
            
        except Exception as e:
            print(f"   ❌ SystemGenerator failed: {e}")
            # Fall back to manual template generator approach
            return self._fallback_manual_template_generator(pdb, molecules, force_field)
    
    def _fallback_manual_template_generator(self, pdb, molecules, force_field):
        """
        Fallback method using manual template generator registration
        """
        print(f"   🔄 Falling back to manual template generator...")
        
        # Create base force field
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Create and register template generator
        if force_field.startswith('gaff'):
            template_generator = GAFFTemplateGenerator(
                molecules=molecules, 
                forcefield=force_field
            )
        else:
            template_generator = SMIRNOFFTemplateGenerator(
                molecules=molecules,
                forcefield=force_field
            )
        
        # Register with force field
        forcefield.registerTemplateGenerator(template_generator.generator)
        print(f"      ✅ Template generator registered")
        
        # Create system
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0*unit.nanometer,
            constraints=HBonds,
            removeCMMotion=True,
            hydrogenMass=4*unit.amu
        )
        
        print(f"   ✅ System created with manual template generator!")
        return system, pdb.topology, pdb.positions
    
    def run_polymer_simulation(self, 
                             pdb_file: str,
                             temperature: float = 310.0,
                             equilibration_steps: int = 1000,
                             production_steps: int = 5000,
                             output_dir: str = "polymer_simulation") -> Dict:
        """
        Run a complete polymer simulation with proper UNL handling
        
        Args:
            pdb_file: Input PDB file
            temperature: Simulation temperature in Kelvin
            equilibration_steps: Number of equilibration steps
            production_steps: Number of production steps  
            output_dir: Output directory
            
        Returns:
            Dictionary with simulation results
        """
        print(f"🚀 Running polymer simulation: {pdb_file}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Step 1: Preprocess PDB
            fixed_pdb = self.fix_pdb_preprocessing(
                pdb_file, 
                str(output_path / "fixed_polymer.pdb")
            )
            
            # Step 2: Create system with proper template generators
            system, topology, positions = self.create_system_with_proper_template_generator(
                fixed_pdb, 
                temperature
            )
            
            # Step 3: Set up simulation
            print(f"   🎮 Setting up simulation...")
            
            # Choose platform
            platform = mm.Platform.getPlatformByName('CUDA')
            properties = {'CudaPrecision': 'mixed'}
            
            # Create integrator
            integrator = mm.LangevinMiddleIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtosecond
            )
            
            # Create simulation
            simulation = Simulation(topology, system, integrator, platform, properties)
            simulation.context.setPositions(positions)
            
            # Minimize energy
            print(f"   🔧 Minimizing energy...")
            simulation.minimizeEnergy()
            
            # Equilibration
            print(f"   🔄 Running equilibration ({equilibration_steps} steps)...")
            simulation.step(equilibration_steps)
            
            # Production with reporting
            print(f"   🏃 Running production ({production_steps} steps)...")
            
            # Set up reporters
            dcd_file = str(output_path / "trajectory.dcd")
            log_file = str(output_path / "simulation.log")
            
            simulation.reporters.append(
                app.DCDReporter(dcd_file, 100)
            )
            simulation.reporters.append(
                app.StateDataReporter(
                    log_file, 100,
                    step=True, time=True, potentialEnergy=True, 
                    temperature=True, speed=True
                )
            )
            
            # Run production
            simulation.step(production_steps)
            
            # Get final state
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            final_energy = state.getPotentialEnergy()
            
            print(f"   ✅ Simulation completed!")
            print(f"      🔋 Final energy: {final_energy}")
            
            # Save final structure
            final_pdb = str(output_path / "final_structure.pdb")
            with open(final_pdb, 'w') as f:
                PDBFile.writeFile(topology, state.getPositions(), f)
            
            return {
                'success': True,
                'final_energy': final_energy,
                'output_dir': str(output_path),
                'trajectory': dcd_file,
                'final_structure': final_pdb,
                'log_file': log_file
            }
            
        except Exception as e:
            print(f"   ❌ Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_dir': str(output_path)
            }
    
    def diagnose_polymer_issues(self, pdb_file: str) -> Dict:
        """
        Diagnose common polymer simulation issues
        
        Args:
            pdb_file: PDB file to diagnose
            
        Returns:
            Dictionary with diagnostic information
        """
        print(f"🔍 Diagnosing polymer simulation issues: {pdb_file}")
        
        issues = []
        recommendations = []
        
        try:
            # Load PDB
            pdb = PDBFile(pdb_file)
            
            # Check for UNL residues
            unl_residues = [res for res in pdb.topology.residues() if res.name == 'UNL']
            
            if not unl_residues:
                issues.append("No UNL residues found - check if polymer is properly named")
                recommendations.append("Ensure polymer residues are named 'UNL' in PDB file")
            else:
                print(f"   ✅ Found {len(unl_residues)} UNL residues")
            
            # Check atom count
            total_atoms = len(list(pdb.topology.atoms()))
            print(f"   📊 Total atoms: {total_atoms}")
            
            if total_atoms > 50000:
                issues.append("Large system may require explicit solvent or more resources")
                recommendations.append("Consider using smaller system or explicit solvent MD")
            
            # Check for missing hydrogens
            hydrogen_count = sum(1 for atom in pdb.topology.atoms() if atom.element.symbol == 'H')
            heavy_atom_count = total_atoms - hydrogen_count
            
            expected_hydrogens = heavy_atom_count * 1.5  # Rough estimate
            
            if hydrogen_count < expected_hydrogens * 0.5:
                issues.append("Possible missing hydrogens")
                recommendations.append("Use PDBFixer to add missing hydrogens")
            
            # Check connectivity
            bond_count = len(list(pdb.topology.bonds()))
            expected_bonds = total_atoms * 0.8  # Rough estimate
            
            if bond_count < expected_bonds * 0.5:
                issues.append("Possible missing connectivity information")
                recommendations.append("Ensure proper bond topology in PDB file")
            
            print(f"   📊 Bonds: {bond_count}")
            print(f"   📊 Hydrogens: {hydrogen_count}/{total_atoms}")
            
            return {
                'total_atoms': total_atoms,
                'unl_residues': len(unl_residues),
                'hydrogen_count': hydrogen_count,
                'bond_count': bond_count,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'issues': [f"Failed to load PDB: {e}"],
                'recommendations': ["Check PDB file format and accessibility"]
            }

def main():
    """Main function for testing the polymer simulation fix"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix polymer simulation issues")
    parser.add_argument("pdb_file", help="Input PDB file")
    parser.add_argument("--diagnose", action="store_true", help="Only diagnose issues")
    parser.add_argument("--temperature", type=float, default=310.0, help="Temperature (K)")
    parser.add_argument("--equilibration-steps", type=int, default=1000, help="Equilibration steps")
    parser.add_argument("--production-steps", type=int, default=5000, help="Production steps")
    parser.add_argument("--output-dir", default="polymer_simulation", help="Output directory")
    
    args = parser.parse_args()
    
    # Create fixer
    fixer = PolymerSimulationFixer()
    
    if args.diagnose:
        # Just diagnose issues
        print(f"🔍 Diagnostic Mode")
        diagnosis = fixer.diagnose_polymer_issues(args.pdb_file)
        
        print(f"\n📋 DIAGNOSTIC RESULTS")
        print(f"=" * 50)
        for key, value in diagnosis.items():
            if key not in ['issues', 'recommendations']:
                print(f"{key}: {value}")
        
        if diagnosis.get('issues'):
            print(f"\n⚠️  ISSUES FOUND:")
            for issue in diagnosis['issues']:
                print(f"   • {issue}")
        
        if diagnosis.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in diagnosis['recommendations']:
                print(f"   • {rec}")
    
    else:
        # Run full simulation
        print(f"🚀 Simulation Mode")
        results = fixer.run_polymer_simulation(
            args.pdb_file,
            temperature=args.temperature,
            equilibration_steps=args.equilibration_steps,
            production_steps=args.production_steps,
            output_dir=args.output_dir
        )
        
        print(f"\n📋 SIMULATION RESULTS")
        print(f"=" * 50)
        
        if results['success']:
            print(f"✅ SUCCESS!")
            print(f"Final energy: {results['final_energy']}")
            print(f"Output directory: {results['output_dir']}")
            print(f"Trajectory: {results['trajectory']}")
            print(f"Final structure: {results['final_structure']}")
        else:
            print(f"❌ FAILED!")
            print(f"Error: {results['error']}")

if __name__ == "__main__":
    main() 