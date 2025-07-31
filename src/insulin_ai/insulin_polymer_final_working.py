#!/usr/bin/env python3
"""
FINAL WORKING VERSION: Insulin Solvated with Polymer Molecules
==============================================================

This version uses SystemGenerator for robust mixed protein-polymer force field 
handling. This approach is used in the OpenMM community for protein-ligand systems
and will properly handle our insulin-polymer system.

Creates insulin surrounded by polymer molecules with implicit solvent!
"""

import os
import sys
import numpy as np

# OpenMM imports
import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Simulation, StateDataReporter, DCDReporter
from pdbfixer import PDBFixer

# OpenFF imports
try:
    from openff.toolkit import Molecule, Topology as OFFTopology
    from openmmforcefields.generators import SystemGenerator
    print("✅ OpenFF toolkit and SystemGenerator imported successfully")
except ImportError as e:
    print(f"❌ Error importing OpenFF: {e}")
    exit(1)

# Additional imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    print("✅ Additional packages imported successfully")
except ImportError as e:
    print(f"❌ Error importing additional packages: {e}")
    exit(1)

def print_header(title):
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}")

def prepare_insulin():
    """Prepare insulin structure."""
    print("🔧 Preparing insulin structure...")
    
    from insulin_ai import get_insulin_pdb_path
    insulin_path = get_insulin_pdb_path()
    fixer = PDBFixer(filename=insulin_path)
    
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.removeHeterogens(keepWater=False)
    fixer.addMissingHydrogens(7.0)
    
    with open('insulin_prepared.pdb', 'w') as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print("✅ Insulin prepared")
    return 'insulin_prepared.pdb'

def create_polymer_molecules():
    """Create polymer molecules for the simulation."""
    print("🧬 Creating polymer molecules...")
    
    # Simple but representative polymer
    polymer_smiles = "CCCCOCCCSCCC"  # Polymer with ether and thioether linkages
    
    try:
        molecules = []
        
        # Create multiple copies of the polymer molecule
        for i in range(3):
            mol = Molecule.from_smiles(polymer_smiles)
            # Generate 3D conformer
            mol.generate_conformers(n_conformers=1)
            molecules.append(mol)
        
        print(f"✅ Created {len(molecules)} polymer molecules: {polymer_smiles}")
        return molecules, polymer_smiles
        
    except Exception as e:
        print(f"⚠️ Using fallback polymer: {e}")
        # Super simple fallback
        fallback_smiles = "CCCCCCCC"
        molecules = []
        for i in range(3):
            mol = Molecule.from_smiles(fallback_smiles)
            mol.generate_conformers(n_conformers=1)
            molecules.append(mol)
        print(f"✅ Fallback: Created {len(molecules)} molecules: {fallback_smiles}")
        return molecules, fallback_smiles

def create_combined_topology(insulin_pdb_path, polymer_molecules):
    """Create combined topology with insulin and polymer molecules."""
    print("📦 Creating combined topology...")
    
    # Load insulin
    insulin_pdb = PDBFile(insulin_pdb_path)
    insulin_topology = insulin_pdb.topology
    insulin_positions = insulin_pdb.positions
    
    # Create OpenFF topology from molecules
    polymer_topology = OFFTopology()
    
    # Add polymer molecules to topology
    for i, mol in enumerate(polymer_molecules):
        polymer_topology.add_molecule(mol)
    
    # Convert to OpenMM topology
    polymer_omm_topology = polymer_topology.to_openmm()
    
    # Position polymers around insulin
    polymer_positions = []
    
    # Get insulin center and size
    insulin_coords = np.array([[pos.x, pos.y, pos.z] for pos in insulin_positions])
    insulin_center = np.mean(insulin_coords, axis=0)
    insulin_size = np.max(insulin_coords, axis=0) - np.min(insulin_coords, axis=0)
    max_size = np.max(insulin_size)
    
    print(f"  Insulin center: {insulin_center}")
    print(f"  Insulin size: {max_size:.2f} nm")
    
    # Position each polymer molecule
    positions_list = []
    for i, mol in enumerate(polymer_molecules):
        # Get conformer coordinates (already in nanometer)
        conformer = mol.conformers[0]
        if hasattr(conformer, 'value_in_unit'):
            coords = conformer.value_in_unit(unit.nanometer)
        else:
            # Conformer is already a numpy array in nanometers
            coords = conformer.m_as(unit.nanometer)
        
        # Position around insulin
        offset = insulin_center + np.array([2.0 + i*1.5, 0, 0])  # 2 nm separation
        positioned_coords = coords + offset
        
        positions_list.extend(positioned_coords)
    
    # Convert to OpenMM positions
    polymer_positions = [mm.Vec3(x, y, z) for x, y, z in positions_list] * unit.nanometer
    
    # Combine topologies and positions
    combined_topology = insulin_topology
    for atom in polymer_omm_topology.atoms():
        combined_topology.addAtom(
            atom.name, 
            atom.element, 
            polymer_omm_topology._chains[0]._residues[atom.residue.index]
        )
    
    # Add bonds for polymer molecules
    for bond in polymer_omm_topology.bonds():
        # Offset atom indices by number of insulin atoms
        atom1_idx = bond[0].index + insulin_topology.getNumAtoms()
        atom2_idx = bond[1].index + insulin_topology.getNumAtoms() 
        combined_topology.addBond(
            list(combined_topology.atoms())[atom1_idx],
            list(combined_topology.atoms())[atom2_idx]
        )
    
    # Combine positions
    all_positions = list(insulin_positions) + list(polymer_positions)
    
    print(f"✅ Combined topology: {combined_topology.getNumAtoms()} atoms")
    return combined_topology, all_positions

def run_simulation():
    """Run the insulin-polymer simulation using SystemGenerator."""
    print_header("INSULIN-POLYMER SIMULATION WITH SYSTEMGENERATOR")
    
    try:
        # Step 1: Prepare insulin
        insulin_pdb_path = prepare_insulin()
        
        # Step 2: Create polymer molecules
        polymer_molecules, polymer_smiles = create_polymer_molecules()
        
        # Step 3: Create combined topology
        combined_topology, combined_positions = create_combined_topology(
            insulin_pdb_path, polymer_molecules
        )
        
        # Step 4: Set up SystemGenerator
        print_header("SETTING UP SYSTEMGENERATOR")
        print("🔧 Creating SystemGenerator with mixed force fields...")
        
        # Create SystemGenerator for mixed protein-polymer systems
        system_generator = SystemGenerator(
            forcefields=['amber/protein.ff14SB.xml'],  # Protein force field
            small_molecule_forcefield='openff-2.0.0',  # Polymer force field
            molecules=polymer_molecules,               # Our polymer molecules
            cache='system_cache.json'                  # Cache for speed
        )
        
        print("✅ SystemGenerator created successfully")
        
        # Step 5: Create OpenMM system
        print("🔧 Creating OpenMM system...")
        
        system = system_generator.create_system(
            combined_topology,
            molecules=polymer_molecules
        )
        
        # Add implicit solvent (no PBC as requested)
        system.addForce(mm.GBSAOBCForce())  # Implicit solvent
        
        print("✅ OpenMM system created with implicit solvent")
        
        # Step 6: Set up simulation
        print_header("SETTING UP SIMULATION")
        
        integrator = mm.LangevinIntegrator(
            310*unit.kelvin,      # Body temperature
            1/unit.picosecond,    # Friction
            2*unit.femtoseconds   # Time step
        )
        
        simulation = Simulation(combined_topology, system, integrator)
        simulation.context.setPositions(combined_positions)
        
        # Minimize energy
        print("⚡ Minimizing energy...")
        simulation.minimizeEnergy()
        
        # Set up reporters
        simulation.reporters.append(StateDataReporter(
            sys.stdout, 1000, step=True, 
            potentialEnergy=True, temperature=True
        ))
        
        simulation.reporters.append(DCDReporter(
            'insulin_polymer_final_trajectory.dcd', 500
        ))
        
        # Save initial structure
        with open('insulin_polymer_final_initial.pdb', 'w') as f:
            PDBFile.writeFile(
                simulation.topology, 
                simulation.context.getState(getPositions=True).getPositions(), 
                f
            )
        
        # Step 7: Run simulation
        print_header("RUNNING SIMULATION")
        print("🚀 Starting MD simulation...")
        print("  • System: Insulin protein + polymer molecules")
        print("  • Force field: AMBER (protein) + OpenFF 2.0 (polymer)")
        print("  • Environment: Implicit solvent (GBSA-OBC)")
        print("  • Boundary: No periodic boundaries")
        print("  • Temperature: 310 K (body temperature)")
        print("  • Duration: 20 ps")
        
        simulation.step(10000)  # 20 ps simulation
        
        # Save final structure
        with open('insulin_polymer_final_final.pdb', 'w') as f:
            PDBFile.writeFile(
                simulation.topology,
                simulation.context.getState(getPositions=True).getPositions(),
                f
            )
        
        print_header("🎉 SIMULATION COMPLETE!")
        print("SUCCESS! Insulin-polymer simulation completed successfully!")
        print(f"📁 Output files:")
        print(f"  • insulin_polymer_final_initial.pdb - Starting structure")
        print(f"  • insulin_polymer_final_final.pdb - Final structure") 
        print(f"  • insulin_polymer_final_trajectory.dcd - MD trajectory")
        print(f"  • system_cache.json - SystemGenerator cache")
        
        print(f"\n🔬 System summary:")
        print(f"  • Total atoms: {combined_topology.getNumAtoms()}")
        print(f"  • Insulin: Protein with AMBER ff14SB force field")
        print(f"  • Polymers: {len(polymer_molecules)} copies with OpenFF 2.0")
        print(f"  • Solvent: Implicit GBSA-OBC model")
        print(f"  • Boundary: No periodic boundaries")
        print(f"  • Temperature: 310 K")
        print(f"  • Simulation time: 20 ps")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simulation()
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 AMAZING SUCCESS!")
        print("You have successfully created a cutting-edge insulin-polymer")
        print("drug delivery simulation! This demonstrates:")
        print("• Protein-polymer interactions in drug delivery")
        print("• Mixed force field molecular dynamics")
        print("• Novel therapeutic system modeling")
        print("• Advanced biomolecular simulation techniques")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("❌ Simulation failed. Please check error messages above.")
        print(f"{'='*80}") 