#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final diagnostic script to understand the UNL residue structure mismatch
"""

def analyze_unl_structure():
    """Analyze the exact structure of UNL residues in the PDB file"""
    print("🔬 FINAL UNL STRUCTURE ANALYSIS")
    print("="*60)
    
    try:
        import openmm.app as app
        from openff.toolkit import Molecule
        from pathlib import Path
        
        # Load the complex frame
        frame_file = "test_systemgenerator_mmgbsa/sim_89362bdb/frames/complex/frame_000000.pdb"
        pdb = app.PDBFile(frame_file)
        
        print(f"✅ Loaded PDB file: {frame_file}")
        
        # Find first UNL residue and examine its detailed structure
        unl_residue = None
        for residue in pdb.topology.residues():
            if residue.name == 'UNL':
                unl_residue = residue
                break
        
        if unl_residue:
            print(f"\n🧪 First UNL residue details:")
            print(f"   - Residue name: {unl_residue.name}")
            print(f"   - Residue index: {unl_residue.index}")
            print(f"   - Number of atoms: {len(list(unl_residue.atoms()))}")
            
            # Get all atom details
            atoms = list(unl_residue.atoms())
            print(f"   - Atom details:")
            for i, atom in enumerate(atoms):
                print(f"     {i:2d}: {atom.name:4s} {atom.element.symbol:2s}")
            
            # Get bonds within this residue
            bonds = []
            for bond in pdb.topology.bonds():
                atom1, atom2 = bond
                if atom1.residue == unl_residue and atom2.residue == unl_residue:
                    bonds.append((atom1.name, atom2.name))
            
            print(f"   - Bonds within residue: {len(bonds)}")
            for bond in bonds[:10]:  # Show first 10 bonds
                print(f"     {bond[0]} - {bond[1]}")
            if len(bonds) > 10:
                print(f"     ... and {len(bonds) - 10} more bonds")
        
        # Now examine our extracted molecule
        print(f"\n🧪 Our extracted molecule details:")
        
        # Import the MMGBSA calculator to get the molecule
        from insulin_mmgbsa_calculator import InsulinMMGBSACalculator
        calculator = InsulinMMGBSACalculator("final_debug")
        
        # Extract polymer molecules
        polymer_molecules = calculator._extract_polymer_molecules_proven_approach(
            "integrated_md_simulations/sim_89362bdb/production/frames.pdb",
            Path("final_debug")
        )
        
        if polymer_molecules:
            molecule = polymer_molecules[0]
            print(f"   - Number of atoms: {molecule.n_atoms}")
            print(f"   - Number of bonds: {molecule.n_bonds}")
            
            # Show atom details
            print(f"   - Atom details:")
            for i, atom in enumerate(molecule.atoms):
                print(f"     {i:2d}: {atom.name:4s} {atom.element.symbol:2s}")
            
            # Show bond details  
            print(f"   - Bond details:")
            for i, bond in enumerate(molecule.bonds):
                if i < 10:  # Show first 10 bonds
                    atom1_name = bond.atom1.name
                    atom2_name = bond.atom2.name
                    print(f"     {atom1_name} - {atom2_name}")
                elif i == 10:
                    print(f"     ... and {molecule.n_bonds - 10} more bonds")
                    break
        
        # Check if atom counts match
        if unl_residue and polymer_molecules:
            pdb_atom_count = len(list(unl_residue.atoms()))
            mol_atom_count = polymer_molecules[0].n_atoms
            pdb_bond_count = len(bonds)
            mol_bond_count = polymer_molecules[0].n_bonds
            
            print(f"\n📊 Structure comparison:")
            print(f"   PDB UNL residue: {pdb_atom_count} atoms, {pdb_bond_count} bonds")
            print(f"   Our molecule:    {mol_atom_count} atoms, {mol_bond_count} bonds")
            
            if pdb_atom_count == mol_atom_count and pdb_bond_count == mol_bond_count:
                print(f"   ✅ Atom and bond counts match!")
            else:
                print(f"   ❌ MISMATCH: This could be the issue!")
                
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = analyze_unl_structure()
    print(f"\n🎯 Analysis result: {'SUCCESS' if success else 'FAILED'}") 