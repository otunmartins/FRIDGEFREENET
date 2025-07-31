#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic script to understand the SystemGenerator UNL residue recognition issue

This script will help us debug exactly what's happening with the molecule/residue matching
"""

import traceback
from pathlib import Path

def debug_unl_residue_issue():
    """Debug the UNL residue recognition issue step by step"""
    print("🔍 DEBUGGING UNL RESIDUE RECOGNITION ISSUE")
    print("="*60)
    
    try:
        # Import required modules
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SystemGenerator
        from openmm.app import PDBFile
        import openmm.app as app
        
        print("✅ Imports successful")
        
        # Step 1: Load a real UNL frame file to examine its structure
        frame_file = "test_systemgenerator_mmgbsa/sim_89362bdb/frames/complex/frame_000000.pdb"
        
        if not Path(frame_file).exists():
            print(f"❌ Frame file not found: {frame_file}")
            return False
        
        print(f"✅ Found frame file: {frame_file}")
        
        # Load PDB and examine UNL residues
        pdb = PDBFile(frame_file)
        print(f"✅ Loaded PDB file")
        
        # Examine the topology for UNL residues
        unl_residues = []
        for residue in pdb.topology.residues():
            if residue.name == 'UNL':
                unl_residues.append(residue)
        
        print(f"📊 Found {len(unl_residues)} UNL residues in PDB")
        
        if unl_residues:
            first_unl = unl_residues[0]
            print(f"🧪 First UNL residue details:")
            print(f"   - Residue name: {first_unl.name}")
            print(f"   - Residue index: {first_unl.index}")
            print(f"   - Number of atoms: {len(list(first_unl.atoms()))}")
            
            atom_names = [atom.name for atom in first_unl.atoms()]
            print(f"   - Atom names: {atom_names[:10]}...")  # Show first 10
        
        # Step 2: Force regeneration of molecule with proper atom names
        print(f"\n🔧 Force regenerating UNL molecule with proper atom names...")
        
        # Import the calculator to use its extraction method
        from insulin_mmgbsa_calculator import InsulinMMGBSACalculator
        calculator = InsulinMMGBSACalculator("debug_temp")
        
        # Extract polymer molecules directly (this method returns molecules with correct atom names)
        polymer_molecules = calculator._extract_polymer_molecules_proven_approach(
            "integrated_md_simulations/sim_89362bdb/production/frames.pdb",
            Path("debug_temp")
        )
        
        if not polymer_molecules:
            print(f"❌ Failed to extract polymer molecules")
            return False
        
        molecule = polymer_molecules[0]  # Use the first extracted molecule
        print(f"✅ Extracted molecule directly with atom names")
        
        print(f"🧪 Molecule details:")
        print(f"   - Atoms: {molecule.n_atoms}")
        print(f"   - Bonds: {molecule.n_bonds}")
        print(f"   - SMILES: {molecule.to_smiles()}")
        
        # Get atom names from molecule (if available)
        try:
            atom_names_mol = [atom.name for atom in molecule.atoms]
            print(f"   - Molecule atom names: {atom_names_mol[:10]}...")
        except:
            print(f"   - Molecule atom names: Not available")
        
        # Step 3: Try the new ForceField + SMIRNOFFTemplateGenerator approach
        print(f"\n🔧 Testing ForceField + SMIRNOFFTemplateGenerator approach...")
        
        # Import the calculator to use its ForceField creation method
        forcefield = calculator._create_proven_forcefield_with_ligand_templates(polymer_molecules)
        print(f"✅ ForceField with templates created")
        
        # Step 4: Test system creation with each component separately
        print(f"\n🧪 Testing ForceField system creation for each component...")
        
        # Test 1: Complex system (should contain both insulin and UNL)
        print(f"   Testing complex system (insulin + UNL)...")
        try:
            complex_system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds
            )
            print(f"   ✅ SUCCESS: Complex system created!")
            
            # Now test individual components using the split frame files
            frames_base_dir = Path("test_systemgenerator_mmgbsa/sim_89362bdb/frames")
            
            # Test 2: Receptor-only system (should contain only insulin)
            print(f"   Testing receptor-only system (insulin only)...")
            receptor_file = frames_base_dir / "receptor" / "frame_000000.pdb"
            if receptor_file.exists():
                receptor_pdb = PDBFile(str(receptor_file))
                try:
                    receptor_system = forcefield.createSystem(
                        receptor_pdb.topology,
                        nonbondedMethod=app.NoCutoff,
                        constraints=app.HBonds
                    )
                    print(f"   ✅ SUCCESS: Receptor system created!")
                except Exception as e:
                    print(f"   ❌ FAILED: Receptor system failed: {str(e)}")
            else:
                print(f"   ⚠️ Receptor file not found: {receptor_file}")
            
            # Test 3: Ligand-only system (should contain only UNL)
            print(f"   Testing ligand-only system (UNL only)...")
            ligand_file = frames_base_dir / "ligand" / "frame_000000.pdb"
            if ligand_file.exists():
                ligand_pdb = PDBFile(str(ligand_file))
                try:
                    ligand_system = forcefield.createSystem(
                        ligand_pdb.topology,
                        nonbondedMethod=app.NoCutoff,
                        constraints=app.HBonds
                    )
                    print(f"   ✅ SUCCESS: Ligand system created!")
                    return True
                except Exception as e:
                    print(f"   ❌ FAILED: Ligand system failed: {str(e)}")
                    print(f"      This is likely where the UNL template issue occurs!")
            else:
                print(f"   ⚠️ Ligand file not found: {ligand_file}")
            
            return False
            
        except Exception as e:
            print(f"   ❌ FAILED: Complex system failed: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_unl_residue_issue()
    print(f"\n🎯 Debugging result: {'SUCCESS' if success else 'FAILED'}") 