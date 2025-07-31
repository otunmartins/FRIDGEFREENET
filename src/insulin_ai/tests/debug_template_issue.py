#!/usr/bin/env python3
"""
Debugging script for UNL residue template registration issue

This script will help us understand exactly what's happening with the
template registration and where the breakdown occurs.
"""

import traceback
import tempfile
from pathlib import Path

# Import required modules
try:
    import openmm as mm
    import openmm.app as app
    from openmm.app import PDBFile, ForceField
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    print("✅ All required modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def create_test_pdb_with_unl():
    """Create a minimal test PDB file with UNL residues"""
    pdb_content = """HEADER    TEST PDB WITH UNL RESIDUE
ATOM      1  C1  UNL     1      -1.234   2.345   0.000  1.00 20.00           C  
ATOM      2  C2  UNL     1      -0.123   1.234   0.000  1.00 20.00           C  
ATOM      3  C3  UNL     1       1.000   2.100   0.000  1.00 20.00           C  
ATOM      4  H1  UNL     1      -2.123   2.890   0.000  1.00 20.00           H  
ATOM      5  H2  UNL     1      -0.234   0.345   0.000  1.00 20.00           H  
ATOM      6  H3  UNL     1       1.890   1.555   0.000  1.00 20.00           H  
CONECT    1    2    4
CONECT    2    1    3    5
CONECT    3    2    6
CONECT    4    1
CONECT    5    2
CONECT    6    3
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        return f.name

def debug_template_registration():
    """Debug the template registration process step by step"""
    
    print("🔍 Step 1: Creating test PDB with UNL residue...")
    pdb_file = create_test_pdb_with_unl()
    print(f"   Created: {pdb_file}")
    
    print("\n🔍 Step 2: Loading PDB file...")
    try:
        pdb = PDBFile(pdb_file)
        print(f"   ✅ PDB loaded with {len(list(pdb.topology.atoms()))} atoms")
        
        # Analyze topology
        print("   📊 Topology analysis:")
        for residue in pdb.topology.residues():
            print(f"      Residue: {residue.name} (ID: {residue.id})")
            print(f"         Atoms: {[atom.name for atom in residue.atoms()]}")
            
    except Exception as e:
        print(f"   ❌ Failed to load PDB: {e}")
        return False
    
    print("\n🔍 Step 3: Creating simple polymer molecule...")
    try:
        # Create a simple alkane chain that should match our test structure
        molecule = Molecule.from_smiles("CCC")  # Simple 3-carbon chain
        print(f"   ✅ Created molecule: {molecule.to_smiles()}")
        print(f"      Atoms: {molecule.n_atoms}, Bonds: {molecule.n_bonds}")
    except Exception as e:
        print(f"   ❌ Failed to create molecule: {e}")
        return False
    
    print("\n🔍 Step 4: Testing different template registration approaches...")
    
    # Approach 1: Simple SMIRNOFFTemplateGenerator (like working examples)
    print("\n   🧪 Approach 1: Simple SMIRNOFFTemplateGenerator")
    try:
        forcefield1 = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Pass molecule directly (not as list) like in working examples
        template_gen1 = SMIRNOFFTemplateGenerator(molecules=molecule)
        forcefield1.registerTemplateGenerator(template_gen1.generator)
        
        print("      ✅ Template generator created and registered")
        
        # Try to create system
        system1 = forcefield1.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        print("      ✅ SUCCESS: System created with simple approach!")
        return True
        
    except Exception as e:
        print(f"      ❌ Simple approach failed: {e}")
        print(f"         Details: {traceback.format_exc()}")
    
    # Approach 2: With residue_templates mapping
    print("\n   🧪 Approach 2: With residue_templates mapping")
    try:
        forcefield2 = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Try the residue_templates approach
        residue_templates = {'UNL': molecule}
        template_gen2 = SMIRNOFFTemplateGenerator(
            molecules=[molecule],
            residue_templates=residue_templates
        )
        forcefield2.registerTemplateGenerator(template_gen2.generator)
        
        print("      ✅ Template generator with residue mapping created")
        
        # Try to create system
        system2 = forcefield2.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        print("      ✅ SUCCESS: System created with residue mapping!")
        return True
        
    except Exception as e:
        print(f"      ❌ Residue mapping approach failed: {e}")
        print(f"         Details: {traceback.format_exc()}")
    
    # Approach 3: Direct ForceField.createSystem with residueTemplates
    print("\n   🧪 Approach 3: Direct ForceField.createSystem with residueTemplates")
    try:
        forcefield3 = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        template_gen3 = SMIRNOFFTemplateGenerator(molecules=[molecule])
        forcefield3.registerTemplateGenerator(template_gen3.generator)
        
        # Use the residueTemplates parameter of createSystem
        topology_residues = {}
        for residue in pdb.topology.residues():
            if residue.name == 'UNL':
                topology_residues[residue] = 'UNL'  # Map residue object to template name
        
        system3 = forcefield3.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            residueTemplates=topology_residues
        )
        print("      ✅ SUCCESS: System created with ForceField residueTemplates!")
        return True
        
    except Exception as e:
        print(f"      ❌ ForceField residueTemplates approach failed: {e}")
        print(f"         Details: {traceback.format_exc()}")
    
    print("\n🔍 Step 5: Inspecting template generator state...")
    try:
        # Check what templates are actually registered
        print("   📋 Checking registered generators...")
        generators = forcefield1.getGenerators()
        print(f"      Found {len(generators)} generators")
        
        # Check for unmatched residues
        unmatched = forcefield1.getUnmatchedResidues(pdb.topology)
        print(f"      Unmatched residues: {[res.name for res in unmatched]}")
        
    except Exception as e:
        print(f"   ❌ Error inspecting state: {e}")
    
    return False

def test_alternative_molecule_creation():
    """Test different ways to create the polymer molecule"""
    
    print("\n🔍 Testing different molecule creation approaches...")
    
    # Method 1: From SMILES
    print("\n   🧪 Method 1: From SMILES")
    try:
        mol1 = Molecule.from_smiles("CCCCCCCCCC")  # 10-carbon chain
        print(f"      ✅ Created from SMILES: {mol1.n_atoms} atoms")
    except Exception as e:
        print(f"      ❌ SMILES method failed: {e}")
    
    # Method 2: From file (if we have one)
    print("\n   🧪 Method 2: From SDF file (if available)")
    # This would require an existing SDF file
    print("      ⏭️ Skipping - would need existing SDF file")

if __name__ == "__main__":
    print("🧮 UNL Residue Template Registration Debugger")
    print("=" * 50)
    
    success = debug_template_registration()
    
    if not success:
        test_alternative_molecule_creation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Found a working approach!")
    else:
        print("❌ All approaches failed - need to investigate further")
    
    print("�� Debug complete") 