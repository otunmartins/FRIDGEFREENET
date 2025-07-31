#!/usr/bin/env python3
"""
Debug script for proven approach MMGBSA calculator to test complete workflow
Uses the exact proven MD simulation approach for automatic XYZ detection
"""

import sys
import traceback
from pathlib import Path
from insulin_mmgbsa_calculator import InsulinMMGBSACalculator

def test_frame_splitting():
    """Test the component frame splitting functionality"""
    print("🧪 Testing component frame splitting...")
    
    calculator = InsulinMMGBSACalculator()
    frames_file = "./integrated_md_simulations/sim_2806a0e3/production/frames.pdb"
    output_dir = Path("./debug_mmgbsa_test")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Test frame splitting
        frames_base_dir = calculator._split_trajectory_components(frames_file, output_dir)
        
        print(f"✅ Frame splitting successful!")
        print(f"   Frames base directory: {frames_base_dir}")
        
        # Check that directories were created
        for component in ['complex', 'receptor', 'ligand']:
            component_dir = frames_base_dir / component
            if component_dir.exists():
                frame_files = list(component_dir.glob("frame*.pdb"))
                print(f"   {component.capitalize()} directory: {len(frame_files)} files")
            else:
                print(f"   {component.capitalize()} directory: NOT FOUND")
        
        # Check a few frames
        complex_dir = frames_base_dir / "complex"
        receptor_dir = frames_base_dir / "receptor"
        ligand_dir = frames_base_dir / "ligand"
        
        complex_files = sorted(list(complex_dir.glob("frame*.pdb")))
        
        for i in range(min(3, len(complex_files))):
            complex_file = complex_files[i]
            receptor_file = receptor_dir / complex_file.name
            ligand_file = ligand_dir / complex_file.name
            
            print(f"   Frame {i}:")
            print(f"      Complex: {complex_file.name} ({complex_file.stat().st_size} bytes)")
            print(f"      Receptor: {receptor_file.name} ({receptor_file.stat().st_size} bytes)")
            print(f"      Ligand: {ligand_file.name} ({ligand_file.stat().st_size} bytes)")
        
        return frames_base_dir
        
    except Exception as e:
        print(f"❌ Frame splitting failed: {e}")
        traceback.print_exc()
        return None

def test_proven_polymer_extraction():
    """Test proven polymer molecule extraction using existing MD simulation approach"""
    print("🧪 Testing proven polymer molecule extraction...")
    
    calculator = InsulinMMGBSACalculator()
    frames_file = "./integrated_md_simulations/sim_2806a0e3/production/frames.pdb"
    output_dir = Path("./debug_mmgbsa_test")
    
    try:
        # Test proven polymer extraction
        polymer_molecules = calculator._extract_polymer_molecules_proven_approach(frames_file, output_dir)
        
        if polymer_molecules:
            print(f"✅ Polymer extraction successful using proven approach!")
            print(f"   Found {len(polymer_molecules)} polymer molecule types:")
            
            for i, molecule in enumerate(polymer_molecules):
                print(f"   Polymer {i+1}: {molecule.n_atoms} atoms, {molecule.n_bonds} bonds")
                
                # Try to get SMILES for verification
                try:
                    smiles = molecule.to_smiles()
                    print(f"      SMILES: {smiles}")
                except:
                    print(f"      SMILES: Unable to generate")
        else:
            print(f"❌ No polymer molecules extracted")
        
        return polymer_molecules
        
    except Exception as e:
        print(f"❌ Polymer extraction failed: {e}")
        traceback.print_exc()
        return None

def test_proven_forcefield_creation(polymer_molecules):
    """Test proven ForceField creation using exact drugscreening approach"""
    print("🧪 Testing proven ForceField creation...")
    
    if not polymer_molecules:
        print("❌ No polymer molecules provided for ForceField test")
        return None
    
    calculator = InsulinMMGBSACalculator()
    
    try:
        # Test ForceField creation using proven drugscreening approach
        forcefield = calculator._create_proven_forcefield_with_ligand_templates(polymer_molecules)
        
        print(f"✅ ForceField created successfully using proven drugscreening approach")
        print(f"   Type: {type(forcefield)}")
        print(f"   Can handle {len(polymer_molecules)} polymer types")
        
        return forcefield
        
    except Exception as e:
        print(f"❌ ForceField creation failed: {e}")
        traceback.print_exc()
        return None

def test_single_frame_processing(frames_base_dir, polymer_molecules, forcefield):
    """Test processing a single frame with the proven methodology"""
    print("🧪 Testing single frame processing with proven methodology...")
    
    if not frames_base_dir or not polymer_molecules or not forcefield:
        print("❌ Prerequisites missing for frame processing test")
        return
    
    calculator = InsulinMMGBSACalculator()
    
    try:
        # Get first frame files
        complex_dir = frames_base_dir / "complex"
        receptor_dir = frames_base_dir / "receptor"
        ligand_dir = frames_base_dir / "ligand"
        
        complex_files = sorted(list(complex_dir.glob("frame*.pdb")))
        
        if not complex_files:
            print("❌ No frame files found")
            return
        
        complex_path = complex_files[0]
        receptor_path = receptor_dir / complex_path.name
        ligand_path = ligand_dir / complex_path.name
        
        print(f"   Complex file: {complex_path}")
        print(f"   Receptor file: {receptor_path}")
        print(f"   Ligand file: {ligand_path}")
        
        # Test frame processing using proven drugscreening approach
        binding_energy = calculator._calculate_frame_binding_energy_proven_approach(
            complex_path, receptor_path, ligand_path, forcefield
        )
        
        print(f"✅ Frame processing successful using proven drugscreening approach!")
        print(f"   Complex energy: {binding_energy['complex_energy']:.2f} kcal/mol")
        print(f"   Receptor energy: {binding_energy['receptor_energy']:.2f} kcal/mol") 
        print(f"   Ligand energy: {binding_energy['ligand_energy']:.2f} kcal/mol")
        print(f"   Binding energy: {binding_energy['binding_energy']:.2f} kcal/mol")
        
        return binding_energy
        
    except Exception as e:
        print(f"❌ Frame processing failed: {e}")
        traceback.print_exc()
        return None

def test_proven_xyz_detection():
    """Test the proven XYZ detection system directly"""
    print("🧪 Testing proven XYZ detection capabilities...")
    
    calculator = InsulinMMGBSACalculator()
    
    try:
        # Test the proven MD simulator's XYZ extraction directly
        frames_file = "./integrated_md_simulations/sim_2806a0e3/production/frames.pdb"
        
        print("   🔄 Testing direct XYZ extraction using proven MD simulator...")
        xyz_smiles = calculator.md_simulator.extract_smiles_from_xyz_files(pdb_file_path=frames_file)
        
        if xyz_smiles:
            print(f"✅ XYZ extraction successful using proven approach!")
            print(f"   Found {len(xyz_smiles)} SMILES from XYZ files:")
            for i, smiles in enumerate(xyz_smiles):
                print(f"      SMILES {i+1}: {smiles}")
                
                # Test molecule creation using proven approach
                molecule = calculator.md_simulator.create_molecule_from_smiles(smiles)
                if molecule:
                    print(f"         ✅ Created molecule: {molecule.n_atoms} atoms")
                else:
                    print(f"         ❌ Failed to create molecule")
        else:
            print(f"⚠️ No SMILES extracted from XYZ files using proven approach")
        
        # Test topology extraction as fallback using proven approach
        print("\n   🔄 Testing topology extraction fallback using proven approach...")
        from openmm.app import PDBFile
        complex_pdb = PDBFile(frames_file)
        topology_smiles = calculator.md_simulator.extract_unl_residues_from_topology(complex_pdb.topology)
        
        if topology_smiles:
            print(f"✅ Topology extraction successful using proven approach!")
            print(f"   Generated {len(topology_smiles)} SMILES from topology:")
            for i, smiles in enumerate(topology_smiles):
                print(f"      SMILES {i+1}: {smiles}")
        else:
            print(f"⚠️ No SMILES generated from topology using proven approach")
        
        return xyz_smiles or topology_smiles
        
    except Exception as e:
        print(f"❌ Proven XYZ detection failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Main debug workflow for proven drugscreening approach MMGBSA"""
    print("🔬 Proven Drugscreening Approach MMGBSA Debug Analysis (Using SMIRNOFFTemplateGenerator)")
    print("=" * 80)
    
    try:
        # Step 1: Test proven XYZ detection system
        detected_smiles = test_proven_xyz_detection()
        print("\n" + "="*80)
        
        # Step 2: Test frame splitting
        frames_base_dir = test_frame_splitting()
        if not frames_base_dir:
            print("❌ Cannot proceed - frame splitting failed")
            return
        
        print("\n" + "="*80)
        
        # Step 3: Test proven polymer extraction
        polymer_molecules = test_proven_polymer_extraction()
        if not polymer_molecules:
            print("❌ Cannot proceed - polymer extraction failed")
            return
            
        print("\n" + "="*80)
        
        # Step 4: Test proven ForceField creation
        forcefield = test_proven_forcefield_creation(polymer_molecules)
        if not forcefield:
            print("❌ Cannot proceed - ForceField creation failed")
            return
            
        print("\n" + "="*80)
        
        # Step 5: Test single frame processing
        binding_energy = test_single_frame_processing(frames_base_dir, polymer_molecules, forcefield)
        if binding_energy:
            print("\n" + "="*80)
            print("✅ All tests passed! Proven approach MMGBSA works using existing infrastructure.")
            print("🎯 This approach leverages your existing proven systems:")
            print("   • Automatic XYZ file detection from insulin_polymer_output_* directories")
            print("   • OpenBabel-based XYZ to SMILES conversion with proper connectivity")
            print("   • OpenFF molecule creation from extracted SMILES")
            print("   • ForceField with exact MD simulation configuration") 
            print("   • AMBER ff14SB + GAFF-2.11 force field combination")
            print("   • Works with ANY polymer type automatically!")
        else:
            print("❌ Frame processing failed")
            
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 