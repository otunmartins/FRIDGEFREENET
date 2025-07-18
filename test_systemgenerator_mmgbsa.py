#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the updated SystemGenerator-based MMGBSA calculator

This script validates that the MMGBSA calculator works correctly with the
new SystemGenerator approach instead of manual template registration.
"""

import os
import sys
import time
import traceback
from pathlib import Path

def test_systemgenerator_mmgbsa():
    """Test the SystemGenerator-based MMGBSA calculator"""
    print("\n" + "="*70)
    print("🧪 TESTING SYSTEMGENERATOR-BASED MMGBSA CALCULATOR")
    print("="*70)
    
    try:
        # Import the updated calculator
        from insulin_mmgbsa_calculator import InsulinMMGBSACalculator
        
        print("✅ Successfully imported InsulinMMGBSACalculator")
        
        # Initialize calculator
        calculator = InsulinMMGBSACalculator("test_systemgenerator_mmgbsa")
        print("✅ Successfully initialized MMGBSA calculator")
        
        # Test basic functionality
        assert calculator.output_dir.exists(), "Output directory not created"
        assert calculator.platform is not None, "Platform not initialized"
        print(f"✅ Platform: {calculator.platform.getName()}")
        
        # Test on real simulation data
        simulation_dir = "integrated_md_simulations"
        simulation_id = "sim_89362bdb"
        
        # Check if test data exists
        test_frames = Path(simulation_dir) / simulation_id / "production" / "frames.pdb"
        if not test_frames.exists():
            print(f"⚠️ Test data not found: {test_frames}")
            print("   Creating minimal test data...")
            return test_minimal_functionality(calculator)
        
        print(f"✅ Found test data: {test_frames}")
        print(f"   File size: {test_frames.stat().st_size / (1024*1024):.1f} MB")
        
        # Test the MMGBSA calculation
        print("\n🧮 Starting MMGBSA calculation with SystemGenerator...")
        
        def progress_callback(message):
            print(f"   {message}")
        
        # Calculate binding energy
        start_time = time.time()
        results = calculator.calculate_binding_energy(
            simulation_dir=simulation_dir,
            simulation_id=simulation_id,
            output_callback=progress_callback
        )
        end_time = time.time()
        
        # Validate results
        if results and results.get('success', False):
            print(f"\n✅ MMGBSA calculation completed successfully!")
            print(f"⏱️ Calculation time: {end_time - start_time:.1f} seconds")
            print(f"🧮 Results:")
            print(f"   - Frames processed: {results.get('number_of_frames', 'N/A')}")
            print(f"   - Raw binding energy: {results.get('raw_binding_energy', 'N/A'):.2f} ± {results.get('binding_energy_std', 0):.2f} kcal/mol")
            print(f"   - Entropy correction: {results.get('entropy_correction', 'N/A'):.2f} kcal/mol")
            print(f"   - Corrected binding energy: {results.get('corrected_binding_energy', 'N/A'):.2f} kcal/mol")
            print(f"   - Method: {results.get('method', 'N/A')}")
            
            return True
        else:
            print(f"\n❌ MMGBSA calculation failed")
            if results:
                print(f"   Error: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        traceback.print_exc()
        return False

def test_minimal_functionality(calculator):
    """Test minimal functionality without full simulation data"""
    print("\n🔬 Testing minimal functionality...")
    
    try:
        # Test polymer molecule extraction from a simple PDB
        test_pdb_content = """MODEL        1
ATOM      1  N   GLY A   1      20.154  16.967  25.449  1.00 30.00           N  
ATOM      2  CA  GLY A   1      19.030  16.067  25.462  1.00 30.00           C  
ATOM      3  C   GLY A   1      17.670  16.768  25.462  1.00 30.00           C  
ATOM      4  O   GLY A   1      17.605  17.989  25.462  1.00 30.00           O  
HETATM    5  C1  UNL A   2      15.000  15.000  15.000  1.00 30.00           C  
HETATM    6  C2  UNL A   2      16.000  15.000  15.000  1.00 30.00           C  
HETATM    7  C3  UNL A   2      17.000  15.000  15.000  1.00 30.00           C  
HETATM    8  C4  UNL A   2      18.000  15.000  15.000  1.00 30.00           C  
ENDMDL
"""
        
        # Create temporary PDB file
        temp_pdb = calculator.output_dir / "test_polymer.pdb"
        with open(temp_pdb, 'w') as f:
            f.write(test_pdb_content)
        
        print(f"✅ Created test PDB: {temp_pdb}")
        
        # Test polymer molecule extraction
        polymer_molecules = calculator._extract_polymer_molecules_proven_approach(
            str(temp_pdb), 
            calculator.output_dir
        )
        
        if polymer_molecules:
            print(f"✅ Successfully extracted {len(polymer_molecules)} polymer molecules")
            for i, mol in enumerate(polymer_molecules):
                print(f"   Molecule {i+1}: {mol.n_atoms} atoms, {mol.n_bonds} bonds")
                print(f"   SMILES: {mol.to_smiles()}")
        else:
            print("⚠️ No polymer molecules extracted (fallback molecules created)")
            return False
        
        # Test SystemGenerator creation
        system_generator = calculator._create_proven_forcefield_with_ligand_templates(polymer_molecules)
        
        if system_generator:
            print("✅ Successfully created SystemGenerator")
            print("✅ Minimal functionality test: PASSED")
            return True
        else:
            print("❌ Failed to create SystemGenerator")
            return False
            
    except Exception as e:
        print(f"❌ Minimal functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test all required dependencies for SystemGenerator approach"""
    print("\n" + "="*70)
    print("🔍 TESTING DEPENDENCIES FOR SYSTEMGENERATOR APPROACH")
    print("="*70)
    
    dependencies = {
        'OpenMM': False,
        'OpenFF Toolkit': False,
        'OpenMM ForceFields': False,
        'SystemGenerator': False,
        'pandas': False,
        'numpy': False
    }
    
    # Test OpenMM
    try:
        import openmm as mm
        import openmm.app as app
        dependencies['OpenMM'] = True
        print(f"✅ OpenMM: Available (version: {mm.__version__})")
    except ImportError:
        print("❌ OpenMM: Not available")
    
    # Test OpenFF Toolkit
    try:
        from openff.toolkit import Molecule
        dependencies['OpenFF Toolkit'] = True
        print("✅ OpenFF Toolkit: Available")
    except ImportError:
        print("❌ OpenFF Toolkit: Not available")
    
    # Test OpenMM ForceFields
    try:
        import openmmforcefields
        dependencies['OpenMM ForceFields'] = True
        print("✅ OpenMM ForceFields: Available")
    except ImportError:
        print("❌ OpenMM ForceFields: Not available")
    
    # Test SystemGenerator specifically
    try:
        from openmmforcefields.generators import SystemGenerator
        dependencies['SystemGenerator'] = True
        print("✅ SystemGenerator: Available")
    except ImportError:
        print("❌ SystemGenerator: Not available")
    
    # Test pandas
    try:
        import pandas as pd
        dependencies['pandas'] = True
        print("✅ pandas: Available")
    except ImportError:
        print("❌ pandas: Not available")
    
    # Test numpy
    try:
        import numpy as np
        dependencies['numpy'] = True
        print("✅ numpy: Available")
    except ImportError:
        print("❌ numpy: Not available")
    
    # Check critical dependencies
    critical_deps = ['OpenMM', 'OpenFF Toolkit', 'OpenMM ForceFields', 'SystemGenerator', 'pandas', 'numpy']
    missing_critical = [dep for dep in critical_deps if not dependencies[dep]]
    
    if missing_critical:
        print(f"\n❌ Critical dependencies missing: {missing_critical}")
        return False
    else:
        print("\n✅ All critical dependencies available for SystemGenerator approach")
        return True

def main():
    """Run all tests"""
    print("🚀 STARTING SYSTEMGENERATOR MMGBSA VALIDATION TESTS")
    print("="*70)
    
    all_tests_passed = True
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Cannot proceed - missing critical dependencies")
        return False
    
    # Test the MMGBSA calculator
    mmgbsa_ok = test_systemgenerator_mmgbsa()
    if not mmgbsa_ok:
        all_tests_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED - SystemGenerator MMGBSA approach is working!")
    else:
        print("❌ SOME TESTS FAILED - Need to debug SystemGenerator approach")
    print("="*70)
    
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 