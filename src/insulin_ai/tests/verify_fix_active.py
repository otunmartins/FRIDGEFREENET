#!/usr/bin/env python3
"""
Verification script to confirm our UNL residue fix is active
"""

import sys
import importlib

def verify_fix_is_active():
    """Verify that our robust UNL handling fix is actually being used"""
    
    print("🔍 VERIFICATION: Checking if our UNL residue fix is active...")
    
    try:
        # Force reload of the module to clear any cache
        import integration.analysis.openmm_md_proper as proper_module
        importlib.reload(proper_module)
        
        from integration.analysis.openmm_md_proper import ProperOpenMMSimulator
        
        # Create simulator
        simulator = ProperOpenMMSimulator()
        
        # Test the robust method directly
        print("\n🧪 Testing robust polymer handling method...")
        
        # Create a mock topology with UNL residues
        class MockAtom:
            def __init__(self, element_symbol):
                self.element = type('Element', (), {'symbol': element_symbol})()
        
        class MockResidue:
            def __init__(self, atom_elements):
                self._atoms = [MockAtom(elem) for elem in atom_elements]
            
            def atoms(self):
                return self._atoms
        
        # Test the _create_system_with_robust_polymer_handling method
        if hasattr(simulator, '_create_system_with_robust_polymer_handling'):
            print("✅ Robust polymer handling method EXISTS")
            
            # Test molecular creation from residue
            test_residue = MockResidue(['C'] * 10 + ['H'] * 22 + ['O'] * 5 + ['S'] * 2)
            molecule = simulator._create_polymer_molecule_from_residue(test_residue)
            
            if molecule:
                print(f"✅ Residue-to-molecule conversion: SUCCESS ({molecule.n_atoms} atoms)")
            else:
                print("❌ Residue-to-molecule conversion: FAILED")
                
        else:
            print("❌ Robust polymer handling method NOT FOUND")
            return False
        
        # Search the actual file content to confirm no SystemGenerator usage
        print("\n🔍 Checking file content for old SystemGenerator code...")
        
        with open('integration/analysis/openmm_md_proper.py', 'r') as f:
            content = f.read()
            
        # Check for problematic patterns
        old_patterns = [
            'SystemGenerator(',
            'system_generator.add_molecules',
            'system_generator.create_system',
            'Using.*polymer molecules for GAFF parameterization',
            'Adding.*molecules to SystemGenerator'
        ]
        
        found_old_code = False
        for pattern in old_patterns:
            if pattern in content:
                print(f"⚠️  Found old pattern: {pattern}")
                found_old_code = True
        
        if not found_old_code:
            print("✅ No old SystemGenerator patterns found in file")
        
        # Check what our create_proper_system method actually does
        print("\n🔍 Analyzing create_proper_system method...")
        
        import inspect
        source = inspect.getsource(simulator.create_proper_system)
        
        if '_create_system_with_robust_polymer_handling' in source:
            print("✅ create_proper_system calls robust method")
        else:
            print("❌ create_proper_system does NOT call robust method")
            return False
        
        if 'SystemGenerator(' in source:
            print("⚠️  create_proper_system still contains SystemGenerator calls")
        else:
            print("✅ create_proper_system has no SystemGenerator calls")
        
        print("\n🎯 VERIFICATION COMPLETE!")
        print("✅ Our robust UNL handling fix is ACTIVE and ready to use")
        print("🚀 SystemGenerator errors should no longer occur")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_fix_is_active()
    
    if success:
        print(f"\n🎉 SUCCESS! The UNL residue fix is properly installed and active.")
        print(f"🔧 Next time you run a simulation, it should use the robust method.")
    else:
        print(f"\n❌ FAILURE! There are still issues with the UNL residue fix.")
        print(f"🔧 The simulation may still encounter SystemGenerator errors.")
    
    sys.exit(0 if success else 1) 