#!/usr/bin/env python3
"""
Debug script to understand which code path is being executed in create_proper_system
"""

def test_code_path_diagnosis():
    """Test which code path is being executed"""
    
    print("🔍 DEBUG: Testing create_proper_system code path...")
    
    try:
        from integration.analysis.openmm_md_proper import ProperOpenMMSimulator
        
        simulator = ProperOpenMMSimulator()
        
        # Create a mock composition to test the logic
        test_composition = {
            'needs_gaff': True,  # This should trigger the robust method
            'some_other_key': 'value'
        }
        
        print(f"🔍 DEBUG: Test composition: {test_composition}")
        print(f"🔍 DEBUG: needs_gaff = {test_composition.get('needs_gaff', 'NOT_SET')}")
        
        # Check if the robust method exists
        if hasattr(simulator, '_create_system_with_robust_polymer_handling'):
            print("✅ _create_system_with_robust_polymer_handling method exists")
        else:
            print("❌ _create_system_with_robust_polymer_handling method NOT FOUND")
            
        # Check what methods are available
        methods = [method for method in dir(simulator) if 'create' in method.lower()]
        print(f"🔍 Available 'create' methods: {methods}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_systemgenerator_imports():
    """Check if SystemGenerator is still being imported somewhere"""
    
    print("\n🔍 DEBUG: Checking SystemGenerator imports...")
    
    try:
        # Check what's being imported
        from integration.analysis.openmm_md_proper import SystemGenerator
        print("⚠️  SystemGenerator is still imported in openmm_md_proper")
    except ImportError:
        print("✅ SystemGenerator not imported in openmm_md_proper")
    
    try:
        from openmmforcefields.generators import SystemGenerator
        print("✅ SystemGenerator available from openmmforcefields")
    except ImportError:
        print("❌ SystemGenerator not available from openmmforcefields")

if __name__ == "__main__":
    success = test_code_path_diagnosis()
    check_systemgenerator_imports()
    
    print(f"\n🎯 Key Questions:")
    print(f"   1. Is the robust method being called?")
    print(f"   2. Is composition['needs_gaff'] = True?")
    print(f"   3. Is there old SystemGenerator code still executing?")
    print(f"   4. Are there cached Python files causing issues?")
    
    if success:
        print(f"\n💡 Next step: Run actual simulation with debug output enabled")
    else:
        print(f"\n❌ Debug setup failed - check imports and method availability") 