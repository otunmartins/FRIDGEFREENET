#!/usr/bin/env python3
"""
Debug script to check auto-corrector availability and session state
"""

import streamlit as st
import os

# Check if auto-corrector can be imported
try:
    from psmiles_auto_corrector import create_psmiles_auto_corrector
    AUTOCORRECTOR_AVAILABLE = True
    print("✅ AUTOCORRECTOR_AVAILABLE = True")
except ImportError as e:
    AUTOCORRECTOR_AVAILABLE = False
    print(f"❌ AUTOCORRECTOR_AVAILABLE = False: {e}")

# Check if we can create an auto-corrector
if AUTOCORRECTOR_AVAILABLE:
    try:
        ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        
        print(f"🔧 Testing auto-corrector creation...")
        print(f"   Model: {ollama_model}")
        print(f"   Host: {ollama_host}")
        
        auto_corrector = create_psmiles_auto_corrector(
            ollama_model=ollama_model,
            ollama_host=ollama_host
        )
        
        print(f"✅ Auto-corrector created successfully!")
        print(f"   Type: {type(auto_corrector)}")
        print(f"   Has correct_psmiles method: {hasattr(auto_corrector, 'correct_psmiles')}")
        
        # Test with a simple case
        print(f"\n🧪 Testing correction...")
        test_result = auto_corrector.correct_psmiles("C[*]O[*]")
        print(f"   Test successful: {test_result.get('success', False)}")
        print(f"   Generated corrections: {test_result.get('correction_count', 0)}")
        
    except Exception as e:
        print(f"❌ Auto-corrector creation failed: {e}")
        print(f"   Error type: {type(e).__name__}")
else:
    print("❌ Cannot test auto-corrector - import failed")

print("\n" + "="*50)
print("Debugging complete!")

if __name__ == "__main__":
    # This would run when executed directly
    pass 