#!/usr/bin/env python3
"""
Debug script to test the PSMILES diversification system directly.
This helps identify the exact failure point outside of the Streamlit app.
"""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_diversification():
    """Test the diversification system with a simple example."""
    
    print("🔬 Testing PSMILES Diversification System")
    print("=" * 50)
    
    try:
        # Step 1: Initialize the PSMILES generator
        print("Step 1: Initializing PSMILES Generator...")
        from insulin_ai import PSMILESGenerator
        
        generator = PSMILESGenerator(
            model_type='openai',
            openai_model='gpt-3.5-turbo',  # Use cheaper model for testing
            temperature=0.7,
            enable_diverse_generation=True
        )
        
        print(f"✅ Generator initialized. Diverse generation available: {generator.diverse_generation_available}")
        
        if not generator.diverse_generation_available:
            print("❌ Diverse generation not available. Cannot continue test.")
            return False
            
        # Step 2: Test basic generation first
        print("\nStep 2: Testing basic generation...")
        basic_result = generator.generate_psmiles(
            description="Sulfur",
            num_candidates=1,
            validate=True
        )
        print(f"Basic generation result: {basic_result['success']}")
        if basic_result['success']:
            print(f"Generated: {basic_result['candidates'][0]['psmiles']}")
        
        # Step 3: Test diverse generation
        print("\nStep 3: Testing diverse generation...")
        diverse_result = generator.generate_truly_diverse_candidates(
            base_request="Sulfur",
            num_candidates=3,
            enable_functionalization=True,
            diversity_threshold=0.4,
            max_retries=1  # Reduced for testing
        )
        
        print(f"Diverse generation result: {diverse_result.get('success', False)}")
        
        if diverse_result.get('success'):
            candidates = diverse_result.get('candidates', [])
            print(f"Generated {len(candidates)} diverse candidates:")
            for i, candidate in enumerate(candidates):
                print(f"  {i+1}. {candidate.get('psmiles', 'N/A')}")
        else:
            print(f"❌ Diverse generation failed: {diverse_result.get('error', 'Unknown error')}")
            if 'debug_info' in diverse_result:
                print(f"Debug info: {diverse_result['debug_info']}")
        
        return diverse_result.get('success', False)
        
    except Exception as e:
        print(f"❌ Test failed with exception: {type(e).__name__}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_individual_components():
    """Test individual components of the diversification system."""
    
    print("\n🧪 Testing Individual Components")
    print("=" * 50)
    
    try:
        # Test prompt diversification
        print("Testing PromptDiversifier...")
        from core.psmiles_diversification import PromptDiversifier, CandidateConfig
        
        diversifier = PromptDiversifier()
        config = CandidateConfig(
            base_request="Sulfur",
            num_candidates=3,
            enable_functionalization=True
        )
        
        prompts = diversifier.generate_diverse_prompts(config)
        print(f"Generated {len(prompts)} diverse prompts:")
        for i, prompt_data in enumerate(prompts):
            print(f"  {i+1}. {prompt_data['prompt']}")
        
        # Test functionalization engine
        print("\nTesting FunctionalizationEngine...")
        from core.psmiles_diversification import FunctionalizationEngine
        
        engine = FunctionalizationEngine()
        test_candidate = {
            'psmiles': '[*]CSC[*]',
            'valid': True
        }
        
        result = engine.functionalize_candidate(test_candidate, config)
        print(f"Functionalization result: {len(result.get('functionalized_variants', []))} variants")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {type(e).__name__}: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 Starting PSMILES Diversification Debug Session")
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test full system
        system_ok = test_diversification()
        
        if system_ok:
            print("\n✅ All tests passed! Diversification system is working.")
        else:
            print("\n❌ System test failed. Check error messages above.")
    else:
        print("\n❌ Component tests failed. Cannot proceed to system test.") 