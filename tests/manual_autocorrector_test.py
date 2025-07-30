#!/usr/bin/env python3
"""
Manual PSMILES Auto-Corrector Test
==================================

Run this script to test auto-corrections on your failing PSMILES structures
without needing to use the Streamlit interface.
"""

from psmiles_auto_corrector import create_psmiles_auto_corrector
from psmiles_processor import PSMILESProcessor
import uuid

def test_structure_with_corrections(psmiles, description=""):
    """Test a PSMILES structure and show auto-corrections if it fails."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {psmiles}")
    if description:
        print(f"   Description: {description}")
    
    # First, test if the original works
    processor = PSMILESProcessor()
    session_id = str(uuid.uuid4())
    
    original_result = processor.process_psmiles_workflow(psmiles, session_id, "test")
    
    if original_result.get('success') and original_result.get('svg_content'):
        print("✅ Original structure works - no correction needed!")
        return True
    else:
        print(f"❌ Original structure fails: {original_result.get('error', 'Unknown error')}")
        
        # Generate corrections
        print("\n🔧 Generating auto-corrections...")
        corrector = create_psmiles_auto_corrector()
        correction_result = corrector.correct_psmiles(
            psmiles, 
            error_details=original_result.get('error', 'Visualization failed')
        )
        
        if correction_result['success'] and correction_result['corrections']:
            print(f"✅ Generated {len(correction_result['corrections'])} corrections:")
            
            working_corrections = []
            for i, correction in enumerate(correction_result['corrections'][:3], 1):
                corrected_psmiles = correction['corrected']
                confidence = correction['confidence']
                fix_description = correction['fix_description']
                
                print(f"\n   🔧 Correction {i}: {corrected_psmiles}")
                print(f"      Confidence: {confidence:.1%}")
                print(f"      Fix: {fix_description}")
                
                # Test the correction
                test_result = processor.process_psmiles_workflow(corrected_psmiles, session_id, "test")
                
                if test_result.get('success') and test_result.get('svg_content'):
                    print(f"      ✅ SUCCESS - This correction works!")
                    working_corrections.append(corrected_psmiles)
                else:
                    print(f"      ❌ FAILED - {test_result.get('error', 'Still fails')}")
            
            if working_corrections:
                print(f"\n🎉 {len(working_corrections)} working corrections found!")
                print(f"   Best working correction: {working_corrections[0]}")
                return True
            else:
                print(f"\n⚠️  No working corrections found - structure may need manual review")
                return False
        else:
            print("❌ No corrections generated")
            if 'analysis' in correction_result and correction_result['analysis']:
                analysis = correction_result['analysis']
                if analysis.get('issues'):
                    print("   Detected Issues:")
                    for issue in analysis['issues']:
                        print(f"   - {issue['description']}")
            return False

def main():
    """Test the failing structures from your examples."""
    print("🧪 PSMILES Auto-Corrector Manual Test")
    print("====================================")
    
    # Your failing structures
    test_cases = [
        {
            'psmiles': 'C(O)C(=O)N[*]CC(=O)[*]',
            'description': 'Structure 4 - Nitrogen directly connected to [*]'
        },
        {
            'psmiles': 'C(O)CO[*]NC(=O)COC(=O)C[*]S', 
            'description': 'Structure 5 - Oxygen connected to [*] + terminal sulfur'
        },
        {
            'psmiles': 'SC(=O)C(O)C[*]NNC(=O)CC[*]',
            'description': 'Structure 1 - Complex structure with sulfur at start'
        },
        {
            'psmiles': 'C(O)CS[*]NC(=O)CC(O)[*]',
            'description': 'Structure 2 - Sulfur directly connected to [*]'
        }
    ]
    
    working_count = 0
    for i, test_case in enumerate(test_cases, 1):
        if test_structure_with_corrections(test_case['psmiles'], test_case['description']):
            working_count += 1
    
    print(f"\n{'='*60}")
    print(f"🎯 Summary: {working_count}/{len(test_cases)} structures have working corrections")
    print("For structures with working corrections, you can use them in your research!")

if __name__ == "__main__":
    main() 