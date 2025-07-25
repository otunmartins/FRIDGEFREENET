#!/usr/bin/env python3
"""
Instant PSMILES Corrector
========================

This module provides immediate PSMILES correction using the proper approach:
1. Extract SMILES from failed PSMILES (remove [*])
2. Validate and clean the SMILES
3. Convert to proper PSMILES as [*]SMILES[*]
"""

def instant_correct_psmiles(failed_psmiles, psmiles_processor=None):
    """
    Apply instant PSMILES correction using SMILES→PSMILES approach.
    
    Args:
        failed_psmiles (str): The PSMILES that failed visualization
        psmiles_processor: Optional processor to test corrections
        
    Returns:
        Dict with correction results
    """
    
    results = {
        'success': False,
        'original': failed_psmiles,
        'corrections': [],
        'method_used': None,
        'error': None
    }
    
    try:
        # Method 1: SMILES→PSMILES conversion
        print(f"🔄 Trying SMILES→PSMILES conversion for: {failed_psmiles}")
        
        # Extract potential SMILES (remove [*])
        potential_smiles = failed_psmiles.replace('[*]', '')
        print(f"   Extracted SMILES: {potential_smiles}")
        
        # Try RDKit validation and cleaning
        try:
            from rdkit import Chem
            
            # Parse and validate SMILES
            mol = Chem.MolFromSmiles(potential_smiles)
            if mol is not None:
                # Clean and canonicalize SMILES
                clean_smiles = Chem.MolToSmiles(mol)
                
                # Convert to proper PSMILES
                corrected_psmiles = f"[*]{clean_smiles}[*]"
                
                print(f"✅ SMILES validation successful:")
                print(f"   Original SMILES: {potential_smiles}")
                print(f"   Clean SMILES: {clean_smiles}")
                print(f"   Final PSMILES: {corrected_psmiles}")
                
                correction = {
                    'corrected': corrected_psmiles,
                    'method': 'smiles_to_psmiles',
                    'confidence': 0.95,
                    'description': f'SMILES→PSMILES conversion: {clean_smiles}',
                    'original_smiles': potential_smiles,
                    'clean_smiles': clean_smiles
                }
                
                # Test if processor is available
                if psmiles_processor:
                    test_result = psmiles_processor.process_psmiles_workflow(
                        corrected_psmiles, "test_session", "validation"
                    )
                    correction['validation_success'] = test_result.get('success', False)
                    correction['validation_error'] = test_result.get('error', None)
                    
                    if correction['validation_success']:
                        print(f"✅ Validation successful - structure works!")
                    else:
                        print(f"❌ Validation failed: {correction['validation_error']}")
                else:
                    correction['validation_success'] = None
                    print(f"⚠️  No processor available for validation")
                
                results['corrections'].append(correction)
                results['method_used'] = 'smiles_to_psmiles'
                results['success'] = True
                
            else:
                print(f"❌ Invalid SMILES: {potential_smiles}")
                
        except ImportError:
            print(f"⚠️  RDKit not available, skipping SMILES validation")
        except Exception as e:
            print(f"⚠️  SMILES validation failed: {e}")
        
        # Method 2: Simple cleaning approach (if SMILES approach didn't work)
        if not results['success']:
            print(f"🔄 Trying simple cleaning approach...")
            
            # Common fixes
            simple_corrections = []
            
            # Fix 1: Remove problematic heteroatoms at connection points
            if 'N[*]' in failed_psmiles or 'O[*]' in failed_psmiles or 'S[*]' in failed_psmiles:
                # Try to move connection points to carbon atoms
                cleaned = failed_psmiles
                cleaned = cleaned.replace('N[*]', '[*]N')
                cleaned = cleaned.replace('O[*]', '[*]O') 
                cleaned = cleaned.replace('S[*]', '[*]S')
                if cleaned != failed_psmiles:
                    simple_corrections.append({
                        'corrected': cleaned,
                        'method': 'move_connection_points',
                        'confidence': 0.7,
                        'description': 'Moved [*] away from heteroatoms'
                    })
            
            # Fix 2: Simplify to core structure
            if potential_smiles:
                # Try basic carbon chain patterns
                simple_patterns = [
                    f"[*]CC[*]",  # Simple alkyl
                    f"[*]CCC[*]", # Propyl chain
                    f"[*]C(=O)C[*]", # Ketone
                    f"[*]COC[*]", # Ether
                ]
                
                for pattern in simple_patterns:
                    simple_corrections.append({
                        'corrected': pattern,
                        'method': 'simple_pattern',
                        'confidence': 0.4,
                        'description': f'Simplified to {pattern}'
                    })
            
            # Test simple corrections if processor available
            for correction in simple_corrections:
                if psmiles_processor:
                    test_result = psmiles_processor.process_psmiles_workflow(
                        correction['corrected'], "test_session", "simple_validation"
                    )
                    correction['validation_success'] = test_result.get('success', False)
                    correction['validation_error'] = test_result.get('error', None)
                    
                    if correction['validation_success']:
                        print(f"✅ Simple correction works: {correction['corrected']}")
                        results['corrections'].append(correction)
                        results['method_used'] = 'simple_cleaning'
                        results['success'] = True
                        break  # Stop at first working correction
                    else:
                        print(f"❌ Simple correction failed: {correction['corrected']}")
                else:
                    results['corrections'].append(correction)
            
            if results['corrections'] and not results['method_used']:
                results['method_used'] = 'simple_cleaning'
                results['success'] = True
        
        if not results['success']:
            results['error'] = "No working corrections found"
            print(f"❌ No working corrections found for: {failed_psmiles}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Error in correction process: {e}")
    
    return results

def apply_instant_corrections_ui(failed_psmiles, psmiles_processor=None):
    """
    Apply instant corrections and return UI-friendly results.
    This is the function to call from Streamlit.
    """
    
    correction_results = instant_correct_psmiles(failed_psmiles, psmiles_processor)
    
    ui_results = {
        'has_corrections': correction_results['success'],
        'original': failed_psmiles,
        'method_used': correction_results['method_used'],
        'corrections': [],
        'error': correction_results['error']
    }
    
    for correction in correction_results['corrections']:
        ui_correction = {
            'psmiles': correction['corrected'],
            'confidence': correction['confidence'],
            'description': correction['description'],
            'method': correction['method'],
            'works': correction.get('validation_success', None)
        }
        
        # Add method-specific details
        if correction['method'] == 'smiles_to_psmiles':
            ui_correction['original_smiles'] = correction.get('original_smiles', '')
            ui_correction['clean_smiles'] = correction.get('clean_smiles', '')
        
        ui_results['corrections'].append(ui_correction)
    
    return ui_results

if __name__ == "__main__":
    # Test with your failing examples
    test_cases = [
        "C(O)C(=O)N[*]CC(=O)[*]",
        "C(O)CO[*]NC(=O)COC(=O)C[*]S",
        "SC(=O)C(O)C[*]NNC(=O)CC[*]",
        "C(O)CS[*]NC(=O)CC(O)[*]"
    ]
    
    print("🧪 Testing Instant PSMILES Corrector")
    print("="*50)
    
    for psmiles in test_cases:
        print(f"\n🔧 Testing: {psmiles}")
        result = instant_correct_psmiles(psmiles)
        
        if result['success']:
            print(f"✅ Success using {result['method_used']}")
            for correction in result['corrections']:
                print(f"   → {correction['corrected']} (confidence: {correction['confidence']:.1%})")
        else:
            print(f"❌ Failed: {result['error']}") 