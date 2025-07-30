"""
PSP Debugging Utilities for Investigating Tuple Errors

This module provides tools to debug the mysterious C++ binding error where PSP
receives a tuple instead of a string for SMILES data.

Author: AI-Driven Material Discovery Team
"""

import os
import uuid
import pandas as pd
import glob
from typing import Dict, Optional, Tuple, Any
import traceback


def debug_data_types(psmiles: Any, description: str = "Unknown") -> Dict[str, Any]:
    """
    Debug function to thoroughly examine the data types and structure
    of the psmiles parameter at various stages.
    """
    debug_info = {
        'description': description,
        'type': type(psmiles).__name__,
        'repr': repr(psmiles),
        'str': str(psmiles),
        'is_string': isinstance(psmiles, str),
        'is_tuple': isinstance(psmiles, tuple),
        'is_list': isinstance(psmiles, list),
        'length': len(psmiles) if hasattr(psmiles, '__len__') else 'No length',
    }
    
    # Additional analysis for tuples/lists
    if isinstance(psmiles, (tuple, list)) and len(psmiles) > 0:
        debug_info['first_element_type'] = type(psmiles[0]).__name__
        debug_info['first_element_repr'] = repr(psmiles[0])
        debug_info['all_elements'] = [repr(item) for item in psmiles]
    
    print(f"\n🔍 DEBUG DATA ANALYSIS - {description}")
    print("=" * 60)
    for key, value in debug_info.items():
        print(f"{key:20}: {value}")
    print("=" * 60)
    
    return debug_info


def test_dataframe_creation(psmiles: Any) -> Dict[str, Any]:
    """
    Test DataFrame creation to see where the tuple issue might come from.
    """
    print(f"\n🧪 Testing DataFrame creation with psmiles...")
    
    debug_data_types(psmiles, "Input to DataFrame creation")
    
    # Create the DataFrame exactly as in molecule_builder_utils.py
    input_data = {
        'ID': ['polymer'],
        'smiles': [psmiles],
        'LeftCap': ['H'],
        'RightCap': ['H'],
        'Loop': [False]
    }
    
    print(f"\n📊 DataFrame input_data:")
    for key, value in input_data.items():
        print(f"  {key}: {repr(value)} (type: {type(value[0]).__name__})")
    
    input_df = pd.DataFrame(input_data)
    
    print(f"\n📋 Created DataFrame:")
    print(input_df)
    print(f"\nDataFrame dtypes:")
    print(input_df.dtypes)
    
    # Test extraction exactly as PSP does
    unit_name = 'polymer'
    ID = 'ID'
    SMILES = 'smiles'
    
    # Step 1: Filter the DataFrame
    filtered_df = input_df[input_df[ID] == unit_name]
    print(f"\n🔍 Filtered DataFrame:")
    print(filtered_df)
    
    # Step 2: Get the SMILES column
    smiles_series = filtered_df[SMILES]
    print(f"\n📝 SMILES Series:")
    print(f"Type: {type(smiles_series)}")
    print(f"Values: {smiles_series.values}")
    print(f"Values type: {type(smiles_series.values)}")
    
    # Step 3: Extract values[0]
    extracted_value = smiles_series.values[0]
    debug_data_types(extracted_value, "Extracted from DataFrame")
    
    return {
        'input_psmiles': debug_data_types(psmiles, "Silent"),
        'extracted_value': debug_data_types(extracted_value, "Silent"),
        'extraction_successful': isinstance(extracted_value, str)
    }


def try_chainbuilder_alternative(psmiles: str, length: int = 5, output_dir: Optional[str] = None) -> Dict:
    """
    Test ChainBuilder as an alternative to MoleculeBuilder to see if it handles the data differently.
    """
    print(f"\n🔗 Testing ChainBuilder as alternative...")
    
    try:
        # Import PSP ChainBuilder
        import psp.ChainBuilder as cb
        
        # Create unique output directory if not provided
        if output_dir is None:
            output_dir = f'chainbuilder_test_{uuid.uuid4().hex[:8]}'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create input DataFrame for ChainBuilder
        input_data = {
            'ID': ['polymer'],
            'smiles': [psmiles],
        }
        
        input_df = pd.DataFrame(input_data)
        
        print(f"   🔧 Using PSP ChainBuilder...")
        print(f"   📥 Input PSMILES: '{psmiles}'")
        print(f"   📏 Chain length: {length}")
        print(f"   📁 Output directory: {output_dir}")
        
        # Test data extraction first
        test_result = test_dataframe_creation(psmiles)
        
        # Create ChainBuilder
        chain_builder = cb.Builder(
            Dataframe=input_df,
            ID_col='ID',
            SMILES_col='smiles',
            OutDir=output_dir,
            Length=[length],       # ChainBuilder uses list format for Length
            NumConf=1,
            NCores=1,              # Use single core for testing
            Method='Dimer',        # Use Dimer method (simpler than SA)
            MonomerAng='low',      # Use low resolution for faster testing
            DimerAng='low'
        )
        
        # Build the polymer structure
        try:
            print(f"   ⚡ Starting ChainBuilder.BuildChain()...")
            results = chain_builder.BuildChain()
            print(f"   ✅ PSP ChainBuilder completed successfully")
            
            # Analyze results
            print(f"   📊 ChainBuilder Results:")
            print(results)
            
            # Look for generated files
            chain_files = []
            
            # ChainBuilder typically creates different file patterns
            for pattern in ["*.xyz", "*.pdb", "*.vasp", "*POSCAR*"]:
                pattern_path = os.path.join(output_dir, "**", pattern)
                found_files = glob.glob(pattern_path, recursive=True)
                chain_files.extend(found_files)
            
            return {
                'success': True,
                'method': 'ChainBuilder_success',
                'output_dir': output_dir,
                'results_dataframe': results,
                'generated_files': chain_files,
                'data_extraction_test': test_result
            }
            
        except Exception as build_error:
            error_msg = str(build_error)
            print(f"   ❌ ChainBuilder Build() failed: {error_msg}")
            
            # Check if it's the same C++ binding error
            if "C++ rvalue" in error_msg and "tuple" in error_msg:
                print(f"   🚨 SAME TUPLE ERROR in ChainBuilder!")
                return {
                    'success': False,
                    'error': f'ChainBuilder has same C++ tuple error: {error_msg}',
                    'method': 'ChainBuilder_same_error',
                    'data_extraction_test': test_result
                }
            else:
                return {
                    'success': False,
                    'error': f'ChainBuilder different error: {error_msg}',
                    'method': 'ChainBuilder_different_error',
                    'data_extraction_test': test_result
                }
        
    except ImportError as e:
        return {
            'success': False,
            'error': f'ChainBuilder import failed: {e}',
            'method': 'ChainBuilder_import_failed'
        }


def create_safe_psmiles_wrapper(original_psmiles: Any) -> str:
    """
    Create a safe string version of psmiles that will definitely work with PSP.
    """
    print(f"\n🛡️ Creating safe PSMILES wrapper...")
    
    debug_data_types(original_psmiles, "Original input")
    
    # Handle different input types
    if isinstance(original_psmiles, str):
        safe_psmiles = original_psmiles.strip()
    elif isinstance(original_psmiles, (tuple, list)) and len(original_psmiles) > 0:
        # Extract first element if it's a tuple/list
        safe_psmiles = str(original_psmiles[0]).strip()
        print(f"   ⚠️ Converted from {type(original_psmiles).__name__} to string")
    else:
        # Force string conversion
        safe_psmiles = str(original_psmiles).strip()
        print(f"   ⚠️ Force converted {type(original_psmiles).__name__} to string")
    
    # Validate PSMILES format
    if not safe_psmiles:
        safe_psmiles = "[*]C[*]"  # Minimal fallback
        print(f"   🔧 Used minimal fallback PSMILES")
    
    debug_data_types(safe_psmiles, "Safe PSMILES output")
    
    return safe_psmiles


def comprehensive_psp_debug(psmiles: Any, length: int = 5) -> Dict[str, Any]:
    """
    Comprehensive debugging of PSP issues with multiple approaches.
    """
    print(f"\n🔬 COMPREHENSIVE PSP DEBUGGING SESSION")
    print("=" * 80)
    
    results = {
        'input_analysis': debug_data_types(psmiles, "Silent"),
        'dataframe_test': None,
        'chainbuilder_test': None,
        'safe_wrapper_test': None,
        'recommendations': []
    }
    
    # Test 1: DataFrame creation
    try:
        results['dataframe_test'] = test_dataframe_creation(psmiles)
    except Exception as e:
        results['dataframe_test'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test 2: Safe wrapper
    try:
        safe_psmiles = create_safe_psmiles_wrapper(psmiles)
        results['safe_wrapper_test'] = {
            'safe_psmiles': safe_psmiles,
            'original_type': type(psmiles).__name__,
            'success': True
        }
    except Exception as e:
        results['safe_wrapper_test'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Test 3: ChainBuilder alternative
    if results['safe_wrapper_test'] and results['safe_wrapper_test'].get('success'):
        safe_psmiles = results['safe_wrapper_test']['safe_psmiles']
        try:
            results['chainbuilder_test'] = try_chainbuilder_alternative(safe_psmiles, length)
        except Exception as e:
            results['chainbuilder_test'] = {'error': str(e), 'traceback': traceback.format_exc()}
    
    # Generate recommendations
    if isinstance(psmiles, tuple):
        results['recommendations'].append("CRITICAL: Input is a tuple - trace where this comes from!")
    
    if results['dataframe_test'] and not results['dataframe_test'].get('extraction_successful'):
        results['recommendations'].append("DataFrame extraction is corrupting the string")
    
    if results['chainbuilder_test'] and results['chainbuilder_test'].get('success'):
        results['recommendations'].append("ChainBuilder works - consider as alternative to MoleculeBuilder")
    
    if results['safe_wrapper_test'] and results['safe_wrapper_test'].get('success'):
        results['recommendations'].append("Safe wrapper conversion works - implement this fix")
    
    print(f"\n📋 DEBUGGING SUMMARY:")
    print(f"Input type: {type(psmiles).__name__}")
    print(f"DataFrame test: {'✅' if results['dataframe_test'] and results['dataframe_test'].get('extraction_successful') else '❌'}")
    print(f"ChainBuilder test: {'✅' if results['chainbuilder_test'] and results['chainbuilder_test'].get('success') else '❌'}")
    print(f"Safe wrapper: {'✅' if results['safe_wrapper_test'] and results['safe_wrapper_test'].get('success') else '❌'}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return results


# Testing function
if __name__ == "__main__":
    # Test with different inputs
    test_cases = [
        "[*]C=CS(=O)(=O)COC([*])=O",  # Normal string
        ("[*]C=CS(=O)(=O)COC([*])=O", "extra_data"),  # Tuple (problematic)
        ["[*]C=CS(=O)(=O)COC([*])=O"],  # List
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n" + "="*50)
        print(f"TEST CASE {i+1}: {type(test_case).__name__}")
        print(f"="*50)
        comprehensive_psp_debug(test_case) 