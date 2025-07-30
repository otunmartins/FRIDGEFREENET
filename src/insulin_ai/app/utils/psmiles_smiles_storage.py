#!/usr/bin/env python3
"""
PSMILES to SMILES Storage Utility

This module handles the conversion of PSMILES to full SMILES strings during generation
and ensures they are stored together for efficient MD simulation workflows.
"""

import streamlit as st
from typing import Dict, Optional, Any, List
from datetime import datetime
import traceback

# Import the converter
try:
    from utils.psmiles_to_smiles_converter import PSMILESConverter
    CONVERTER_AVAILABLE = True
except ImportError:
    CONVERTER_AVAILABLE = False


class PSMILESWithSMILESStorage:
    """Utility class to handle PSMILES generation with SMILES storage"""
    
    def __init__(self):
        self.converter = PSMILESConverter() if CONVERTER_AVAILABLE else None
        
    def convert_and_store_psmiles(self, psmiles: str) -> Dict[str, Any]:
        """
        Convert PSMILES to SMILES and return both for storage.
        
        Args:
            psmiles: PSMILES string to convert
            
        Returns:
            Dict containing both PSMILES and SMILES with conversion metadata
        """
        result = {
            'psmiles': psmiles,
            'smiles': None,
            'smiles_conversion_success': False,
            'smiles_conversion_method': None,
            'smiles_conversion_error': None,
            'conversion_metadata': {}
        }
        
        if not self.converter:
            result['smiles_conversion_error'] = "PSMILESConverter not available"
            return result
            
        try:
            print(f"🧬 Converting PSMILES to SMILES: {psmiles}")
            conversion_result = self.converter.convert_psmiles_to_smiles(psmiles)
            
            if conversion_result['success'] and conversion_result['best_smiles']:
                result['smiles'] = conversion_result['best_smiles']
                result['smiles_conversion_success'] = True
                result['smiles_conversion_method'] = conversion_result['best_method']
                result['conversion_metadata'] = {
                    'all_methods_tried': list(conversion_result['conversions'].keys()),
                    'successful_methods': [
                        method for method, data in conversion_result['conversions'].items() 
                        if data.get('valid', False)
                    ],
                    'atom_count': conversion_result['conversions'].get(conversion_result['best_method'], {}).get('atoms', 'unknown'),
                    'conversion_description': conversion_result['conversions'].get(conversion_result['best_method'], {}).get('description', 'unknown')
                }
                print(f"✅ SMILES conversion successful: {result['smiles']} (method: {result['smiles_conversion_method']})")
            else:
                result['smiles_conversion_error'] = conversion_result.get('error', 'Unknown conversion error')
                print(f"❌ SMILES conversion failed: {result['smiles_conversion_error']}")
                
        except Exception as e:
            result['smiles_conversion_error'] = str(e)
            print(f"❌ SMILES conversion exception: {e}")
            traceback.print_exc()
            
        return result
    
    def enhance_candidate_with_smiles(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance a PSMILES candidate dictionary with SMILES information.
        
        Args:
            candidate: Existing candidate dict with 'psmiles' key
            
        Returns:
            Enhanced candidate dict with SMILES information added
        """
        psmiles = candidate.get('psmiles')
        if not psmiles:
            print("⚠️ No PSMILES found in candidate")
            return candidate
            
        # Convert PSMILES to SMILES
        conversion_data = self.convert_and_store_psmiles(psmiles)
        
        # Add SMILES data to candidate
        candidate.update(conversion_data)
        
        # Add timestamp for when SMILES was generated
        candidate['smiles_generated_at'] = datetime.now().isoformat()
        
        return candidate


def enhance_psmiles_generation_with_smiles_storage(
    generation_function,
    material_request: str, 
    *args, 
    **kwargs
) -> Dict[str, Any]:
    """
    Wrapper function to enhance any PSMILES generation function with SMILES storage.
    
    Args:
        generation_function: The original PSMILES generation function
        material_request: Material request string
        *args, **kwargs: Arguments to pass to generation function
        
    Returns:
        Enhanced result dict with SMILES included
    """
    # Initialize storage utility
    storage = PSMILESWithSMILESStorage()
    
    # Call original generation function
    print(f"🚀 Enhanced PSMILES generation with SMILES storage for: {material_request}")
    result = generation_function(material_request, *args, **kwargs)
    
    # Enhance result with SMILES
    if result and 'psmiles' in result:
        conversion_data = storage.convert_and_store_psmiles(result['psmiles'])
        result.update(conversion_data)
        
        # If we have candidates list, enhance each candidate
        if 'candidates' in result and isinstance(result['candidates'], list):
            enhanced_candidates = []
            for candidate in result['candidates']:
                if isinstance(candidate, str):
                    # Convert string PSMILES to dict format
                    candidate_dict = {'psmiles': candidate}
                    enhanced_candidate = storage.enhance_candidate_with_smiles(candidate_dict)
                    enhanced_candidates.append(enhanced_candidate)
                elif isinstance(candidate, dict) and 'psmiles' in candidate:
                    enhanced_candidate = storage.enhance_candidate_with_smiles(candidate.copy())
                    enhanced_candidates.append(enhanced_candidate)
                else:
                    enhanced_candidates.append(candidate)
            result['candidates'] = enhanced_candidates
            
        print(f"✅ Enhanced PSMILES generation completed with SMILES storage")
    else:
        print(f"⚠️ Could not enhance result - no PSMILES found")
        
    return result


def get_stored_smiles_for_psmiles(psmiles: str) -> Optional[str]:
    """
    Retrieve stored SMILES for a given PSMILES from session state candidates.
    
    Args:
        psmiles: PSMILES string to find SMILES for
        
    Returns:
        SMILES string if found, None otherwise
    """
    if 'psmiles_candidates' not in st.session_state:
        return None
        
    for candidate in st.session_state.psmiles_candidates:
        if candidate.get('psmiles') == psmiles and candidate.get('smiles'):
            print(f"✅ Found stored SMILES for PSMILES {psmiles}: {candidate['smiles']}")
            return candidate['smiles']
            
    return None


def get_smiles_for_md_simulation(psmiles: str) -> Optional[str]:
    """
    Get SMILES string for MD simulation use. First tries stored SMILES, 
    then falls back to conversion if needed.
    
    Args:
        psmiles: PSMILES string
        
    Returns:
        SMILES string for MD simulation use
    """
    # First try to get from stored candidates
    stored_smiles = get_stored_smiles_for_psmiles(psmiles)
    if stored_smiles:
        print(f"🎯 Using pre-stored SMILES for MD simulation: {stored_smiles}")
        return stored_smiles
        
    # Fallback to conversion
    print(f"⚠️ No stored SMILES found, converting PSMILES to SMILES for MD simulation")
    storage = PSMILESWithSMILESStorage()
    conversion_data = storage.convert_and_store_psmiles(psmiles)
    
    if conversion_data['smiles_conversion_success']:
        print(f"✅ Converted PSMILES to SMILES for MD simulation: {conversion_data['smiles']}")
        return conversion_data['smiles']
    else:
        print(f"❌ Failed to convert PSMILES to SMILES for MD simulation: {conversion_data['smiles_conversion_error']}")
        return None
        

def update_session_candidates_with_smiles():
    """
    Update existing session state candidates to include SMILES if not already present.
    This is a utility function for migrating existing data.
    """
    if 'psmiles_candidates' not in st.session_state:
        return
        
    storage = PSMILESWithSMILESStorage()
    updated_count = 0
    
    for i, candidate in enumerate(st.session_state.psmiles_candidates):
        if 'smiles' not in candidate or not candidate.get('smiles'):
            print(f"🔄 Adding SMILES to existing candidate {i}")
            enhanced_candidate = storage.enhance_candidate_with_smiles(candidate)
            st.session_state.psmiles_candidates[i] = enhanced_candidate
            updated_count += 1
            
    if updated_count > 0:
        print(f"✅ Updated {updated_count} existing candidates with SMILES")
    else:
        print(f"ℹ️ All existing candidates already have SMILES") 