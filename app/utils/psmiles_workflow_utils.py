"""
PSMILES Workflow Utilities

This module contains functions for managing interactive PSMILES workflows,
including dimerization, copolymerization, functional group addition, and
visualization of workflow results.
"""

import streamlit as st
import numpy as np
import random
import re
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit.components.v1 as components


def escape_psmiles_for_markdown(psmiles: str) -> str:
    """Escape asterisks in PSMILES to prevent markdown interpretation."""
    if psmiles is None:
        return "None"
    return str(psmiles).replace('*', r'\*')


def display_psmiles_workflow(result, context="main"):
    """Display PSMILES workflow results with interactive options."""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Escape PSMILES for markdown display
        original_psmiles = result.get('original_psmiles', 'N/A')
        canonical_psmiles = result.get('canonical_psmiles', 'N/A')
        escaped_original = escape_psmiles_for_markdown(original_psmiles) if original_psmiles != 'N/A' else 'N/A'
        escaped_canonical = escape_psmiles_for_markdown(canonical_psmiles) if canonical_psmiles != 'N/A' else 'N/A'
        st.markdown(f"**Original PSMILES:** `{escaped_original}`")
        st.markdown(f"**Canonical PSMILES:** `{escaped_canonical}`")
        
        # Show compound type and any special notes
        compound_type = result.get('type', 'unknown')
        if compound_type == 'organometallic':
            st.info("🏗️ **Organometallic Compound Detected** - Enhanced natural language generation with limited workflow functionality")
        elif compound_type == 'organic':
            st.success("🧪 **Organic Polymer** - Full workflow functionality available")
        
        if result.get('note'):
            st.warning(f"📝 **Note:** {result['note']}")
        
        if result.get('operation'):
            st.markdown(f"**Operation:** {result['operation']}")
        
        # Display SVG if available
        if result.get('svg_content'):
            st.markdown("### 🧪 Structure Visualization")
            
            # Clean SVG content for better Streamlit compatibility
            svg_content = result['svg_content']
            if svg_content.startswith('<?xml'):
                # Remove XML declaration for Streamlit compatibility
                svg_start = svg_content.find('<svg')
                if svg_start > 0:
                    svg_content = svg_content[svg_start:]
            
            # Use HTML component for better SVG rendering
            components.html(f"""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div style="background: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    {svg_content}
                </div>
            </div>
            """, height=400)
            
            st.session_state.svg_content = result['svg_content']
        
        # Store current PSMILES for operations
        st.session_state.current_psmiles = result.get('canonical_psmiles')
        st.session_state.psmiles_workflow_active = True
    
    with col2:
        # Show different options based on compound type
        compound_type = result.get('type', 'unknown')
        
        if compound_type == 'organometallic':
            st.markdown("### ⚠️ Limited Actions Available")
            st.markdown("**🏗️ Organometallic Compound**")
            
            st.info("This compound contains metal atoms. The psmiles library has limited support for organometallic compounds.")
            
            st.markdown("**✅ Available:**")
            st.markdown("- View structure information")
            st.markdown("- Export PSMILES string")
            st.markdown("- Use in Polymer Builder mode")
            
            st.markdown("**❌ Not Available:**")
            st.markdown("- Dimerization")
            st.markdown("- Copolymerization")
            st.markdown("- Fingerprint generation")
            st.markdown("- InChI generation")
            
            st.markdown("**💡 Suggestions:**")
            if st.button("🔄 Extract Organic Parts", key=f"extract_organic_{context}"):
                # Extract organic parts from the PSMILES
                psmiles = result.get('canonical_psmiles', '')
                # Simple extraction: remove metal atoms and their brackets
                metal_pattern = r'\[[\w\+\-]+\]'
                organic_parts = re.sub(metal_pattern, '', psmiles)
                # Clean up any double dots or empty parts
                organic_parts = re.sub(r'\.+', '.', organic_parts)
                organic_parts = organic_parts.strip('.')
                
                if organic_parts and '[*]' in organic_parts:
                    st.success(f"🧪 Extracted organic parts: `{escape_psmiles_for_markdown(organic_parts)}`")
                    st.info("You can use this simpler organic structure for full workflow functionality.")
                else:
                    st.warning("Could not extract meaningful organic parts from this structure.")
        
        else:
            # Standard organic polymer options
            st.markdown("### 🎯 Available Actions")
            
            # Dimerization options
            st.markdown("**🔗 Dimerization**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Connect Star 0", key=f"dimer_0_{context}"):
                    perform_dimerization(0)
            with col_b:
                if st.button("Connect Star 1", key=f"dimer_1_{context}"):
                    perform_dimerization(1)
            
            # Copolymerization
            st.markdown("**🧬 Copolymerization**")
            second_psmiles = st.text_input("Second PSMILES:", placeholder="e.g., [*]CC(=O)[*]", key=f"copolymer_input_{context}")
            pattern = st.selectbox("Connection Pattern:", 
                                 options=["[1,1]", "[0,1]", "[1,0]", "[0,0]"],
                                 key=f"copolymer_pattern_{context}")
            
            if st.button("Create Copolymer", key=f"copolymer_btn_{context}") and second_psmiles:
                pattern_list = eval(pattern)
                perform_copolymerization(second_psmiles, pattern_list)
            
            # Functional group addition
            st.markdown("**🧪 Add Functional Groups**")
            
            description = st.text_input("Functional group:", placeholder="e.g., hydroxyl groups", key=f"fg_desc_{context}")
            if st.button("➕ Add Functional Groups", key=f"add_fg_{context}") and description:
                perform_functional_group_addition(description)
            
            # Analysis options
            st.markdown("**🔬 Analysis**")
            if st.button("Generate Fingerprints", key=f"fingerprints_{context}"):
                generate_fingerprints()
            
            if st.button("Get InChI", key=f"inchi_{context}"):
                get_inchi_info()
        
        # Reset workflow (available for both types)
        st.markdown("---")
        if st.button("🔄 Reset Workflow", key=f"reset_workflow_{context}"):
            st.session_state.psmiles_workflow_active = False
            st.session_state.current_psmiles = None
            st.session_state.svg_content = None
            st.rerun()


def perform_dimerization(star_index):
    """Perform dimerization operation."""
    if not st.session_state.systems_initialized:
        st.error("Systems not initialized")
        return
    
    session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
    if not session_psmiles:
        st.error("No PSMILES in session")
        return
    
    psmiles_index = len(session_psmiles) - 1
    result = st.session_state.psmiles_processor.perform_dimerization(
        st.session_state.session_id, psmiles_index, star_index
    )
    
    if result['success']:
        st.success(f"✅ Dimerization complete! Connected star {star_index}")
        
        # Update workflow with new result
        st.session_state.current_psmiles = result['canonical_psmiles']
        st.session_state.svg_content = result.get('svg_content')
        st.session_state.workflow_result = result
        
        # Add to material library
        from .app_utils import add_to_material_library
        add_to_material_library(
            result['canonical_psmiles'],
            {
                'thermal_stability': np.random.uniform(0.5, 0.9),
                'biocompatibility': np.random.uniform(0.6, 0.95),
                'insulin_stability_score': np.random.uniform(0.4, 0.85)
            },
            'dimerization',
            f"Dimerized {result['parent_psmiles']} at star {star_index}"
        )
        
        st.rerun()
    else:
        st.error(f"❌ Dimerization failed: {result['error']}")


def perform_copolymerization(second_psmiles, pattern):
    """Perform copolymerization operation."""
    if not st.session_state.systems_initialized:
        st.error("Systems not initialized")
        return
    
    session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
    if not session_psmiles:
        st.error("No PSMILES in session")
        return
    
    psmiles_index = len(session_psmiles) - 1
    result = st.session_state.psmiles_processor.perform_copolymerization(
        st.session_state.session_id, psmiles_index, second_psmiles, pattern
    )
    
    if result['success']:
        st.success("✅ Copolymerization complete!")
        
        # Update workflow with new result
        st.session_state.current_psmiles = result['canonical_psmiles']
        st.session_state.svg_content = result.get('svg_content')
        st.session_state.workflow_result = result
        
        # Add to material library
        from .app_utils import add_to_material_library
        add_to_material_library(
            result['canonical_psmiles'],
            {
                'thermal_stability': np.random.uniform(0.5, 0.9),
                'biocompatibility': np.random.uniform(0.6, 0.95),
                'insulin_stability_score': np.random.uniform(0.4, 0.85)
            },
            'copolymer',
            f"Copolymer of {result['parent_psmiles1']} and {second_psmiles}"
        )
        
        st.rerun()
    else:
        st.error(f"❌ Copolymerization failed: {result['error']}")


def perform_functional_group_addition(description):
    """Perform functional group addition via copolymerization."""
    if not st.session_state.systems_initialized:
        st.error("Systems not initialized")
        return
    
    # Generate functional group PSMILES using LLM
    try:
        functional_group_request = f"functional group for {description}"
        fg_result = st.session_state.psmiles_generator.generate_psmiles(request=functional_group_request)
        
        if fg_result.get('success') and fg_result.get('psmiles'):
            functional_group_psmiles = fg_result['psmiles']
            st.success(f"Generated functional group: {functional_group_psmiles}")
        else:
            st.error(f"Failed to generate functional group: {fg_result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        st.error(f"Error generating functional group: {str(e)}")
        return
    
    # Perform copolymerization with the functional group
    connection_patterns = [[0, 1], [1, 0], [0, 0], [1, 1]]
    chosen_pattern = random.choice(connection_patterns)
    
    session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
    if not session_psmiles:
        st.error("No PSMILES in session")
        return
    
    psmiles_index = len(session_psmiles) - 1
    result = st.session_state.psmiles_processor.perform_copolymerization(
        st.session_state.session_id, psmiles_index, functional_group_psmiles, chosen_pattern
    )
    
    if result['success']:
        st.success("✅ Functional group addition complete!")
        
        # Update workflow with new result
        st.session_state.current_psmiles = result['canonical_psmiles']
        st.session_state.svg_content = result.get('svg_content')
        st.session_state.workflow_result = result
        
        # Add to material library
        from .app_utils import add_to_material_library
        add_to_material_library(
            result['canonical_psmiles'],
            {
                'thermal_stability': np.random.uniform(0.5, 0.9),
                'biocompatibility': np.random.uniform(0.6, 0.95),
                'insulin_stability_score': np.random.uniform(0.4, 0.85)
            },
            'functional_group',
            f"Added functional group to {result['parent_psmiles1']}"
        )
        
        st.rerun()
    else:
        st.error(f"❌ Addition failed: {result['error']}")


def generate_fingerprints():
    """Generate fingerprints for current PSMILES."""
    if not st.session_state.systems_initialized:
        st.error("Systems not initialized")
        return
    
    session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
    if not session_psmiles:
        st.error("No PSMILES in session")
        return
    
    psmiles_index = len(session_psmiles) - 1
    result = st.session_state.psmiles_processor.get_fingerprints(
        st.session_state.session_id, psmiles_index, ['ci', 'rdkit']
    )
    
    if result['success']:
        st.success("✅ Fingerprints generated successfully!")
        
        # Display fingerprint information
        st.markdown("### 🔬 Fingerprint Analysis")
        
        for fp_type, fp_data in result['fingerprints'].items():
            st.markdown(f"**{fp_type.upper()} Fingerprint:**")
            if isinstance(fp_data, dict):
                st.write(f"- Length: {len(fp_data)}")
                st.write(f"- Sample: {str(dict(list(fp_data.items())[:5]))}...")
            else:
                st.write(f"- {fp_data}")
            
    else:
        st.error(f"❌ Fingerprint generation failed: {result['error']}")


def get_inchi_info():
    """Get InChI information for current PSMILES."""
    if not st.session_state.systems_initialized:
        st.error("Systems not initialized")
        return
    
    session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
    if not session_psmiles:
        st.error("No PSMILES in session")
        return
    
    psmiles_index = len(session_psmiles) - 1
    result = st.session_state.psmiles_processor.get_inchi_info(
        st.session_state.session_id, psmiles_index
    )
    
    if result['success']:
        st.success("✅ InChI information generated successfully!")
        
        # Display InChI information
        st.markdown("### 🧪 InChI Information")
        st.code(result['inchi'])
        st.markdown(f"**InChI Key:** `{result['inchi_key']}`")
        st.info("The InChI (International Chemical Identifier) provides a unique textual identifier for the chemical structure.")
        
    else:
        st.error(f"❌ InChI generation failed: {result['error']}") 