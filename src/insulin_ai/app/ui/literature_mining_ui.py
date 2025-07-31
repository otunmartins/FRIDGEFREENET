"""
Literature Mining UI Module for Insulin-AI App

This module provides the user interface for literature mining with LLM analysis,
adaptive query generation, and active learning integration.

Author: AI-Driven Material Discovery Team
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

def render_literature_mining_ui():
    """
    Render the complete Literature Mining UI interface
    
    Returns:
        None
    """
    st.subheader("📚 Literature Mining with LLM Analysis")
    
    # Import required functions with fallback handling
    try:
        from app.services.system_service import check_systems_initialized
        from app.utils.app_utils import add_to_material_library
        from insulin_ai.integration.rag_literature_mining import RAGLiteratureMiningSystem
        
        # Test RAG system initialization
        rag_system = RAGLiteratureMiningSystem()
        systems_initialized = check_systems_initialized()
        rag_available = True
    except ImportError as e:
        st.error("⚠️ Required services not available. Please check system configuration.")
        st.info(f"Import error: {e}")
        systems_initialized = False
        rag_available = False
    except Exception as e:
        st.warning(f"⚠️ RAG system initialization issue: {e}")
        systems_initialized = check_systems_initialized() if 'check_systems_initialized' in locals() else False
        rag_available = False
    
    if not systems_initialized:
        st.error("⚠️ AI systems not initialized. Please restart the application.")
        st.stop()
    
    if not rag_available:
        st.warning("⚠️ RAG Literature Mining system not fully available. Using fallback mode.")
        st.info("💡 To enable full RAG capabilities, ensure OpenAI API key is configured and dependencies are installed.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_query_generation_interface()
    
    with col2:
        render_iteration_history()
        render_example_queries()


def render_query_generation_interface():
    """Render the adaptive query generation interface"""
    st.markdown("### Adaptive Query Generation")
    
    # Use previous iteration context if available
    if st.session_state.get('iteration_feedback'):
        st.info("📈 Using feedback from previous iterations to refine search strategy")
        
        with st.expander("Previous Iteration Insights"):
            feedback = st.session_state.iteration_feedback
            st.write("**Top Performing Materials:**", feedback.get('top_materials', []))
            st.write("**Successful Mechanisms:**", feedback.get('mechanisms', []))
    
    # Query configuration
    query_type = st.selectbox(
        "Literature Focus",
        ["Thermal Stabilization", "Insulin-Polymer Interactions", "Transdermal Delivery", 
         "Protein Aggregation Prevention", "Glass Transition Optimization"]
    )
    
    # Initialize the text area key for suggested queries
    if 'research_query_text' not in st.session_state:
        st.session_state.research_query_text = ""
    
    # Check if an example query was clicked
    if st.session_state.get('example_query'):
        st.session_state.research_query_text = st.session_state.example_query
        st.session_state.example_query = None
    
    user_query = st.text_area(
        "Research Query",
        value=st.session_state.research_query_text,
        placeholder="e.g., polymer matrices for insulin thermal stabilization at ambient temperature",
        height=100,
        key="research_query_input"
    )
    
    # Update the session state with current text area value
    st.session_state.research_query_text = user_query
    
    # Advanced search parameters
    with st.expander("Search Parameters"):
        search_strategy = st.selectbox(
            "Search Strategy",
            ["Comprehensive (3000 tokens)", "Fast (1000 tokens)", "Focused (specific mechanisms)"]
        )
        
        include_recent = st.checkbox("Focus on recent publications (2020+)", value=True)
        include_patents = st.checkbox("Include patent literature")
    
    # Mining execution
    if st.button("🔍 Mine Literature", type="primary"):
        if user_query:
            # Clear previous results when starting new mining
            st.session_state.current_mining_result = None
            execute_literature_mining(user_query, query_type, search_strategy)
        else:
            st.warning("Please enter a research query.")
    
    # Display persistent results section
    render_persistent_results()


def execute_literature_mining(user_query: str, query_type: str, search_strategy: str):
    """Execute the literature mining process with LLM analysis"""
    
    with st.spinner("Analyzing literature with LLM..."):
        # Use iteration context for adaptive mining
        iteration_context = st.session_state.get('iteration_feedback', None)
        
        try:
            # Import literature mining function with fallback
            try:
                from app.services.literature_service import literature_mining_with_llm
                mining_result = literature_mining_with_llm(user_query, iteration_context)
            except ImportError:
                # Fallback to core functionality if service not available
                mining_result = fallback_literature_mining(user_query, iteration_context)
            
            # Store iteration
            if 'literature_iterations' not in st.session_state:
                st.session_state.literature_iterations = []
            
            iteration_data = {
                'query': user_query,
                'result': mining_result,
                'timestamp': datetime.now().isoformat(),
                'iteration': len(st.session_state.literature_iterations) + 1,
                'query_type': query_type,
                'search_strategy': search_strategy
            }
            
            st.session_state.literature_iterations.append(iteration_data)
            
            # Store current result for persistent display
            st.session_state.current_mining_result = iteration_data
            
            # Add materials to library
            process_mining_candidates(mining_result)
            
            st.success("✅ Literature mining completed! Results are now persistent and can be viewed below.")
            
        except Exception as e:
            st.error(f"Literature mining failed: {str(e)}")
            st.info("Please try again with a different query or check system configuration.")


def display_mining_results(mining_result: Dict[str, Any]):
    """Display the results of literature mining analysis"""
    
    # Check if this is real RAG analysis or simulation
    analysis_method = mining_result.get('analysis_method', 'simulation')
    is_rag_powered = analysis_method == 'rag_powered'
    
    st.markdown(f"""
    <div class="llm-response">
        <h4>🤖 Literature Analysis Results {'🔬 (Real AI Analysis)' if is_rag_powered else '⚡ (Simulation Mode)'}</h4>
        <p><strong>Papers Analyzed:</strong> {mining_result.get('papers_analyzed', 0)}</p>
        <p><strong>Analysis Method:</strong> {'Real Semantic Scholar + OpenAI RAG' if is_rag_powered else 'Simulated Results'}</p>
        <p><strong>Insights:</strong> {mining_result['insights']}</p>
        <p><strong>Materials Found:</strong> {', '.join(mining_result['materials_found'])}</p>
        <p><strong>Key Mechanisms:</strong> {', '.join(mining_result['stabilization_mechanisms'])}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display PSMILES generation prompt if available
    psmiles_prompt = mining_result.get('psmiles_generation_prompt', '')
    if psmiles_prompt:
        st.markdown("### 🧬 PSMILES Generation Prompt")
        st.info("Copy this prompt and use it in the PSMILES Generation tab to create targeted polymers:")
        
        with st.expander("📋 Copy PSMILES Generation Prompt", expanded=False):
            st.text_area(
                "PSMILES Generation Prompt",
                value=psmiles_prompt,
                height=300,
                help="Copy this prompt and paste it into the PSMILES Generation tab"
            )
            
            if st.button("📋 Copy to Clipboard", key="copy_psmiles_prompt"):
                # Store in session state for access by PSMILES tab
                st.session_state.literature_psmiles_prompt = psmiles_prompt
                st.success("✅ Prompt copied! Go to PSMILES Generation tab to use it.")
    
    elif is_rag_powered:
        st.warning("⚠️ PSMILES prompt generation failed - but analysis was successful!")


def process_mining_candidates(mining_result: Dict[str, Any]):
    """Process and add mining candidates to material library"""
    
    try:
        from app.utils.app_utils import add_to_material_library
        
        if mining_result.get('material_candidates'):
            for material in mining_result['material_candidates'][:3]:
                properties = {
                    'thermal_stability': np.random.uniform(0.5, 0.9),
                    'biocompatibility': np.random.uniform(0.6, 0.95),
                    'insulin_stability_score': np.random.uniform(0.4, 0.9)
                }
                
                material_name = material.get('material_name', 'Unknown')
                psmiles = '[*]CC[*]'  # Placeholder - would need PSMILES generation
                
                add_to_material_library(psmiles, properties, 'literature', material_name)
    except ImportError:
        st.warning("⚠️ Material library functions not available")


def render_mining_action_buttons(mining_result: Dict[str, Any]):
    """Render action buttons for mining results"""
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("➡️ Generate PSMILES"):
            if mining_result['materials_found']:
                st.session_state.selected_material = mining_result['materials_found'][0]
                st.success(f"Selected: {st.session_state.selected_material}")
    
    with col_b:
        if st.button("🔄 Add to Active Learning"):
            if 'active_learning_queue' not in st.session_state:
                st.session_state.active_learning_queue = []
            
            st.session_state.active_learning_queue.append({
                'type': 'literature_insight',
                'content': mining_result,
                'priority': 0.8,
                'timestamp': datetime.now().isoformat()
            })
            st.success("Added to learning queue!")
    
    with col_c:
        if st.button("📊 Update Feedback"):
            if 'iteration_feedback' not in st.session_state:
                st.session_state.iteration_feedback = {}
                
            st.session_state.iteration_feedback.update({
                'mechanisms': mining_result['stabilization_mechanisms'],
                'top_materials': mining_result['materials_found'][:3]
            })
            st.success("Feedback updated!")


def render_persistent_results():
    """Render persistent literature mining results"""
    
    if st.session_state.get('current_mining_result'):
        # Header with clear option
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📚 Current Literature Mining Results")
        with col2:
            if st.button("🗑️ Clear Results", help="Clear current results to start fresh"):
                st.session_state.current_mining_result = None
                st.rerun()
        
        current_result = st.session_state.current_mining_result
        if current_result:  # Check again after potential clearing
            mining_result = current_result['result']
            
            # Display the results persistently
            display_mining_results(mining_result)
            
            # Render action buttons
            render_mining_action_buttons(mining_result)
            
            # Add metadata about the query
            with st.expander("📋 Query Details"):
                st.write(f"**Query:** {current_result['query']}")
                st.write(f"**Focus:** {current_result.get('query_type', 'N/A')}")
                st.write(f"**Strategy:** {current_result.get('search_strategy', 'N/A')}")
                st.write(f"**Timestamp:** {current_result['timestamp']}")
                st.write(f"**Iteration:** {current_result['iteration']}")
    else:
        st.info("🔍 Run literature mining to see persistent results here.")


def render_iteration_history():
    """Render the literature iteration history sidebar with clickable results"""
    
    st.markdown("### Literature Iteration History")
    
    if st.session_state.get('literature_iterations'):
        st.info("💡 Click on any iteration to view those results")
        
        for i, iteration in enumerate(st.session_state.literature_iterations[-5:]):  # Show last 5
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="iteration-card">
                    <strong>Iteration {iteration['iteration']}</strong><br>
                    <small>Query: {iteration['query'][:50]}...</small><br>
                    <small>Found: {len(iteration['result']['materials_found'])} materials</small><br>
                    <small>Time: {iteration['timestamp'][:16]}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("📖 View", key=f"view_iteration_{iteration['iteration']}"):
                    st.session_state.current_mining_result = iteration
                    st.rerun()
    else:
        st.info("No iterations yet. Start mining literature!")


def render_example_queries():
    """Render common example queries for insulin stabilization"""
    
    st.markdown("### Common Insulin Stabilization Queries")
    
    example_queries = [
        "trehalose insulin thermal protection mechanisms",
        "PEG conjugation protein stability enhancement", 
        "chitosan hydrogel insulin delivery patch",
        "PLGA microsphere insulin encapsulation"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"📝 {example}", key=f"lit_example_{i}"):
            st.session_state.example_query = example
            st.rerun()


def fallback_literature_mining(user_query: str, iteration_context: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Fallback literature mining implementation when services are not available
    
    Args:
        user_query: The research query from the user
        iteration_context: Optional context from previous iterations
        
    Returns:
        Dictionary containing mining results
    """
    
    # Simulate literature mining results for development/testing
    simulated_results = {
        'papers_analyzed': np.random.randint(15, 45),
        'insights': f"Analysis of recent literature on '{user_query}' reveals promising polymer-based approaches for insulin stabilization.",
        'materials_found': [
            'Trehalose-PEG conjugates',
            'Chitosan-insulin complexes', 
            'PLGA microspheres',
            'Cyclodextrin inclusions'
        ],
        'stabilization_mechanisms': [
            'Hydrogen bonding stabilization',
            'Hydrophobic interactions',
            'Controlled release kinetics',
            'Thermal protection'
        ],
        'material_candidates': [
            {'material_name': 'Trehalose-PEG matrix', 'confidence': 0.85},
            {'material_name': 'Chitosan hydrogel', 'confidence': 0.78},
            {'material_name': 'PLGA encapsulation', 'confidence': 0.72}
        ]
    }
    
    return simulated_results


# CSS Styles for the module
def inject_literature_mining_styles():
    """Inject CSS styles for the literature mining UI"""
    
    st.markdown("""
    <style>
    .llm-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .iteration-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: box-shadow 0.2s;
    }
    
    .iteration-card:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# Module initialization
if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="Literature Mining UI Test", layout="wide")
    inject_literature_mining_styles()
    render_literature_mining_ui() 