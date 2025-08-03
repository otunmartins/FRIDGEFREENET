"""
Simple Active Learning UI Module

This module provides the user interface for the simple active learning system,
connecting literature mining → PSMILES generation → MD simulation → new prompt.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Set up logging
logger = logging.getLogger(__name__)

def map_ui_properties_to_computed_properties(ui_properties: List[str]) -> Dict[str, float]:
    """
    Map UI property names to actual computed property names with target values.
    
    ONLY includes properties with REAL MD computational implementations.
    All hardcoded/mock properties have been removed.
    
    Args:
        ui_properties: List of property names from the UI
        
    Returns:
        Dictionary mapping computed property names to target values
    """
    
    # Mapping from UI display names to actual computed property names and target values
    # ONLY properties with real MD computational implementations in:
    # - src/insulin_ai/integration/analysis/insulin_comprehensive_analyzer.py
    property_mapping = {
        # Structural Properties (REAL implementations)
        "RMSF (Å)": ("rmsf_polymer", 1.5),  # Line 698: md.rmsf() calculation
        "Hydrogen Bond Count": ("hydrogen_bond_count", 50.0),  # Line 744-750: md.baker_hubbard() calculation
        
        # Transport Properties (REAL implementations)  
        "Diffusion Coefficient Insulin (cm²/s)": ("diffusion_coefficient_drug", 1.0e-6),  # Line 880-990: MSD analysis of insulin movement through polymer matrix
    }
    
    # Convert UI properties to target properties dict
    target_properties = {}
    for prop in ui_properties:
        if prop in property_mapping:
            computed_prop, target_value = property_mapping[prop]
            target_properties[computed_prop] = target_value
        else:
            logger.warning(f"Unknown property: {prop}")
    
    return target_properties


def render_active_learning():
    """Main function to render the active learning interface with inner tabs for results"""
    st.title("🤖 Active Learning Material Discovery System")
    
    # Check for required systems
    if not st.session_state.get('systems_initialized'):
        st.warning("⚠️ Please ensure all systems are initialized from the Framework Overview page.")
        return
    
    if not st.session_state.get('literature_miner'):
        st.error("❌ Literature mining system not available. Please check your configuration.")
        return
    
    if not st.session_state.get('md_integration_available'):
        st.warning("""
        ⚠️ **MD Integration not available**
        
        Active Learning requires MD simulation capabilities. Please ensure:
        - GAFF force field files are properly installed
        - AMBER tools are available  
        - OpenMM is properly configured
        
        You can still test literature mining and PSMILES generation components independently.
        """)
        return
    
    # Create inner tabs similar to MD simulation structure
    tab1, tab2 = st.tabs(["🚀 Active Learning", "📊 Results"])
    
    with tab1:
        render_active_learning_main_tab()
    
    with tab2:
        # Import and render the results UI within this tab
        from insulin_ai.app.ui.active_learning_results_ui import render_active_learning_results_tab
        render_active_learning_results_tab()


def render_active_learning_main_tab():
    """Render the main active learning configuration and execution tab"""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Active Learning Configuration")
        
        # Basic configuration
        max_iterations = st.number_input("Max Iterations", min_value=1, max_value=20, value=5)
        storage_path = st.text_input("Storage Path", value="active_learning_results")
        
        st.markdown("---")
        
        # Literature Mining Settings (Enhanced with all options from Literature Mining UI)
        st.subheader("📚 Literature Mining Settings")
        lit_max_papers = st.number_input("Max Papers to Analyze", min_value=5, max_value=50, value=10, 
                                       help="Number of papers to retrieve and analyze per iteration")
        
        # Search strategy from Literature Mining UI
        lit_search_strategy = st.selectbox("Search Strategy", 
                                         ["Comprehensive (3000 tokens)", "Fast (1000 tokens)", "Focused (specific mechanisms)"],
                                         index=0, help="Literature search depth and token limit")
        
        lit_recent_only = st.checkbox("Focus on recent publications (2020+)", value=True,
                                    help="Only search papers from recent years")
        lit_include_patents = st.checkbox("Include patent literature", value=False,
                                        help="Include patents in literature search")
        
        # 🔧 NEW: Autocorrect configuration section
        st.markdown("##### 🔧 Autocorrect Settings")
        lit_enable_autocorrect = st.checkbox("Enable Forbidden Element Autocorrect", value=True,
                                           help="Automatically remove/replace Si, Al, Ge from PSMILES prompts (boron allowed)")
        
        if lit_enable_autocorrect:
            # Forbidden elements (Si, Al, Ge)
            st.markdown("**Forbidden Elements:** Si, Al, Ge")
            
            # Replacement strategy
            lit_replacement_strategy = st.selectbox(
                "Replacement Strategy",
                ["Remove entirely", "Replace with carbon", "Replace with oxygen", "Replace with nitrogen"],
                index=1,
                help="How to handle forbidden elements in generated PSMILES"
            )
        else:
            lit_replacement_strategy = "Remove entirely"
        
        st.markdown("---")
        
        # PSMILES Generation Settings
        st.subheader("🧪 PSMILES Generation Settings")
        psmiles_max_candidates = st.number_input("Max PSMILES per iteration", min_value=1, max_value=10, value=3)
        psmiles_creativity_level = st.selectbox("Creativity Level", 
                                               ["Conservative", "Balanced", "Creative"], 
                                               index=1)
        psmiles_enable_validation = st.checkbox("Enable PSMILES Validation", value=True,
                                               help="Validate generated PSMILES for chemical correctness")
        
        # Autorepair configuration (from PSMILES Generation UI)
        psmiles_enable_autorepair = st.checkbox("Enable Auto-Repair", value=True,
                                               help="Automatically fix invalid PSMILES structures")
        
        if psmiles_enable_autorepair:
            psmiles_max_repair_attempts = st.number_input("Max Repair Attempts", min_value=1, max_value=5, value=3)
        else:
            psmiles_max_repair_attempts = 1
        
        st.markdown("---")
        
        # MD Simulation Settings (Enhanced with dual GAFF+AMBER options)
        st.subheader("⚛️ MD Simulation Settings")
        
        # System configuration
        md_simulation_method = st.selectbox("Simulation Method", 
                                          ["Dual GAFF+AMBER (Recommended)"], 
                                          index=0,
                                          help="Dual method uses GAFF for polymers and AMBER for insulin. This is the only supported method in the active learning loop.",
                                          disabled=True)
        
        md_temperature = st.number_input("Temperature (K)", min_value=250, max_value=400, value=310,
                                       help="Physiological temperature = 310K")
        md_equilibration_steps = st.number_input("Equilibration Steps", min_value=1000, max_value=50000, value=10000)
        md_production_steps = st.number_input("Production Steps", min_value=5000, max_value=500000, value=50000)
        md_save_interval = st.number_input("Save Interval", min_value=100, max_value=5000, value=1000)
        
        # Enhanced polymer configuration (from dual GAFF+AMBER system)
        st.markdown("##### 🧬 Polymer Configuration")
        md_polymer_chain_length = st.number_input("Polymer Chain Length", min_value=5, max_value=50, value=20,
                                                 help="Number of repeat units per polymer chain")
        md_num_polymer_chains = st.number_input("Number of Polymer Chains", min_value=1, max_value=1000, value=3,
                                               help="Multiple chains for realistic polymer matrix")
        
        # Simulation limits
        md_max_simulations = st.number_input("Max Simulations per Iteration", min_value=1, max_value=5, value=2,
                                           help="Number of simulations to run per generated PSMILES")
        md_timeout = st.number_input("Simulation Timeout (minutes)", min_value=5, max_value=120, value=30)
        
        # Calculate total simulation time
        total_time_ns = (md_equilibration_steps + md_production_steps) * 0.002  # 2 fs timestep
        st.info(f"💡 Total simulation time: ~{total_time_ns:.1f} ns per simulation")
        
        st.markdown("---")
        
        # Advanced Settings
        st.subheader("⚙️ Advanced Settings")
        enable_parallel_processing = st.checkbox("Enable Parallel Processing", value=False,
                                                help="Run multiple components in parallel (experimental)")
        save_intermediate_results = st.checkbox("Save Intermediate Results", value=True,
                                               help="Save results after each step for debugging")
        enable_detailed_logging = st.checkbox("Enable Detailed Logging", value=True,
                                             help="Generate detailed logs for troubleshooting")
        
        # Configuration summary for the learning loop
        config = {
            'max_iterations': max_iterations,
            'storage_path': storage_path,
            # Top-level model configuration for orchestrator
            'openai_model': st.session_state.get('openai_model', 'gpt-4o-mini'),
            'temperature': st.session_state.get('temperature', 0.7),
            'literature_mining': {
                'max_papers': lit_max_papers,
                'search_strategy': lit_search_strategy,
                'recent_only': lit_recent_only,
                'include_patents': lit_include_patents,
                'enable_autocorrect': lit_enable_autocorrect,
                'replacement_strategy': lit_replacement_strategy,
                'openai_model': st.session_state.get('openai_model', 'gpt-4o-mini'),
            },
            'psmiles_generation': {
                'max_candidates': psmiles_max_candidates,
                'creativity_level': psmiles_creativity_level,
                'enable_validation': psmiles_enable_validation,
                'enable_autorepair': psmiles_enable_autorepair,
                'max_repair_attempts': psmiles_max_repair_attempts
            },
            'md_simulation': {
                'simulation_method': md_simulation_method,
                'temperature': md_temperature,
                'equilibration_steps': md_equilibration_steps,
                'production_steps': md_production_steps,
                'save_interval': md_save_interval,
                'polymer_chain_length': md_polymer_chain_length,
                'num_polymer_chains': md_num_polymer_chains,
                'max_simulations': md_max_simulations,
                'timeout_minutes': md_timeout,
                'total_time_ns': total_time_ns
            },
            'advanced': {
                'enable_parallel_processing': enable_parallel_processing,
                'save_intermediate_results': save_intermediate_results,
                'enable_detailed_logging': enable_detailed_logging
            }
        }
    
    # Main interface
    st.header("🚀 Automated Active Learning Loop")
    
    # Show which existing tabs will be used
    st.markdown("### 🔗 Integration with Existing Tabs:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📚 Literature Mining**")
        st.markdown("Uses: `literature_mining_with_llm`")
        st.markdown("*Same as Literature Mining tab*")
    
    with col2:
        st.markdown("**🧪 PSMILES Generation**") 
        st.markdown("Uses: `process_psmiles_workflow_with_autorepair`")
        st.markdown("*Same as PSMILES Generation tab*")
    
    with col3:
        st.markdown("**⚛️ MD Simulation**")
        st.markdown("Uses: `DualGaffAmberIntegration`")
        st.markdown("*Same as MD Simulation tab*")
    
    st.markdown("---")
    
    # Input parameters
    col1, col2 = st.columns(2)

    with col1:
        initial_prompt = st.text_area(
            "Initial Research Prompt",
            value="Design a biodegradable polymer for insulin delivery that is both biocompatible and stable.",
            height=100,
            help="Describe what kind of material you want to discover"
        )

    with col2:
        st.markdown("### 📋 Active Learning Focus")
        
        # Only include properties with REAL MD computational implementations
        st.markdown("**🔬 Structural Properties (Real MD Calculations):**")
        structural_properties = st.multiselect(
            "Structural Analysis",
            ["RMSF (Å)", "Hydrogen Bond Count"],
            default=[],  # No defaults - user must choose
            key="structural_props",
            help="Select structural properties to optimize"
        )
        
        st.markdown("**🚶 Transport Properties (Real MD Calculations):**")
        transport_properties = st.multiselect(
            "Transport Analysis", 
            ["Diffusion Coefficient Insulin (cm²/s)"],
            default=[],  # No defaults - user must choose
            key="transport_props",
            help="Select transport properties to optimize - tracks insulin movement through polymer matrix"
        )
        
        # Combine all selected properties
        all_properties = structural_properties + transport_properties
        
        # Validation
        if not all_properties:
            st.warning("⚠️ Please select at least one property to optimize!")
        else:
            st.success(f"✅ Optimizing {len(all_properties)} properties")
        
        st.markdown("**⚙️ Convergence Settings:**")
        convergence_threshold = st.number_input(
            "Convergence Threshold (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Stop when property improvement < threshold"
        )
    
    st.markdown("---")
    
    # Execute active learning
    execute_col1, execute_col2 = st.columns([3, 1])
    
    with execute_col1:
        st.markdown("**Ready to start automated material discovery?**")
        st.markdown("This will run literature mining → PSMILES generation → MD simulation in a loop")
    
    with execute_col2:
        start_learning = st.button("🚀 Start Active Learning", type="primary", use_container_width=True)
    
    # Enhanced execution with full workflow
    if start_learning:
        if not initial_prompt.strip():
            st.error("❌ Please provide an initial research prompt")
        elif not all_properties:
            st.error("❌ Please select at least one property to optimize!")
        else:
            # Store configuration in session state
            st.session_state.active_learning_config = config
            st.session_state.active_learning_prompt = initial_prompt
            st.session_state.active_learning_focus = all_properties
            st.session_state.convergence_threshold = convergence_threshold
            
            # Start the learning process
            run_active_learning_loop(initial_prompt, config)


def run_active_learning_loop(initial_prompt, config):
    """Run the active learning loop that connects existing tabs"""
    
    # Use the orchestrator that connects existing tab functions
    from src.insulin_ai.core.active_learning.simple_orchestrator import SimpleActiveLearningOrchestrator
    
    # Create orchestrator that uses existing tab workflows with full configuration
    orchestrator = SimpleActiveLearningOrchestrator(
        max_iterations=config['max_iterations'],
        storage_path=config['storage_path'],
        config=config  # Pass full configuration
    )
    
    # Create progress tracking containers
    status_container = st.container()
    progress_container = st.container()
    iteration_container = st.container()
    
    # Progress tracking
    progress_bar = progress_container.progress(0)
    status_text = status_container.empty()
    iteration_display = iteration_container.empty()
    
    # Callback for iteration updates
    def iteration_callback(state):
        iteration_num = state['iteration']
        progress = min(iteration_num / config['max_iterations'], 1.0)
        progress_bar.progress(progress)
        
        status_text.write(f"**Iteration {iteration_num}**: {state['status']} - Score: {state.get('overall_score', 0):.3f}")
        
        # Display iteration details
        with iteration_display.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Iteration", iteration_num)
                st.metric("Status", state['status'])
            
            with col2:
                st.metric("Overall Score", f"{state.get('overall_score', 0):.3f}")
                if 'improvement_over_previous' in state:
                    st.metric("Improvement", f"{state['improvement_over_previous']:.3f}")
            
            with col3:
                st.metric("Errors", len(state.get('errors', [])))
                st.metric("Warnings", 0)
            
            # Show which existing tab functions are being used (defensive coding)
            simulation_results = state.get('simulation_results') or {}
            components = {
                "Literature Mining (existing tab)": state.get('literature_results') is not None,
                "PSMILES Generation (existing tab)": state.get('generated_molecules') is not None,
                "MD Simulation (existing tab)": state.get('simulation_results') is not None,
                "Property Calculation": simulation_results.get('properties_computed') is not None if isinstance(simulation_results, dict) else False,
                "New Prompt Generation": state.get('new_prompt') is not None
            }
            
            st.markdown("**Existing Tab Integration Status:**")
            component_cols = st.columns(5)
            for i, (name, completed) in enumerate(components.items()):
                with component_cols[i]:
                    icon = "✅" if completed else "⏳"
                    display_name = name.split()[0]  # Show first word
                    st.markdown(f"{icon} {display_name}")
    
    # Completion callback
    def completion_callback(results):
        status_text.success(f"🎉 Active Learning Loop Completed!")
        progress_bar.progress(1.0)
        
        # Display final results
        with iteration_display.container():
            st.success("**🎉 Active Learning Loop Completed Successfully!**")
            st.info("✅ **All existing tab functions used without modifications**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Iterations", results.get('total_iterations', 'N/A'))
            
            with col2:
                st.metric("Success Rate", f"{results.get('success_rate', 0)*100:.1f}%")
            
            with col3:
                st.metric("Best Score", f"{results.get('best_score', 0):.3f}")
            
            with col4:
                st.metric("Runtime", f"{results.get('total_runtime', 0):.1f} seconds")
    
    # Register callbacks
    orchestrator.add_iteration_callback(iteration_callback)
    orchestrator.add_completion_callback(completion_callback)
    
    # Store orchestrator in session state for monitoring
    st.session_state.active_learning_orchestrator = orchestrator
    st.session_state.active_learning_storage_path = config['storage_path']
    
    # Get target properties from config if available
    target_properties_ui = st.session_state.get('active_learning_focus', [])
    target_properties = map_ui_properties_to_computed_properties(target_properties_ui)

    # Run the active learning loop that uses existing tabs
    try:
        status_text.info("🚀 Starting active learning loop...")
        st.info("🔄 **Active Learning Loop**: Automatically running your existing Literature Mining → PSMILES Generation → MD Simulation tabs in sequence.")
        
        # Run the loop that connects existing tab workflows
        with st.spinner("🔄 Running active learning loop using existing tabs..."):
            results = orchestrator.run_simple_loop(
                initial_prompt=initial_prompt,
                target_properties=target_properties,
            )
        
        # Process and display results
        if results:
            completion_callback(results)
            st.success("✅ Active learning loop completed successfully using existing tab workflows!")
            
            # Display summary
            if 'summary' in results:
                summary = results['summary']
                st.metric("Total Iterations", summary.get('total_iterations', 'N/A'))
                st.metric("Success Rate", f"{summary.get('success_rate', 0)*100:.1f}%")
                st.metric("Best Score", f"{summary.get('best_score', 0):.3f}")
                st.metric("Runtime", f"{summary.get('total_runtime', 0):.1f} seconds")
        else:
            st.warning("⚠️ Active learning loop completed but returned no results.")
        
    except Exception as e:
        st.error(f"❌ Error running active learning loop: {e}")
        st.exception(e)
        
        # Show troubleshooting information
        with st.expander("🔍 Troubleshooting Information"):
            st.code(f"""
Error: {type(e).__name__}: {str(e)}

This error occurred while running the simple active learning loop.
Common issues:
1. Missing dependencies (check OpenMM, RDKit, etc.)
2. OpenAI API key issues
3. Insufficient memory or disk space
4. Network connectivity issues for literature mining

Check the logs above for more details.
            """)
            
            # Show system status
            st.write("**System Status:**")
            try:
                st.write(f"- Orchestrator: {type(orchestrator).__name__}")
                st.write(f"- Storage Path: {orchestrator.storage_path}")
            except Exception as status_e:
                st.write(f"Could not get system status: {status_e}")


def render_active_learning_runner(max_iterations, convergence_patience, score_threshold, 
                                max_memory_gb, max_time_hours, min_confidence, max_error_rate):
    """Render the active learning runner interface"""
    
    st.subheader("🔄 Simple Active Learning Runner")
    st.markdown("*Run autonomous material discovery loops*")
    
    # Input parameters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        initial_prompt = st.text_area(
            "Research Objective",
            value="Design a biodegradable polymer for insulin delivery",
            height=100,
            help="Describe the material discovery goal"
        )
        
        # Active learning focus
        with st.expander("🎯 Active Learning Approach"):
            st.markdown("""
            **The system optimizes based on:**
            - MD simulation observables (RMSD, energy, density, etc.)
            - Literature-guided material insights
            - Iterative prompt refinement
            - Structural and dynamic properties from simulations
            
            **⚠️ Note:** Only real computed properties are used.
            No estimated or random properties are generated.
            """)
    
    with col2:
        storage_path = st.text_input("Output Directory", value="simple_active_learning_output")
        
        if st.button("🚀 Run Simple Loop", type="primary"):
            # This now properly uses the main active learning loop
            config = st.session_state.get('active_learning_config', {})
            config.update({
                'max_iterations': max_iterations,
                'storage_path': storage_path,
            })
            run_active_learning_loop(
                initial_prompt=initial_prompt,
                config=config
            )


def render_system_status():
    """Render system status for simple active learning"""
    st.subheader("📊 Simple System Status")
    
    # Check if orchestrator is in session state
    if 'active_learning_orchestrator' in st.session_state:
        orchestrator = st.session_state.active_learning_orchestrator
        storage_path = st.session_state.get('active_learning_storage_path', 'Unknown')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Orchestrator Type", "Simple")
            st.metric("Storage Path", storage_path)
            st.metric("Max Iterations", orchestrator.max_iterations)
        
        with col2:
            st.metric("Results Available", len(orchestrator.results))
            if orchestrator.results:
                latest_result = orchestrator.results[-1]
                st.metric("Latest Status", latest_result.get('status', 'Unknown'))
                st.metric("Latest Score", f"{latest_result.get('overall_score', 0):.3f}")
    else:
        st.info("No active learning session running")


# Legacy compatibility functions (kept for backward compatibility)
def render_active_learning_configuration():
    """Legacy function for backward compatibility"""
    st.info("Using simplified configuration. See sidebar for options.")

def render_iteration_details(state):
    """Legacy function for backward compatibility"""
    pass

def render_results_visualization(results):
    """Legacy function for backward compatibility"""
    pass 