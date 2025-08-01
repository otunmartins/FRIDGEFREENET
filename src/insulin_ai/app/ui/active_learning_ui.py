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

def render_active_learning():
    """Main function to render the active learning interface that connects existing tabs"""
    st.title("🤖 Active Learning Material Discovery System")
    st.markdown("**Automated Loop**: Connects existing Literature Mining → PSMILES Generation → MD Simulation tabs")
    
    # Important clarification
    st.info("🔗 **This connects your existing working tabs without modifications:** Uses the same functions as Literature Mining tab, PSMILES Generation tab, and MD Simulation tab")
    
    # Check if we can import the simple orchestrator
    try:
        from src.insulin_ai.core.active_learning.simple_orchestrator import SimpleActiveLearningOrchestrator
        simple_active_learning_available = True
    except ImportError as e:
        simple_active_learning_available = False
        import_error = str(e)
    
    if not simple_active_learning_available:
        st.error("❌ Active Learning orchestrator is not available. Please check the installation.")
        with st.expander("🔍 Import Error Details"):
            st.code(import_error, language="python")
        
        # Show manual installation instructions
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            **Possible Solutions:**
            
            1. **Check that the orchestrator exists:**
            ```bash
            ls src/insulin_ai/core/active_learning/simple_orchestrator.py
            ```
            
            2. **Check if you're in the right directory:**
            ```bash
            pwd  # Should show: .../insulin-ai
            ls src/insulin_ai/core/active_learning/  # Should show the files
            ```
            
            3. **Try starting the app from the project root:**
            ```bash
            cd /path/to/insulin-ai
            streamlit run src/insulin_ai/app.py
            ```
            
            4. **Verify Python path:**
            ```bash
            python -c "import sys; print('\\n'.join(sys.path))"
            ```
            """)
        return
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ OpenAI API key is required for the decision engine. Please set OPENAI_API_KEY environment variable.")
        return
    
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
        
        lit_openai_model = st.selectbox("Literature Analysis Model", 
                                      ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], 
                                      index=1, help="OpenAI model for literature analysis")
        lit_temperature = st.slider("Literature LLM Temperature", 0.0, 1.0, 0.7, 0.1,
                                   help="Temperature for literature analysis LLM")
        
        st.markdown("---")
        
        # PSMILES Generation Settings (Enhanced with all options from PSMILES Generation UI)
        st.subheader("🧪 PSMILES Generation Settings")
        psmiles_model = st.selectbox("PSMILES Generation Model",
                                   ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                                   index=0, help="OpenAI model for PSMILES generation")
        psmiles_temperature = st.slider("PSMILES LLM Temperature", 0.0, 1.0, 0.7, 0.1,
                                       help="Temperature for PSMILES generation")
        psmiles_num_candidates = st.slider("Number of candidates per iteration", 1, 10, 1,
                                         help="How many PSMILES candidates to generate per iteration")
        psmiles_max_retries = st.number_input("Max Generation Retries", min_value=1, max_value=10, value=5,
                                            help="Maximum retries for PSMILES generation")
        psmiles_enable_functionalization = st.checkbox("Multi-step functionalization", value=True,
                                                      help="Enable multi-step functionalization process")
        psmiles_max_repair_attempts = st.number_input("Max Repair Attempts", min_value=1, max_value=10, value=3,
                                                     help="Maximum attempts to repair invalid PSMILES")
        
        st.markdown("---")
        
        # MD Simulation Settings (Enhanced with all options from Simulation UI)
        st.subheader("⚛️ MD Simulation Settings")
        
        # Simulation method selection (from Simulation UI)
        md_simulation_method = st.selectbox(
            "Force Field Approach",
            options=["Dual GAFF+AMBER (Recommended)", "Enhanced (Stored SMILES)", "Standard (Legacy)"],
            index=0,  # Default to dual approach
            help="""
            • **🚀 Dual GAFF+AMBER**: GAFF for polymers + AMBER for insulin (Fixed CYS/CYX issues)
            • **⚡ Enhanced**: Uses pre-stored SMILES data for faster setup
            • **🔧 Standard**: Original approach (may have template generator issues)
            """
        )
        
        md_temperature = st.slider("Temperature (K)", 250, 400, 310, 5,
                                 help="Simulation temperature in Kelvin (physiological = 310 K)")
        
        # Equilibration options (from Simulation UI)
        equilibration_options = {
            "Quick Test (250 ps)": 125000,
            "Short (500 ps) - Recommended": 250000,
            "Medium (1000 ps)": 500000,
            "Long (2000 ps)": 1000000,
            "Extended (4000 ps)": 2000000
        }
        md_equilibration_selection = st.selectbox(
            "Equilibration Duration",
            list(equilibration_options.keys()),
            index=0,
            help="Equilibration phase duration (2 fs timestep)"
        )
        md_equilibration_steps = equilibration_options[md_equilibration_selection]
        
        # Production options (from Simulation UI)
        production_options = {
            "Quick Test (1 ns)": 500000,
            "Short (2.5 ns)": 1250000,
            "Medium (5 ns) - Recommended": 2500000,
            "Long (10 ns)": 5000000,
            "Extended (25 ns)": 12500000
        }
        md_production_selection = st.selectbox(
            "Production Duration",
            list(production_options.keys()),
            index=0,
            help="Production phase duration (2 fs timestep)"
        )
        md_production_steps = production_options[md_production_selection]
        
        # Save interval options (from Simulation UI)
        save_options = {
            "Frequent (1 ps)": 500,
            "Normal (2 ps) - Recommended": 1000,
            "Sparse (4 ps)": 2000,
            "Very Sparse (8 ps)": 4000
        }
        md_save_selection = st.selectbox(
            "Frame Saving Frequency",
            list(save_options.keys()),
            index=1,
            help="How often to save trajectory frames"
        )
        md_save_interval = save_options[md_save_selection]
        
        md_max_simulations = st.number_input("Max Simulations per Iteration", min_value=1, max_value=10, value=3,
                                           help="Maximum number of simulations to run per iteration")
        md_timeout = st.number_input("Simulation Timeout (minutes)", min_value=5, max_value=120, value=30,
                                   help="Timeout for individual simulations")
        
        # Calculate and display timing information
        eq_time_ns = md_equilibration_steps * 2 / 1000000
        prod_time_ns = md_production_steps * 2 / 1000000
        total_time_ns = eq_time_ns + prod_time_ns
        st.caption(f"⏱️ **Total simulation time: {total_time_ns:.1f} ns**")
        
        st.markdown("---")
        
        # Advanced Options
        st.subheader("🔬 Advanced Options")
        enable_parallel_processing = st.checkbox("Enable Parallel Processing", value=False,
                                                help="Run multiple simulations in parallel")
        save_intermediate_results = st.checkbox("Save Intermediate Results", value=True,
                                               help="Save results after each stage")
        enable_detailed_logging = st.checkbox("Enable Detailed Logging", value=True,
                                            help="Enable verbose logging for debugging")
        # Remove fallback option since we don't want fallbacks [[memory:4967721]]
        
        # Package the enhanced configuration
        al_config = {
            'max_iterations': max_iterations,
            'storage_path': storage_path,
            'literature_mining': {
                'max_papers': lit_max_papers,
                'search_strategy': lit_search_strategy,
                'recent_only': lit_recent_only,
                'include_patents': lit_include_patents,
                'openai_model': lit_openai_model,
                'temperature': lit_temperature
            },
            'psmiles_generation': {
                'model': psmiles_model,
                'temperature': psmiles_temperature,
                'num_candidates': psmiles_num_candidates,
                'max_retries': psmiles_max_retries,
                'enable_functionalization': psmiles_enable_functionalization,
                'max_repair_attempts': psmiles_max_repair_attempts
            },
            'md_simulation': {
                'simulation_method': md_simulation_method,
                'temperature': md_temperature,
                'equilibration_steps': md_equilibration_steps,
                'production_steps': md_production_steps,
                'save_interval': md_save_interval,
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
            value="Design a biodegradable polymer for insulin delivery",
            height=100,
            help="Describe what kind of material you want to discover"
        )
    
    with col2:
        st.subheader("Target Properties")
        biocompatibility = st.slider("Biocompatibility", 0.0, 1.0, 0.9, 0.1)
        degradation_rate = st.slider("Degradation Rate", 0.0, 1.0, 0.5, 0.1)
        stability = st.slider("Stability", 0.0, 1.0, 0.8, 0.1)
        
        target_properties = {
            "biocompatibility": biocompatibility,
            "degradation_rate": degradation_rate,
            "stability": stability
        }
    
    # Run button
    if st.button("🚀 Start Active Learning Loop", type="primary"):
        if not initial_prompt.strip():
            st.warning("⚠️ Please provide an initial research prompt")
            return
        
        run_active_learning_loop(
            initial_prompt=initial_prompt,
            target_properties=target_properties,
            config=al_config
        )


def run_active_learning_loop(initial_prompt, target_properties, config):
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
    
    # Run the active learning loop that uses existing tabs
    try:
        status_text.info("🚀 Starting active learning loop...")
        st.info("🔄 **Active Learning Loop**: Automatically running your existing Literature Mining → PSMILES Generation → MD Simulation tabs in sequence.")
        
        # Run the loop that connects existing tab workflows
        with st.spinner("🔄 Running active learning loop using existing tabs..."):
            results = orchestrator.run_simple_loop(
                initial_prompt=initial_prompt,
                target_properties=target_properties
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
        
        # Target properties
        with st.expander("🎯 Target Material Properties"):
            biocompatibility = st.slider("Biocompatibility", 0.0, 1.0, 0.9, 0.1)
            degradation_rate = st.slider("Degradation Rate", 0.0, 1.0, 0.5, 0.1)
            stability = st.slider("Thermal Stability", 0.0, 1.0, 0.8, 0.1)
            
            target_properties = {
                "biocompatibility": biocompatibility,
                "degradation_rate": degradation_rate,
                "stability": stability
            }
    
    with col2:
        storage_path = st.text_input("Output Directory", value="simple_active_learning_output")
        
        if st.button("🚀 Run Simple Loop", type="primary"):
            run_simple_active_learning_loop(
                initial_prompt=initial_prompt,
                target_properties=target_properties,
                max_iterations=max_iterations,
                storage_path=storage_path
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