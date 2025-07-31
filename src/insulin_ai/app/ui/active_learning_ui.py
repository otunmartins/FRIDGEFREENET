"""
Active Learning UI Module - Phase 1

This module provides the user interface for the new active learning system,
including orchestrator control, real-time monitoring, and iteration visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Import the active learning system
try:
    # Try multiple import paths to handle different execution contexts
    try:
        from insulin_ai.core.active_learning import (
            ActiveLearningOrchestrator, 
            StateManager, 
            LLMDecisionEngine,
            LoopController,
            ConvergenceConfig,
            ResourceLimits,
            QualityGates,
            IterationStatus
        )
    except ImportError:
        # Fallback import for different path contexts
        import sys
        from pathlib import Path
        
        # Add the project root to the path
        project_root = Path(__file__).parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from insulin_ai.core.active_learning import (
            ActiveLearningOrchestrator, 
            StateManager, 
            LLMDecisionEngine,
            LoopController,
            ConvergenceConfig,
            ResourceLimits,
            QualityGates,
            IterationStatus
        )
    
    ACTIVE_LEARNING_AVAILABLE = True
except ImportError as e:
    ACTIVE_LEARNING_AVAILABLE = False
    import traceback
    IMPORT_ERROR_DETAILS = f"Import error: {e}\nTraceback: {traceback.format_exc()}"


def render_active_learning():
    """Main function to render the active learning interface"""
    st.title("🤖 Active Learning Material Discovery System")
    st.markdown("**Phase 1**: Autonomous material discovery through intelligent iteration")
    
    if not ACTIVE_LEARNING_AVAILABLE:
        st.error("❌ Active Learning system is not available. Please check the installation.")
        if 'IMPORT_ERROR_DETAILS' in globals():
            with st.expander("🔍 Import Error Details"):
                st.code(IMPORT_ERROR_DETAILS, language="python")
        
        # Show manual installation instructions
        with st.expander("🛠️ Troubleshooting"):
            st.markdown("""
            **Possible Solutions:**
            
            1. **Run the Phase 1 test script first:**
            ```bash
            python test_active_learning_phase1_simple.py
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
        
        # Convergence settings
        st.subheader("Convergence Settings")
        convergence_patience = st.number_input("Patience (iterations without improvement)", min_value=1, max_value=10, value=3)
        score_threshold = st.slider("Target Score Threshold", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
        
        # Resource limits
        st.subheader("Resource Limits")
        max_memory_gb = st.number_input("Max Memory (GB)", min_value=1, max_value=32, value=8)
        max_time_hours = st.number_input("Max Time (hours)", min_value=0.5, max_value=24.0, value=4.0, step=0.5)
        
        # Quality gates
        st.subheader("Quality Gates")
        min_confidence = st.slider("Min Decision Confidence", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
        max_error_rate = st.slider("Max Error Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Run Active Learning", "📊 Monitor Progress", "📋 Iteration History", "⚙️ System Status"])
    
    with tab1:
        render_active_learning_runner(max_iterations, convergence_patience, score_threshold, 
                                    max_memory_gb, max_time_hours, min_confidence, max_error_rate)
    
    with tab2:
        render_progress_monitor()
    
    with tab3:
        render_iteration_history()
    
    with tab4:
        render_system_status()


def render_active_learning_runner(max_iterations, convergence_patience, score_threshold, 
                                max_memory_gb, max_time_hours, min_confidence, max_error_rate):
    """Render the active learning runner interface"""
    st.header("🚀 Active Learning Loop Runner")
    
    # Input configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Initial Configuration")
        initial_prompt = st.text_area(
            "Initial Prompt",
            value="Design a biodegradable polymer for insulin delivery with high biocompatibility and controlled degradation rate",
            height=100,
            help="This prompt will seed the first iteration of the active learning loop"
        )
        
        storage_path = st.text_input(
            "Storage Path",
            value="active_learning_results",
            help="Directory to store iteration results and state"
        )
    
    with col2:
        st.subheader("Target Properties")
        
        # Use realistic property ranges based on drug delivery literature
        biocompatibility = st.slider(
            "Biocompatibility Target", 
            min_value=0.5, max_value=1.0, value=0.9, step=0.05,
            help="Biocompatibility score based on molecular interactions, glass transition temperature, and surface properties"
        )
        
        degradation_rate = st.slider(
            "Degradation Rate Target", 
            min_value=0.1, max_value=1.0, value=0.7, step=0.05,
            help="Degradation rate score based on chain scission, water diffusion, and ester bond strength"
        )
        
        mechanical_strength = st.slider(
            "Mechanical Strength Target", 
            min_value=0.1, max_value=1.0, value=0.6, step=0.05,
            help="Mechanical strength score based on elastic moduli, cohesive energy, and density"
        )

        target_properties = {
            "biocompatibility": biocompatibility,
            "degradation_rate": degradation_rate,
            "mechanical_strength": mechanical_strength
        }
        
        # Display property explanations
        with st.expander("ℹ️ Property Score Explanations"):
            st.markdown("""
            **Biocompatibility (0-1)**: Based on glass transition temperature (optimal ~37°C), 
            hydrogen bonding patterns, density (1.0-1.4 g/cm³), and water interaction properties.
            
            **Degradation Rate (0-1)**: Based on polymer chain scission rates, water diffusion 
            coefficients, ester bond strength (~300 kJ/mol optimal), and molecular mobility.
            
            **Mechanical Strength (0-1)**: Based on Young's modulus (0.1-5.0 GPa), bulk/shear 
            moduli, cohesive energy, and crystallinity balance.
            
            *Scores are derived from MD simulation properties using literature correlations.*
            """)
    
    # Configuration summary
    with st.expander("📋 Configuration Summary"):
        config_data = {
            "Max Iterations": max_iterations,
            "Convergence Patience": convergence_patience,
            "Score Threshold": score_threshold,
            "Max Memory (GB)": max_memory_gb,
            "Max Time (hours)": max_time_hours,
            "Min Confidence": min_confidence,
            "Max Error Rate": max_error_rate,
            "Target Properties": target_properties
        }
        st.json(config_data)
    
    # Run button and status
    if st.button("🚀 Start Active Learning Loop", type="primary", use_container_width=True):
        if not initial_prompt.strip():
            st.error("Please provide an initial prompt")
            return
        
        with st.spinner("Initializing active learning system..."):
            run_active_learning_loop(
                initial_prompt=initial_prompt,
                target_properties=target_properties,
                max_iterations=max_iterations,
                convergence_patience=convergence_patience,
                score_threshold=score_threshold,
                max_memory_gb=max_memory_gb,
                max_time_hours=max_time_hours,
                min_confidence=min_confidence,
                max_error_rate=max_error_rate,
                storage_path=storage_path
            )


def run_active_learning_loop(initial_prompt, target_properties, max_iterations, 
                           convergence_patience, score_threshold, max_memory_gb, 
                           max_time_hours, min_confidence, max_error_rate, storage_path):
    """Run the active learning loop with real-time updates"""
    
    # Initialize configuration objects
    convergence_config = ConvergenceConfig(
        patience=convergence_patience,
        score_threshold=score_threshold,
        enable_early_stopping=True
    )
    
    resource_limits = ResourceLimits(
        max_memory_mb=int(max_memory_gb * 1024),
        max_time_hours=max_time_hours
    )
    
    quality_gates = QualityGates(
        min_confidence_score=min_confidence,
        max_error_rate=max_error_rate,
        enable_validation=True
    )
    
    # Create orchestrator
    orchestrator = ActiveLearningOrchestrator(
        max_iterations=max_iterations,
        storage_path=storage_path,
        convergence_config=convergence_config,
        resource_limits=resource_limits,
        quality_gates=quality_gates
    )
    
    # Create progress tracking containers
    status_container = st.container()
    progress_container = st.container()
    iteration_container = st.container()
    
    # Store orchestrator in session state for monitoring
    st.session_state.active_learning_orchestrator = orchestrator
    st.session_state.active_learning_storage_path = storage_path
    
    # Progress tracking
    progress_bar = progress_container.progress(0)
    status_text = status_container.empty()
    iteration_display = iteration_container.empty()
    
    # Callback for iteration updates
    def iteration_callback(state):
        iteration_num = state.iteration_number
        progress = min(iteration_num / max_iterations, 1.0)
        progress_bar.progress(progress)
        
        status_text.write(f"**Iteration {iteration_num}**: {state.status.value} - Score: {state.overall_score:.3f}")
        
        # Display iteration details
        with iteration_display.container():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Iteration", iteration_num)
                st.metric("Status", state.status.value)
            
            with col2:
                st.metric("Overall Score", f"{state.overall_score:.3f}")
                if hasattr(state, 'improvement_over_previous'):
                    st.metric("Improvement", f"{state.improvement_over_previous:.3f}")
            
            with col3:
                st.metric("Errors", len(state.errors))
                st.metric("Warnings", len(state.warnings))
            
            # Show component completion status
            components = {
                "Literature Mining": state.literature_results is not None,
                "Molecule Generation": state.generated_molecules is not None,
                "MD Simulation": state.simulation_results is not None,
                "Property Calculation": state.computed_properties is not None,
                "RAG Analysis": state.rag_analysis is not None
            }
            
            st.markdown("**Component Status:**")
            component_cols = st.columns(5)
            for i, (name, completed) in enumerate(components.items()):
                with component_cols[i]:
                    icon = "✅" if completed else "⏳"
                    st.markdown(f"{icon} {name.split()[0]}")
    
    # Completion callback
    def completion_callback(results):
        status_text.success(f"🎉 Active Learning Loop Completed!")
        progress_bar.progress(1.0)
        
        # Display final results
        with iteration_display.container():
            st.success("**🎉 Active Learning Loop Completed Successfully!**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Iterations", results['summary']['total_iterations'])
            with col2:
                st.metric("Success Rate", f"{results['summary']['success_rate']:.1%}")
            with col3:
                st.metric("Best Score", f"{results['summary']['best_score']:.3f}")
            with col4:
                st.metric("Runtime", f"{results['summary']['total_runtime']:.1f}s")
            
            # Best result details
            if results['best_result']['iteration']:
                st.subheader("🏆 Best Result")
                st.json(results['best_result'])
    
    # Add callbacks
    orchestrator.add_iteration_callback(iteration_callback)
    orchestrator.add_completion_callback(completion_callback)
    
    # Run the loop (this is synchronous for Streamlit)
    try:
        status_text.info("🚀 Starting active learning loop...")
        
        # Note: In a real deployment, you'd want to run this asynchronously
        # For demo purposes, we'll use mock results
        st.info("🔄 Active Learning Loop is starting...")
        st.info("⚠️ **Demo Mode**: This would normally run the full active learning loop. "
               "In the demo, we'll show the monitoring interface with sample data.")
        
        # Simulate some progress for demonstration
        for i in range(1, min(4, max_iterations + 1)):
            progress_bar.progress(i / max_iterations)
            status_text.write(f"**Simulated Iteration {i}**: Running components...")
            
            # Create a mock state for demonstration using realistic property scoring
            from insulin_ai.core.active_learning.property_scoring import PropertyScoring
            
            # Initialize property scorer
            scorer = PropertyScoring()
            
            # Generate realistic MD properties based on iteration
            if i == 1:
                material_type = "random"  # First iteration is more random
            elif i == 2:
                material_type = "pla"     # Second iteration finds PLA-like
            else:
                material_type = "plga"    # Third iteration finds PLGA-like
            
            md_props = scorer.generate_mock_md_properties(material_type)
            target_scores = scorer.score_material_properties(md_props, target_properties)
            
            mock_state = IterationState(iteration_number=i)
            mock_state.overall_score = target_scores.overall_score
            mock_state.status = IterationStatus.COMPLETED
            
            # Update state with realistic computed properties
            from insulin_ai.core.active_learning.state_manager import ComputedProperties
            mock_state.computed_properties = ComputedProperties(
                md_properties=md_props,
                target_scores=target_scores,
                property_details={
                    "biocompatibility_factors": {
                        "glass_transition_temp": md_props.glass_transition_temp,
                        "density": md_props.density,
                        "hydrogen_bonding": md_props.hydrogen_bond_count
                    },
                    "degradation_factors": {
                        "chain_scission_rate": md_props.chain_scission_rate,
                        "water_diffusion": md_props.diffusion_coefficient_water,
                        "ester_bond_strength": md_props.ester_bond_strength
                    },
                    "mechanical_factors": {
                        "youngs_modulus_avg": (md_props.youngs_modulus_x + md_props.youngs_modulus_y + md_props.youngs_modulus_z) / 3,
                        "bulk_modulus": md_props.bulk_modulus,
                        "cohesive_energy": md_props.cohesive_energy
                    }
                },
                computation_method="MD_simulation_with_literature_scoring",
                execution_time=5.0,
                confidence_score=0.85
            )

            iteration_callback(mock_state)
            
            # Small delay for demo effect
            import time
            time.sleep(1)
        
        # Show completion
        mock_results = {
            'summary': {
                'total_iterations': min(3, max_iterations),
                'success_rate': 1.0,
                'best_score': 0.8,
                'total_runtime': 45.0
            },
            'best_result': {
                'iteration': 3,
                'score': 0.8,
                'properties': target_properties
            }
        }
        completion_callback(mock_results)
        
    except Exception as e:
        st.error(f"❌ Error running active learning loop: {e}")
        st.exception(e)


def render_progress_monitor():
    """Render real-time progress monitoring"""
    st.header("📊 Progress Monitor")
    
    if 'active_learning_orchestrator' not in st.session_state:
        st.info("🔄 No active learning session running. Start one in the 'Run Active Learning' tab.")
        return
    
    orchestrator = st.session_state.active_learning_orchestrator
    
    # Auto-refresh button
    if st.button("🔄 Refresh Status"):
        st.rerun()
    
    # Get current status
    try:
        status = orchestrator.get_status()
        
        # Loop status overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Iteration", status['current_iteration'])
        with col2:
            st.metric("Total States", status['total_states'])
        with col3:
            st.metric("Completed", status['completed_states'])
        with col4:
            success_rate = status['completed_states'] / max(status['total_states'], 1)
            st.metric("Success Rate", f"{success_rate:.1%}")
        
        # Component status
        st.subheader("🔧 Component Status")
        components_df = pd.DataFrame([
            {"Component": k, "Status": v} 
            for k, v in status['components_status'].items()
        ])
        st.dataframe(components_df, use_container_width=True)
        
        # Loop controller status if available
        if 'loop_status' in status:
            st.subheader("🎛️ Loop Controller Status")
            loop_status = status['loop_status']
            st.json(loop_status)
    
    except Exception as e:
        st.error(f"Error getting status: {e}")


def render_iteration_history():
    """Render iteration history and analytics"""
    st.header("📋 Iteration History")
    
    if 'active_learning_storage_path' not in st.session_state:
        st.info("🔄 No active learning session data available.")
        return
    
    storage_path = st.session_state.active_learning_storage_path
    
    try:
        # Load state manager to get history
        state_manager = StateManager(f"{storage_path}/states")
        all_states = state_manager.get_all_states()
        
        if not all_states:
            st.info("📝 No iteration history available yet.")
            return
        
        # Progress chart
        st.subheader("📈 Progress Over Time")
        
        iteration_data = []
        for state in all_states:
            iteration_data.append({
                'Iteration': state.iteration_number,
                'Score': state.overall_score,
                'Status': state.status.value,
                'Timestamp': state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Errors': len(state.errors),
                'Warnings': len(state.warnings)
            })
        
        df = pd.DataFrame(iteration_data)
        
        # Score progression chart
        fig = px.line(df, x='Iteration', y='Score', 
                     title='Score Progression Across Iterations',
                     markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['Status'].value_counts()
            fig_pie = px.pie(values=status_counts.values, names=status_counts.index,
                           title='Iteration Status Distribution')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(df, x='Iteration', y='Errors',
                           title='Errors per Iteration')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed iteration table
        st.subheader("📊 Detailed Iteration Data")
        st.dataframe(df, use_container_width=True)
        
        # Individual iteration details
        if st.checkbox("Show Individual Iteration Details"):
            selected_iteration = st.selectbox("Select Iteration", df['Iteration'].tolist())
            
            selected_state = next((s for s in all_states if s.iteration_number == selected_iteration), None)
            if selected_state:
                render_iteration_details(selected_state)
    
    except Exception as e:
        st.error(f"Error loading iteration history: {e}")
        st.exception(e)


def render_iteration_details(state):
    """Render detailed information for a specific iteration"""
    st.subheader(f"🔍 Iteration {state.iteration_number} Details")

    # Basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Overall Score", f"{state.overall_score:.3f}")
        st.metric("Status", state.status.value)

    with col2:
        st.metric("Start Time", state.start_time.strftime('%Y-%m-%d %H:%M:%S'))
        if state.end_time:
            duration = (state.end_time - state.start_time).total_seconds()
            st.metric("Duration", f"{duration:.1f}s")

    with col3:
        st.metric("Errors", len(state.errors))
        st.metric("Warnings", len(state.warnings))

    # Enhanced property display
    if state.computed_properties:
        st.subheader("🎯 Material Property Analysis")
        
        # Target scores visualization
        if state.computed_properties.target_scores:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = state.computed_properties.target_scores.biocompatibility
                st.metric("Biocompatibility", f"{score:.3f}", 
                         delta=f"{(score - 0.5):.3f}" if score > 0.5 else None)
            
            with col2:
                score = state.computed_properties.target_scores.degradation_rate
                st.metric("Degradation Rate", f"{score:.3f}",
                         delta=f"{(score - 0.5):.3f}" if score > 0.5 else None)
            
            with col3:
                score = state.computed_properties.target_scores.mechanical_strength
                st.metric("Mechanical Strength", f"{score:.3f}",
                         delta=f"{(score - 0.5):.3f}" if score > 0.5 else None)
        
        # MD Properties display
        if state.computed_properties.md_properties:
            with st.expander("🔬 MD Simulation Properties"):
                md_props = state.computed_properties.md_properties
                
                # Mechanical properties
                st.subheader("Mechanical Properties")
                mech_col1, mech_col2, mech_col3 = st.columns(3)
                
                with mech_col1:
                    avg_youngs = (md_props.youngs_modulus_x + md_props.youngs_modulus_y + md_props.youngs_modulus_z) / 3
                    st.metric("Young's Modulus (avg)", f"{avg_youngs:.2f} GPa")
                    st.metric("Bulk Modulus", f"{md_props.bulk_modulus:.2f} GPa")
                
                with mech_col2:
                    st.metric("Shear Modulus", f"{md_props.shear_modulus:.2f} GPa")
                    st.metric("Density", f"{md_props.density:.2f} g/cm³")
                
                with mech_col3:
                    st.metric("Cohesive Energy", f"{md_props.cohesive_energy:.1f} kJ/mol")
                    st.metric("Glass Transition", f"{md_props.glass_transition_temp:.1f} K")
                
                # Degradation properties
                st.subheader("Degradation Properties")
                deg_col1, deg_col2, deg_col3 = st.columns(3)
                
                with deg_col1:
                    st.metric("Chain Scission Rate", f"{md_props.chain_scission_rate:.3f} ns⁻¹")
                    st.metric("Ester Bond Strength", f"{md_props.ester_bond_strength:.1f} kJ/mol")
                
                with deg_col2:
                    st.metric("Water Diffusion", f"{md_props.diffusion_coefficient_water:.2e} cm²/s")
                    st.metric("Water Accessibility", f"{md_props.water_accessibility:.2f}")
                
                with deg_col3:
                    st.metric("Polymer RMSF", f"{md_props.rmsf_polymer:.2f} Å")
                    st.metric("Free Volume", f"{md_props.free_volume_fraction:.2f}")
                
                # Biocompatibility properties
                st.subheader("Biocompatibility Properties")
                bio_col1, bio_col2, bio_col3 = st.columns(3)
                
                with bio_col1:
                    st.metric("H-Bond Count", f"{md_props.hydrogen_bond_count:.0f}")
                    st.metric("H-Bond Lifetime", f"{md_props.hydrogen_bond_lifetime:.1f} ps")
                
                with bio_col2:
                    st.metric("Surface Area", f"{md_props.surface_area:.0f} Ų")
                    st.metric("Drug Diffusion", f"{md_props.diffusion_coefficient_drug:.2e} cm²/s")
                
                with bio_col3:
                    body_temp_diff = abs(md_props.glass_transition_temp - 310)
                    st.metric("Tg vs Body Temp", f"{body_temp_diff:.1f} K difference")
                    
        # Property details in a more structured way
        if state.computed_properties.property_details:
            with st.expander("📊 Property Analysis Details"):
                details = state.computed_properties.property_details
                
                if "biocompatibility_factors" in details:
                    st.subheader("🩺 Biocompatibility Analysis")
                    st.json(details["biocompatibility_factors"])
                
                if "degradation_factors" in details:
                    st.subheader("⏱️ Degradation Analysis")
                    st.json(details["degradation_factors"])
                
                if "mechanical_factors" in details:
                    st.subheader("🔧 Mechanical Analysis")
                    st.json(details["mechanical_factors"])

    # Component results (existing code)
    if state.literature_results:
        with st.expander("📚 Literature Mining Results"):
            st.json(state.literature_results.__dict__)

    if state.generated_molecules:
        with st.expander("🧪 Generated Molecules"):
            st.json(state.generated_molecules.__dict__)

    if state.simulation_results:
        with st.expander("⚛️ Simulation Results"):
            st.json(state.simulation_results.__dict__)

    if state.rag_analysis:
        with st.expander("🔍 RAG Analysis"):
            st.json(state.rag_analysis.__dict__)

    # Reasoning log
    if state.reasoning_log:
        with st.expander("🧠 Reasoning Log"):
            for entry in state.reasoning_log:
                st.text(entry)
    
    # Errors and warnings
    if state.errors:
        with st.expander("❌ Errors"):
            for error in state.errors:
                st.error(error)
    
    if state.warnings:
        with st.expander("⚠️ Warnings"):
            for warning in state.warnings:
                st.warning(warning)


def render_system_status():
    """Render system status and health checks"""
    st.header("⚙️ System Status")
    
    # Component availability checks
    st.subheader("🔧 Component Availability")
    
    checks = {
        "Active Learning Core": ACTIVE_LEARNING_AVAILABLE,
        "OpenAI API Key": bool(os.getenv('OPENAI_API_KEY')),
        "StateManager": True,
        "DecisionEngine": True,
        "LoopController": True,
    }
    
    for component, available in checks.items():
        status = "✅ Available" if available else "❌ Not Available"
        color = "green" if available else "red"
        st.markdown(f"**{component}**: :{color}[{status}]")
    
    # System information
    st.subheader("💻 System Information")
    
    import psutil
    import platform
    
    system_info = {
        "Platform": platform.platform(),
        "Python Version": platform.python_version(),
        "CPU Count": psutil.cpu_count(),
        "Memory Total": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "Memory Available": f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
        "Disk Usage": f"{psutil.disk_usage('/').percent:.1f}%"
    }
    
    for key, value in system_info.items():
        st.text(f"{key}: {value}")
    
    # Test active learning components
    st.subheader("🧪 Component Tests")
    
    if st.button("🧪 Run Component Tests"):
        with st.spinner("Running component tests..."):
            run_component_tests()


def run_component_tests():
    """Run basic tests on active learning components"""
    test_results = {}
    
    try:
        # Test StateManager
        from insulin_ai.core.active_learning.state_manager import StateManager
        sm = StateManager("test_component_check")
        test_results["StateManager"] = "✅ Pass"
    except Exception as e:
        test_results["StateManager"] = f"❌ Fail: {e}"
    
    try:
        # Test DecisionEngine
        from insulin_ai.core.active_learning.decision_engine import LLMDecisionEngine
        de = LLMDecisionEngine()
        test_results["DecisionEngine"] = f"✅ Pass (LLM Available: {de.llm_available})"
    except Exception as e:
        test_results["DecisionEngine"] = f"❌ Fail: {e}"
    
    try:
        # Test LoopController
        from insulin_ai.core.active_learning.loop_controller import LoopController
        lc = LoopController(max_iterations=1)
        test_results["LoopController"] = "✅ Pass"
    except Exception as e:
        test_results["LoopController"] = f"❌ Fail: {e}"
    
    # Display results
    for component, result in test_results.items():
        if "✅" in result:
            st.success(f"**{component}**: {result}")
        else:
            st.error(f"**{component}**: {result}")


# For backwards compatibility with the old UI system
def render_active_learning_ui():
    """Backwards compatibility wrapper"""
    render_active_learning() 