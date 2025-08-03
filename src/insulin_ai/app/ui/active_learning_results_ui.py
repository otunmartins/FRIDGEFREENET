"""
Active Learning Results Review UI Module

This module provides comprehensive visualization and analysis of active learning results,
including progress tracking, material discovery summary, and AI-powered analysis reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ActiveLearningResults:
    """Container for active learning results"""
    iterations: List[Dict[str, Any]]
    total_iterations: int
    success_rate: float
    best_score: float
    best_material: Optional[Dict[str, Any]]
    progress_data: pd.DataFrame
    materials_discovered: List[Dict[str, Any]]
    
def render_active_learning_results():
    """Main function to render the active learning results review interface"""
    st.title("📊 Active Learning Results Review")
    st.markdown("**Comprehensive Analysis**: Review progress, discovered materials, and generate AI-powered insights")
    
    # Handle special views first (similar to simulation UI)
    # 3D Visualization view
    if 'show_results_3d_visualization' in st.session_state:
        viz_data = st.session_state.show_results_3d_visualization
        
        # Back button
        if st.button("⬅️ Back to Results Analysis", type="secondary"):
            del st.session_state.show_results_3d_visualization
            st.rerun()
        
        # Show 3D visualization
        st.markdown("### 🧬 3D Molecular Trajectory Analysis")
        st.markdown(f"**Simulation:** {viz_data['simulation_id']}")
        if 'molecule_id' in viz_data:
            st.markdown(f"**Molecule:** {viz_data['molecule_id']}")
        
        try:
            # Try to import and use the PDB visualizer from simulation UI
            from insulin_ai.app.ui.simulation_ui import PDB_VISUALIZER_AVAILABLE, render_pdb_visualizer
            
            if PDB_VISUALIZER_AVAILABLE:
                render_pdb_visualizer(viz_data['trajectory_file'], viz_data['simulation_id'])
            else:
                st.error("❌ 3D visualizer not available. Please check dependencies.")
                st.info("💡 You can manually view the trajectory file:")
                st.code(viz_data['trajectory_file'])
        except Exception as e:
            st.error(f"❌ Error rendering 3D visualization: {str(e)}")
            st.info("💡 You can manually view the trajectory file:")
            st.code(viz_data['trajectory_file'])
            with st.expander("Error Details"):
                st.exception(e)
        
        return  # Exit early when showing 3D view
    
    # Post-processing Results view
    if 'show_postprocessing_results' in st.session_state:
        pp_data = st.session_state.show_postprocessing_results
        
        # Back button
        if st.button("⬅️ Back to Results Analysis", type="secondary"):
            del st.session_state.show_postprocessing_results
            st.rerun()
        
        # Show post-processing results
        st.markdown("### 📊 Post-Processing Results Dashboard")
        st.markdown(f"**Iteration:** {pp_data['iteration']}")
        st.markdown(f"**Molecule:** {pp_data['molecule_id']}")
        
        try:
            # Import and use post-processing dashboard from simulation UI
            from insulin_ai.app.ui.simulation_ui import render_postprocessing_results_dashboard
            
            # Create a simulation ID for the dashboard
            simulation_id = f"iter_{pp_data['iteration']}_{pp_data['molecule_id']}"
            render_postprocessing_results_dashboard(simulation_id)
            
        except Exception as e:
            st.error(f"❌ Error displaying post-processing results: {str(e)}")
            
            # Fallback: show raw simulation results
            st.markdown("**Raw Simulation Results:**")
            st.json(pp_data['simulation_results'])
            
            with st.expander("Error Details"):
                st.exception(e)
        
        return  # Exit early when showing post-processing view
    
    # Results directory selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        results_path = st.text_input(
            "Results Directory Path",
            value="active_learning_results",
            help="Path to the directory containing active learning results"
        )
    
    with col2:
        load_requested = st.button("🔍 Load Results", type="primary")
    
    # Handle loading outside column context for proper full-width display
    if load_requested:
        load_and_display_results(results_path)
    
    # Auto-load if results exist (also outside column context)
    elif Path(results_path).exists() and not st.session_state.get('results_loaded'):
        with st.spinner("🔄 Auto-loading existing results..."):
            load_and_display_results(results_path)
    
    # Display existing results if already loaded
    elif st.session_state.get('results_loaded') and 'active_learning_results' in st.session_state:
        results_data = st.session_state.active_learning_results
        
        # Add separator for better visual organization
        st.markdown("---")
        
        # Display results sections in full width
        display_results_overview(results_data)
        display_progress_visualization(results_data)
        display_materials_discovery(results_data)
        display_detailed_iteration_analysis(results_data)
        generate_ai_comprehensive_report(results_data, results_path)

def load_and_display_results(results_path: str):
    """Load active learning results into session state"""
    
    try:
        results_data = load_active_learning_results(results_path)
        if results_data is None:
            st.warning("⚠️ No valid active learning results found in the specified directory.")
            st.session_state.results_loaded = False
            return
        
        st.session_state.results_loaded = True
        st.session_state.active_learning_results = results_data
        st.success(f"✅ Successfully loaded {results_data.total_iterations} iterations from {results_path}")
        
        # Trigger rerun to display the results
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        logger.error(f"Error loading active learning results: {e}")
        st.session_state.results_loaded = False
        st.exception(e)

def load_active_learning_results(results_path: str) -> Optional[ActiveLearningResults]:
    """Load active learning results from the specified directory"""
    
    results_dir = Path(results_path)
    if not results_dir.exists():
        return None
    
    # Find iteration result files
    iteration_files = list(results_dir.glob("iteration_*_results.json"))
    
    if not iteration_files:
        return None
    
    iterations = []
    materials_discovered = []
    
    # Load each iteration
    for file_path in sorted(iteration_files):
        try:
            with open(file_path, 'r') as f:
                iteration_data = json.load(f)
                iterations.append(iteration_data)
                
                # Extract discovered materials
                if 'generated_molecules' in iteration_data and iteration_data['generated_molecules'].get('success'):
                    molecules = iteration_data['generated_molecules'].get('molecules', [])
                    for mol in molecules:
                        materials_discovered.append({
                            'iteration': iteration_data.get('iteration', 0),
                            'molecule_id': mol.get('id', 'unknown'),
                            'psmiles': mol.get('psmiles', ''),
                            'smiles': mol.get('smiles', ''),
                            'confidence': mol.get('confidence', 0.0),
                            'description': mol.get('description', ''),
                            'properties': iteration_data.get('simulation_results', {}).get('properties_computed', {}),
                            'overall_score': iteration_data.get('overall_score', 0.0),
                            'simulation_success': iteration_data.get('simulation_results', {}).get('success', False)
                        })
        
        except Exception as e:
            logger.warning(f"Could not load iteration file {file_path}: {e}")
            continue
    
    if not iterations:
        return None
    
    # Create progress DataFrame
    progress_data = create_progress_dataframe(iterations)
    
    # Calculate summary statistics
    successful_iterations = len([it for it in iterations if it.get('status') == 'completed'])
    success_rate = successful_iterations / len(iterations) if iterations else 0
    
    scores = [it.get('overall_score', 0) for it in iterations]
    best_score = max(scores) if scores else 0
    
    # Find best material
    best_material = None
    if materials_discovered:
        best_material = max(materials_discovered, key=lambda x: x['overall_score'])
    
    return ActiveLearningResults(
        iterations=iterations,
        total_iterations=len(iterations),
        success_rate=success_rate,
        best_score=best_score,
        best_material=best_material,
        progress_data=progress_data,
        materials_discovered=materials_discovered
    )

def create_progress_dataframe(iterations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame for progress visualization"""
    
    progress_data = []
    
    for iteration in iterations:
        row = {
            'Iteration': iteration.get('iteration', 0),
            'Overall Score': iteration.get('overall_score', 0),
            'Status': iteration.get('status', 'unknown'),
            'Literature Success': iteration.get('literature_results', {}).get('success', False),
            'PSMILES Success': iteration.get('generated_molecules', {}).get('success', False),
            'Simulation Success': iteration.get('simulation_results', {}).get('success', False),
            'Papers Found': iteration.get('literature_results', {}).get('papers_found', 0),
            'Molecules Generated': iteration.get('generated_molecules', {}).get('num_generated', 0),
            'Successful Simulations': iteration.get('simulation_results', {}).get('successful_simulations', 0),
            'Timestamp': iteration.get('timestamp', ''),
            'Errors': len(iteration.get('errors', [])),
        }
        
        # Extract properties if available
        properties = iteration.get('simulation_results', {}).get('properties_computed', {})
        for prop_name, prop_value in properties.items():
            row[f'Property_{prop_name}'] = prop_value
            
        progress_data.append(row)
    
    return pd.DataFrame(progress_data)

def display_results_overview(results: ActiveLearningResults):
    """Display high-level overview of results with candidate structures and simulation results"""
    
    st.header("📈 Results Overview")
    
    # Find latest successful iteration
    latest_successful = None
    for iteration in reversed(results.iterations):
        if (iteration.get('status') == 'completed' and 
            iteration.get('simulation_results', {}).get('success', False)):
            latest_successful = iteration
            break
    
    if not latest_successful:
        # Fallback to any completed iteration
        for iteration in reversed(results.iterations):
            if iteration.get('status') == 'completed':
                latest_successful = iteration
                break
    
    # Key metrics - similar to PSMILES tab
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Iterations", results.total_iterations)
        
    with col2:
        st.metric("Success Rate", f"{results.success_rate*100:.1f}%")
        
    with col3:
        st.metric("Best Score", f"{results.best_score:.3f}")
        
    with col4:
        st.metric("Materials Discovered", len(results.materials_discovered))
    
    # Latest iteration results - similar to PSMILES tab structure
    if latest_successful:
        st.subheader("🎯 Latest Successful Iteration Results")
        
        iteration_num = latest_successful.get('iteration', 'Unknown')
        st.success(f"✅ Showing results from Iteration {iteration_num}")
        
        # Show generated molecules if available
        generated_molecules = latest_successful.get('generated_molecules', {})
        if generated_molecules.get('success') and generated_molecules.get('molecules'):
            molecules = generated_molecules['molecules']
            st.info(f"🧪 Generated {len(molecules)} candidate molecules")
            
            # Display candidates with expandable details
            for i, molecule in enumerate(molecules[:5], 1):  # Show first 5 candidates
                mol_id = molecule.get('id', f'molecule_{i}')
                psmiles_preview = molecule.get('psmiles', molecule.get('smiles', 'Unknown'))[:50]
                confidence = molecule.get('confidence', 0.0)
                
                with st.expander(f"⭐ Candidate {i}: {mol_id} - {psmiles_preview}{'...' if len(psmiles_preview) >= 50 else ''}", expanded=(i == 1)):
                    
                    # Candidate structure details
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Chemical Structure:**")
                        if molecule.get('psmiles'):
                            st.code(f"PSMILES: {molecule['psmiles']}")
                        if molecule.get('smiles'):
                            st.code(f"SMILES: {molecule['smiles']}")
                        
                        st.markdown(f"**Description:** {molecule.get('description', 'N/A')}")
                        st.markdown(f"**Confidence:** {confidence:.3f}")
                        
                    with col2:
                        st.markdown("**Candidate Actions:**")
                        
                        # Check for simulation results
                        sim_results = latest_successful.get('simulation_results', {})
                        if sim_results.get('success'):
                            
                            # View MD Trajectory button
                            if st.button(f"🎬 View MD Trajectory", key=f"traj_{mol_id}_{iteration_num}"):
                                # Try to find trajectory file
                                sim_files = sim_results.get('simulation_files', {})
                                trajectory_file = None
                                
                                for file_type, file_path in sim_files.items():
                                    if ('trajectory' in file_type.lower() or 
                                        'frames' in file_type.lower() or 
                                        file_path.endswith('.pdb')):
                                        trajectory_file = file_path
                                        break
                                
                                if trajectory_file and os.path.exists(trajectory_file):
                                    st.session_state.show_results_3d_visualization = {
                                        'trajectory_file': trajectory_file,
                                        'simulation_id': f"iter_{iteration_num}_{mol_id}",
                                        'molecule_id': mol_id
                                    }
                                    st.rerun()
                                else:
                                    st.error("❌ Trajectory file not found")
                            
                            # View Post-processing Results button
                            if st.button(f"📊 View Post-processing", key=f"postproc_{mol_id}_{iteration_num}"):
                                st.session_state.show_postprocessing_results = {
                                    'iteration': iteration_num,
                                    'molecule_id': mol_id,
                                    'simulation_results': sim_results
                                }
                                st.rerun()
                        else:
                            st.warning("⚠️ No simulation results available")
            
            if len(molecules) > 5:
                st.info(f"📝 Showing first 5 of {len(molecules)} candidates. View detailed analysis below for all candidates.")
        
        # Simulation status and metrics
        sim_results = latest_successful.get('simulation_results', {})
        if sim_results:
            st.subheader("⚛️ Simulation Results Summary")
            
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            
            with sim_col1:
                success_status = "✅ Success" if sim_results.get('success') else "❌ Failed"
                st.metric("Simulation Status", success_status)
            
            with sim_col2:
                total_atoms = sim_results.get('total_atoms', 'N/A')
                st.metric("Total Atoms", f"{total_atoms:,}" if isinstance(total_atoms, int) else total_atoms)
            
            with sim_col3:
                performance = sim_results.get('performance', 0)
                st.metric("Performance", f"{performance:.1f} ns/day" if performance > 0 else "N/A")
            
            # Properties computed - filter out fake abstract properties
            properties = sim_results.get('properties_computed', {})
            fake_properties = ['biocompatibility', 'stability', 'degradation_rate']
            
            # Filter out fake properties
            real_properties = {k: v for k, v in properties.items() 
                             if k.lower() not in fake_properties}
            
            if real_properties:
                st.markdown("**Real Computed Properties:**")
                prop_cols = st.columns(min(4, len(real_properties)))
                for i, (prop_name, prop_value) in enumerate(real_properties.items()):
                    if i < len(prop_cols):
                        with prop_cols[i]:
                            if isinstance(prop_value, (int, float)):
                                st.metric(prop_name.replace('_', ' ').title(), f"{prop_value:.3f}")
                            else:
                                st.metric(prop_name.replace('_', ' ').title(), str(prop_value))
            
            # Show real simulation metrics instead of abstract properties
            real_sim_properties = {}
            
            # Extract real properties from simulation results
            if sim_results.get('simulation_results'):
                for sim_result in sim_results['simulation_results']:
                    if sim_result.get('success'):
                        # Add real MD simulation properties
                        if 'final_energy' in sim_result:
                            real_sim_properties['Final Energy (kJ/mol)'] = sim_result['final_energy']
                        if 'temperature' in sim_result:
                            real_sim_properties['Temperature (K)'] = sim_result['temperature']
                        if 'pressure' in sim_result:
                            real_sim_properties['Pressure (atm)'] = sim_result['pressure']
                        if 'density' in sim_result:
                            real_sim_properties['Density (g/cm³)'] = sim_result['density']
                        if 'total_time_s' in sim_result:
                            real_sim_properties['Simulation Time (s)'] = sim_result['total_time_s']
                        
                        # Look for analysis results with real computed properties
                        analysis = sim_result.get('analysis_results', {})
                        if analysis.get('success'):
                            key_metrics = analysis.get('key_metrics', {})
                            
                            # Add real computed metrics
                            if 'energy_change' in key_metrics:
                                real_sim_properties['Energy Change (kJ/mol)'] = key_metrics['energy_change']
                            if 'rmsd_avg' in key_metrics:
                                real_sim_properties['RMSD (Å)'] = key_metrics['rmsd_avg']
                            if 'rmsf_avg' in key_metrics:
                                real_sim_properties['RMSF (Å)'] = key_metrics['rmsf_avg']
                            if 'radius_of_gyration' in key_metrics:
                                real_sim_properties['Radius of Gyration (Å)'] = key_metrics['radius_of_gyration']
                            
                            # Add thermodynamic properties if available
                            comprehensive = analysis.get('comprehensive_analysis', {})
                            if comprehensive.get('thermodynamics'):
                                thermo = comprehensive['thermodynamics']
                                if 'heat_capacity' in thermo:
                                    real_sim_properties['Heat Capacity'] = thermo['heat_capacity']
                                if 'thermal_expansion' in thermo:
                                    real_sim_properties['Thermal Expansion'] = thermo['thermal_expansion']
            
            # Display real simulation properties if we found any
            if real_sim_properties:
                st.markdown("**Real MD Simulation Properties:**")
                prop_cols = st.columns(min(4, len(real_sim_properties)))
                for i, (prop_name, prop_value) in enumerate(real_sim_properties.items()):
                    if i < len(prop_cols):
                        with prop_cols[i]:
                            if isinstance(prop_value, (int, float)):
                                if abs(prop_value) > 1000:
                                    st.metric(prop_name, f"{prop_value:.1f}")
                                else:
                                    st.metric(prop_name, f"{prop_value:.3f}")
                            else:
                                st.metric(prop_name, str(prop_value))
    
    else:
        st.warning("⚠️ No successful iterations found. Please check the detailed iteration analysis below.")
    
    # Component success summary (keeping this as it's useful)
    st.subheader("🎯 Component Success Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status pie chart
        status_counts = results.progress_data['Status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Iteration Status Distribution"
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Component success rates
        component_success = {
            'Literature Mining': results.progress_data['Literature Success'].mean(),
            'PSMILES Generation': results.progress_data['PSMILES Success'].mean(),
            'MD Simulation': results.progress_data['Simulation Success'].mean()
        }
        
        fig_components = px.bar(
            x=list(component_success.keys()),
            y=list(component_success.values()),
            title="Component Success Rates",
            labels={'x': 'Component', 'y': 'Success Rate'}
        )
        fig_components.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_components, use_container_width=True)

def display_progress_visualization(results: ActiveLearningResults):
    """Display detailed progress visualization"""
    
    st.header("📊 Progress Visualization")
    
    # Score progression
    fig_score = go.Figure()
    
    fig_score.add_trace(go.Scatter(
        x=results.progress_data['Iteration'],
        y=results.progress_data['Overall Score'],
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig_score.update_layout(
        title="Overall Score Progression",
        xaxis_title="Iteration",
        yaxis_title="Overall Score",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_score, use_container_width=True)
    
    # Properties progression (if available)
    property_columns = [col for col in results.progress_data.columns if col.startswith('Property_')]
    
    # Filter out fake/abstract properties
    real_property_columns = []
    fake_properties = ['biocompatibility', 'stability', 'degradation_rate']
    
    for prop_col in property_columns:
        prop_name = prop_col.replace('Property_', '').lower()
        if prop_name not in fake_properties:
            real_property_columns.append(prop_col)
    
    if real_property_columns:
        st.subheader("🔬 Real MD Simulation Properties Progression")
        
        fig_props = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, prop_col in enumerate(real_property_columns):
            prop_name = prop_col.replace('Property_', '')
            fig_props.add_trace(go.Scatter(
                x=results.progress_data['Iteration'],
                y=results.progress_data[prop_col],
                mode='lines+markers',
                name=prop_name.title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig_props.update_layout(
            title="Real MD Simulation Properties Over Iterations",
            xaxis_title="Iteration",
            yaxis_title="Property Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_props, use_container_width=True)
    elif property_columns:
        st.info("ℹ️ Only abstract properties found. Run actual MD simulations to see real computed properties.")
    
    # Detailed metrics table
    st.subheader("📋 Detailed Progress Table")
    
    display_columns = ['Iteration', 'Overall Score', 'Status', 'Papers Found', 
                      'Molecules Generated', 'Successful Simulations', 'Errors']
    
    # Add only real property columns if they exist
    display_columns.extend(real_property_columns)
    
    st.dataframe(
        results.progress_data[display_columns],
        use_container_width=True,
        hide_index=True
    )

def display_materials_discovery(results: ActiveLearningResults):
    """Display discovered materials analysis"""
    
    st.header("🧪 Materials Discovery Analysis")
    
    if not results.materials_discovered:
        st.info("ℹ️ No materials were successfully discovered in this run.")
        return
    
    # Best material highlight
    if results.best_material:
        st.subheader("🏆 Best Material Discovered")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Iteration**: {results.best_material['iteration']}")
            st.markdown(f"**Molecule ID**: {results.best_material['molecule_id']}")
            st.markdown(f"**Description**: {results.best_material['description']}")
            st.markdown(f"**Overall Score**: {results.best_material['overall_score']:.3f}")
            st.markdown(f"**Confidence**: {results.best_material['confidence']:.3f}")
            
            # Chemical structure
            st.markdown("**Chemical Structure**:")
            st.code(f"PSMILES: {results.best_material['psmiles']}")
            st.code(f"SMILES: {results.best_material['smiles']}")
        
        with col2:
            # Properties radar chart - filter out fake properties
            if results.best_material['properties']:
                properties = results.best_material['properties']
                fake_properties = ['biocompatibility', 'stability', 'degradation_rate']
                
                # Filter out fake properties
                real_properties = {k: v for k, v in properties.items() 
                                 if k.lower() not in fake_properties}
                
                if real_properties:
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(real_properties.values()),
                        theta=list(real_properties.keys()),
                        fill='toself',
                        name='Best Material'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, max(real_properties.values()) * 1.1] if real_properties else [0, 1]
                            )
                        ),
                        title="Real Material Properties Profile",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("ℹ️ No real computed properties available for radar chart")
            else:
                st.info("ℹ️ No properties data available")
    
    # All materials summary
    st.subheader("📋 All Discovered Materials")
    
    materials_df = pd.DataFrame(results.materials_discovered)
    
    if not materials_df.empty:
        # Filter out unsuccessful simulations option
        show_only_successful = st.checkbox("Show only materials with successful simulations", value=False)
        
        if show_only_successful:
            materials_df = materials_df[materials_df['simulation_success'] == True]
        
        if materials_df.empty:
            st.info("ℹ️ No materials with successful simulations found.")
        else:
            # Display columns selection
            display_cols = ['iteration', 'molecule_id', 'overall_score', 'confidence', 
                          'simulation_success', 'description']
            
            st.dataframe(
                materials_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Score distribution
            fig_scores = px.histogram(
                materials_df,
                x='overall_score',
                title="Distribution of Material Scores",
                nbins=10
            )
            st.plotly_chart(fig_scores, use_container_width=True)

def display_detailed_iteration_analysis(results: ActiveLearningResults):
    """Display detailed analysis of specific iterations"""
    
    st.header("🔍 Detailed Iteration Analysis")
    
    # Iteration selector
    iteration_options = [f"Iteration {it['iteration']}" for it in results.iterations]
    selected_iteration = st.selectbox("Select Iteration to Analyze", iteration_options)
    
    if selected_iteration:
        iteration_num = int(selected_iteration.split()[-1])
        iteration_data = next((it for it in results.iterations if it['iteration'] == iteration_num), None)
        
        if iteration_data:
            # Iteration overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Status", iteration_data.get('status', 'Unknown'))
                st.metric("Overall Score", f"{iteration_data.get('overall_score', 0):.3f}")
            
            with col2:
                lit_success = iteration_data.get('literature_results', {}).get('success', False)
                st.metric("Literature Mining", "✅ Success" if lit_success else "❌ Failed")
                
                psmiles_success = iteration_data.get('generated_molecules', {}).get('success', False)
                st.metric("PSMILES Generation", "✅ Success" if psmiles_success else "❌ Failed")
            
            with col3:
                sim_success = iteration_data.get('simulation_results', {}).get('success', False)
                st.metric("MD Simulation", "✅ Success" if sim_success else "❌ Failed")
                
                errors = len(iteration_data.get('errors', []))
                st.metric("Errors", errors)
            
            # Detailed sections
            with st.expander("📚 Literature Mining Results"):
                lit_results = iteration_data.get('literature_results', {})
                if lit_results:
                    st.json(lit_results)
                else:
                    st.info("No literature results available")
            
            with st.expander("🧪 Generated Molecules"):
                mol_results = iteration_data.get('generated_molecules', {})
                if mol_results:
                    st.json(mol_results)
                else:
                    st.info("No molecule generation results available")
            
            with st.expander("⚛️ Simulation Results"):
                sim_results = iteration_data.get('simulation_results', {})
                if sim_results:
                    st.json(sim_results)
                else:
                    st.info("No simulation results available")

def generate_ai_comprehensive_report(results: ActiveLearningResults, results_path: str):
    """Generate AI-powered comprehensive analysis report"""
    
    st.header("🤖 AI-Powered Comprehensive Report")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ OpenAI API key required for report generation. Please set OPENAI_API_KEY environment variable.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Generate a comprehensive analysis report using OpenAI to interpret the active learning results.")
    
    with col2:
        if st.button("🤖 Generate AI Report", type="primary"):
            generate_ai_report(results, results_path)
    
    # Display existing report if available
    report_file = Path(results_path) / "ai_comprehensive_report.md"
    if report_file.exists():
        st.subheader("📄 Latest AI Analysis Report")
        
        try:
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            st.markdown(report_content)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"active_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_report_standalone"
            )
            
        except Exception as e:
            st.error(f"Error loading existing report: {e}")

def generate_ai_report(results: ActiveLearningResults, results_path: str):
    """Generate comprehensive AI analysis report"""
    
    with st.spinner("🤖 Generating comprehensive AI analysis report..."):
        try:
            # Prepare data summary for AI analysis
            analysis_prompt = create_analysis_prompt(results)
            
            # Generate report using OpenAI
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scientific AI assistant specializing in materials discovery and active learning. Generate a comprehensive analysis report of active learning results for insulin delivery material discovery. Be scientific, detailed, and provide actionable insights."
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            report_content = response.choices[0].message.content
            
            # Save report
            report_file = Path(results_path) / "ai_comprehensive_report.md"
            with open(report_file, 'w') as f:
                f.write(f"# Active Learning Results - Comprehensive AI Analysis\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report_content)
            
            st.success("✅ AI report generated successfully!")
            
            # Display the new report immediately instead of using st.rerun()
            st.subheader("📄 Newly Generated AI Analysis Report")
            st.markdown(report_content)
            
            # Download button for the new report
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"active_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_report_new_standalone"
            )
            
        except Exception as e:
            st.error(f"❌ Error generating AI report: {e}")
            logger.error(f"Error generating AI report: {e}")

def create_analysis_prompt(results: ActiveLearningResults) -> str:
    """Create analysis prompt for AI report generation"""
    
    # Summarize key data
    best_material_info = ""
    if results.best_material:
        best_material_info = f"""
Best Material Discovered:
- Iteration: {results.best_material['iteration']}
- Overall Score: {results.best_material['overall_score']:.3f}
- PSMILES: {results.best_material['psmiles']}
- SMILES: {results.best_material['smiles']}
- Description: {results.best_material['description']}
- Properties: {results.best_material['properties']}
- Simulation Success: {results.best_material['simulation_success']}
"""
    
    # Extract iteration summaries
    iteration_summaries = []
    for iteration in results.iterations:
        summary = f"""
Iteration {iteration.get('iteration', 'Unknown')}:
- Status: {iteration.get('status', 'Unknown')}
- Overall Score: {iteration.get('overall_score', 0):.3f}
- Prompt: {iteration.get('prompt', 'Not available')[:200]}...
- Literature Success: {iteration.get('literature_results', {}).get('success', False)}
- PSMILES Success: {iteration.get('generated_molecules', {}).get('success', False)}
- Simulation Success: {iteration.get('simulation_results', {}).get('success', False)}
- Errors: {len(iteration.get('errors', []))}
"""
        iteration_summaries.append(summary)
    
    progress_summary = f"""
Overall Progress:
- Total Iterations: {results.total_iterations}
- Success Rate: {results.success_rate*100:.1f}%
- Best Score Achieved: {results.best_score:.3f}
- Total Materials Discovered: {len(results.materials_discovered)}
"""
    
    prompt = f"""
Please analyze the following active learning results for insulin delivery material discovery and provide a comprehensive scientific report.

{progress_summary}

{best_material_info}

Iteration Details:
{chr(10).join(iteration_summaries)}

Please provide a comprehensive analysis covering:

1. **Executive Summary**: Overall performance and key achievements
2. **Scientific Analysis**: 
   - Material discovery effectiveness
   - Chemical structure insights
   - Property optimization trends
3. **Technical Performance**:
   - Component success rates (literature mining, PSMILES generation, simulation)
   - Convergence patterns
   - Error analysis
4. **Material Properties Analysis**:
   - Best materials discovered
   - Property-structure relationships
   - Biocompatibility and stability insights
5. **Recommendations**:
   - Optimization strategies
   - Future research directions
   - Parameter tuning suggestions
6. **Conclusions**: Key findings and scientific value

Format the report in clear markdown with appropriate headers, bullet points, and scientific terminology. Include specific numbers and data points from the results.
"""
    
    return prompt

# Helper functions for backward compatibility
def render_results_overview():
    """Render just the overview section"""
    if 'active_learning_results' in st.session_state:
        display_results_overview(st.session_state.active_learning_results)
    else:
        st.info("No active learning results loaded.")

def render_materials_analysis():
    """Render just the materials analysis section"""
    if 'active_learning_results' in st.session_state:
        display_materials_discovery(st.session_state.active_learning_results)
    else:
        st.info("No active learning results loaded.") 

def render_active_learning_results_tab():
    """Render active learning results within a tab (not as standalone page)"""
    
    # Handle special views first (similar to simulation UI)
    # 3D Visualization view
    if 'show_results_3d_visualization' in st.session_state:
        viz_data = st.session_state.show_results_3d_visualization
        
        # Back button
        if st.button("⬅️ Back to Results Analysis", type="secondary", key="back_from_3d_viz_tab"):
            del st.session_state.show_results_3d_visualization
            st.rerun()
        
        # Show 3D visualization
        st.markdown("### 🧬 3D Molecular Trajectory Analysis")
        st.markdown(f"**Simulation:** {viz_data['simulation_id']}")
        if 'molecule_id' in viz_data:
            st.markdown(f"**Molecule:** {viz_data['molecule_id']}")
        
        try:
            # Try to import and use the PDB visualizer from simulation UI
            from insulin_ai.app.ui.simulation_ui import PDB_VISUALIZER_AVAILABLE, render_pdb_visualizer
            
            if PDB_VISUALIZER_AVAILABLE:
                render_pdb_visualizer(viz_data['trajectory_file'], viz_data['simulation_id'])
            else:
                st.error("❌ 3D visualizer not available. Please check dependencies.")
                st.info("💡 You can manually view the trajectory file:")
                st.code(viz_data['trajectory_file'])
        except Exception as e:
            st.error(f"❌ Error rendering 3D visualization: {str(e)}")
            st.info("💡 You can manually view the trajectory file:")
            st.code(viz_data['trajectory_file'])
            with st.expander("Error Details"):
                st.exception(e)
        
        return  # Exit early when showing 3D view
    
    # Post-processing Results view
    if 'show_postprocessing_results' in st.session_state:
        pp_data = st.session_state.show_postprocessing_results
        
        # Back button
        if st.button("⬅️ Back to Results Analysis", type="secondary", key="back_from_postproc_tab"):
            del st.session_state.show_postprocessing_results
            st.rerun()
        
        # Show post-processing results
        st.markdown("### 📊 Post-Processing Results Dashboard")
        st.markdown(f"**Iteration:** {pp_data['iteration']}")
        st.markdown(f"**Molecule:** {pp_data['molecule_id']}")
        
        try:
            # Import and use post-processing dashboard from simulation UI
            from insulin_ai.app.ui.simulation_ui import render_postprocessing_results_dashboard
            
            # Create a simulation ID for the dashboard
            simulation_id = f"iter_{pp_data['iteration']}_{pp_data['molecule_id']}"
            render_postprocessing_results_dashboard(simulation_id)
            
        except Exception as e:
            st.error(f"❌ Error displaying post-processing results: {str(e)}")
            
            # Fallback: show raw simulation results
            st.markdown("**Raw Simulation Results:**")
            st.json(pp_data['simulation_results'])
            
            with st.expander("Error Details"):
                st.exception(e)
        
        return  # Exit early when showing post-processing view
    
    # Results directory selection
    st.header("📊 Active Learning Results")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        results_path = st.text_input(
            "Results Directory Path",
            value="active_learning_results",
            help="Path to the directory containing active learning results",
            key="results_path_tab"
        )
    
    with col2:
        load_requested = st.button("🔍 Load Results", type="primary", key="load_results_tab")
    
    # Handle loading outside column context for proper full-width display
    if load_requested:
        load_and_display_results_tab(results_path)
    
    # Auto-load if results exist (also outside column context)
    elif Path(results_path).exists() and not st.session_state.get('results_loaded_tab'):
        with st.spinner("🔄 Auto-loading existing results..."):
            load_and_display_results_tab(results_path)
    
    # Display existing results if already loaded
    elif st.session_state.get('results_loaded_tab') and 'active_learning_results_tab' in st.session_state:
        results_data = st.session_state.active_learning_results_tab
        
        # Add separator for better visual organization
        st.markdown("---")
        
        # Display simplified results sections for tab version
        display_results_quick_overview(results_data)
        display_materials_with_svg(results_data)
        generate_enhanced_ai_report(results_data, results_path)


def load_and_display_results_tab(results_path: str):
    """Load active learning results into session state for tab version"""
    
    try:
        results_data = load_active_learning_results(results_path)
        if results_data is None:
            st.warning("⚠️ No valid active learning results found in the specified directory.")
            st.session_state.results_loaded_tab = False
            return
        
        st.session_state.results_loaded_tab = True
        st.session_state.active_learning_results_tab = results_data
        st.success(f"✅ Successfully loaded {results_data.total_iterations} iterations from {results_path}")
        
        # Trigger rerun to display the results
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error loading results: {e}")
        logger.error(f"Error loading active learning results: {e}")
        st.session_state.results_loaded_tab = False
        st.exception(e)


def display_results_quick_overview(results: ActiveLearningResults):
    """Display quick overview of properties computed from MD simulations"""
    
    st.header("🏆 Best Materials Discovered")
    
    # Find latest successful iteration
    latest_successful = None
    for iteration in reversed(results.iterations):
        if (iteration.get('status') == 'completed' and 
            iteration.get('simulation_results', {}).get('success', False)):
            latest_successful = iteration
            break
    
    if not latest_successful:
        # Fallback to any completed iteration
        for iteration in reversed(results.iterations):
            if iteration.get('status') == 'completed':
                latest_successful = iteration
                break
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Iterations", results.total_iterations)
        
    with col2:
        st.metric("Success Rate", f"{results.success_rate*100:.1f}%")
        
    with col3:
        st.metric("Best Score", f"{results.best_score:.3f}")
        
    with col4:
        st.metric("Materials Discovered", len(results.materials_discovered))
    
    # Display candidate structures with MD properties and action buttons
    if latest_successful:
        st.subheader("🧪 Latest Discovered Materials")
        
        iteration_num = latest_successful.get('iteration', 'Unknown')
        st.success(f"✅ Showing results from Iteration {iteration_num}")
        
        # Show generated molecules if available
        generated_molecules = latest_successful.get('generated_molecules', {})
        if generated_molecules.get('success') and generated_molecules.get('molecules'):
            molecules = generated_molecules['molecules']
            
            # Display candidates with action buttons
            for i, molecule in enumerate(molecules[:3], 1):  # Show first 3 candidates
                mol_id = molecule.get('id', f'molecule_{i}')
                psmiles = molecule.get('psmiles', molecule.get('smiles', 'Unknown'))
                confidence = molecule.get('confidence', 0.0)
                
                with st.expander(f"⭐ Candidate {i}: {mol_id}", expanded=(i == 1)):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("**Chemical Structure:**")
                        if molecule.get('psmiles'):
                            st.code(f"PSMILES: {molecule['psmiles']}")
                        if molecule.get('smiles'):
                            st.code(f"SMILES: {molecule['smiles']}")
                        
                        # Display MD computed properties
                        sim_results = latest_successful.get('simulation_results', {})
                        if sim_results.get('success'):
                            st.markdown("**MD Computed Properties:**")
                            display_md_properties_quick(sim_results)
                        
                        st.markdown(f"**Description:** {molecule.get('description', 'N/A')}")
                        st.markdown(f"**Confidence:** {confidence:.3f}")
                        
                    with col2:
                        st.markdown("**Actions:**")
                        
                        # Check for simulation results
                        if sim_results.get('success'):
                            
                            # View MD Trajectory button
                            if st.button(f"🎬 View MD Trajectory", key=f"traj_tab_{mol_id}_{iteration_num}"):
                                # Try to find trajectory file
                                sim_files = sim_results.get('simulation_files', {})
                                trajectory_file = None
                                
                                for file_type, file_path in sim_files.items():
                                    if ('trajectory' in file_type.lower() or 
                                        'frames' in file_type.lower() or 
                                        file_path.endswith('.pdb')):
                                        trajectory_file = file_path
                                        break
                                
                                if trajectory_file and os.path.exists(trajectory_file):
                                    st.session_state.show_results_3d_visualization = {
                                        'trajectory_file': trajectory_file,
                                        'simulation_id': f"iter_{iteration_num}_{mol_id}",
                                        'molecule_id': mol_id
                                    }
                                    st.rerun()
                                else:
                                    st.error("❌ Trajectory file not found")
                            
                            # View Post-processing Results button
                            if st.button(f"📊 View Post-processing", key=f"postproc_tab_{mol_id}_{iteration_num}"):
                                st.session_state.show_postprocessing_results = {
                                    'iteration': iteration_num,
                                    'molecule_id': mol_id,
                                    'simulation_results': sim_results
                                }
                                st.rerun()
                        else:
                            st.warning("⚠️ No simulation results available")
    else:
        st.warning("⚠️ No successful iterations found.")


def display_md_properties_quick(sim_results: Dict[str, Any]):
    """Display quick overview of MD properties"""
    
    # Extract real simulation metrics
    real_properties = {}
    
    if sim_results.get('simulation_results'):
        for sim_result in sim_results['simulation_results']:
            if sim_result.get('success'):
                # Add real MD simulation properties
                if 'final_energy' in sim_result:
                    real_properties['Energy (kJ/mol)'] = sim_result['final_energy']
                if 'temperature' in sim_result:
                    real_properties['Temperature (K)'] = sim_result['temperature']
                if 'density' in sim_result:
                    real_properties['Density (g/cm³)'] = sim_result['density']
                
                # Look for analysis results
                analysis = sim_result.get('analysis_results', {})
                if analysis.get('success'):
                    key_metrics = analysis.get('key_metrics', {})
                    
                    if 'rmsd_avg' in key_metrics:
                        real_properties['RMSD (Å)'] = key_metrics['rmsd_avg']
                    if 'radius_of_gyration' in key_metrics:
                        real_properties['Rg (Å)'] = key_metrics['radius_of_gyration']
    
    # Display properties in a compact way
    if real_properties:
        cols = st.columns(min(3, len(real_properties)))
        for i, (prop_name, prop_value) in enumerate(real_properties.items()):
            if i < len(cols):
                with cols[i]:
                    if isinstance(prop_value, (int, float)):
                        st.metric(prop_name, f"{prop_value:.3f}")
                    else:
                        st.metric(prop_name, str(prop_value))


def display_materials_with_svg(results: ActiveLearningResults):
    """Display discovered materials with SVG structures"""
    
    st.header("🧪 All Discovered Materials")
    
    if not results.materials_discovered:
        st.info("ℹ️ No materials were successfully discovered in this run.")
        return
    
    # Group materials by iteration
    materials_by_iteration = {}
    for material in results.materials_discovered:
        iteration = material['iteration']
        if iteration not in materials_by_iteration:
            materials_by_iteration[iteration] = []
        materials_by_iteration[iteration].append(material)
    
    # Display materials by iteration
    for iteration in sorted(materials_by_iteration.keys()):
        materials = materials_by_iteration[iteration]
        st.subheader(f"📑 Iteration {iteration} ({len(materials)} materials)")
        
        for i, material in enumerate(materials):
            with st.expander(f"Material {i+1}: {material['molecule_id']}", expanded=False):
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**Chemical Structure:**")
                    psmiles = material['psmiles']
                    smiles = material['smiles']
                    
                    if psmiles:
                        st.code(f"PSMILES: {psmiles}")
                    if smiles:
                        st.code(f"SMILES: {smiles}")
                    
                    # Try to generate SVG for the structure
                    try:
                        svg_html = generate_molecule_svg(psmiles if psmiles else smiles)
                        if svg_html:
                            st.markdown("**Structure Visualization:**")
                            st.components.v1.html(svg_html, height=300)
                    except Exception as e:
                        st.info("SVG visualization not available")
                
                with col2:
                    st.markdown("**Properties:**")
                    st.markdown(f"**Score:** {material['overall_score']:.3f}")
                    st.markdown(f"**Confidence:** {material['confidence']:.3f}")
                    st.markdown(f"**Description:** {material['description']}")
                    st.markdown(f"**Simulation Success:** {'✅' if material['simulation_success'] else '❌'}")
                    
                    # Display real properties if available
                    if material['properties']:
                        fake_properties = ['biocompatibility', 'stability', 'degradation_rate']
                        real_props = {k: v for k, v in material['properties'].items() 
                                    if k.lower() not in fake_properties}
                        
                        if real_props:
                            st.markdown("**MD Properties:**")
                            for prop_name, prop_value in real_props.items():
                                if isinstance(prop_value, (int, float)):
                                    st.text(f"{prop_name}: {prop_value:.3f}")
                                else:
                                    st.text(f"{prop_name}: {prop_value}")


def generate_molecule_svg(molecule_string: str) -> Optional[str]:
    """Generate SVG representation of molecule from PSMILES or SMILES"""
    
    try:
        # Try to use RDKit for SVG generation
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        
        # Parse the molecule
        mol = None
        
        # Try parsing as SMILES first (more likely to work)
        if '[*]' not in molecule_string:  # Regular SMILES
            mol = Chem.MolFromSmiles(molecule_string)
        else:  # PSMILES - convert to SMILES approximation
            # Simple conversion: remove [*] attachment points
            clean_smiles = molecule_string.replace('[*]', '')
            if clean_smiles:
                mol = Chem.MolFromSmiles(clean_smiles)
        
        if mol is not None:
            # Generate SVG
            drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            
            # Return HTML to embed SVG
            return f'<div style="text-align: center;">{svg}</div>'
            
    except ImportError:
        # RDKit not available
        pass
    except Exception as e:
        # Parsing failed
        pass
    
    return None


def generate_enhanced_ai_report(results: ActiveLearningResults, results_path: str):
    """Generate enhanced AI-powered comprehensive analysis report with better organization"""
    
    st.header("🤖 AI Comprehensive Analysis Report")
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("❌ OpenAI API key required for report generation. Please set OPENAI_API_KEY environment variable.")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("Generate a comprehensive analysis report covering all discovered polymers with detailed comparisons.")
    
    with col2:
        if st.button("🤖 Generate Enhanced Report", type="primary", key="generate_ai_report_tab"):
            generate_enhanced_ai_report_content(results, results_path)
    
    # Display existing report if available
    report_file = Path(results_path) / "ai_comprehensive_report.md"
    if report_file.exists():
        st.subheader("📄 Latest AI Analysis Report")
        
        try:
            with open(report_file, 'r') as f:
                report_content = f.read()
            
            # Display report content in a proper container
            st.markdown(report_content)
            
            # Download button with unique key
            st.download_button(
                label="📥 Download Report",
                data=report_content,
                file_name=f"active_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_ai_report_tab"
            )
            
        except Exception as e:
            st.error(f"Error loading existing report: {e}")


def generate_enhanced_ai_report_content(results: ActiveLearningResults, results_path: str):
    """Generate enhanced comprehensive AI analysis report with polymer-focused structure"""
    
    with st.spinner("🤖 Generating enhanced AI analysis report..."):
        try:
            # Prepare enhanced data summary for AI analysis
            analysis_prompt = create_enhanced_analysis_prompt(results)
            
            # Generate report using OpenAI
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a scientific AI assistant specializing in materials discovery and polymer chemistry for insulin delivery systems. 

Generate a comprehensive analysis report that:
1. Discusses each polymer discovered in detail
2. Compares polymers and explains why one is better than another
3. Organizes content by polymer with clear sections
4. Includes chemical structure analysis
5. Provides scientific insights and recommendations

Be scientific, detailed, and provide actionable insights. Focus on the specific polymers discovered and their relative merits."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            report_content = response.choices[0].message.content
            
            # Save report
            report_file = Path(results_path) / "ai_comprehensive_report.md"
            with open(report_file, 'w') as f:
                f.write(f"# Active Learning Results - Enhanced AI Analysis\n\n")
                f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report_content)
            
            st.success("✅ Enhanced AI report generated successfully!")
            
            # Display the new report in a more organized way
            st.subheader("📄 Newly Generated Enhanced Analysis Report")
            
            # Create a container for better display
            with st.container():
                st.markdown(report_content)
            
            # Download button for the new report with unique key
            st.download_button(
                label="📥 Download Enhanced Report",
                data=report_content,
                file_name=f"enhanced_active_learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key="download_enhanced_report_tab"
            )
            
        except Exception as e:
            st.error(f"❌ Error generating enhanced AI report: {e}")
            logger.error(f"Error generating enhanced AI report: {e}")


def create_enhanced_analysis_prompt(results: ActiveLearningResults) -> str:
    """Create enhanced analysis prompt for AI report generation focused on polymers"""
    
    # Group materials by polymer type/structure
    polymer_analysis = {}
    
    for material in results.materials_discovered:
        psmiles = material['psmiles']
        iteration = material['iteration']
        
        # Create a key for grouping similar polymers
        polymer_key = f"Iteration_{iteration}_{material['molecule_id']}"
        
        polymer_analysis[polymer_key] = {
            'iteration': iteration,
            'molecule_id': material['molecule_id'],
            'psmiles': psmiles,
            'smiles': material['smiles'],
            'description': material['description'],
            'overall_score': material['overall_score'],
            'confidence': material['confidence'],
            'simulation_success': material['simulation_success'],
            'properties': material['properties']
        }
    
    # Create polymer comparison section
    polymer_summaries = []
    for polymer_key, polymer_data in polymer_analysis.items():
        summary = f"""
### Polymer: {polymer_data['molecule_id']} (Iteration {polymer_data['iteration']})

**Chemical Structure:**
- PSMILES: {polymer_data['psmiles']}
- SMILES: {polymer_data['smiles']}

**Performance Metrics:**
- Overall Score: {polymer_data['overall_score']:.3f}
- Confidence: {polymer_data['confidence']:.3f}
- Simulation Success: {polymer_data['simulation_success']}

**Description:** {polymer_data['description']}

**Properties:** {polymer_data['properties']}

**Generation Method:** {polymer_data.get('generation_method', 'Active learning iteration')}
"""
        polymer_summaries.append(summary)
    
    # Overall progress summary
    progress_summary = f"""
## Overall Active Learning Progress

- **Total Iterations:** {results.total_iterations}
- **Success Rate:** {results.success_rate*100:.1f}%
- **Best Score Achieved:** {results.best_score:.3f}
- **Total Polymers Discovered:** {len(results.materials_discovered)}
- **Successful Simulations:** {len([m for m in results.materials_discovered if m['simulation_success']])}
"""
    
    # Best material highlight
    best_material_info = ""
    if results.best_material:
        best_material_info = f"""
## Best Performing Polymer

**Winner:** {results.best_material['molecule_id']} from Iteration {results.best_material['iteration']}

**Why this polymer is the best:**
- Highest Overall Score: {results.best_material['overall_score']:.3f}
- Chemical Structure: {results.best_material['psmiles']}
- Description: {results.best_material['description']}
- Simulation Success: {results.best_material['simulation_success']}
- Key Properties: {results.best_material['properties']}
"""
    
    prompt = f"""
Please analyze the following active learning results for insulin delivery polymer discovery and provide a comprehensive scientific report.

{progress_summary}

{best_material_info}

## Discovered Polymers Analysis

{chr(10).join(polymer_summaries)}

## Analysis Instructions

Please provide a comprehensive analysis with the following structure:

### 1. Executive Summary
- Overall performance of the active learning system
- Key achievements and best polymer discovered
- Success metrics and convergence analysis

### 2. Polymer Discovery Analysis
For each polymer discovered, provide:
- Chemical structure analysis (what makes it unique)
- Performance metrics and simulation results
- Potential for insulin delivery applications
- Advantages and limitations

### 3. Comparative Analysis
- Rank all polymers by performance
- Explain why the best polymer outperforms others
- Identify structure-property relationships
- Compare chemical features (functional groups, backbone, side chains)

### 4. Scientific Insights
- What chemical features correlate with better performance?
- How did the active learning process evolve over iterations?
- What patterns emerged in successful vs unsuccessful designs?

### 5. Recommendations for Future Work
- Suggested modifications to improve top polymers
- New polymer designs to explore
- Active learning parameter optimization
- Experimental validation priorities

### 6. Technical Performance Assessment
- Simulation quality and reliability
- Convergence behavior of the active learning loop
- Computational efficiency and bottlenecks

Format the report in clear markdown with appropriate headers, bullet points, and scientific terminology. Include specific chemical structures (PSMILES/SMILES) and numerical data from the results.

Focus particularly on:
- Why polymer X is better than polymer Y (with specific reasons)
- Chemical rationale for performance differences
- Structure-activity relationships discovered
- Actionable insights for material optimization
"""
    
    return prompt 