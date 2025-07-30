"""
Material Evaluation UI Module for Insulin-AI App

This module provides the user interface for material evaluation, performance analysis,
composite scoring, property analysis, and material ranking/selection.

Author: AI-Driven Material Discovery Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Optional, Any

def render_material_evaluation_ui():
    """
    Render the complete Material Evaluation UI interface
    
    Returns:
        None
    """
    st.subheader("📊 Material Evaluation & Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Composite Scoring", "Property Analysis", "Ranking & Selection"])
    
    with tab1:
        render_composite_scoring_tab()
    
    with tab2:
        render_property_analysis_tab()
    
    with tab3:
        render_ranking_selection_tab()


def render_composite_scoring_tab():
    """Render the composite scoring tab"""
    st.markdown("### Insulin Delivery Composite Scoring")
    
    # Scoring weights
    st.markdown("#### 🎚️ Scoring Weight Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        thermal_weight = st.slider("Thermal Stability Weight", 0.0, 1.0, 0.4)
    with col2:
        biocompat_weight = st.slider("Biocompatibility Weight", 0.0, 1.0, 0.3)
    with col3:
        release_weight = st.slider("Release Control Weight", 0.0, 1.0, 0.3)
    
    # Normalize weights
    total_weight = thermal_weight + biocompat_weight + release_weight
    if total_weight > 0:
        thermal_weight /= total_weight
        biocompat_weight /= total_weight
        release_weight /= total_weight
    
    # Calculate composite scores for all materials
    if len(st.session_state.get('material_library', pd.DataFrame())) > 0:
        calculate_and_display_composite_scores(thermal_weight, biocompat_weight, release_weight)
    else:
        st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")


def calculate_and_display_composite_scores(thermal_weight: float, biocompat_weight: float, release_weight: float):
    """Calculate and display composite scores for materials"""
    
    material_library = st.session_state.material_library
    
    # Calculate composite scores
    material_library['composite_score'] = (
        thermal_weight * material_library['thermal_stability'] +
        biocompat_weight * material_library['biocompatibility'] +
        release_weight * material_library['release_control']
    )
    
    # Display top performers
    top_materials = material_library.nlargest(10, 'composite_score')
    
    st.markdown("#### 🏆 Top 10 Materials by Composite Score")
    
    # Enhanced display with color coding
    for idx, (_, material) in enumerate(top_materials.iterrows()):
        score = material['composite_score']
        color = '#28a745' if score > 0.8 else '#ffc107' if score > 0.6 else '#dc3545'
        
        st.markdown(f"""
        <div style="background: {color}15; border-left: 4px solid {color}; padding: 1rem; margin: 0.5rem 0; border-radius: 5px;">
            <strong>#{idx+1} Material ID: {material['material_id']}</strong><br>
            <code>PSMILES: {escape_psmiles_for_markdown(material['psmiles'])}</code><br>
            <strong>Composite Score: {score:.3f}</strong><br>
            Thermal: {material['thermal_stability']:.2f} | 
            Biocompat: {material['biocompatibility']:.2f} | 
            Release: {material['release_control']:.2f}<br>
            <small>Source: {material['source']} | Insulin Stability: {material['insulin_stability_score']:.2f}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Scoring distribution
    fig_score_dist = px.histogram(
        material_library,
        x='composite_score',
        color='source',
        title="Distribution of Composite Scores by Source",
        nbins=25
    )
    st.plotly_chart(fig_score_dist, use_container_width=True)


def render_property_analysis_tab():
    """Render the multi-property analysis tab"""
    st.markdown("### Multi-Property Analysis")
    
    if len(st.session_state.get('material_library', pd.DataFrame())) > 0:
        material_library = st.session_state.material_library
        
        # Property correlation matrix
        props_for_correlation = ['thermal_stability', 'biocompatibility', 'release_control', 
                               'uncertainty_score', 'insulin_stability_score']
        
        if 'composite_score' in material_library.columns:
            props_for_correlation.append('composite_score')
        
        # Display correlation matrix
        render_correlation_matrix(material_library, props_for_correlation)
        
        # Interactive property space exploration
        render_property_space_exploration(material_library, props_for_correlation)
        
        # Statistical analysis
        render_statistical_analysis(material_library, props_for_correlation)
        
    else:
        st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")


def render_correlation_matrix(material_library: pd.DataFrame, props_for_correlation: List[str]):
    """Render the property correlation matrix"""
    
    correlation_matrix = material_library[props_for_correlation].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Property Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def render_property_space_exploration(material_library: pd.DataFrame, props_for_correlation: List[str]):
    """Render interactive property space exploration"""
    
    st.markdown("#### 🔍 Interactive Property Space")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_prop = st.selectbox("X-axis Property", props_for_correlation, index=0)
        y_prop = st.selectbox("Y-axis Property", props_for_correlation, index=1)
    
    with col2:
        color_prop = st.selectbox("Color Property", props_for_correlation, index=-1)
        size_prop = st.selectbox("Size Property", props_for_correlation, index=3)
    
    fig_properties = px.scatter(
        material_library,
        x=x_prop,
        y=y_prop,
        color=color_prop,
        size=size_prop,
        hover_data=['material_id', 'psmiles', 'source'],
        title=f"{y_prop.title()} vs {x_prop.title()}",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_properties, use_container_width=True)


def render_statistical_analysis(material_library: pd.DataFrame, props_for_correlation: List[str]):
    """Render statistical summary and source comparison"""
    
    st.markdown("#### 📈 Statistical Summary")
    
    stats_summary = material_library[props_for_correlation].describe()
    st.dataframe(stats_summary.round(3))
    
    # Source comparison
    source_comparison = material_library.groupby('source')[props_for_correlation].mean()
    
    fig_source_comp = px.bar(
        source_comparison.reset_index(),
        x='source',
        y=props_for_correlation[-1],  # Use the last property (composite_score if available)
        title=f"Average {props_for_correlation[-1].title()} by Material Source",
        color='source'
    )
    st.plotly_chart(fig_source_comp, use_container_width=True)


def render_ranking_selection_tab():
    """Render the material ranking and selection tab"""
    st.markdown("### Material Ranking & Selection")
    
    if len(st.session_state.get('material_library', pd.DataFrame())) > 0:
        material_library = st.session_state.material_library
        
        # Advanced filtering interface
        filtered_materials = render_advanced_filtering(material_library)
        
        if len(filtered_materials) > 0:
            # Material selection interface
            render_material_selection_interface(filtered_materials)
        else:
            st.warning("No materials meet the current filter criteria. Please adjust the filters.")
    else:
        st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")


def render_advanced_filtering(material_library: pd.DataFrame) -> pd.DataFrame:
    """Render advanced material filtering interface and return filtered materials"""
    
    st.markdown("#### 🎯 Advanced Material Filtering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_thermal = st.slider("Min Thermal Stability", 0.0, 1.0, 0.5)
        min_biocompat = st.slider("Min Biocompatibility", 0.0, 1.0, 0.6)
    
    with col2:
        max_uncertainty = st.slider("Max Uncertainty", 0.0, 1.0, 0.8)
        source_filter = st.multiselect(
            "Material Sources",
            material_library['source'].unique(),
            default=material_library['source'].unique()
        )
    
    with col3:
        min_insulin_stability = st.slider("Min Insulin Stability", 0.0, 1.0, 0.4)
        if 'composite_score' in material_library.columns:
            min_composite = st.slider("Min Composite Score", 0.0, 1.0, 0.6)
        else:
            min_composite = 0.0
    
    # Apply filters
    filtered_materials = material_library[
        (material_library['thermal_stability'] >= min_thermal) &
        (material_library['biocompatibility'] >= min_biocompat) &
        (material_library['uncertainty_score'] <= max_uncertainty) &
        (material_library['source'].isin(source_filter)) &
        (material_library['insulin_stability_score'] >= min_insulin_stability)
    ].copy()
    
    if 'composite_score' in material_library.columns:
        filtered_materials = filtered_materials[filtered_materials['composite_score'] >= min_composite]
    
    st.write(f"**{len(filtered_materials)} materials** meet criteria (from {len(material_library)} total)")
    
    return filtered_materials


def render_material_selection_interface(filtered_materials: pd.DataFrame):
    """Render the material selection interface for experimental validation"""
    
    # Rank filtered materials
    sort_column = 'composite_score' if 'composite_score' in filtered_materials.columns else 'insulin_stability_score'
    filtered_materials = filtered_materials.sort_values(sort_column, ascending=False)
    
    # Selection interface
    st.markdown("#### 📋 Material Selection for Experimental Validation")
    
    selection_strategy = st.selectbox(
        "Selection Strategy",
        ["Top Performers", "Diverse Portfolio", "High Uncertainty", "Balanced Approach"]
    )
    
    n_select = st.number_input("Number to Select", 1, min(20, len(filtered_materials)), 5)
    
    # Apply selection strategy
    selected_materials = apply_selection_strategy(filtered_materials, selection_strategy, n_select)
    
    # Display selected materials
    display_selected_materials(selected_materials, selection_strategy)
    
    # Export and action buttons
    render_selection_action_buttons(selected_materials)


def apply_selection_strategy(filtered_materials: pd.DataFrame, strategy: str, n_select: int) -> pd.DataFrame:
    """Apply the selected material selection strategy"""
    
    if strategy == "Top Performers":
        return filtered_materials.head(n_select)
    elif strategy == "Diverse Portfolio":
        # Select diverse materials across property space
        return filtered_materials.sample(n=min(n_select, len(filtered_materials)))
    elif strategy == "High Uncertainty":
        return filtered_materials.nlargest(n_select, 'uncertainty_score')
    else:  # Balanced Approach
        # Mix of top performers and high uncertainty
        top_half = filtered_materials.head(n_select//2)
        uncertain_half = filtered_materials.nlargest(n_select - len(top_half), 'uncertainty_score')
        return pd.concat([top_half, uncertain_half]).drop_duplicates()


def display_selected_materials(selected_materials: pd.DataFrame, selection_strategy: str):
    """Display the selected materials"""
    
    st.markdown(f"#### 🎯 Selected Materials ({selection_strategy})")
    
    # Display selected materials
    for idx, (_, material) in enumerate(selected_materials.iterrows()):
        composite_score = material.get('composite_score', 'N/A')
        st.markdown(f"""
        <div class="iteration-card">
            <strong>Selection #{idx+1}: Material {material['material_id']}</strong><br>
            <code>{escape_psmiles_for_markdown(material['psmiles'])}</code><br>
            <strong>Composite Score: {composite_score:.3f if composite_score != 'N/A' else 'N/A'}</strong><br>
            Thermal: {material['thermal_stability']:.2f} | 
            Biocompat: {material['biocompatibility']:.2f} | 
            Insulin: {material['insulin_stability_score']:.2f}<br>
            <small>Uncertainty: {material['uncertainty_score']:.2f} | Source: {material['source']}</small>
        </div>
        """, unsafe_allow_html=True)


def render_selection_action_buttons(selected_materials: pd.DataFrame):
    """Render action buttons for selected materials"""
    
    st.markdown("#### 🚀 Next Steps")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Selection"):
            st.success("Selected materials exported for experimental validation!")
    
    with col2:
        if st.button("🧬 Submit for MD Simulation"):
            st.success("Materials submitted for molecular dynamics simulation!")
    
    with col3:
        if st.button("🔄 Add to Active Learning"):
            if 'active_learning_queue' not in st.session_state:
                st.session_state.active_learning_queue = []
                
            for _, material in selected_materials.iterrows():
                st.session_state.active_learning_queue.append({
                    'type': 'selected_material',
                    'content': material.to_dict(),
                    'priority': material.get('composite_score', material['insulin_stability_score']),
                    'timestamp': datetime.now().isoformat()
                })
            st.success("Selected materials added to active learning queue!")
            st.rerun()


def escape_psmiles_for_markdown(psmiles: str) -> str:
    """
    Escape PSMILES string for safe markdown display
    
    Args:
        psmiles: The PSMILES string to escape
        
    Returns:
        Escaped PSMILES string
    """
    if not isinstance(psmiles, str):
        return str(psmiles)
    
    # Basic escaping for markdown
    escaped = psmiles.replace('*', '\\*').replace('_', '\\_').replace('[', '\\[').replace(']', '\\]')
    return escaped


# CSS Styles for the module
def inject_material_evaluation_styles():
    """Inject CSS styles for the material evaluation UI"""
    
    st.markdown("""
    <style>
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
    
    .property-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .score-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .score-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .score-low {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# Module initialization
if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(page_title="Material Evaluation UI Test", layout="wide")
    inject_material_evaluation_styles()
    render_material_evaluation_ui() 