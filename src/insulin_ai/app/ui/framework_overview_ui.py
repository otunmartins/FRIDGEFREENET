"""
Framework Overview UI Module for Insulin-AI Application

This module handles the main dashboard/overview page that displays
system metrics, component status, and framework architecture.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional


def render_framework_metrics():
    """Render the main framework metrics in a dashboard style"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        iterations_count = len(st.session_state.literature_iterations)
        st.markdown(
            f'<div class="metric-insulin"><h3>{iterations_count}</h3><p>Literature Iterations</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        psmiles_count = len(st.session_state.psmiles_candidates)
        st.markdown(
            f'<div class="metric-insulin"><h3>{psmiles_count}</h3><p>PSMILES Generated</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        if len(st.session_state.material_library) > 0:
            high_scoring = (st.session_state.material_library['insulin_stability_score'] > 0.7).sum()
        else:
            high_scoring = 0
        st.markdown(
            f'<div class="metric-insulin"><h3>{high_scoring}</h3><p>High-Performance Materials</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        queue_count = len(st.session_state.active_learning_queue)
        st.markdown(
            f'<div class="metric-insulin"><h3>{queue_count}</h3><p>Active Learning Queue</p></div>',
            unsafe_allow_html=True
        )


def render_framework_components():
    """Render the framework components overview"""
    st.subheader("🧬 Framework Components")
    
    framework_components = [
        {
            'name': 'Literature Mining (LLM Analysis)',
            'description': 'Ollama-based semantic analysis of scientific literature for insulin stabilization mechanisms',
            'status': 'Active' if st.session_state.systems_initialized else 'Offline',
            'color': '#4CAF50' if st.session_state.systems_initialized else '#f44336'
        },
        {
            'name': 'PSMILES Generation',
            'description': 'AI-driven polymer structure generation with multi-validation pipeline',
            'status': 'Active' if st.session_state.systems_initialized else 'Offline',
            'color': '#4CAF50' if st.session_state.systems_initialized else '#f44336'
        },
        {
            'name': 'Active Learning',
            'description': 'Uncertainty-driven material discovery with iterative feedback loops',
            'status': 'Active',
            'color': '#4CAF50'
        },
        {
            'name': 'Material Evaluation',
            'description': 'Multi-criteria assessment: thermal stability, biocompatibility, insulin interaction',
            'status': 'Active',
            'color': '#4CAF50'
        },
        {
            'name': 'MD Simulation',
            'description': 'OpenMM-based molecular dynamics for insulin-polymer interaction analysis',
            'status': 'Active',
            'color': '#4CAF50'
        }
    ]
    
    for component in framework_components:
        st.markdown(f"""
        <div style="border-left: 4px solid {component['color']}; padding: 15px; margin: 10px 0; background-color: #f8f9fa;">
            <h4 style="color: {component['color']}; margin-bottom: 5px;">{component['name']}</h4>
            <p style="margin-bottom: 5px;">{component['description']}</p>
            <small style="color: {component['color']};">Status: {component['status']}</small>
        </div>
        """, unsafe_allow_html=True)


def render_active_learning_cycle():
    """Render the active learning cycle visualization"""
    st.subheader("🔄 Active Learning Cycle")
    
    # Create a visual representation of the active learning cycle
    cycle_steps = [
        "📚 Literature Mining",
        "🧪 PSMILES Generation", 
        "🔬 Material Evaluation",
        "📊 Uncertainty Analysis",
        "🎯 Next Iteration Planning"
    ]
    
    st.markdown("**Current Active Learning Pipeline:**")
    
    cols = st.columns(len(cycle_steps))
    for i, (step, col) in enumerate(zip(cycle_steps, cols)):
        with col:
            if i < len(st.session_state.literature_iterations):
                st.markdown(f"✅ **{step}**")
            else:
                st.markdown(f"⏳ **{step}**")


def render_recent_activity():
    """Render recent system activity"""
    st.subheader("📈 Recent Activity")
    
    # Combine different types of activities
    activities = []
    
    # Literature iterations
    for iteration in st.session_state.literature_iterations[-3:]:
        activities.append({
            'type': 'Literature Mining',
            'timestamp': iteration.get('timestamp', datetime.now().isoformat()),
            'description': f"Query: {iteration['query'][:50]}...",
            'icon': '📚'
        })
    
    # PSMILES candidates
    for candidate in st.session_state.psmiles_candidates[-3:]:
        activities.append({
            'type': 'PSMILES Generation',
            'timestamp': candidate.get('timestamp', datetime.now().isoformat()),
            'description': f"Generated: {candidate.get('psmiles', 'Unknown')}",
            'icon': '🧪'
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    
    if activities:
        for activity in activities[:5]:  # Show last 5 activities
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-radius: 5px; background-color: #f0f2f6;">
                {activity['icon']} **{activity['type']}** - {activity['description']}
                <br><small style="color: #666;">
                    {datetime.fromisoformat(activity['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
                </small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activity. Start by running Literature Mining or PSMILES Generation.")


def render_material_library_overview():
    """Render overview of the current material library"""
    st.subheader("🗃️ Material Library Overview")
    
    if len(st.session_state.material_library) > 0:
        df = st.session_state.material_library
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_thermal = df['thermal_stability'].mean()
            st.metric("Avg Thermal Stability", f"{avg_thermal:.2f}")
        
        with col2:
            avg_biocompat = df['biocompatibility'].mean()
            st.metric("Avg Biocompatibility", f"{avg_biocompat:.2f}")
        
        with col3:
            avg_insulin = df['insulin_stability_score'].mean()
            st.metric("Avg Insulin Stability", f"{avg_insulin:.2f}")
        
        # Recent materials
        st.markdown("**Recent Materials:**")
        recent_materials = df.tail(5)
        for _, material in recent_materials.iterrows():
            st.markdown(f"""
            <div style="padding: 8px; margin: 4px 0; border-radius: 4px; background-color: #e8f4fd;">
                **ID:** {material['material_id']} | 
                **PSMILES:** {material['psmiles']} | 
                **Insulin Score:** {material['insulin_stability_score']:.2f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Material library is empty. Generate materials using Literature Mining and PSMILES Generation.")


def render_system_health():
    """Render system health indicators"""
    st.subheader("🏥 System Health")
    
    # Check various system components
    health_indicators = []
    
    # Systems initialization
    if st.session_state.systems_initialized:
        health_indicators.append(("AI Systems", "✅ Initialized", "#4CAF50"))
    else:
        health_indicators.append(("AI Systems", "❌ Not Initialized", "#f44336"))
    
    # Session state components
    required_components = [
        'literature_iterations', 'psmiles_candidates', 
        'active_learning_queue', 'material_library'
    ]
    
    for component in required_components:
        if component in st.session_state:
            health_indicators.append((component.replace('_', ' ').title(), "✅ Available", "#4CAF50"))
        else:
            health_indicators.append((component.replace('_', ' ').title(), "❌ Missing", "#f44336"))
    
    # Display health indicators
    for name, status, color in health_indicators:
        st.markdown(f"""
        <div style="padding: 8px; margin: 4px 0; border-left: 4px solid {color};">
            **{name}:** <span style="color: {color};">{status}</span>
        </div>
        """, unsafe_allow_html=True)


def render_quick_actions():
    """Render quick action buttons for common tasks"""
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Literature Mining", type="primary"):
            st.switch_page("Literature Mining (LLM)")
    
    with col2:
        if st.button("🧪 Generate PSMILES", type="primary"):
            st.switch_page("PSMILES Generation")
    
    with col3:
        if st.button("🔬 Run MD Simulation", type="primary"):
            st.switch_page("MD Simulation")


def render_framework_overview():
    """
    Render the complete framework overview page
    
    This is the main dashboard that provides an overview of the entire
    insulin-AI discovery framework.
    """
    st.subheader("🔄 Active Learning Framework Architecture")
    
    # Main metrics dashboard
    render_framework_metrics()
    
    st.markdown("---")
    
    # Framework components overview
    render_framework_components()
    
    st.markdown("---")
    
    # Active learning cycle visualization
    render_active_learning_cycle()
    
    st.markdown("---")
    
    # Two-column layout for detailed sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_recent_activity()
        render_system_health()
    
    with col2:
        render_material_library_overview()
        render_quick_actions()
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 14px;">
        <p>🧬 <strong>Insulin-AI Framework</strong> - AI-Powered Drug Delivery System Discovery</p>
        <p>Integrating LLM-based literature mining, PSMILES generation, and MD simulation for insulin stabilization</p>
    </div>
    """, unsafe_allow_html=True) 