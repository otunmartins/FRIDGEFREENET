"""
Comprehensive Analysis UI Module

This module provides comprehensive analysis across all framework components,
including cross-component insights, system performance analysis, and integrated reporting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def render_system_overview_tab():
    """Render the system overview tab with high-level metrics"""
    st.markdown("### 🎯 System Performance Overview")
    
    # High-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        literature_count = len(st.session_state.get('literature_iterations', []))
        st.metric("Literature Iterations", literature_count, 
                 delta=f"+{literature_count-max(0, literature_count-1)}" if literature_count > 0 else None)
    
    with col2:
        psmiles_count = len(st.session_state.get('psmiles_candidates', []))
        st.metric("PSMILES Generated", psmiles_count,
                 delta=f"+{psmiles_count-max(0, psmiles_count-1)}" if psmiles_count > 0 else None)
    
    with col3:
        material_count = len(st.session_state.get('material_library', pd.DataFrame()))
        st.metric("Materials Evaluated", material_count,
                 delta=f"+{material_count-max(0, material_count-1)}" if material_count > 0 else None)
    
    with col4:
        simulations_count = len(st.session_state.get('simulation_results', []))
        st.metric("MD Simulations", simulations_count,
                 delta=f"+{simulations_count-max(0, simulations_count-1)}" if simulations_count > 0 else None)
    
    # System health indicators
    st.markdown("### 🏥 System Health")
    
    health_col1, health_col2 = st.columns(2)
    
    with health_col1:
        # Check system initialization
        systems_status = {
            "Literature Mining": hasattr(st.session_state, 'literature_miner') and st.session_state.literature_miner is not None,
            "PSMILES Generator": hasattr(st.session_state, 'psmiles_processor') and st.session_state.psmiles_processor is not None,
            "MD Integration": hasattr(st.session_state, 'md_integration') and st.session_state.md_integration is not None,
            "Session State": st.session_state.get('systems_initialized', False)
        }
        
        st.markdown("**AI Systems Status:**")
        for system, status in systems_status.items():
            status_emoji = "✅" if status else "❌"
            st.write(f"{status_emoji} {system}")
    
    with health_col2:
        # Memory usage estimation
        session_size = len(str(st.session_state))
        st.metric("Session State Size", f"{session_size:,} chars")
        
        # Data freshness
        if literature_count > 0:
            last_lit = st.session_state.literature_iterations[-1]['timestamp']
            last_lit_time = datetime.fromisoformat(last_lit)
            time_since = datetime.now() - last_lit_time
            st.metric("Last Literature Update", f"{time_since.seconds//3600}h {(time_since.seconds//60)%60}m ago")


def render_cross_component_analysis_tab():
    """Render cross-component analysis and correlations"""
    st.markdown("### 🔗 Cross-Component Analysis")
    
    if (len(st.session_state.get('literature_iterations', [])) > 0 and 
        len(st.session_state.get('psmiles_candidates', [])) > 0):
        
        # Literature-PSMILES correlation analysis
        st.markdown("#### 📚➡️🧪 Literature to PSMILES Pipeline Efficiency")
        
        # Calculate pipeline metrics
        lit_iterations = st.session_state.literature_iterations
        psmiles_candidates = st.session_state.psmiles_candidates
        
        pipeline_data = []
        for lit_iter in lit_iterations:
            lit_time = datetime.fromisoformat(lit_iter['timestamp'])
            
            # Find PSMILES generated within 1 hour after this literature iteration
            related_psmiles = [
                p for p in psmiles_candidates
                if abs(datetime.fromisoformat(p['timestamp']).timestamp() - lit_time.timestamp()) < 3600
            ]
            
            if related_psmiles:
                avg_performance = np.mean([
                    np.mean(list(p['properties'].values())) 
                    for p in related_psmiles
                ])
            else:
                avg_performance = 0.0
            
            pipeline_data.append({
                'iteration': len(pipeline_data) + 1,
                'materials_found': len(lit_iter['result']['materials_found']),
                'psmiles_generated': len(related_psmiles),
                'avg_performance': avg_performance,
                'efficiency': len(related_psmiles) / max(1, len(lit_iter['result']['materials_found']))
            })
        
        if pipeline_data:
            pipeline_df = pd.DataFrame(pipeline_data)
            
            # Pipeline efficiency chart
            fig_pipeline = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Materials Found vs PSMILES Generated', 'Pipeline Efficiency',
                               'Average Performance Trend', 'Efficiency Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Materials vs PSMILES
            fig_pipeline.add_trace(
                go.Scatter(x=pipeline_df['materials_found'], y=pipeline_df['psmiles_generated'],
                          mode='markers', name='Iterations', marker=dict(size=10)),
                row=1, col=1
            )
            
            # Efficiency over time
            fig_pipeline.add_trace(
                go.Scatter(x=pipeline_df['iteration'], y=pipeline_df['efficiency'],
                          mode='lines+markers', name='Efficiency'),
                row=1, col=2
            )
            
            # Performance trend
            fig_pipeline.add_trace(
                go.Scatter(x=pipeline_df['iteration'], y=pipeline_df['avg_performance'],
                          mode='lines+markers', name='Avg Performance'),
                row=2, col=1
            )
            
            # Efficiency distribution
            fig_pipeline.add_trace(
                go.Histogram(x=pipeline_df['efficiency'], name='Efficiency Dist'),
                row=2, col=2
            )
            
            fig_pipeline.update_layout(height=600, showlegend=False,
                                     title_text="Literature-PSMILES Pipeline Analysis")
            st.plotly_chart(fig_pipeline, use_container_width=True)
            
            # Summary statistics
            st.markdown("#### 📊 Pipeline Summary Statistics")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Average Efficiency", f"{pipeline_df['efficiency'].mean():.2f}",
                         help="PSMILES generated per literature material found")
            
            with summary_col2:
                st.metric("Best Performance", f"{pipeline_df['avg_performance'].max():.3f}",
                         help="Highest average material performance")
            
            with summary_col3:
                st.metric("Total Pipeline Runs", len(pipeline_df),
                         help="Number of literature→PSMILES cycles completed")
    
    else:
        st.info("Complete at least one full literature mining → PSMILES generation cycle to see cross-component analysis")


def render_trend_analysis_tab():
    """Render trend analysis and predictive insights"""
    st.markdown("### 📈 Trend Analysis & Predictions")
    
    # Material performance trends
    if len(st.session_state.get('psmiles_candidates', [])) > 2:
        st.markdown("#### 🧪 Material Performance Trends")
        
        candidates = st.session_state.psmiles_candidates
        
        # Extract time series data
        trend_data = []
        for candidate in candidates:
            timestamp = datetime.fromisoformat(candidate['timestamp'])
            props = candidate['properties']
            
            trend_data.append({
                'timestamp': timestamp,
                'thermal_stability': props['thermal_stability'],
                'biocompatibility': props['biocompatibility'],
                'insulin_binding': props['insulin_binding'],
                'composite_score': 0.4 * props['thermal_stability'] + 
                                0.3 * props['biocompatibility'] + 
                                0.3 * props['insulin_binding']
            })
        
        trend_df = pd.DataFrame(trend_data)
        trend_df = trend_df.sort_values('timestamp')
        
        # Trend visualization
        fig_trends = go.Figure()
        
        properties = ['thermal_stability', 'biocompatibility', 'insulin_binding', 'composite_score']
        colors = ['blue', 'green', 'orange', 'red']
        
        for prop, color in zip(properties, colors):
            fig_trends.add_trace(go.Scatter(
                x=trend_df['timestamp'],
                y=trend_df[prop],
                mode='lines+markers',
                name=prop.replace('_', ' ').title(),
                line=dict(color=color)
            ))
        
        fig_trends.update_layout(
            title="Material Property Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Property Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Simple trend analysis
        st.markdown("#### 🔮 Trend Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # Recent performance
            recent_scores = trend_df['composite_score'].tail(5).mean()
            earlier_scores = trend_df['composite_score'].head(5).mean()
            
            if recent_scores > earlier_scores:
                trend_direction = "📈 Improving"
                trend_color = "green"
            else:
                trend_direction = "📉 Declining"
                trend_color = "red"
            
            st.markdown(f"**Overall Trend:** <span style='color: {trend_color}'>{trend_direction}</span>", 
                       unsafe_allow_html=True)
            st.write(f"Recent avg: {recent_scores:.3f}")
            st.write(f"Earlier avg: {earlier_scores:.3f}")
        
        with insight_col2:
            # Best property
            property_means = {
                'Thermal Stability': trend_df['thermal_stability'].mean(),
                'Biocompatibility': trend_df['biocompatibility'].mean(),
                'Insulin Binding': trend_df['insulin_binding'].mean()
            }
            
            best_property = max(property_means, key=property_means.get)
            worst_property = min(property_means, key=property_means.get)
            
            st.write(f"**Strongest Property:** {best_property} ({property_means[best_property]:.3f})")
            st.write(f"**Weakest Property:** {worst_property} ({property_means[worst_property]:.3f})")
    
    else:
        st.info("Generate more materials to see trend analysis (minimum 3 required)")


def render_optimization_recommendations_tab():
    """Render optimization recommendations based on analysis"""
    st.markdown("### 🎯 Optimization Recommendations")
    
    # Analyze current state and provide recommendations
    recommendations = []
    priorities = []
    
    # Check literature mining effectiveness
    lit_count = len(st.session_state.get('literature_iterations', []))
    if lit_count == 0:
        recommendations.append("🔍 Start with literature mining to identify promising materials")
        priorities.append("High")
    elif lit_count < 3:
        recommendations.append("📚 Conduct more literature mining iterations for better coverage")
        priorities.append("Medium")
    
    # Check PSMILES generation
    psmiles_count = len(st.session_state.get('psmiles_candidates', []))
    if psmiles_count == 0:
        recommendations.append("🧪 Generate PSMILES candidates from literature insights")
        priorities.append("High")
    elif psmiles_count < 10:
        recommendations.append("⚗️ Generate more PSMILES variants for better exploration")
        priorities.append("Medium")
    
    # Check material evaluation
    if len(st.session_state.get('material_library', pd.DataFrame())) == 0:
        recommendations.append("📊 Evaluate generated materials for performance assessment")
        priorities.append("High")
    
    # Check MD simulations
    sim_count = len(st.session_state.get('simulation_results', []))
    if sim_count == 0 and psmiles_count > 0:
        recommendations.append("🔬 Run MD simulations on promising materials")
        priorities.append("Medium")
    
    # Active learning recommendations
    if (psmiles_count > 5 and 
        not st.session_state.get('iteration_feedback')):
        recommendations.append("🎯 Use active learning to refine material search strategy")
        priorities.append("High")
    
    # Performance-based recommendations
    if psmiles_count > 0:
        candidates = st.session_state.psmiles_candidates
        avg_performance = np.mean([
            np.mean(list(c['properties'].values())) 
            for c in candidates
        ])
        
        if avg_performance < 0.6:
            recommendations.append("⚡ Current materials show low performance - refine search criteria")
            priorities.append("High")
        elif avg_performance > 0.8:
            recommendations.append("✨ Excellent performance! Focus on scaling and optimization")
            priorities.append("Low")
    
    # Display recommendations
    if recommendations:
        st.markdown("#### 📋 Actionable Recommendations")
        
        for rec, priority in zip(recommendations, priorities):
            priority_color = {
                "High": "red",
                "Medium": "orange", 
                "Low": "green"
            }[priority]
            
            st.markdown(f"""
            <div style="padding: 10px; margin: 5px 0; border-left: 4px solid {priority_color}; background-color: rgba(255,255,255,0.1);">
                <strong style="color: {priority_color};">[{priority} Priority]</strong><br>
                {rec}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.success("🎉 System is running optimally! Continue with current workflow.")
    
    # Resource optimization
    st.markdown("#### 🔧 Resource Optimization")
    
    # Estimate computational resources
    estimated_time = lit_count * 2 + psmiles_count * 0.5 + sim_count * 30  # minutes
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Estimated Compute Time", f"{estimated_time:.1f} min",
                 help="Approximate time spent on computations")
        
        efficiency_score = psmiles_count / max(1, lit_count) if lit_count > 0 else 0
        st.metric("Pipeline Efficiency", f"{efficiency_score:.2f}",
                 help="PSMILES generated per literature iteration")
    
    with col2:
        # Memory usage recommendations
        session_size = len(str(st.session_state))
        if session_size > 100000:  # Large session
            st.warning("⚠️ Large session state detected. Consider periodic cleanup.")
        else:
            st.success("✅ Session state size is optimal")
        
        # Suggest next actions
        st.markdown("**Suggested Next Action:**")
        if not recommendations:
            st.write("🔄 Continue current active learning cycle")
        else:
            st.write(f"🎯 {recommendations[0].replace('🔍 ', '').replace('📚 ', '').replace('🧪 ', '')}")


def render_comprehensive_analysis():
    """
    Render the complete comprehensive analysis page
    
    This provides system-wide analysis, cross-component insights, and optimization recommendations.
    """
    st.subheader("🔍 Comprehensive System Analysis")
    
    st.markdown("""
    This analysis provides insights across all framework components, helping you optimize 
    the active learning workflow and identify areas for improvement.
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "System Overview", 
        "Cross-Component Analysis", 
        "Trend Analysis", 
        "Optimization Recommendations"
    ])
    
    with tab1:
        render_system_overview_tab()
    
    with tab2:
        render_cross_component_analysis_tab()
    
    with tab3:
        render_trend_analysis_tab()
    
    with tab4:
        render_optimization_recommendations_tab() 