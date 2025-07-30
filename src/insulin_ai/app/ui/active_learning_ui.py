"""
Active Learning UI Module

This module provides the user interface for the active learning feedback loop,
including iteration analysis, feedback integration, and learning queue management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from datetime import datetime
from typing import Dict, List, Any, Optional


def escape_psmiles_for_markdown(psmiles: str) -> str:
    """
    Escape PSMILES string for safe markdown display
    
    Args:
        psmiles: PSMILES string to escape
        
    Returns:
        Escaped PSMILES string safe for markdown
    """
    if not psmiles:
        return ""
    
    # Escape characters that could interfere with markdown
    escaped = psmiles.replace("*", "\\*").replace("_", "\\_")
    return escaped


def render_iteration_analysis_tab():
    """Render the iteration analysis tab for active learning"""
    st.markdown("### Active Learning Cycle Analysis")
    
    if st.session_state.literature_iterations and st.session_state.psmiles_candidates:
        # Calculate iteration performance
        iteration_data = []
        for i, lit_iter in enumerate(st.session_state.literature_iterations):
            # Find corresponding PSMILES candidates
            corresponding_candidates = [
                c for c in st.session_state.psmiles_candidates 
                if abs(datetime.fromisoformat(c['timestamp']).timestamp() - 
                       datetime.fromisoformat(lit_iter['timestamp']).timestamp()) < 3600
            ]
            
            if corresponding_candidates:
                avg_performance = np.mean([
                    np.mean(list(c['properties'].values())) 
                    for c in corresponding_candidates
                ])
            else:
                avg_performance = 0.5
            
            iteration_data.append({
                'iteration': i + 1,
                'performance': avg_performance,
                'materials_found': len(lit_iter['result']['materials_found']),
                'psmiles_generated': len(corresponding_candidates),
                'query': lit_iter['query'][:50] + "..."
            })
        
        if iteration_data:
            iteration_df = pd.DataFrame(iteration_data)
            
            # Performance trend
            fig_trend = px.line(
                iteration_df,
                x='iteration',
                y='performance',
                title="Active Learning Performance Trend",
                markers=True
            )
            fig_trend.add_hline(y=0.7, line_dash="dash", line_color="red", 
                              annotation_text="Target Performance Threshold")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Iteration details
            st.subheader("📈 Iteration Performance Details")
            st.dataframe(iteration_df)
            
            # Best performing iteration
            best_iter = iteration_df.loc[iteration_df['performance'].idxmax()]
            st.success(f"🏆 Best Iteration: #{int(best_iter['iteration'])} with performance {best_iter['performance']:.3f}")
    
    else:
        st.info("Complete at least one literature mining + PSMILES generation cycle to see analysis")
    
    # Active learning parameters
    st.subheader("🔧 Active Learning Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        exploration_rate = st.slider("Exploration vs Exploitation", 0.0, 1.0, 0.6)
        uncertainty_threshold = st.slider("Uncertainty Threshold", 0.0, 1.0, 0.7)
    
    with col2:
        batch_size = st.number_input("Batch Size", 1, 20, 5)
        convergence_criterion = st.selectbox(
            "Convergence Criterion",
            ["Performance Plateau", "Uncertainty Reduction", "Diversity Coverage"]
        )


def render_feedback_integration_tab():
    """Render the feedback integration tab for active learning"""
    st.markdown("### Feedback Integration & Strategy Refinement")
    
    # Generate feedback from current materials
    if st.session_state.psmiles_candidates:
        # Analyze successful patterns
        candidates = st.session_state.psmiles_candidates
        
        # Calculate composite scores
        composite_scores = []
        for candidate in candidates:
            props = candidate['properties']
            score = (0.4 * props['thermal_stability'] + 
                    0.3 * props['biocompatibility'] + 
                    0.3 * props['insulin_binding'])
            composite_scores.append(score)
        
        # Identify top performers
        top_indices = np.argsort(composite_scores)[-3:]
        top_candidates = [candidates[i] for i in top_indices]
        
        st.markdown("#### 🏆 Top Performing Materials")
        for i, candidate in enumerate(top_candidates):
            st.markdown(f"""
            <div class="iteration-card">
                <strong>Rank {i+1}: {candidate['id']}</strong><br>
                <code>{escape_psmiles_for_markdown(candidate['psmiles'])}</code><br>
                <small>Score: {composite_scores[top_indices[i]]:.3f} | Mode: {candidate['generation_mode']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Extract patterns
        successful_patterns = {
            'functional_groups': [],
            'backbone_structures': [],
            'connection_strategies': []
        }
        
        for candidate in top_candidates:
            psmiles = candidate['psmiles']
            # Simple pattern extraction
            if 'O' in psmiles:
                successful_patterns['functional_groups'].append('oxygen_containing')
            if 'N' in psmiles:
                successful_patterns['functional_groups'].append('nitrogen_containing')
            if '=' in psmiles:
                successful_patterns['backbone_structures'].append('double_bond')
            if 'C(=O)' in psmiles:
                successful_patterns['functional_groups'].append('carbonyl')
        
        st.markdown("#### 🎯 Identified Successful Patterns")
        for pattern_type, patterns in successful_patterns.items():
            if patterns:
                unique_patterns = list(set(patterns))
                st.write(f"**{pattern_type.title()}:** {', '.join(unique_patterns)}")
        
        # Update feedback for next iteration
        if st.button("🔄 Update Iteration Feedback"):
            st.session_state.iteration_feedback = {
                'top_materials': [c['psmiles'] for c in top_candidates],
                'successful_patterns': successful_patterns,
                'target_properties': {
                    'thermal_stability': np.mean([c['properties']['thermal_stability'] for c in top_candidates]),
                    'biocompatibility': np.mean([c['properties']['biocompatibility'] for c in top_candidates])
                },
                'mechanisms': ['thermal_protection', 'protein_stabilization', 'controlled_release'],
                'timestamp': datetime.now().isoformat()
            }
            st.success("✅ Feedback updated for next iteration!")
    
    # Literature mining guidance
    st.markdown("#### 📚 Next Literature Mining Focus")
    if st.session_state.iteration_feedback:
        feedback = st.session_state.iteration_feedback
        
        suggested_queries = [
            f"thermal stabilization {pattern}" 
            for pattern in feedback.get('successful_patterns', {}).get('functional_groups', ['polymer'])
        ]
        
        st.write("**Suggested Research Queries:**")
        for query in suggested_queries[:3]:
            st.code(query)
    else:
        st.info("Generate and evaluate materials first to get targeted suggestions")


def render_learning_queue_tab():
    """Render the learning queue management tab"""
    st.markdown("### Active Learning Queue Management")
    
    if st.session_state.active_learning_queue:
        # Sort by priority
        sorted_queue = sorted(
            st.session_state.active_learning_queue, 
            key=lambda x: x['priority'], 
            reverse=True
        )
        
        st.write(f"**{len(sorted_queue)} items** in active learning queue")
        
        for i, item in enumerate(sorted_queue[:10]):  # Show top 10
            with st.expander(f"Priority {item['priority']:.2f} - {item['type']}"):
                st.write(f"**Type:** {item['type']}")
                st.write(f"**Priority:** {item['priority']:.3f}")
                st.write(f"**Added:** {item['timestamp']}")
                
                if item['type'] == 'literature_insight':
                    content = item['content']
                    st.write(f"**Query:** {content['query']}")
                    st.write(f"**Materials:** {', '.join(content['materials_found'][:3])}")
                elif item['type'] == 'psmiles_candidate':
                    content = item['content']
                    st.code(escape_psmiles_for_markdown(content['psmiles']))
                    st.write(f"**Request:** {content['request']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚀 Process", key=f"process_{i}"):
                        st.success("Processing initiated!")
                
                with col2:
                    if st.button("⬆️ Prioritize", key=f"prioritize_{i}"):
                        item['priority'] = min(1.0, item['priority'] + 0.1)
                        st.success("Priority increased!")
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.active_learning_queue.remove(item)
                        st.rerun()
        
        # Queue statistics
        st.subheader("📊 Queue Statistics")
        queue_df = pd.DataFrame([
            {
                'type': item['type'],
                'priority': item['priority']
            }
            for item in st.session_state.active_learning_queue
        ])
        
        if not queue_df.empty:
            fig_queue = px.histogram(
                queue_df,
                x='type',
                color='type',
                title="Queue Composition by Type"
            )
            st.plotly_chart(fig_queue, use_container_width=True)
    else:
        st.info("No items in active learning queue. Add insights from literature mining or PSMILES generation!")
        
        # Quick add suggestions
        st.markdown("#### 🎯 Quick Add to Queue")
        if st.button("Add Random High-Priority Research Direction"):
            research_directions = [
                "trehalose-based insulin stabilization mechanisms",
                "PEG-insulin conjugation thermal protection",
                "chitosan hydrogel patch formulation optimization",
                "PLGA microsphere insulin encapsulation"
            ]
            
            selected_direction = random.choice(research_directions)
            st.session_state.active_learning_queue.append({
                'type': 'research_direction',
                'content': {'query': selected_direction},
                'priority': np.random.uniform(0.7, 0.9),
                'timestamp': datetime.now().isoformat()
            })
            st.success(f"Added: {selected_direction}")
            st.rerun()


def render_active_learning():
    """
    Render the complete active learning page
    
    This includes iteration analysis, feedback integration, and learning queue management.
    """
    st.subheader("🎯 Active Learning Feedback Loop")
    
    # Initialize session state for active learning
    if 'active_learning_queue' not in st.session_state:
        st.session_state.active_learning_queue = []
    if 'iteration_feedback' not in st.session_state:
        st.session_state.iteration_feedback = {}
    if 'literature_iterations' not in st.session_state:
        st.session_state.literature_iterations = []
    if 'psmiles_candidates' not in st.session_state:
        st.session_state.psmiles_candidates = []
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Iteration Analysis", "Feedback Integration", "Learning Queue"])
    
    with tab1:
        render_iteration_analysis_tab()
    
    with tab2:
        render_feedback_integration_tab()
    
    with tab3:
        render_learning_queue_tab() 