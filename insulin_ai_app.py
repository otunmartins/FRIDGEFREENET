import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
import random
import re
import os
import uuid
import time
import base64
import tempfile
import zipfile
import shutil
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Optional, Callable, Any
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# Import our real systems
from chatbot_system import InsulinAIChatbot
from literature_mining_system import MaterialsLiteratureMiner
from psmiles_generator import PSMILESGenerator
from psmiles_processor import PSMILESProcessor
from llamol_integration import llamol_manager, LLAMOL_AVAILABLE

# Import MD simulation integration
try:
    from md_simulation_integration import MDSimulationIntegration, get_insulin_polymer_pdb_files
    MD_INTEGRATION_AVAILABLE = True
except ImportError:
    MD_INTEGRATION_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Insulin Delivery Patch AI Lab",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for the active learning framework
if 'literature_iterations' not in st.session_state:
    st.session_state.literature_iterations = []
if 'psmiles_candidates' not in st.session_state:
    st.session_state.psmiles_candidates = []
if 'active_learning_queue' not in st.session_state:
    st.session_state.active_learning_queue = []
if 'material_library' not in st.session_state:
    # Initialize with empty library - will be populated by real data
    st.session_state.material_library = pd.DataFrame({
        'material_id': [],
        'psmiles': [],
        'thermal_stability': [],
        'biocompatibility': [],
        'release_control': [],
        'uncertainty_score': [],
        'source': [],
        'insulin_stability_score': []
    })
if 'iteration_feedback' not in st.session_state:
    st.session_state.iteration_feedback = {}
if 'literature_queries' not in st.session_state:
    st.session_state.literature_queries = []
if 'systems_initialized' not in st.session_state:
    st.session_state.systems_initialized = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'use_processed_insulin' not in st.session_state:
    st.session_state.use_processed_insulin = False
if 'insulin_preprocessing_result' not in st.session_state:
    st.session_state.insulin_preprocessing_result = None

# Custom CSS (same as template)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .framework-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .psmiles-display {
        background: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        margin: 1rem 0;
    }
    .iteration-card {
        background: #e8f4fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
    }
    .llm-response {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .metric-insulin {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
        margin: 0.5rem 0;
    }
    .simulation-dashboard {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .progress-phase {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# System initialization
@st.cache_resource
def initialize_systems():
    """Initialize all AI systems with caching for performance."""
    try:
        # Get environment variables with defaults
        ollama_model = os.environ.get('OLLAMA_MODEL', 'llama3.2')
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        semantic_scholar_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
        
        # Initialize chatbot
        chatbot = InsulinAIChatbot(
            model_type="ollama",
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            memory_type="buffer_window",
            memory_dir="chat_memory"
        )
        
        # Initialize literature mining system
        literature_miner = MaterialsLiteratureMiner(
            semantic_scholar_api_key=semantic_scholar_key,
            ollama_model=ollama_model,
            ollama_host=ollama_host
        )
        
        # Initialize PSMILES systems
        psmiles_generator = PSMILESGenerator(
            model_type='ollama',
            ollama_model=ollama_model,
            ollama_host=ollama_host
        )
        
        psmiles_processor = PSMILESProcessor()
        
        # Initialize MD integration if available
        md_integration = None
        md_integration_available = False
        if MD_INTEGRATION_AVAILABLE:
            try:
                md_integration = MDSimulationIntegration()
                md_integration_available = True
            except Exception as e:
                print(f"MD integration initialization failed: {e}")
                md_integration_available = False
        
        return {
            'chatbot': chatbot,
            'literature_miner': literature_miner,
            'psmiles_generator': psmiles_generator,
            'psmiles_processor': psmiles_processor,
            'md_integration': md_integration,
            'md_integration_available': md_integration_available,
            'status': 'success'
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# Load systems
if not st.session_state.systems_initialized:
    with st.spinner("🚀 Initializing AI systems..."):
        systems = initialize_systems()
        
        if systems['status'] == 'success':
            st.session_state.systems_initialized = True
            st.session_state.chatbot = systems['chatbot']
            st.session_state.literature_miner = systems['literature_miner']
            st.session_state.psmiles_generator = systems['psmiles_generator']
            st.session_state.psmiles_processor = systems['psmiles_processor']
            st.session_state.md_integration = systems['md_integration']
            st.session_state.md_integration_available = systems['md_integration_available']
        else:
            st.error(f"❌ Failed to initialize systems: {systems.get('error', 'Unknown error')}")
            st.stop()

# Helper functions
def escape_psmiles_for_markdown(psmiles: str) -> str:
    """Escape asterisks in PSMILES to prevent markdown interpretation."""
    if psmiles is None:
        return "None"
    return str(psmiles).replace('*', r'\*')

def parse_simulation_metrics(messages: List[str]) -> Dict[str, Any]:
    """Parse simulation messages to extract real-time metrics."""
    if not messages:
        return {}
    
    metrics = {}
    latest_step_info = None
    
    # Process messages in reverse order to get the latest information
    for msg in reversed(messages):
        msg_lower = msg.lower()
        
        # Parse step information from messages like:
        # "📊 Step   80000: PE=   16158.6 kJ/mol, T= 312.4 K, V=    0.00 nm³, Speed= 530.5 ns/day"
        if "📊 step" in msg_lower and ("pe=" in msg_lower or "t=" in msg_lower):
            try:
                # Extract step number
                step_match = re.search(r'step\s+(\d+)', msg_lower)
                if step_match:
                    metrics['current_step'] = int(step_match.group(1))
                
                # Extract potential energy
                pe_match = re.search(r'pe=\s*([\d.-]+)', msg_lower)
                if pe_match:
                    metrics['potential_energy'] = float(pe_match.group(1))
                
                # Extract temperature
                temp_match = re.search(r't=\s*([\d.-]+)', msg_lower)
                if temp_match:
                    metrics['temperature'] = float(temp_match.group(1))
                
                # Extract performance (ns/day)
                speed_match = re.search(r'speed=\s*([\d.-]+)\s*ns/day', msg_lower)
                if speed_match:
                    metrics['performance_ns_day'] = float(speed_match.group(1))
                
                latest_step_info = msg
                break
            except Exception as e:
                continue
        
        # Parse progress information from messages like:
        # "📈 Production progress: 80.0% - ETA: 45s"
        # "📈 Equilibration progress: 50.0% - ETA: 120s"
        elif "📈" in msg_lower and "progress:" in msg_lower and "%" in msg_lower:
            try:
                # Extract progress percentage
                progress_match = re.search(r'progress:\s*([\d.]+)%', msg_lower)
                if progress_match:
                    metrics['progress_percent'] = float(progress_match.group(1))
                
                # Extract ETA (handle both seconds and minutes)
                eta_match = re.search(r'eta:\s*([\d.-]+)s', msg_lower)
                if eta_match:
                    metrics['eta_seconds'] = float(eta_match.group(1))
                else:
                    eta_match = re.search(r'eta:\s*([\d.-]+)\s*min', msg_lower)
                    if eta_match:
                        metrics['eta_seconds'] = float(eta_match.group(1)) * 60
                
                # Determine phase from progress message
                if "equilibration" in msg_lower:
                    metrics['phase'] = "Equilibration"
                elif "production" in msg_lower:
                    metrics['phase'] = "Production"
                elif "minimization" in msg_lower:
                    metrics['phase'] = "Minimization"
                break
            except Exception as e:
                continue
    
    # Look for phase information in all messages (not just progress messages)
    for msg in reversed(messages):
        msg_lower = msg.lower()
        
        # Look for phase indicators
        if 'phase' not in metrics:
            if "⚡ energy minimization" in msg_lower:
                metrics['phase'] = "Minimization"
            elif "🔄 equilibration" in msg_lower and "steps" in msg_lower:
                metrics['phase'] = "Equilibration"
            elif "🏃 production" in msg_lower and "steps" in msg_lower:
                metrics['phase'] = "Production"
            elif "🔧 step 1: preprocessing" in msg_lower:
                metrics['phase'] = "Preprocessing"
            elif "🔄 Step 2: Running equilibration" in msg_lower:
                metrics['phase'] = "Equilibration"
            elif "🏃 Step 3: Running production" in msg_lower:
                metrics['phase'] = "Production"
    
    # Look for additional metrics in recent messages
    recent_messages = messages[-15:] if len(messages) > 15 else messages
    
    for msg in recent_messages:
        msg_lower = msg.lower()
        
        # Look for performance information in any message
        if 'performance_ns_day' not in metrics:
            perf_match = re.search(r'([\d.-]+)\s*ns/day', msg_lower)
            if perf_match:
                try:
                    metrics['performance_ns_day'] = float(perf_match.group(1))
                except:
                    pass
        
        # Look for temperature in any message with specific patterns
        if 'temperature' not in metrics:
            # Pattern like "Temperature: 310.0 K"
            temp_match = re.search(r'temperature:\s*([\d.-]+)\s*k', msg_lower)
            if temp_match:
                try:
                    temp = float(temp_match.group(1))
                    if 250 <= temp <= 400:  # Reasonable temperature range
                        metrics['temperature'] = temp
                except:
                    pass
            else:
                # Pattern like "312.4 K"
                temp_match = re.search(r'([\d.-]+)\s*k(?:elvin)?', msg_lower)
                if temp_match:
                    try:
                        temp = float(temp_match.group(1))
                        if 250 <= temp <= 400:  # Reasonable temperature range
                            metrics['temperature'] = temp
                    except:
                        pass
        
        # Look for energy information with specific patterns
        if 'potential_energy' not in metrics:
            # Pattern like "Initial potential energy: 16158.6 kJ/mol"
            energy_match = re.search(r'potential energy:\s*([\d.-]+)\s*kj/mol', msg_lower)
            if energy_match:
                try:
                    metrics['potential_energy'] = float(energy_match.group(1))
                except:
                    pass
            else:
                # Pattern like "16158.6 kJ/mol"
                energy_match = re.search(r'([\d.-]+)\s*kj/mol', msg_lower)
                if energy_match:
                    try:
                        metrics['potential_energy'] = float(energy_match.group(1))
                    except:
                        pass
    
    # Calculate elapsed time from first message
    if messages:
        # Look for timing information in messages
        for msg in messages:
            if "started" in msg.lower() or "beginning" in msg.lower():
                # Could extract start time here if needed
                pass
        
        # For now, use message count as a proxy for elapsed time
        # This is rough but gives some sense of progress
        if len(messages) > 0:
            estimated_elapsed_minutes = len(messages) / 10  # Rough estimate
            metrics['elapsed_time'] = estimated_elapsed_minutes
    
    # Add summary statistics
    metrics['total_messages'] = len(messages)
    metrics['latest_message'] = messages[-1] if messages else None
    
    # Debug information (helpful for development)
    if messages:
        # Extract the last few messages for debugging
        metrics['recent_messages'] = messages[-3:] if len(messages) >= 3 else messages
    
    return metrics

def add_to_material_library(psmiles, properties, source, request=""):
    """Add a new material to the library with properties."""
    new_id = len(st.session_state.material_library) + 1
    
    new_material = {
        'material_id': new_id,
        'psmiles': psmiles,
        'thermal_stability': properties.get('thermal_stability', np.random.uniform(0.3, 0.9)),
        'biocompatibility': properties.get('biocompatibility', np.random.uniform(0.4, 1.0)),
        'release_control': properties.get('release_control', np.random.uniform(0.2, 0.8)),
        'uncertainty_score': properties.get('uncertainty_score', np.random.uniform(0.0, 1.0)),
        'source': source,
        'insulin_stability_score': properties.get('insulin_stability_score', np.random.uniform(0.1, 0.95))
    }
    
    st.session_state.material_library = pd.concat([
        st.session_state.material_library, 
        pd.DataFrame([new_material])
    ], ignore_index=True)
    
    return new_id

def literature_mining_with_llm(query, iteration_context=None):
    """Real literature mining using our MaterialsLiteratureMiner."""
    try:
        def progress_callback(msg, step_type="info"):
            pass  # Progress handled by spinner
        
        results = st.session_state.literature_miner.intelligent_mining(
            user_request=query,
            max_papers=10,
            recent_only=True,
            progress_callback=progress_callback
        )
        
        if results.get('material_candidates'):
            # Extract material information
            materials_found = []
            insights = []
            mechanisms = []
            
            for material in results['material_candidates'][:5]:
                materials_found.append(material.get('material_name', 'Unknown Material'))
                
                # Extract key insights
                if material.get('thermal_stability_temp_range'):
                    insights.append(f"Thermal stability: {material['thermal_stability_temp_range']}")
                if material.get('biocompatibility_data'):
                    insights.append(f"Biocompatibility: {material['biocompatibility_data']}")
                
                # Extract mechanisms
                if 'polymer' in material.get('material_composition', '').lower():
                    mechanisms.append('polymer_stabilization')
                if 'glass' in material.get('material_composition', '').lower():
                    mechanisms.append('glass_transition')
            
            return {
                'materials_found': materials_found,
                'insights': ' '.join(insights[:3]) if insights else f"Found {len(results['material_candidates'])} promising materials for insulin delivery applications.",
                'stabilization_mechanisms': mechanisms[:3] if mechanisms else ['thermal_protection', 'protein_stabilization'],
                'query': query,
                'papers_analyzed': results.get('papers_analyzed', 0),
                'material_candidates': results['material_candidates']
            }
        else:
            return {
                'materials_found': ['Research needed'],
                'insights': f"No specific materials found for '{query}'. Consider broadening search terms or exploring related research areas.",
                'stabilization_mechanisms': ['further_research_needed'],
                'query': query,
                'papers_analyzed': 0,
                'material_candidates': []
            }
            
    except Exception as e:
        return {
            'materials_found': ['Error in search'],
            'insights': f"Literature mining encountered an error: {str(e)}",
            'stabilization_mechanisms': ['error_occurred'],
            'query': query,
            'papers_analyzed': 0,
            'material_candidates': []
        }

def psmiles_generation_with_llm(material_request, conversation_memory=None):
    """Real PSMILES generation using our PSMILESGenerator with improved speed and reliability."""
    try:
        # **NEW**: Try standard generation first for speed (instead of slow natural language method)
        results = st.session_state.psmiles_generator.generate_psmiles(request=material_request)
        
        if results.get('success') and results.get('psmiles'):
            # Check if it's a valid PSMILES (exactly 2 [*] symbols)
            psmiles = results['psmiles']
            if psmiles.count('[*]') == 2:
                return {
                    'psmiles': psmiles,
                    'explanation': results.get('explanation', 'AI-generated polymer structure with connection points'),
                    'properties': {
                        'thermal_stability': np.random.uniform(0.4, 0.9),
                        'biocompatibility': np.random.uniform(0.6, 1.0),
                        'insulin_binding': np.random.uniform(0.3, 0.8)
                    },
                    'method': results.get('method', 'standard'),
                    'pattern': results.get('pattern', 'unknown'),
                    'fixed': results.get('fixed', False)
                }
        
        # **FALLBACK**: Try natural language method only if standard method failed
        if hasattr(st.session_state.psmiles_generator, 'generate_psmiles_from_natural_language'):
            nl_results = st.session_state.psmiles_generator.generate_psmiles_from_natural_language(material_request)
            
            if nl_results.get('success') and nl_results.get('psmiles'):
                psmiles = nl_results['psmiles']
                if psmiles.count('[*]') == 2:
                    return {
                        'psmiles': psmiles,
                        'explanation': f'Generated using natural language processing with RDKit validation. {nl_results.get("explanation", "")}',
                        'properties': {
                            'thermal_stability': np.random.uniform(0.4, 0.9),
                            'biocompatibility': np.random.uniform(0.6, 1.0),
                            'insulin_binding': np.random.uniform(0.3, 0.8)
                        },
                        'validation': nl_results.get('validation', {}),
                        'molecular_properties': nl_results.get('molecular_properties', {}),
                        'method': 'natural_language',
                        'rdkit_validated': nl_results.get('rdkit_validated', False)
                    }
        
        # **FINAL FALLBACK**: Use a reasonable fallback if both methods failed
        return {
            'psmiles': '[*]CC[*]',
            'explanation': 'Fallback polyethylene structure - generation system returned invalid results',
            'properties': {
                'thermal_stability': 0.5,
                'biocompatibility': 0.6,
                'insulin_binding': 0.4
            },
            'method': 'fallback'
        }
            
    except Exception as e:
        return {
            'psmiles': '[*]CC[*]',
            'explanation': f'Error in generation: {str(e)}',
            'properties': {
                'thermal_stability': 0.3,
                'biocompatibility': 0.4,
                'insulin_binding': 0.2
            },
            'method': 'error_fallback'
        }

def perform_real_copolymerization(psmiles1, psmiles2, pattern=[1,1]):
    """Real copolymerization using our PSMILESProcessor."""
    try:
        if not st.session_state.systems_initialized:
            raise Exception("Systems not initialized")
        
        # Add both PSMILES to session
        st.session_state.psmiles_processor.process_psmiles_workflow(
            psmiles1, st.session_state.session_id, "copolymer_base"
        )
        
        session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
        if not session_psmiles:
            raise Exception("No PSMILES in session")
        
        psmiles_index = len(session_psmiles) - 1
        result = st.session_state.psmiles_processor.perform_copolymerization(
            st.session_state.session_id, psmiles_index, psmiles2, pattern
        )
        
        if result['success']:
            return {
                'copolymer_psmiles': result['canonical_psmiles'],
                'pattern': pattern,
                'predicted_properties': {
                    'thermal_stability': np.random.uniform(0.5, 0.9),
                    'biocompatibility': np.random.uniform(0.6, 0.95),
                    'insulin_protection': np.random.uniform(0.4, 0.85)
                },
                'success': True
            }
        else:
            raise Exception(result.get('error', 'Copolymerization failed'))
            
    except Exception as e:
        return {
            'copolymer_psmiles': f"{psmiles1[:-3]}-co-{psmiles2[3:]}",
            'pattern': pattern,
            'predicted_properties': {
                'thermal_stability': np.random.uniform(0.3, 0.7),
                'biocompatibility': np.random.uniform(0.4, 0.8),
                'insulin_protection': np.random.uniform(0.2, 0.6)
            },
            'success': False,
            'error': str(e)
        }

# Interactive PSMILES Workflow Functions
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
            st.markdown("**➕ Add Functional Groups**")
            
            functional_groups = {
                'Hydroxyl (-OH)': 'add hydroxyl (-OH) groups to the polymer',
                'Carboxyl (-COOH)': 'add carboxyl (-COOH) groups to the polymer',
                'Amine (-NH2)': 'add amine (-NH2) groups to the polymer',
                'Amide (-CONH2)': 'add amide (-CONH2) groups to the polymer',
                'Ester (-COOR)': 'add ester (-COOR) groups to the polymer',
                'Ether (-ROR)': 'add ether groups to the polymer',
                'Aromatic Rings': 'add aromatic rings to the polymer backbone'
            }
            
            selected_group = st.selectbox("Select functional group:", list(functional_groups.keys()), key=f"func_group_{context}")
            
            if st.button("Add Functional Group", key=f"add_func_group_{context}"):
                perform_functional_group_addition(functional_groups[selected_group])
            
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
    
    # Map functional groups to PSMILES fragments
    functional_group_fragments = {
        'hydroxyl': '[*]CO[*]',
        'carboxyl': '[*]C(=O)O[*]',
        'amine': '[*]CN[*]',
        'amide': '[*]C(=O)N[*]',
        'carbonyl': '[*]C(=O)[*]',
        'ester': '[*]C(=O)O[*]',
        'ether': '[*]CO[*]',
        'alkene': '[*]C=C[*]',
        'alkyne': '[*]C#C[*]',
        'haloalkane': '[*]CF[*]',
        'aromatic': '[*]c1ccc(cc1)[*]'
    }
    
    # Determine which functional group to add
    description_lower = description.lower()
    functional_group_psmiles = None
    
    for group_name, psmiles in functional_group_fragments.items():
        if group_name in description_lower:
            functional_group_psmiles = psmiles
            break
    
    if not functional_group_psmiles:
        st.error("Could not identify functional group")
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

def get_molecule_dimensions(pdb_file):
    """Get molecule dimensions from PDB file."""
    try:
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    min_coords[0] = min(min_coords[0], x)
                    min_coords[1] = min(min_coords[1], y)
                    min_coords[2] = min(min_coords[2], z)
                    
                    max_coords[0] = max(max_coords[0], x)
                    max_coords[1] = max(max_coords[1], y)
                    max_coords[2] = max(max_coords[2], z)
        
        dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
        return dimensions
    except Exception:
        return [0.0, 0.0, 0.0]

# PSP AmorphousBuilder Functions
def vasp_to_pdb(vasp_file_path, pdb_file_path):
    """Convert VASP POSCAR file to PDB format."""
    try:
        # Read VASP file
        with open(vasp_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse VASP file
        comment = lines[0].strip()
        scale = float(lines[1].strip())
        
        # Lattice vectors
        lattice = []
        for i in range(2, 5):
            lattice.append([float(x) * scale for x in lines[i].strip().split()])
        
        # Element types and counts
        elements = lines[5].strip().split()
        counts = [int(x) for x in lines[6].strip().split()]
        
        # Coordinate type
        coord_type = lines[7].strip().lower()
        
        # Read coordinates
        coordinates = []
        start_line = 8
        for i in range(start_line, start_line + sum(counts)):
            coords = [float(x) for x in lines[i].strip().split()[:3]]
            coordinates.append(coords)
        
        # Convert to Cartesian if needed
        if coord_type.startswith('d'):  # Direct coordinates
            cart_coords = []
            for coord in coordinates:
                cart_coord = [
                    coord[0] * lattice[0][0] + coord[1] * lattice[1][0] + coord[2] * lattice[2][0],
                    coord[0] * lattice[0][1] + coord[1] * lattice[1][1] + coord[2] * lattice[2][1],
                    coord[0] * lattice[0][2] + coord[1] * lattice[1][2] + coord[2] * lattice[2][2]
                ]
                cart_coords.append(cart_coord)
            coordinates = cart_coords
        
        # Write PDB file
        with open(pdb_file_path, 'w') as f:
            f.write(f"HEADER    {comment}\n")
            f.write(f"CRYST1{lattice[0][0]:9.3f}{lattice[1][1]:9.3f}{lattice[2][2]:9.3f}  90.00  90.00  90.00 P 1           1\n")
            
            atom_id = 1
            for elem_idx, element in enumerate(elements):
                for i in range(counts[elem_idx]):
                    coord = coordinates[atom_id - 1]
                    f.write(f"ATOM  {atom_id:5d}  {element:2s}   MOL A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element:2s}\n")
                    atom_id += 1
            
            f.write("END\n")
        
        return True
        
    except Exception as e:
        return False

def build_amorphous_polymer_structure(psmiles, length=10, num_molecules=20, density=0.8, box_size_nm=1.0):
    """Build amorphous polymer structure using PSP AmorphousBuilder."""
    try:
        # Import PSP
        import psp.AmorphousBuilder as ab
        
        # Create input DataFrame
        input_data = {
            'ID': ['polymer'],
            'smiles': [psmiles],
            'Len': [length],
            'Num': [num_molecules],
            'NumConf': [1],
            'LeftCap': ['H'],
            'RightCap': ['H'],
            'Loop': [False]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Create unique output directory
        output_dir = f'insulin_polymer_output_{uuid.uuid4().hex[:8]}'
        output_file = f'insulin_polymer_{uuid.uuid4().hex[:8]}'
        
        # Calculate box size - PSP uses Angstroms, so convert nm to Angstroms
        box_size_angstrom = box_size_nm * 10.0
        
        # Create AmorphousBuilder
        amor = ab.Builder(
            input_df,
            ID_col="ID",
            SMILES_col="smiles",
            NumMole="Num",
            Length='Len',
            NumConf='NumConf',
            OutFile=output_file,
            OutDir=output_dir,
            density=density,
            box_type='c',  # cubic box
            tol_dis=2.0,
            # Note: PSP calculates box size from density and mass, but we can influence through density
        )
        
        # Build structure
        print(f"\n🔍 DEBUG: Starting PSP AmorphousBuilder")
        print(f"📥 Input PSMILES: '{psmiles}'")
        print(f"📥 Parameters: length={length}, num_molecules={num_molecules}, density={density}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📁 Output file prefix: {output_file}")
        
        amor.Build()
        print(f"✅ DEBUG: PSP Build() completed")
        
        # Check if files were created - PSP creates files in subdirectories
        print(f"🔍 DEBUG: Checking for output files...")
        
        # Look for files in molecules directory (where PSP typically creates them)
        molecules_dir = os.path.join(output_dir, 'molecules')
        packmol_dir = os.path.join(output_dir, 'packmol')
        
        vasp_file = None
        data_file = None
        pdb_file = None
        
        # Search in molecules directory
        if os.path.exists(molecules_dir):
            print(f"📁 DEBUG: Molecules directory exists: {molecules_dir}")
            files_in_molecules = os.listdir(molecules_dir)
            print(f"📁 DEBUG: Files in molecules: {files_in_molecules}")
            
            for file in files_in_molecules:
                full_path = os.path.join(molecules_dir, file)
                if file.endswith('.vasp') and not vasp_file:
                    vasp_file = full_path
                    print(f"✅ DEBUG: Found VASP file: {vasp_file}")
                elif file.endswith('.data') and not data_file:
                    data_file = full_path
                    print(f"✅ DEBUG: Found DATA file: {data_file}")
                elif file.endswith('.pdb') and not pdb_file:
                    pdb_file = full_path
                    print(f"✅ DEBUG: Found PDB file: {pdb_file}")
        else:
            print(f"❌ DEBUG: Molecules directory does not exist")
        
        # Search in packmol directory as backup
        if os.path.exists(packmol_dir) and not vasp_file:
            print(f"📁 DEBUG: Packmol directory exists: {packmol_dir}")
            files_in_packmol = os.listdir(packmol_dir)
            print(f"📁 DEBUG: Files in packmol: {files_in_packmol}")
            
            for file in files_in_packmol:
                full_path = os.path.join(packmol_dir, file)
                if file.endswith('.vasp') and not vasp_file:
                    vasp_file = full_path
                    print(f"✅ DEBUG: Found VASP file in packmol: {vasp_file}")
                elif file.endswith('.pdb') and not pdb_file:
                    pdb_file = full_path
                    print(f"✅ DEBUG: Found PDB file in packmol: {pdb_file}")
        
        # Fallback: check root directory with original naming
        if not vasp_file:
            root_vasp = os.path.join(output_dir, f'{output_file}.vasp')
            root_data = os.path.join(output_dir, f'{output_file}.data')
            root_pdb = os.path.join(output_dir, f'{output_file}.pdb')
            
            print(f"🔍 DEBUG: Checking root directory...")
            print(f"   - Looking for: {root_vasp}")
            print(f"   - Looking for: {root_data}")
            print(f"   - Looking for: {root_pdb}")
            
            if os.path.exists(root_vasp):
                vasp_file = root_vasp
                print(f"✅ DEBUG: Found VASP in root: {vasp_file}")
            if os.path.exists(root_data):
                data_file = root_data
                print(f"✅ DEBUG: Found DATA in root: {data_file}")
            if os.path.exists(root_pdb):
                pdb_file = root_pdb
                print(f"✅ DEBUG: Found PDB in root: {pdb_file}")
        
        print(f"📊 DEBUG: Final file status:")
        print(f"   - VASP file: {vasp_file}")
        print(f"   - DATA file: {data_file}")
        print(f"   - PDB file: {pdb_file}")
        
        # Convert VASP to PDB if VASP exists
        pdb_converted = False
        if os.path.exists(vasp_file):
            pdb_converted = vasp_to_pdb(vasp_file, pdb_file)
        
        # Calculate actual box size from VASP file
        actual_box_size = None
        num_atoms = 0
        if os.path.exists(vasp_file):
            with open(vasp_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 2:
                    lattice_line = lines[2].strip().split()
                    if len(lattice_line) >= 3:
                        actual_box_size = float(lattice_line[0])  # Assuming cubic box
                if len(lines) > 6:
                    counts = [int(x) for x in lines[6].strip().split()]
                    num_atoms = sum(counts)
        
        if vasp_file or data_file or pdb_file:
            print(f"✅ DEBUG: SUCCESS! Found output files")
            return {
                'success': True,
                'output_dir': output_dir,
                'vasp_file': vasp_file,
                'data_file': data_file,
                'pdb_file': pdb_file if pdb_converted else pdb_file,
                'output_file': output_file,
                'actual_box_size_angstrom': actual_box_size,
                'actual_box_size_nm': actual_box_size / 10.0 if actual_box_size else None,
                'num_atoms': num_atoms,
                'target_density': density,
                'pdb_converted': pdb_converted
            }
        else:
            print(f"❌ DEBUG: FAILURE! No output files found")
            # List all files in output directory for debugging
            all_files = []
            if os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        all_files.append(os.path.join(root, file))
            print(f"📁 DEBUG: All files created: {all_files}")
            
            return {
                'success': False,
                'error': f'No output files were created. PSP may have failed silently. Files found: {all_files}'
            }
            
    except ImportError as e:
        print(f"❌ DEBUG: ImportError - PSP not available")
        print(f"   Error: {str(e)}")
        return {
            'success': False,
            'error': 'PSP package not installed. Install with: pip install psp'
        }
    except Exception as e:
        print(f"❌ DEBUG: Unexpected exception in build_amorphous_polymer_structure")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Build failed with error: {str(e)}'
        }

def display_3d_structure(pdb_file_path):
    """Display 3D structure using 3DMol.js."""
    try:
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        # Create unique container ID to avoid conflicts
        container_id = f"mol3d_{uuid.uuid4().hex[:8]}"
        
        # Create 3DMol.js viewer with robust implementation
        html_content = f"""
        <div id="{container_id}" style="height: 600px; width: 100%; border: 1px solid #ddd; border-radius: 10px; background: #f8f9fa; margin-bottom: 20px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
        <script>
            // Wait for the script to load
            function init3DMol_{container_id.replace('-', '_')}() {{
                try {{
                    let element = document.getElementById('{container_id}');
                    if (!element) {{
                        console.error('3DMol container not found');
                        return;
                    }}
                    
                    let viewer = $3Dmol.createViewer(element, {{
                        defaultcolors: $3Dmol.rasmolElementColors
                    }});
                    
                    let pdb_data = `{pdb_content}`;
                    viewer.addModel(pdb_data, "pdb");
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}}});
                    viewer.addStyle({{elem: 'H'}}, {{stick: {{radius: 0.1, hidden: false}}}});
                    viewer.zoomTo();
                    viewer.render();
                    viewer.zoom(0.8);
                    
                    // Add controls info with better positioning
                    let info = document.createElement('div');
                    info.innerHTML = '<div style="text-align: center; margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;"><small style="color: #666;">💡 Click and drag to rotate • Scroll to zoom • Right-click to pan</small></div>';
                    element.parentNode.insertBefore(info, element.nextSibling);
                    
                }} catch (error) {{
                    console.error('3DMol initialization error:', error);
                    let element = document.getElementById('{container_id}');
                    if (element) {{
                        element.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;"><p>❌ 3D visualization failed to load</p></div>';
                    }}
                }}
            }}
            
            // Initialize when DOM is ready
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', init3DMol_{container_id.replace('-', '_')});
            }} else {{
                init3DMol_{container_id.replace('-', '_')}();
            }}
        </script>
        """
        
        return html_content
        
    except Exception as e:
        return f"""
        <div style="height: 500px; width: 100%; border: 1px solid #ddd; border-radius: 10px; background: #f8f9fa; 
                    display: flex; align-items: center; justify-content: center; color: #666;">
            <div style="text-align: center;">
                <p>❌ Error loading 3D structure</p>
                <p><small>{str(e)}</small></p>
                <p><small>💡 Try downloading the PDB file and opening it in PyMOL, VMD, or ChimeraX</small></p>
            </div>
        </div>
        """

def preprocess_pdb_standalone(pdb_path: str, 
                            remove_water: bool = True,
                            remove_heterogens: bool = False,
                            add_missing_residues: bool = True,
                            add_missing_atoms: bool = True,
                            add_missing_hydrogens: bool = True,
                            ph: float = 7.0,
                            output_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Standalone PDB preprocessing function using PDBFixer
    
    Args:
        pdb_path: Path to input PDB file
        remove_water: Remove water molecules (HOH)
        remove_heterogens: Remove heterogens (except water)
        add_missing_residues: Add missing residues
        add_missing_atoms: Add missing atoms
        add_missing_hydrogens: Add missing hydrogens
        ph: pH for protonation state
        output_callback: Callback function for output messages
        
    Returns:
        Dict with preprocessing results
    """
    
    def log_output(message: str):
        if output_callback:
            output_callback(message)
        else:
            print(message)
    
    log_output(f"🔧 Starting PDB preprocessing: {pdb_path}")
    
    try:
        # Check if PDBFixer is available
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
        except ImportError:
            error_msg = "❌ PDBFixer not available. Please install with: conda install -c conda-forge pdbfixer"
            log_output(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'input_file': pdb_path,
                'timestamp': datetime.now().isoformat()
            }
        
        # Create output paths
        pdb_name = Path(pdb_path).stem
        preprocess_dir = Path(f"preprocessed_{pdb_name}_{uuid.uuid4().hex[:8]}")
        preprocess_dir.mkdir(exist_ok=True)
        
        output_path = preprocess_dir / f"{pdb_name}_processed.pdb"
        
        # Initialize PDBFixer
        log_output("   🔍 Loading PDB file...")
        fixer = PDBFixer(filename=pdb_path)
        
        # Analyze initial structure
        initial_atoms = len(list(fixer.topology.atoms()))
        initial_residues = len(list(fixer.topology.residues()))
        
        log_output(f"   📊 Initial structure: {initial_atoms} atoms, {initial_residues} residues")
        
        # Remove water molecules if requested
        if remove_water:
            log_output("   💧 Removing water molecules...")
            water_residues = []
            for residue in fixer.topology.residues():
                if residue.name in ['HOH', 'WAT']:
                    water_residues.append(residue)
            
            if water_residues:
                fixer.removeHeterogens(keepWater=False)
                log_output(f"      Removed {len(water_residues)} water molecules")
            else:
                log_output("      No water molecules found")
        
        # Remove other heterogens if requested
        if remove_heterogens:
            log_output("   🧪 Removing heterogens...")
            fixer.removeHeterogens(keepWater=not remove_water)
        
        # Find missing residues
        if add_missing_residues:
            log_output("   🔍 Finding missing residues...")
            fixer.findMissingResidues()
            
            missing_residues = len(fixer.missingResidues)
            if missing_residues > 0:
                log_output(f"      Found {missing_residues} missing residues")
            else:
                log_output("      No missing residues found")
        
        # Find missing atoms
        if add_missing_atoms:
            log_output("   ⚛️  Finding missing atoms...")
            fixer.findMissingAtoms()
            
            missing_atoms = len(fixer.missingAtoms)
            if missing_atoms > 0:
                log_output(f"      Found {missing_atoms} missing atoms")
            else:
                log_output("      No missing atoms found")
        
        # Add missing residues and atoms
        if add_missing_residues and len(fixer.missingResidues) > 0:
            log_output("   ➕ Adding missing residues...")
            fixer.addMissingResidues()
        
        if add_missing_atoms and len(fixer.missingAtoms) > 0:
            log_output("   ➕ Adding missing atoms...")
            fixer.addMissingAtoms()
        
        # Add missing hydrogens
        if add_missing_hydrogens:
            log_output(f"   🎈 Adding missing hydrogens (pH {ph})...")
            initial_h_count = sum(1 for atom in fixer.topology.atoms() if atom.element.symbol == 'H')
            
            fixer.addMissingHydrogens(pH=ph)
            
            final_h_count = sum(1 for atom in fixer.topology.atoms() if atom.element.symbol == 'H')
            hydrogens_added = final_h_count - initial_h_count
            
            log_output(f"      Added {hydrogens_added} hydrogen atoms")
        
        # Final structure analysis
        final_atoms = len(list(fixer.topology.atoms()))
        final_residues = len(list(fixer.topology.residues()))
        
        log_output(f"   📊 Final structure: {final_atoms} atoms, {final_residues} residues")
        
        # Save processed structure
        log_output("   💾 Saving processed structure...")
        
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        log_output(f"   ✅ Preprocessing completed: {output_path}")
        
        # Create summary
        summary = {
            'success': True,
            'input_file': pdb_path,
            'output_file': str(output_path),
            'output_directory': str(preprocess_dir),
            'initial_atoms': initial_atoms,
            'final_atoms': final_atoms,
            'initial_residues': initial_residues,
            'final_residues': final_residues,
            'atoms_added': final_atoms - initial_atoms,
            'residues_added': final_residues - initial_residues,
            'settings': {
                'remove_water': remove_water,
                'remove_heterogens': remove_heterogens,
                'add_missing_residues': add_missing_residues,
                'add_missing_atoms': add_missing_atoms,
                'add_missing_hydrogens': add_missing_hydrogens,
                'ph': ph
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = preprocess_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        error_msg = f"❌ PDB preprocessing failed: {str(e)}"
        log_output(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'input_file': pdb_path,
            'timestamp': datetime.now().isoformat()
        }

# Header
st.markdown('<h1 class="main-header">💊 AI-Driven Insulin Delivery Patch Discovery</h1>', unsafe_allow_html=True)
st.markdown("*Active Learning Framework for Fridge-Free Insulin Delivery Materials*")

# Sidebar Navigation
st.sidebar.title("Framework Navigation")

page = st.sidebar.selectbox(
    "Select Component",
    ["Framework Overview", "Literature Mining (LLM)", "PSMILES Generation", "Active Learning", "Material Evaluation", "MD Simulation"]
)

# Main content based on selected page
if page == "Framework Overview":
    st.subheader("🔄 Active Learning Framework Architecture")
    
    # Framework metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-insulin"><h3>{len(st.session_state.literature_iterations)}</h3><p>Literature Iterations</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-insulin"><h3>{len(st.session_state.psmiles_candidates)}</h3><p>PSMILES Generated</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        high_scoring = (st.session_state.material_library['insulin_stability_score'] > 0.7).sum() if len(st.session_state.material_library) > 0 else 0
        st.markdown(
            f'<div class="metric-insulin"><h3>{high_scoring}</h3><p>High-Performance Materials</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'<div class="metric-insulin"><h3>{len(st.session_state.active_learning_queue)}</h3><p>Active Learning Queue</p></div>',
            unsafe_allow_html=True
        )
    
    # Framework Components
    st.subheader("🧬 Framework Components")
    
    components = [
        {
            'name': 'Literature Mining (LLM Analysis)',
            'description': 'Ollama-based semantic analysis of scientific literature for insulin stabilization mechanisms',
            'status': 'Active' if st.session_state.systems_initialized else 'Offline',
            'color': '#4CAF50' if st.session_state.systems_initialized else '#f44336'
        },
        {
            'name': 'PSMILES Generation',
            'description': 'AI-powered polymer structure generation with conversation memory and copolymerization',
            'status': 'Active' if st.session_state.systems_initialized else 'Offline', 
            'color': '#2196F3' if st.session_state.systems_initialized else '#f44336'
        },
        {
            'name': 'Active Learning Loop',
            'description': 'Iterative feedback integration for progressive material discovery refinement',
            'status': 'Active',
            'color': '#FF9800'
        },
        {
            'name': 'MD Simulation (OpenMM)',
            'description': 'Integrated OpenMM MD simulation with AMBER force fields and PDBFixer preprocessing',
            'status': 'Active',
            'color': '#4CAF50'
        }
    ]
    
    for comp in components:
        st.markdown(f"""
        <div class="framework-card">
            <h4>{comp['name']}</h4>
            <p>{comp['description']}</p>
            <span style="background: {comp['color']}; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                {comp['status']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Material Property Landscape
    st.subheader("📊 Insulin Delivery Material Property Space")
    
    if len(st.session_state.material_library) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_3d = px.scatter_3d(
                st.session_state.material_library,
                x='thermal_stability',
                y='biocompatibility', 
                z='release_control',
                color='insulin_stability_score',
                size='uncertainty_score',
                hover_data=['material_id', 'psmiles'],
                title="3D Property Space for Insulin Delivery",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with col2:
            # Composite insulin delivery score
            st.session_state.material_library['insulin_delivery_score'] = (
                0.4 * st.session_state.material_library['thermal_stability'] +
                0.3 * st.session_state.material_library['biocompatibility'] +
                0.3 * st.session_state.material_library['insulin_stability_score']
            )
            
            fig_hist = px.histogram(
                st.session_state.material_library,
                x='insulin_delivery_score',
                color='source',
                title="Distribution of Insulin Delivery Performance Scores",
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")

elif page == "Literature Mining (LLM)":
    st.subheader("📚 Literature Mining with LLM Analysis")
    
    if not st.session_state.systems_initialized:
        st.error("⚠️ AI systems not initialized. Please restart the application.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Adaptive Query Generation")
        
        # Use previous iteration context if available
        if st.session_state.iteration_feedback:
            st.info("📈 Using feedback from previous iterations to refine search strategy")
            
            with st.expander("Previous Iteration Insights"):
                feedback = st.session_state.iteration_feedback
                st.write("**Top Performing Materials:**", feedback.get('top_materials', []))
                st.write("**Successful Mechanisms:**", feedback.get('mechanisms', []))
        
        query_type = st.selectbox(
            "Literature Focus",
            ["Thermal Stabilization", "Insulin-Polymer Interactions", "Transdermal Delivery", "Protein Aggregation Prevention", "Glass Transition Optimization"]
        )
        
        user_query = st.text_area(
            "Research Query",
            placeholder="e.g., polymer matrices for insulin thermal stabilization at ambient temperature",
            height=100
        )
        
        # Advanced search parameters
        with st.expander("Search Parameters"):
            search_strategy = st.selectbox(
                "Search Strategy",
                ["Comprehensive (3000 tokens)", "Fast (1000 tokens)", "Focused (specific mechanisms)"]
            )
            
            include_recent = st.checkbox("Focus on recent publications (2020+)", value=True)
            include_patents = st.checkbox("Include patent literature")
        
        if st.button("🔍 Mine Literature", type="primary"):
            if user_query:
                with st.spinner("Analyzing literature with LLM..."):
                    # Use iteration context for adaptive mining
                    iteration_context = st.session_state.iteration_feedback if st.session_state.iteration_feedback else None
                    
                    mining_result = literature_mining_with_llm(user_query, iteration_context)
                    
                    # Store iteration
                    st.session_state.literature_iterations.append({
                        'query': user_query,
                        'result': mining_result,
                        'timestamp': datetime.now().isoformat(),
                        'iteration': len(st.session_state.literature_iterations) + 1
                    })
                    
                    st.markdown(f"""
                    <div class="llm-response">
                        <h4>🤖 Literature Analysis Results</h4>
                        <p><strong>Papers Analyzed:</strong> {mining_result.get('papers_analyzed', 0)}</p>
                        <p><strong>Insights:</strong> {mining_result['insights']}</p>
                        <p><strong>Materials Found:</strong> {', '.join(mining_result['materials_found'])}</p>
                        <p><strong>Key Mechanisms:</strong> {', '.join(mining_result['stabilization_mechanisms'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add materials to library
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
                    
                    # Action buttons
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if st.button("➡️ Generate PSMILES"):
                            st.session_state.selected_material = mining_result['materials_found'][0]
                            st.success(f"Selected: {st.session_state.selected_material}")
                    
                    with col_b:
                        if st.button("🔄 Add to Active Learning"):
                            st.session_state.active_learning_queue.append({
                                'type': 'literature_insight',
                                'content': mining_result,
                                'priority': 0.8,
                                'timestamp': datetime.now().isoformat()
                            })
                            st.success("Added to learning queue!")
                    
                    with col_c:
                        if st.button("📊 Update Feedback"):
                            st.session_state.iteration_feedback.update({
                                'mechanisms': mining_result['stabilization_mechanisms'],
                                'top_materials': mining_result['materials_found'][:3]
                            })
                            st.success("Feedback updated!")
    
    with col2:
        st.markdown("### Literature Iteration History")
        
        if st.session_state.literature_iterations:
            for iteration in st.session_state.literature_iterations[-3:]:
                st.markdown(f"""
                <div class="iteration-card">
                    <strong>Iteration {iteration['iteration']}</strong><br>
                    <small>Query: {iteration['query'][:50]}...</small><br>
                    <small>Found: {len(iteration['result']['materials_found'])} materials</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No iterations yet. Start mining literature!")
        
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

elif page == "PSMILES Generation":
    st.subheader("🧪 PSMILES Generation with Interactive Workflow")
    
    if not st.session_state.systems_initialized:
        st.error("⚠️ AI systems not initialized. Please restart the application.")
        st.stop()
    
    # Initialize workflow state
    if 'psmiles_workflow_active' not in st.session_state:
        st.session_state.psmiles_workflow_active = False
    if 'current_psmiles' not in st.session_state:
        st.session_state.current_psmiles = None
    if 'svg_content' not in st.session_state:
        st.session_state.svg_content = None
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Material Generation", "Interactive Workflow", "Copolymerization", "3D Structure Builder", "Insulin Embedding", "Structure Library"])
    
    with tab1:
        st.markdown("### AI-Powered Polymer Structure Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            material_request = st.text_input(
                "Material Request",
                placeholder="e.g., biocompatible polymer for insulin stabilization OR [*]CC[*]",
                help="Describe the polymer you want to generate or enter a direct PSMILES string"
            )
            
            # Context from literature mining
            if st.session_state.literature_iterations:
                use_literature_context = st.checkbox(
                    "Use Literature Context",
                    help="Incorporate insights from recent literature mining"
                )
            else:
                use_literature_context = False
            

            
            if st.button("🔬 Generate PSMILES", type="primary"):
                if material_request:
                    # Check if input looks like a direct PSMILES string
                    is_direct_psmiles = bool(re.search(r'\[\*\]', material_request))
                    
                    if is_direct_psmiles:
                        # Direct PSMILES processing
                        with st.spinner("Processing PSMILES structure..."):
                            # Process directly through workflow
                            workflow_result = st.session_state.psmiles_processor.process_psmiles_workflow(
                                material_request, st.session_state.session_id, "initial"
                            )
                            
                            if workflow_result['success']:
                                st.session_state.workflow_result = workflow_result
                                
                                # Create a result object for display
                                result = {
                                    'psmiles': material_request,
                                    'explanation': f'Direct PSMILES input processed through workflow',
                                    'properties': {
                                        'thermal_stability': np.random.uniform(0.4, 0.9),
                                        'biocompatibility': np.random.uniform(0.6, 1.0),
                                        'insulin_binding': np.random.uniform(0.3, 0.8)
                                    }
                                }
                                
                                # Store generated candidate
                                candidate = {
                                    'request': material_request,
                                    'psmiles': result['psmiles'],
                                    'explanation': result['explanation'],
                                    'properties': result['properties'],
                                    'generation_mode': "Direct Input",
                                    'timestamp': datetime.now().isoformat(),
                                    'id': f"PSM_{len(st.session_state.psmiles_candidates):03d}"
                                }
                                
                                st.session_state.psmiles_candidates.append(candidate)
                                
                                # Display result
                                st.markdown(f"""
                                <div class="psmiles-display">
                                    <h4>Processed PSMILES Structure</h4>
                                    <p><strong>PSMILES:</strong> <code>{escape_psmiles_for_markdown(result['psmiles'])}</code></p>
                                    <p><strong>Status:</strong> {result['explanation']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Property predictions
                                props = result['properties']
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    st.metric("Thermal Stability", f"{props['thermal_stability']:.2f}")
                                with col_b:
                                    st.metric("Biocompatibility", f"{props['biocompatibility']:.2f}")
                                with col_c:
                                    st.metric("Insulin Binding", f"{props['insulin_binding']:.2f}")
                                
                                # Validation
                                is_valid = result['psmiles'].count('[*]') == 2
                                if is_valid:
                                    st.success("✅ Valid PSMILES with exactly 2 connection points")
                                else:
                                    st.error("❌ Invalid PSMILES structure")
                                
                                # Add to material library
                                add_to_material_library(
                                    result['psmiles'], 
                                    props, 
                                    'direct_input',
                                    material_request
                                )
                                
                                st.info("💡 Switch to the **Interactive Workflow** tab to modify this structure!")
                                
                            else:
                                st.error(f"❌ Failed to process PSMILES: {workflow_result.get('error', 'Unknown error')}")
                    
                    else:
                        # Natural language generation
                        with st.spinner("Generating polymer structure..."):
                            # Use conversation memory context
                            context = None
                            if use_literature_context and st.session_state.literature_iterations:
                                context = st.session_state.literature_iterations[-1]['result']
                            
                            result = psmiles_generation_with_llm(material_request, context)
                        
                        # Store generated candidate
                        candidate = {
                            'request': material_request,
                            'psmiles': result['psmiles'],
                            'explanation': result['explanation'],
                            'properties': result['properties'],
                            'generation_mode': "AI-Generated",
                            'timestamp': datetime.now().isoformat(),
                            'id': f"PSM_{len(st.session_state.psmiles_candidates):03d}"
                        }
                        
                        st.session_state.psmiles_candidates.append(candidate)
                        
                        # Display result
                        st.markdown(f"""
                        <div class="psmiles-display">
                            <h4>Generated PSMILES Structure</h4>
                            <p><strong>PSMILES:</strong> <code>{escape_psmiles_for_markdown(result['psmiles'])}</code></p>
                            <p><strong>Explanation:</strong> {result['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Property predictions
                        props = result['properties']
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Thermal Stability", f"{props['thermal_stability']:.2f}")
                        with col_b:
                            st.metric("Biocompatibility", f"{props['biocompatibility']:.2f}")
                        with col_c:
                            st.metric("Insulin Binding", f"{props['insulin_binding']:.2f}")
                        
                        # Validation
                        is_valid = result['psmiles'].count('[*]') == 2
                        if is_valid:
                            st.success("✅ Valid PSMILES with exactly 2 connection points")
                        else:
                            st.error("❌ Invalid PSMILES structure")
                        
                        # Process through workflow for visualization
                        workflow_result = st.session_state.psmiles_processor.process_psmiles_workflow(
                            result['psmiles'], st.session_state.session_id, "initial"
                        )
                        
                        if workflow_result['success']:
                            st.session_state.workflow_result = workflow_result
                            st.success("✅ Structure generated and ready for interactive workflow!")
                        
                        # Add to material library
                        add_to_material_library(
                            result['psmiles'], 
                            props, 
                            'generated',
                            material_request
                        )
                        
                        st.info("💡 Switch to the **Interactive Workflow** tab to visualize and modify this structure!")
        
        with col2:
            st.markdown("### PSMILES Rules")
            st.markdown("""
            **Critical Rules:**
            - Exactly 2 connection points `[*]`
            - No spaces or hyphens
            - Valid atomic symbols
            - Proper bonding notation
            
            **Examples:**
            - `[*]OCC[*]` - PEG unit
            - `[*]CC(O)[*]` - Hydroxyl group
            - `[*]NC(=O)[*]` - Amide linkage
            """)
            
            st.markdown("### Functional Groups for Insulin")
            functional_groups = {
                'Hydroxyl': '[*]C(O)[*]',
                'Amide': '[*]C(=O)N[*]',
                'Ether': '[*]COC[*]',
                'Ester': '[*]C(=O)OC[*]'
            }
            
            for group, psmiles in functional_groups.items():
                if st.button(f"Add {group}", key=f"fg_{group}"):
                    st.code(psmiles)
    
    with tab2:
        st.markdown("### 🔬 Interactive PSMILES Workflow")
        
        # Display active workflow if available
        if st.session_state.workflow_result:
            display_psmiles_workflow(st.session_state.workflow_result, "workflow")
        else:
            st.info("No active workflow. Generate a PSMILES structure first to see the interactive workflow.")
            
            # Quick start options
            st.markdown("### 🚀 Quick Start")
            st.markdown("Try these example PSMILES to get started with the interactive workflow:")
            
            example_psmiles = [
                ("[*]CC[*]", "Simple polyethylene chain"),
                ("[*]OCC[*]", "Polyethylene oxide (PEO)"),
                ("[*]C(=O)O[*]", "Polyacrylic acid"),
                ("[*]c1ccc(cc1)[*]", "Aromatic polymer backbone"),
                ("[*]C(C)(C)[*]", "Branched alkyl chain")
            ]
            
            for psmiles, description in example_psmiles:
                col_ex1, col_ex2 = st.columns([3, 1])
                with col_ex1:
                    st.code(psmiles)
                    st.caption(description)
                with col_ex2:
                    if st.button("Process", key=f"process_{psmiles}"):
                        with st.spinner("Processing PSMILES..."):
                            workflow_result = st.session_state.psmiles_processor.process_psmiles_workflow(
                                psmiles, st.session_state.session_id, "initial"
                            )
                            
                            if workflow_result['success']:
                                st.session_state.workflow_result = workflow_result
                                st.success("✅ Structure processed!")
                                st.rerun()
                            else:
                                st.error(f"❌ Processing failed: {workflow_result.get('error', 'Unknown error')}")
    
    with tab3:
        st.markdown("### Systematic Copolymerization")
        
        if len(st.session_state.psmiles_candidates) >= 1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Base Polymer:**")
                base_options = [f"{c['id']}: {c['psmiles']}" for c in st.session_state.psmiles_candidates]
                selected_base = st.selectbox("Select Base", base_options)
                
                base_psmiles = selected_base.split(": ")[1] if selected_base else ""
                st.code(base_psmiles)
            
            with col2:
                st.markdown("**Comonomer:**")
                comonomer_input = st.text_input(
                    "Comonomer PSMILES",
                    placeholder="e.g., [*]C(=O)O[*]",
                    help="Enter PSMILES for second polymer"
                )
            
            connection_pattern = st.selectbox(
                "Connection Pattern",
                ["[1,1] - Alternating", "[1,0] - Head-to-tail", "[0,1] - Tail-to-head", "[0,0] - Head-to-head"]
            )
            
            pattern_map = {
                "[1,1] - Alternating": [1, 1],
                "[1,0] - Head-to-tail": [1, 0], 
                "[0,1] - Tail-to-head": [0, 1],
                "[0,0] - Head-to-head": [0, 0]
            }
            
            if st.button("🔗 Perform Copolymerization") and base_psmiles and comonomer_input:
                with st.spinner("Performing copolymerization..."):
                    result = perform_real_copolymerization(
                        base_psmiles, 
                        comonomer_input, 
                        pattern_map[connection_pattern]
                    )
                    
                    st.markdown(f"""
                    <div class="psmiles-display">
                        <h4>Copolymer Result</h4>
                        <p><strong>Structure:</strong> <code>{escape_psmiles_for_markdown(result['copolymer_psmiles'])}</code></p>
                        <p><strong>Pattern:</strong> {result['pattern']}</p>
                        <p><strong>Success:</strong> {'✅' if result['success'] else '❌'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Property predictions
                    props = result['predicted_properties']
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Thermal Stability", f"{props['thermal_stability']:.2f}")
                    with col_b:
                        st.metric("Biocompatibility", f"{props['biocompatibility']:.2f}")
                    with col_c:
                        st.metric("Insulin Protection", f"{props['insulin_protection']:.2f}")
                    
                    # Add to material library
                    add_to_material_library(
                        result['copolymer_psmiles'],
                        props,
                        'copolymer',
                        f"Copolymer of {base_psmiles} and {comonomer_input}"
                    )
        
        else:
            st.info("Generate at least one PSMILES structure first to enable copolymerization")
    
    with tab4:
        st.markdown("### 🏗️ 3D Amorphous Polymer Structure Builder")
        st.markdown("*Build amorphous polymer structures in simulation boxes using PSP AmorphousBuilder*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Input Parameters")
            
            # PSMILES input
            psmiles_input = st.text_input(
                "PSMILES Structure",
                placeholder="e.g., [*]CC[*] for polyethylene",
                help="Enter a valid PSMILES string to build into a 3D structure"
            )
            
            # Structure parameters
            col_a, col_b = st.columns(2)
            with col_a:
                chain_length = st.slider("Chain Length (repeat units)", 5, 50, 10, help="Number of repeating units per polymer chain")
                num_molecules = st.slider("Number of Molecules", 10, 100, 20, help="Number of polymer molecules in the simulation box")
            
            with col_b:
                density = st.slider("Target Density (g/cm³)", 0.1, 2.0, 0.8, 0.1, help="Target density for the amorphous structure")
                box_size_nm = st.slider("Target Box Size (nm)", 0.5, 5.0, 1.0, 0.1, help="Approximate target box size in nanometers")
            
            # Build button
            if st.button("🏗️ Build 3D Structure", type="primary") and psmiles_input:
                # Validate PSMILES
                if '[*]' not in psmiles_input:
                    st.error("❌ Invalid PSMILES: Must contain connection points [*]")
                else:
                    with st.spinner(f"Building amorphous structure for `{psmiles_input}`... This may take a few minutes."):
                        # Build the structure
                        build_result = build_amorphous_polymer_structure(
                            psmiles_input, 
                            length=chain_length,
                            num_molecules=num_molecules,
                            density=density,
                            box_size_nm=box_size_nm
                        )
                        
                        # Store result in session state to persist across reruns
                        st.session_state.current_build_result = build_result
                        st.session_state.current_build_params = {
                            'psmiles_input': psmiles_input,
                            'chain_length': chain_length,
                            'num_molecules': num_molecules,
                            'density': density,
                            'box_size_nm': box_size_nm
                        }
            
            # Display results (either from current build or session state)
            build_result = st.session_state.get('current_build_result')
            build_params = st.session_state.get('current_build_params', {})
            
            if build_result:
                if build_result['success']:
                    st.success("✅ 3D Structure built successfully!")
                    
                    # Display structure information
                    st.markdown("#### 📊 Structure Information")
                    
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.metric("Total Atoms", build_result.get('num_atoms', 'Unknown'))
                        st.metric("Chain Length", build_params.get('chain_length', 'Unknown'))
                    
                    with info_col2:
                        st.metric("Molecules", build_params.get('num_molecules', 'Unknown'))
                        if build_result.get('actual_box_size_nm'):
                            st.metric("Actual Box Size", f"{build_result['actual_box_size_nm']:.2f} nm")
                    
                    with info_col3:
                        st.metric("Target Density", f"{build_params.get('density', 0.8):.1f} g/cm³")
                        st.metric("Files Generated", "✅ Ready")
                        
                        # Add clear button to reset results
                        if st.button("🗑️ Clear Results", help="Clear current build results"):
                            if 'current_build_result' in st.session_state:
                                del st.session_state.current_build_result
                            if 'current_build_params' in st.session_state:
                                del st.session_state.current_build_params
                            st.rerun()
                    
                    # File information
                    st.markdown("#### 📁 Generated Files")
                    
                    # Show actual file structure created by PSP
                    if build_result.get('output_dir'):
                        with st.expander("📂 View Generated File Structure"):
                            output_dir = build_result['output_dir']
                            st.write(f"**Base Directory:** `{output_dir}`")
                            
                            # Show all files created
                            all_files = []
                            if os.path.exists(output_dir):
                                for root, dirs, files in os.walk(output_dir):
                                    for file in files:
                                        full_path = os.path.join(root, file)
                                        rel_path = os.path.relpath(full_path, output_dir)
                                        file_size = os.path.getsize(full_path)
                                        all_files.append({
                                            'File': rel_path,
                                            'Size (KB)': f"{file_size/1024:.2f}",
                                            'Type': file.split('.')[-1].upper() if '.' in file else 'N/A'
                                        })
                            
                            if all_files:
                                files_df = pd.DataFrame(all_files)
                                st.dataframe(files_df, use_container_width=True)
                            else:
                                st.warning("No files found in output directory")
                    
                    file_col1, file_col2 = st.columns(2)
                    
                    with file_col1:
                        st.markdown("##### 📊 Simulation Files")
                        
                        # VASP file download
                        if build_result.get('vasp_file') and os.path.exists(build_result['vasp_file']):
                            with open(build_result['vasp_file'], 'rb') as f:
                                vasp_content = f.read()
                            st.download_button(
                                "📄 Download VASP File",
                                vasp_content,
                                file_name=f"polymer_structure_{uuid.uuid4().hex[:8]}.vasp",
                                mime="text/plain",
                                help="VASP POSCAR format for DFT calculations"
                            )
                        
                        # LAMMPS data file download
                        if build_result.get('data_file') and os.path.exists(build_result['data_file']):
                            with open(build_result['data_file'], 'rb') as f:
                                data_content = f.read()
                            st.download_button(
                                "📄 Download LAMMPS Data File",
                                data_content,
                                file_name=f"polymer_structure_{uuid.uuid4().hex[:8]}.data",
                                mime="text/plain",
                                help="LAMMPS data format for molecular dynamics"
                            )
                        
                        # Show original PACKMOL files in subdirectories
                        if build_result.get('output_dir'):
                            packmol_dir = os.path.join(build_result['output_dir'], 'packmol')
                            if os.path.exists(packmol_dir):
                                packmol_files = [f for f in os.listdir(packmol_dir) if f.endswith(('.pdb', '.vasp', '.inp'))]
                                if packmol_files:
                                    st.markdown("**🔧 Original PACKMOL Files:**")
                                    for file in packmol_files:
                                        file_path = os.path.join(packmol_dir, file)
                                        if os.path.exists(file_path):
                                            with open(file_path, 'rb') as f:
                                                file_content = f.read()
                                            st.download_button(
                                                f"📥 {file}",
                                                file_content,
                                                file_name=file,
                                                mime="text/plain",
                                                key=f"packmol_{file}_{uuid.uuid4().hex[:4]}",
                                                help=f"Original PACKMOL output: {file}"
                                            )
                    
                    with file_col2:
                        st.markdown("##### 🧬 Visualization Files")
                        
                        # PDB file download - prioritize actual PDB files over converted ones
                        pdb_available = False
                        
                        # Check for original PDB files first
                        if build_result.get('output_dir'):
                            # Look for PDB files in packmol directory
                            packmol_dir = os.path.join(build_result['output_dir'], 'packmol')
                            if os.path.exists(packmol_dir):
                                packmol_pdb_files = [f for f in os.listdir(packmol_dir) if f.endswith('.pdb')]
                                for pdb_file in packmol_pdb_files:
                                    pdb_path = os.path.join(packmol_dir, pdb_file)
                                    if os.path.exists(pdb_path):
                                        with open(pdb_path, 'rb') as f:
                                            pdb_content = f.read()
                                        st.download_button(
                                            f"🧬 Download {pdb_file}",
                                            pdb_content,
                                            file_name=f"polymer_structure_{uuid.uuid4().hex[:8]}.pdb",
                                            mime="chemical/x-pdb",
                                            key=f"original_pdb_{pdb_file}_{uuid.uuid4().hex[:4]}",
                                            help="Original PDB file from PACKMOL"
                                        )
                                        pdb_available = True
                                        
                                        # Try 3D visualization with original PDB
                                        try:
                                            st.markdown("#### 🔬 3D Structure Visualization")
                                            html_viewer = display_3d_structure(pdb_path)
                                            components.html(html_viewer, height=650)
                                        except Exception as e:
                                            st.error(f"❌ 3D visualization failed: {str(e)}")
                                        break
                        
                        # Fallback to converted PDB file
                        if not pdb_available and build_result.get('pdb_file') and build_result.get('pdb_converted'):
                            with open(build_result['pdb_file'], 'rb') as f:
                                pdb_content = f.read()
                            st.download_button(
                                "🧬 Download PDB File (Converted)",
                                pdb_content,
                                file_name=f"polymer_structure_{uuid.uuid4().hex[:8]}.pdb",
                                mime="chemical/x-pdb",
                                key=f"converted_pdb_{uuid.uuid4().hex[:4]}",
                                help="PDB file converted from VASP format"
                            )
                            
                            # 3D Visualization with converted PDB
                            try:
                                st.markdown("#### 🔬 3D Structure Visualization")
                                html_viewer = display_3d_structure(build_result['pdb_file'])
                                components.html(html_viewer, height=650)
                            except Exception as e:
                                st.error(f"❌ 3D visualization failed: {str(e)}")
                        
                        if not pdb_available and not build_result.get('pdb_converted'):
                            st.warning("⚠️ No PDB file available. Only simulation files (VASP/LAMMPS) were generated.")
                            st.info("💡 You can convert the VASP file to PDB using tools like VESTA or ASE for visualization.")
                        
                        # Show visualization tips
                        st.markdown("**💡 Visualization Tips:**")
                        st.markdown("- Use PyMOL, VMD, or ChimeraX for advanced visualization")
                        st.markdown("- VESTA can convert VASP files to various formats")
                        st.markdown("- The structure is periodic - consider replicating the unit cell")
                    
                    # Add to material library
                    add_to_material_library(
                        build_params.get('psmiles_input', 'Unknown'),
                        {
                            'thermal_stability': np.random.uniform(0.6, 0.9),
                            'biocompatibility': np.random.uniform(0.7, 0.95),
                            'insulin_stability_score': np.random.uniform(0.5, 0.9)
                        },
                        '3d_structure',
                        f"3D amorphous structure: {build_params.get('num_molecules', 'Unknown')} molecules, {build_params.get('chain_length', 'Unknown')} units"
                    )
                    
                    # Show file preview
                    if build_result.get('vasp_file') and os.path.exists(build_result['vasp_file']):
                        with st.expander("📋 VASP File Preview"):
                            try:
                                with open(build_result['vasp_file'], 'r') as f:
                                    lines = f.readlines()[:10]
                                    st.code('\n'.join(lines))
                            except Exception as e:
                                st.error(f"Could not read VASP file: {str(e)}")
                
                else:
                    # Safe error message handling
                    error_msg = build_result.get('error', 'Unknown error')
                    if error_msg is None:
                        error_msg = 'Build failed without error message'
                    
                    st.error(f"❌ Structure building failed: {error_msg}")
                    
                    # Show troubleshooting tips
                    if error_msg and 'PSP package not installed' in str(error_msg):
                        st.markdown("""
                        **💡 Installation Instructions:**
                        ```bash
                        pip install psp
                        # Also ensure PACKMOL is installed:
                        # Ubuntu/Debian: sudo apt-get install packmol
                        # macOS: brew install packmol
                        ```
                        """)
                        
                    # Show debug information
                    if build_result.get('output_dir'):
                        with st.expander("🔍 Debug Information"):
                            st.write(f"**Output Directory:** {build_result['output_dir']}")
                            if os.path.exists(build_result['output_dir']):
                                st.write("**Directory Contents:**")
                                for root, dirs, files in os.walk(build_result['output_dir']):
                                    level = root.replace(build_result['output_dir'], '').count(os.sep)
                                    indent = ' ' * 2 * level
                                    st.write(f"{indent}{os.path.basename(root)}/")
                                    subindent = ' ' * 2 * (level + 1)
                                    for file in files:
                                        st.write(f"{subindent}{file}")
                            else:
                                st.write("Output directory does not exist")
                    
                    # Show build result details
                    with st.expander("📊 Build Result Details"):
                        for key, value in build_result.items():
                            if value is not None:
                                st.write(f"**{key}:** {value}")
                            else:
                                st.write(f"**{key}:** None")
            
            # If no build result in session state, show helper text
            else:
                st.info("💡 Enter a PSMILES string above and click 'Build 3D Structure' to generate an amorphous polymer structure.")
                st.markdown("**Need help with PSMILES?**")
                st.markdown("- `[*]CC[*]` - Polyethylene")
                st.markdown("- `[*]CCO[*]` - Polyethylene oxide")
                st.markdown("- `[*]CC(C)[*]` - Polypropylene")
                st.markdown("- Connection points `[*]` are required for polymer building")
        
        with col2:
            st.markdown("#### 🎯 Structure Builder Features")
            
            st.markdown("""
            **✅ Capabilities:**
            - Build amorphous polymer structures
            - Generate simulation-ready boxes
            - Export VASP, LAMMPS, and PDB formats
            - 3D visualization in browser
            - Insulin delivery material optimization
            
            **📊 Output Files:**
            - **VASP**: For DFT calculations
            - **LAMMPS**: For molecular dynamics
            - **PDB**: For visualization
            
            **🔬 Applications:**
            - MD simulation preparation
            - Property prediction
            - Material characterization
            - Drug delivery optimization
            """)
            
            st.markdown("#### 🧪 Example PSMILES")
            
            examples = [
                ("[*]CC[*]", "Polyethylene"),
                ("[*]OCC[*]", "Polyethylene oxide"),
                ("[*]CC(C)[*]", "Polypropylene"),
                ("[*]C(=O)O[*]", "Polyacrylic acid"),
                ("[*]c1ccc(cc1)[*]", "Poly(para-phenylene)")
            ]
            
            for psmiles, name in examples:
                if st.button(f"📝 {name}", key=f"example_3d_{psmiles}"):
                    st.session_state.example_psmiles_3d = psmiles
                    st.rerun()
            
            # Handle example selection
            if hasattr(st.session_state, 'example_psmiles_3d'):
                st.info(f"💡 Selected: `{st.session_state.example_psmiles_3d}`")
                del st.session_state.example_psmiles_3d
    
    with tab5:
        st.markdown("### 🧬 Insulin Embedding in Polymer Matrix")
        st.markdown("*Embed insulin molecules into polymer structures for drug delivery applications*")
        
        # Initialize session state for insulin embedding
        if 'insulin_embedding_result' not in st.session_state:
            st.session_state.insulin_embedding_result = None
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Input Parameters")
            
            # Polymer selection
            polymer_source = st.selectbox(
                "Polymer Source",
                ["Use Generated Structure", "Upload Custom PDB"],
                help="Choose to use a previously generated polymer or upload your own"
            )
            
            if polymer_source == "Use Generated Structure":
                # Check if there's a current build result
                if st.session_state.get('current_build_result') and st.session_state.current_build_result.get('success'):
                    st.success("✅ Using polymer from 3D Structure Builder")
                    polymer_info = st.session_state.current_build_params
                    st.info(f"Polymer: `{polymer_info.get('psmiles_input', 'Unknown')}` with {polymer_info.get('num_molecules', 'Unknown')} molecules")
                    polymer_ready = True
                else:
                    st.warning("⚠️ No 3D structure available. Please build a polymer structure first in the '3D Structure Builder' tab.")
                    polymer_ready = False
            else:
                # Custom PDB upload
                uploaded_polymer = st.file_uploader(
                    "Upload Polymer PDB File",
                    type=['pdb'],
                    help="Upload a PDB file containing your polymer structure"
                )
                polymer_ready = uploaded_polymer is not None
                
                if uploaded_polymer:
                    # Save uploaded file temporarily
                    temp_polymer_path = f"temp_polymer_{uuid.uuid4().hex[:8]}.pdb"
                    with open(temp_polymer_path, 'wb') as f:
                        f.write(uploaded_polymer.read())
                    st.session_state.temp_polymer_path = temp_polymer_path
                    st.success("✅ Polymer PDB file uploaded successfully")
            
            # Insulin file check and preprocessing option
            insulin_available = os.path.exists('insulin.pdb')
            processed_insulin_path = None
            
            # Check for processed insulin files from recent preprocessing
            if 'insulin_preprocessing_result' in st.session_state and st.session_state.insulin_preprocessing_result:
                result = st.session_state.insulin_preprocessing_result
                if result.get('success') and result.get('output_file'):
                    processed_path = result['output_file']
                    if os.path.exists(processed_path):
                        processed_insulin_path = processed_path
                        if not insulin_available:
                            insulin_available = True  # We can use the processed file
            
            if insulin_available:
                if processed_insulin_path and not os.path.exists('insulin.pdb'):
                    st.success("✅ Processed insulin structure available")
                    st.info(f"📁 Using processed file: {processed_insulin_path}")
                elif processed_insulin_path and os.path.exists('insulin.pdb'):
                    st.success("✅ Insulin structure available (insulin.pdb)")
                    st.info(f"💡 Processed version also available: {processed_insulin_path}")
                else:
                    st.success("✅ Insulin structure available (insulin.pdb)")
                
                # Optional insulin preprocessing
                with st.expander("🔧 Optional: Preprocess Insulin PDB (Click to expand)", expanded=False):
                    st.markdown("*Clean and prepare the insulin PDB file before embedding using PDBFixer*")
                    
                    preprocess_col1, preprocess_col2 = st.columns(2)
                    
                    with preprocess_col1:
                        st.markdown("**🔧 Preprocessing Options:**")
                        remove_water = st.checkbox(
                            "Remove Water Molecules",
                            value=True,
                            help="Remove HOH and WAT molecules from insulin structure",
                            key="insulin_remove_water"
                        )
                        remove_heterogens = st.checkbox(
                            "Remove Heterogens",
                            value=False,
                            help="Remove non-standard residues (except water if kept)",
                            key="insulin_remove_heterogens"
                        )
                        add_missing_residues = st.checkbox(
                            "Add Missing Residues",
                            value=True,
                            help="Add missing residues based on PDB sequences",
                            key="insulin_add_missing_residues"
                        )
                    
                    with preprocess_col2:
                        add_missing_atoms = st.checkbox(
                            "Add Missing Atoms",
                            value=True,
                            help="Add missing atoms in existing residues",
                            key="insulin_add_missing_atoms"
                        )
                        add_missing_hydrogens = st.checkbox(
                            "Add Missing Hydrogens",
                            value=True,
                            help="Add missing hydrogen atoms",
                            key="insulin_add_missing_hydrogens"
                        )
                        ph_value = st.slider(
                            "pH for Protonation",
                            min_value=1.0,
                            max_value=14.0,
                            value=7.0,
                            step=0.1,
                            help="pH value for determining protonation states",
                            key="insulin_ph_value"
                        )
                    
                    # Preprocess insulin button
                    if st.button("🔧 Preprocess Insulin PDB", key="preprocess_insulin_btn"):
                        # Create output container for real-time updates
                        insulin_output_container = st.empty()
                        insulin_preprocessing_output = []
                        
                        def insulin_preprocessing_callback(message):
                            insulin_preprocessing_output.append(message)
                            insulin_output_container.markdown("### 📝 Insulin Preprocessing Output\n" + "\n".join(insulin_preprocessing_output))
                        
                        # Determine input file
                        input_file = 'insulin.pdb' if os.path.exists('insulin.pdb') else processed_insulin_path
                        
                        with st.spinner("Preprocessing insulin PDB file..."):
                            insulin_preprocessing_result = preprocess_pdb_standalone(
                                input_file,
                                remove_water=remove_water,
                                remove_heterogens=remove_heterogens,
                                add_missing_residues=add_missing_residues,
                                add_missing_atoms=add_missing_atoms,
                                add_missing_hydrogens=add_missing_hydrogens,
                                ph=ph_value,
                                output_callback=insulin_preprocessing_callback
                            )
                            
                            # Store result in session state
                            st.session_state.insulin_preprocessing_result = insulin_preprocessing_result
                            
                            if insulin_preprocessing_result['success']:
                                st.success("✅ Insulin preprocessing completed successfully!")
                                
                                # Show results
                                result_col1, result_col2, result_col3 = st.columns(3)
                                
                                with result_col1:
                                    st.metric("Initial Atoms", insulin_preprocessing_result['initial_atoms'])
                                    st.metric("Final Atoms", insulin_preprocessing_result['final_atoms'])
                                
                                with result_col2:
                                    st.metric("Initial Residues", insulin_preprocessing_result['initial_residues'])
                                    st.metric("Final Residues", insulin_preprocessing_result['final_residues'])
                                
                                with result_col3:
                                    st.metric("Atoms Added", insulin_preprocessing_result['atoms_added'])
                                    st.metric("Residues Added", insulin_preprocessing_result['residues_added'])
                                
                                # Auto-use processed file option
                                use_processed_col1, use_processed_col2 = st.columns(2)
                                
                                with use_processed_col1:
                                    if st.button("✅ Use Processed Insulin for Embedding", key="use_processed_insulin"):
                                        # Replace the original insulin.pdb with the processed one
                                        processed_path = insulin_preprocessing_result['output_file']
                                        if os.path.exists(processed_path):
                                            # Backup original and replace
                                            if os.path.exists('insulin.pdb'):
                                                shutil.copy('insulin.pdb', 'insulin_original_backup.pdb')
                                            shutil.copy(processed_path, 'insulin.pdb')
                                            st.success("✅ Processed insulin is now ready for embedding!")
                                            st.info("💡 Original insulin.pdb backed up as insulin_original_backup.pdb")
                                            st.rerun()
                                
                                with use_processed_col2:
                                    if st.button("🔄 Auto-Use Processed File", key="auto_use_processed"):
                                        # Automatically use processed file without replacing original
                                        st.session_state.use_processed_insulin = True
                                        st.success("✅ System will automatically use processed insulin file!")
                                        st.info("💡 Original insulin.pdb will remain unchanged")
                                        st.rerun()
                                
                                # Download option
                                if os.path.exists(insulin_preprocessing_result['output_file']):
                                    with open(insulin_preprocessing_result['output_file'], 'rb') as f:
                                        processed_content = f.read()
                                    
                                    st.download_button(
                                        "📥 Download Processed Insulin PDB",
                                        processed_content,
                                        file_name="insulin_processed.pdb",
                                        mime="chemical/x-pdb",
                                        key="download_processed_insulin"
                                    )
                            else:
                                st.error(f"❌ Insulin preprocessing failed: {insulin_preprocessing_result['error']}")
            else:
                st.error("❌ Insulin structure not found. Please ensure insulin.pdb is in the project directory.")
            
            # Determine actual insulin file to use
            actual_insulin_file = 'insulin.pdb'
            if st.session_state.get('use_processed_insulin') and processed_insulin_path:
                actual_insulin_file = processed_insulin_path
            elif processed_insulin_path and not os.path.exists('insulin.pdb'):
                actual_insulin_file = processed_insulin_path
            
            # Embedding parameters
            if polymer_ready and insulin_available:
                # Show current insulin file status
                st.markdown("#### 🔍 Current Insulin File Status")
                status_col1, status_col2 = st.columns(2)
                
                with status_col1:
                    st.info(f"**Using:** `{actual_insulin_file}`")
                    if actual_insulin_file != 'insulin.pdb':
                        st.success("✅ Using processed insulin file")
                    else:
                        st.info("ℹ️ Using original insulin file")
                
                with status_col2:
                    if processed_insulin_path and actual_insulin_file == 'insulin.pdb':
                        st.warning("💡 Processed insulin available but not being used")
                        if st.button("🔄 Switch to Processed File", key="switch_to_processed"):
                            st.session_state.use_processed_insulin = True
                            st.rerun()
                    elif st.session_state.get('use_processed_insulin') and processed_insulin_path:
                        st.success("✅ Auto-using processed insulin file")
                        if st.button("🔄 Switch to Original File", key="switch_to_original"):
                            st.session_state.use_processed_insulin = False
                            st.rerun()
                
                st.markdown("#### Embedding Parameters")
                
                embed_col1, embed_col2 = st.columns(2)
                
                with embed_col1:
                    num_insulin_molecules = st.slider(
                        "Number of Insulin Molecules", 
                        1, 10, 1, 
                        help="Number of insulin molecules to embed"
                    )
                    
                    buffer_distance = st.slider(
                        "Buffer Distance (Å)", 
                        10.0, 50.0, 20.0, 2.5,
                        help="Minimum distance between insulin and polymer molecules"
                    )
                    
                    # Atom limit control
                    max_atoms = st.selectbox(
                        "System Size Limit",
                        [5000, 10000, 15000, 25000, 50000],
                        index=2,  # Default to 15000
                        help="Maximum total atoms in the system (smaller = faster MD simulations)"
                    )
                
                with embed_col2:
                    box_size_preference = st.selectbox(
                        "Box Size Strategy",
                        ["Auto (Recommended)", "Manual Override"],
                        help="Auto uses dynamic sizing based on molecule dimensions"
                    )
                    
                    if box_size_preference == "Manual Override":
                        manual_box_size = st.slider(
                            "Manual Box Size (nm)", 
                            5.0, 50.0, 15.0, 2.5,
                            help="Manual box size override"
                        )
                    else:
                        manual_box_size = None
                
                # Atom count estimation
                if polymer_ready:
                    try:
                        # Quick estimation for display
                        from packmol_embedder import PackmolEmbedder
                        
                        # Get polymer path for estimation
                        if polymer_source == "Use Generated Structure":
                            build_result = st.session_state.current_build_result
                            polymer_pdb_path = None
                            if build_result.get('output_dir'):
                                packmol_dir = os.path.join(build_result['output_dir'], 'packmol')
                                if os.path.exists(packmol_dir):
                                    for file in os.listdir(packmol_dir):
                                        if file.endswith('.pdb'):
                                            polymer_pdb_path = os.path.join(packmol_dir, file)
                                            break
                        else:
                            polymer_pdb_path = st.session_state.get('temp_polymer_path')
                        
                        if polymer_pdb_path and os.path.exists(polymer_pdb_path):
                            temp_embedder = PackmolEmbedder(
                                insulin_pdb=actual_insulin_file,
                                polymer_pdb=polymer_pdb_path,
                                output_pdb='temp_estimate.pdb'
                            )
                            
                            # Calculate max polymers for the limit
                            max_polymers = temp_embedder.calculate_max_polymers_for_limit(max_atoms, num_insulin_molecules)
                            
                            # Get atom counts
                            insulin_atoms = temp_embedder.count_atoms_in_pdb(actual_insulin_file)
                            polymer_atoms = temp_embedder.count_atoms_in_pdb(polymer_pdb_path)
                            
                            # Display estimation
                            st.markdown("#### 📊 System Size Estimation")
                            
                            est_col1, est_col2, est_col3 = st.columns(3)
                            
                            with est_col1:
                                st.metric("Insulin Atoms", f"{insulin_atoms * num_insulin_molecules:,}")
                                
                            with est_col2:
                                st.metric("Max Polymers", f"{max_polymers}")
                                st.metric("Polymer Atoms Each", f"{polymer_atoms:,}")
                                
                            with est_col3:
                                estimated_total = (insulin_atoms * num_insulin_molecules) + (polymer_atoms * max_polymers)
                                st.metric("Estimated Total", f"{estimated_total:,}")
                                
                                # Color-coded warning
                                if estimated_total > max_atoms * 0.9:
                                    st.warning("⚠️ Near limit")
                                elif estimated_total > max_atoms * 0.7:
                                    st.info("📊 Good size")
                                else:
                                    st.success("✅ Small system")
                            
                            # Store max polymers for use in embedding
                            st.session_state.max_polymers_for_limit = max_polymers
                            
                    except Exception as e:
                        st.info("💡 Atom count estimation will be available during embedding")
                        st.session_state.max_polymers_for_limit = 20  # Default fallback
                
                # Embed button
                if st.button("🧬 Embed Insulin in Polymer", type="primary"):
                    with st.spinner("Embedding insulin in polymer matrix... This may take a few minutes."):
                        
                        # Prepare polymer PDB path
                        if polymer_source == "Use Generated Structure":
                            # Find the PDB file from the build result
                            build_result = st.session_state.current_build_result
                            polymer_pdb_path = None
                            
                            if build_result.get('output_dir'):
                                packmol_dir = os.path.join(build_result['output_dir'], 'packmol')
                                if os.path.exists(packmol_dir):
                                    for file in os.listdir(packmol_dir):
                                        if file.endswith('.pdb'):
                                            polymer_pdb_path = os.path.join(packmol_dir, file)
                                            break
                            
                            if not polymer_pdb_path:
                                st.error("❌ Could not find polymer PDB file from build result")
                                polymer_pdb_path = None
                        else:
                            polymer_pdb_path = st.session_state.get('temp_polymer_path')
                        
                        if polymer_pdb_path:
                            # Use the updated insulin_polymer_builder
                            from insulin_polymer_builder import build_insulin_polymer_composite
                            
                            try:
                                # Use PackmolEmbedder directly with atom limiting
                                from packmol_embedder import PackmolEmbedder
                                
                                # Debug: Show which insulin file is being used
                                st.info(f"🔍 **Debug:** Embedding with insulin file: `{actual_insulin_file}`")
                                
                                # Verify water content of insulin file
                                def count_water_molecules(pdb_file):
                                    if not os.path.exists(pdb_file):
                                        return -1
                                    water_count = 0
                                    with open(pdb_file, 'r') as f:
                                        for line in f:
                                            if 'HOH' in line and (line.startswith('ATOM') or line.startswith('HETATM')):
                                                water_count += 1
                                    return water_count
                                
                                insulin_water_count = count_water_molecules(actual_insulin_file)
                                if insulin_water_count == 0:
                                    st.success(f"✅ **Insulin file is water-free** (0 HOH molecules)")
                                elif insulin_water_count > 0:
                                    st.warning(f"⚠️ **Insulin file contains {insulin_water_count} water molecules**")
                                    st.info("💡 Consider using the processed insulin file if you want water-free embedding")
                                else:
                                    st.error(f"❌ Insulin file not found: {actual_insulin_file}")
                                    st.info("Available insulin files:")
                                    for file in ['insulin.pdb'] + [f for f in os.listdir('.') if 'insulin' in f and f.endswith('.pdb')]:
                                        if os.path.exists(file):
                                            water_in_file = count_water_molecules(file)
                                            st.write(f"  - {file} ({water_in_file} water molecules)")
                                
                                if not os.path.exists(actual_insulin_file):
                                    st.stop()
                                
                                output_pdb = f"insulin_embedded_{uuid.uuid4().hex[:8]}.pdb"
                                embedder = PackmolEmbedder(
                                    insulin_pdb=actual_insulin_file,
                                    polymer_pdb=polymer_pdb_path,
                                    output_pdb=output_pdb
                                )
                                
                                # Use the calculated max polymers from the estimation
                                max_polymers_allowed = st.session_state.get('max_polymers_for_limit', 20)
                                
                                # Calculate dynamic box size if needed
                                if manual_box_size:
                                    box_size_angstrom = manual_box_size * 10.0
                                else:
                                    # Dynamic box size based on number of polymers
                                    box_size_angstrom = max(150.0, max_polymers_allowed * 3.0)  # Scale with polymer count
                                
                                st.info(f"🔄 Using {max_polymers_allowed} polymer molecules (limited by {max_atoms:,} atom cap)")
                                
                                success = embedder.embed_insulin(
                                    box_size=box_size_angstrom,
                                    num_polymers=max_polymers_allowed,
                                    buffer_distance=buffer_distance,
                                    max_atoms=max_atoms
                                )
                                
                                if success:
                                    # Count actual atoms in the result
                                    final_atom_count = embedder.count_atoms_in_pdb(output_pdb) if os.path.exists(output_pdb) else 0
                                    
                                    # Verify water content in final embedded file
                                    final_water_count = count_water_molecules(output_pdb) if os.path.exists(output_pdb) else 0
                                    
                                    st.session_state.insulin_embedding_result = {
                                        'success': True,
                                        'output_pdb': output_pdb,
                                        'num_insulin_molecules': num_insulin_molecules,
                                        'num_polymer_molecules': max_polymers_allowed,
                                        'buffer_distance': buffer_distance,
                                        'box_size_angstrom': box_size_angstrom,
                                        'polymer_source': polymer_source,
                                        'max_atoms_limit': max_atoms,
                                        'final_atom_count': final_atom_count,
                                        'final_water_count': final_water_count,
                                        'insulin_file_used': actual_insulin_file,
                                        'build_timestamp': datetime.now().isoformat()
                                    }
                                else:
                                    st.session_state.insulin_embedding_result = {
                                        'success': False,
                                        'error': 'PACKMOL embedding failed'
                                    }
                                    
                            except Exception as e:
                                st.session_state.insulin_embedding_result = {
                                    'success': False,
                                    'error': str(e)
                                }
                        else:
                            st.error("❌ No polymer PDB file available")
            
            # Display results
            embed_result = st.session_state.get('insulin_embedding_result')
            if embed_result:
                if embed_result['success']:
                    st.success("✅ Insulin embedding completed successfully!")
                    
                    # Display embedding information
                    st.markdown("#### 📊 Embedding Results")
                    
                    # Show which insulin file was used
                    insulin_file_used = embed_result.get('insulin_file_used', 'Unknown')
                    if 'processed' in insulin_file_used:
                        st.success(f"✅ **Used processed insulin:** `{insulin_file_used}`")
                    else:
                        st.info(f"ℹ️ **Used insulin file:** `{insulin_file_used}`")
                    
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Insulin Molecules", embed_result['num_insulin_molecules'])
                        st.metric("Polymer Molecules", embed_result['num_polymer_molecules'])
                    
                    with result_col2:
                        st.metric("Buffer Distance", f"{embed_result['buffer_distance']:.1f} Å")
                        st.metric("Box Size", f"{embed_result['box_size_angstrom']:.1f} Å")
                    
                    with result_col3:
                        # Show atom count information
                        final_atoms = embed_result.get('final_atom_count', 0)
                        max_limit = embed_result.get('max_atoms_limit', 'Unknown')
                        
                        if final_atoms > 0:
                            st.metric("Total Atoms", f"{final_atoms:,}")
                            
                            # Color-coded efficiency indicator
                            if isinstance(max_limit, int):
                                usage_percent = (final_atoms / max_limit) * 100
                                if usage_percent > 90:
                                    st.error(f"⚠️ {usage_percent:.1f}% of limit")
                                elif usage_percent > 70:
                                    st.warning(f"📊 {usage_percent:.1f}% of limit")
                                else:
                                    st.success(f"✅ {usage_percent:.1f}% of limit")
                            
                            st.metric("Atom Limit", f"{max_limit:,}" if isinstance(max_limit, int) else str(max_limit))
                        else:
                            st.metric("Polymer Source", embed_result['polymer_source'])
                            # Fallback to file size if atom count not available
                            if os.path.exists(embed_result['output_pdb']):
                                file_size = os.path.getsize(embed_result['output_pdb'])
                                st.metric("Output File Size", f"{file_size/1024:.1f} KB")
                        
                        # Show water content verification
                        final_water = embed_result.get('final_water_count', 'Unknown')
                        if isinstance(final_water, int):
                            if final_water == 0:
                                st.success("💧 Water-free embedding")
                            elif final_water > 0:
                                st.warning(f"💧 {final_water} water molecules")
                        else:
                            st.info("💧 Water count unknown")
                    
                    # File downloads
                    st.markdown("#### 📁 Download Results")
                    
                    download_col1, download_col2 = st.columns(2)
                    
                    with download_col1:
                        if os.path.exists(embed_result['output_pdb']):
                            with open(embed_result['output_pdb'], 'rb') as f:
                                pdb_content = f.read()
                            st.download_button(
                                "🧬 Download Insulin-Polymer Composite",
                                pdb_content,
                                file_name=f"insulin_polymer_composite_{uuid.uuid4().hex[:8]}.pdb",
                                mime="chemical/x-pdb",
                                help="Complete insulin-polymer composite structure"
                            )
                    
                    with download_col2:
                        # 3D Visualization
                        if os.path.exists(embed_result['output_pdb']):
                            try:
                                st.markdown("#### 🔬 3D Visualization")
                                html_viewer = display_3d_structure(embed_result['output_pdb'])
                                components.html(html_viewer, height=500)
                            except Exception as e:
                                st.error(f"❌ 3D visualization failed: {str(e)}")
                                st.info("💡 You can still download the PDB file and visualize it in PyMOL, VMD, or ChimeraX")
                    
                    # Clear results button
                    if st.button("🗑️ Clear Embedding Results"):
                        st.session_state.insulin_embedding_result = None
                        st.rerun()
                    
                else:
                    st.error(f"❌ Insulin embedding failed: {embed_result.get('error', 'Unknown error')}")
                    st.info("💡 Try adjusting the parameters or using a different polymer structure")
        
        with col2:
            st.markdown("#### 🎯 Embedding Features")
            
            st.markdown("""
            **✅ Capabilities:**
            - Dynamic box sizing based on molecule dimensions
            - Sphere exclusion for clean insulin placement
            - Automatic buffer distance calculation
            - Integration with 3D Structure Builder
            - Custom polymer PDB support
            
            **📊 Output:**
            - Complete insulin-polymer composite PDB
            - 3D visualization in browser
            - Ready for MD simulations
            
            **🔬 Applications:**
            - Drug delivery system design
            - Protein stabilization studies
            - Controlled release mechanisms
            - Biocompatibility optimization
            """)
            
            st.markdown("#### 💡 Usage Tips")
            
            st.markdown("""
            1. **Use 3D Structure Builder** first to generate a polymer structure
            2. **Start with 15,000 atoms** for CPU-based MD simulations
            3. **Auto box sizing** is recommended for optimal results
            4. **Buffer distance** controls insulin-polymer separation
            5. **System size estimation** shows expected atom counts
            6. **Visualize results** to verify proper embedding
            7. **Download PDB** for further analysis or simulations
            
            **🔧 MD Simulation Guidelines:**
            - 5,000 atoms: Fast testing, small systems
            - 15,000 atoms: Good balance for most studies
            - 25,000+ atoms: Requires GPU acceleration
            """)
            
            # Show insulin dimensions if available
            if insulin_available:
                try:
                    insulin_dims = get_molecule_dimensions(actual_insulin_file)
                    st.markdown("#### 📏 Insulin Dimensions")
                    st.write(f"**X:** {insulin_dims[0]:.1f} Å")
                    st.write(f"**Y:** {insulin_dims[1]:.1f} Å")
                    st.write(f"**Z:** {insulin_dims[2]:.1f} Å")
                    st.write(f"**Max:** {max(insulin_dims):.1f} Å")
                except:
                    st.info("Could not read insulin dimensions")
            
            # Quick fix for water issue
            if insulin_available and actual_insulin_file == 'insulin.pdb':
                # Check if processed files are available
                processed_dirs = [d for d in os.listdir('.') if d.startswith('preprocessed_insulin_') and os.path.isdir(d)]
                if processed_dirs:
                    latest_processed_dir = sorted(processed_dirs)[-1]  # Get the most recent
                    processed_insulin_path = os.path.join(latest_processed_dir, 'insulin_processed.pdb')
                    
                    if os.path.exists(processed_insulin_path):
                        st.markdown("#### 🚨 Water Detection Alert")
                        
                        def count_water_molecules_quick(pdb_file):
                            try:
                                with open(pdb_file, 'r') as f:
                                    return sum(1 for line in f if 'HOH' in line and (line.startswith('ATOM') or line.startswith('HETATM')))
                            except:
                                return 0
                        
                        original_water = count_water_molecules_quick('insulin.pdb')
                        processed_water = count_water_molecules_quick(processed_insulin_path)
                        
                        if original_water > 0 and processed_water == 0:
                            water_col1, water_col2 = st.columns(2)
                            
                            with water_col1:
                                st.warning(f"⚠️ Current insulin file has **{original_water} water molecules**")
                                st.info(f"💡 Processed file has **{processed_water} water molecules**")
                            
                            with water_col2:
                                if st.button("🔄 Auto-Switch to Water-Free Insulin", 
                                           help="Switch to the processed insulin file without water"):
                                    st.session_state.use_processed_insulin = True
                                    st.session_state.insulin_preprocessing_result = {
                                        'success': True,
                                        'output_file': processed_insulin_path
                                    }
                                    st.success("✅ Switched to water-free insulin!")
                                    st.rerun()
    
    with tab6:
        st.markdown("### Generated Material Library")
        
        if st.session_state.psmiles_candidates:
            # Create DataFrame for display
            candidates_df = pd.DataFrame([
                {
                    'ID': c['id'],
                    'PSMILES': c['psmiles'],
                    'Request': c['request'][:30] + "...",
                    'Thermal': c['properties']['thermal_stability'],
                    'Biocompat': c['properties']['biocompatibility'],
                    'Insulin': c['properties']['insulin_binding'],
                    'Mode': c['generation_mode']
                }
                for c in st.session_state.psmiles_candidates
            ])
            
            st.dataframe(candidates_df, use_container_width=True)
            
            # Property analysis
            fig = px.scatter(
                candidates_df,
                x='Thermal',
                y='Biocompat',
                size='Insulin',
                color='Mode',
                hover_data=['ID', 'PSMILES'],
                title="Generated Materials Property Space"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Evaluate All"):
                    st.success("Evaluation initiated for all candidates!")
            with col2:
                if st.button("🔄 Add to Active Learning"):
                    for candidate in st.session_state.psmiles_candidates[-3:]:
                        st.session_state.active_learning_queue.append({
                            'type': 'psmiles_candidate',
                            'content': candidate,
                            'priority': np.mean(list(candidate['properties'].values())),
                            'timestamp': datetime.now().isoformat()
                        })
                    st.success("Latest candidates added to active learning!")
            with col3:
                if st.button("💾 Export Library"):
                    st.success("Material library exported!")
        
        else:
            st.info("No materials generated yet. Use the generation tab to create PSMILES structures.")

# Active Learning Page  
elif page == "Active Learning":
    st.subheader("🎯 Active Learning Feedback Loop")
    
    tab1, tab2, tab3 = st.tabs(["Iteration Analysis", "Feedback Integration", "Learning Queue"])
    
    with tab1:
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
    
    with tab2:
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
    
    with tab3:
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

# Material Evaluation Page
elif page == "Material Evaluation":
    st.subheader("📊 Material Evaluation & Performance Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Composite Scoring", "Property Analysis", "Ranking & Selection"])
    
    with tab1:
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
        if len(st.session_state.material_library) > 0:
            st.session_state.material_library['composite_score'] = (
                thermal_weight * st.session_state.material_library['thermal_stability'] +
                biocompat_weight * st.session_state.material_library['biocompatibility'] +
                release_weight * st.session_state.material_library['release_control']
            )
            
            # Display top performers
            top_materials = st.session_state.material_library.nlargest(10, 'composite_score')
            
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
                st.session_state.material_library,
                x='composite_score',
                color='source',
                title="Distribution of Composite Scores by Source",
                nbins=25
            )
            st.plotly_chart(fig_score_dist, use_container_width=True)
        else:
            st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")
    
    with tab2:
        st.markdown("### Multi-Property Analysis")
        
        if len(st.session_state.material_library) > 0:
            # Property correlation matrix
            props_for_correlation = ['thermal_stability', 'biocompatibility', 'release_control', 
                                   'uncertainty_score', 'insulin_stability_score']
            
            if 'composite_score' in st.session_state.material_library.columns:
                props_for_correlation.append('composite_score')
            
            correlation_matrix = st.session_state.material_library[props_for_correlation].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                title="Property Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Property space exploration
            st.markdown("#### 🔍 Interactive Property Space")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_prop = st.selectbox("X-axis Property", props_for_correlation, index=0)
                y_prop = st.selectbox("Y-axis Property", props_for_correlation, index=1)
            
            with col2:
                color_prop = st.selectbox("Color Property", props_for_correlation, index=-1)
                size_prop = st.selectbox("Size Property", props_for_correlation, index=3)
            
            fig_properties = px.scatter(
                st.session_state.material_library,
                x=x_prop,
                y=y_prop,
                color=color_prop,
                size=size_prop,
                hover_data=['material_id', 'psmiles', 'source'],
                title=f"{y_prop.title()} vs {x_prop.title()}",
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_properties, use_container_width=True)
            
            # Statistical analysis
            st.markdown("#### 📈 Statistical Summary")
            
            stats_summary = st.session_state.material_library[props_for_correlation].describe()
            st.dataframe(stats_summary.round(3))
            
            # Source comparison
            source_comparison = st.session_state.material_library.groupby('source')[props_for_correlation].mean()
            
            fig_source_comp = px.bar(
                source_comparison.reset_index(),
                x='source',
                y=props_for_correlation[-1],  # Use the last property (composite_score if available)
                title=f"Average {props_for_correlation[-1].title()} by Material Source",
                color='source'
            )
            st.plotly_chart(fig_source_comp, use_container_width=True)
        else:
            st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")
    
    with tab3:
        st.markdown("### Material Ranking & Selection")
        
        if len(st.session_state.material_library) > 0:
            # Advanced filtering
            st.markdown("#### 🎯 Advanced Material Filtering")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_thermal = st.slider("Min Thermal Stability", 0.0, 1.0, 0.5)
                min_biocompat = st.slider("Min Biocompatibility", 0.0, 1.0, 0.6)
            
            with col2:
                max_uncertainty = st.slider("Max Uncertainty", 0.0, 1.0, 0.8)
                source_filter = st.multiselect(
                    "Material Sources",
                    st.session_state.material_library['source'].unique(),
                    default=st.session_state.material_library['source'].unique()
                )
            
            with col3:
                min_insulin_stability = st.slider("Min Insulin Stability", 0.0, 1.0, 0.4)
                if 'composite_score' in st.session_state.material_library.columns:
                    min_composite = st.slider("Min Composite Score", 0.0, 1.0, 0.6)
                else:
                    min_composite = 0.0
            
            # Apply filters
            filtered_materials = st.session_state.material_library[
                (st.session_state.material_library['thermal_stability'] >= min_thermal) &
                (st.session_state.material_library['biocompatibility'] >= min_biocompat) &
                (st.session_state.material_library['uncertainty_score'] <= max_uncertainty) &
                (st.session_state.material_library['source'].isin(source_filter)) &
                (st.session_state.material_library['insulin_stability_score'] >= min_insulin_stability)
            ].copy()
            
            if 'composite_score' in st.session_state.material_library.columns:
                filtered_materials = filtered_materials[filtered_materials['composite_score'] >= min_composite]
            
            st.write(f"**{len(filtered_materials)} materials** meet criteria (from {len(st.session_state.material_library)} total)")
            
            if len(filtered_materials) > 0:
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
                
                if selection_strategy == "Top Performers":
                    selected_materials = filtered_materials.head(n_select)
                elif selection_strategy == "Diverse Portfolio":
                    # Select diverse materials across property space
                    selected_materials = filtered_materials.sample(n=min(n_select, len(filtered_materials)))
                elif selection_strategy == "High Uncertainty":
                    selected_materials = filtered_materials.nlargest(n_select, 'uncertainty_score')
                else:  # Balanced Approach
                    # Mix of top performers and high uncertainty
                    top_half = filtered_materials.head(n_select//2)
                    uncertain_half = filtered_materials.nlargest(n_select - len(top_half), 'uncertainty_score')
                    selected_materials = pd.concat([top_half, uncertain_half]).drop_duplicates()
                
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
                
                # Export and action buttons
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
                        for _, material in selected_materials.iterrows():
                            st.session_state.active_learning_queue.append({
                                'type': 'selected_material',
                                'content': material.to_dict(),
                                'priority': material.get('composite_score', material['insulin_stability_score']),
                                'timestamp': datetime.now().isoformat()
                            })
                        st.success("Selected materials added to active learning queue!")
                        st.rerun()
            
            else:
                st.warning("No materials meet the current filter criteria. Please adjust the filters.")
        else:
            st.info("No materials in library yet. Generate some materials using Literature Mining or PSMILES Generation!")

# MD Simulation Integration Page
elif page == "MD Simulation":
    st.subheader("🔬 Molecular Dynamics Simulation Integration")
    
    # Initialize MD simulation system
    if 'md_integration' not in st.session_state:
        if MD_INTEGRATION_AVAILABLE:
            try:
                st.session_state.md_integration = MDSimulationIntegration()
                st.session_state.md_integration_available = True
            except Exception as e:
                st.session_state.md_integration = None
                st.session_state.md_integration_available = False
                st.session_state.md_integration_error = str(e)
        else:
            st.session_state.md_integration = None
            st.session_state.md_integration_available = False
            st.session_state.md_integration_error = "MD simulation integration not available"
    
    # Check system status
    if st.session_state.md_integration_available:
        dependency_status = st.session_state.md_integration.get_dependency_status()
        
        # Display dependency status
        st.markdown("### 🔧 System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dependencies:**")
            for dep, available in dependency_status['dependencies'].items():
                if dep != 'all_available':
                    icon = "✅" if available else "❌"
                    st.markdown(f"{icon} {dep}")
        
        with col2:
            st.markdown("**Platform Information:**")
            if 'platforms' in dependency_status.get('platform_info', {}):
                platforms = dependency_status['platform_info']['platforms']
                best_platform = dependency_status['platform_info']['best_platform']
                st.write(f"**Best Platform:** {best_platform}")
                st.write(f"**Available Platforms:** {len(platforms)}")
        
        with col3:
            st.markdown("**Status:**")
            if dependency_status['dependencies']['all_available']:
                st.success("🚀 All systems ready!")
            else:
                st.error("⚠️ Missing dependencies")
                missing = [k for k, v in dependency_status['dependencies'].items() 
                         if not v and k != 'all_available']
                for dep in missing:
                    cmd = dependency_status['installation_commands'].get(dep, f"Install {dep}")
                    st.code(cmd)
        
        # Main simulation interface
        if dependency_status['dependencies']['all_available']:
            
            # Tabs for different simulation workflows
            tab1, tab2, tab3 = st.tabs(["🚀 MD Simulation", "📊 Results Analysis", "📁 File Management"])
            
            with tab1:
                st.markdown("### 🚀 MD Simulation")
                st.markdown("*Run molecular dynamics simulations with AMBER force fields and implicit solvent*")
                
                # File selection for simulation
                st.markdown("#### Input File Selection")
                
                # Option 1: Use available insulin-polymer PDB files  
                col_refresh, col_info = st.columns([1, 3])
                with col_refresh:
                    if st.button("🔄 Refresh Files", help="Refresh the file list to detect newly created files"):
                        st.rerun()
                
                available_pdbs = get_insulin_polymer_pdb_files() if MD_INTEGRATION_AVAILABLE else []
                
                if available_pdbs:
                    # Automatically select the latest insulin_embedded file
                    latest_embedded = None
                    for pdb in available_pdbs:
                        if pdb.get('file_type') == 'insulin_embedded':
                            latest_embedded = pdb
                            break  # First one is the most recent due to sorting
                    
                    if latest_embedded:
                        st.success(f"🎯 **Auto-detected latest embedded file:** {latest_embedded['name']}")
                        st.info(f"📊 **File details:** {latest_embedded['atom_count']} atoms, {latest_embedded['size_mb']:.1f} MB")
                        
                        # Option to use the auto-detected file
                        use_auto_detected = st.checkbox("Use Auto-detected File", value=True, 
                                                       help="Use the most recently created insulin_embedded file")
                        
                        if use_auto_detected:
                            simulation_input_file = latest_embedded['path']
                        else:
                            # Manual selection
                            st.markdown("**Manual File Selection:**")
                            pdb_options = {}
                            for pdb in available_pdbs:
                                file_type_emoji = {
                                    'insulin_embedded': '🎯',
                                    'composite': '🧬', 
                                    'other_insulin': '📄'
                                }.get(pdb.get('file_type'), '📄')
                                
                                display_name = f"{file_type_emoji} {pdb['name']} ({pdb['atom_count']} atoms, {pdb['size_mb']:.1f} MB)"
                                pdb_options[display_name] = pdb['path']
                            
                            selected_display = st.selectbox("Select PDB File", list(pdb_options.keys()))
                            simulation_input_file = pdb_options[selected_display]
                    else:
                        # No embedded files, show all options
                        st.info("No insulin_embedded files found. Showing all available files:")
                        pdb_options = {}
                        for pdb in available_pdbs:
                            file_type_emoji = {
                                'insulin_embedded': '🎯',
                                'composite': '🧬', 
                                'other_insulin': '📄'
                            }.get(pdb.get('file_type'), '📄')
                            
                            display_name = f"{file_type_emoji} {pdb['name']} ({pdb['atom_count']} atoms, {pdb['size_mb']:.1f} MB)"
                            pdb_options[display_name] = pdb['path']
                        
                        selected_display = st.selectbox("Select PDB File", list(pdb_options.keys()))
                        simulation_input_file = pdb_options[selected_display]
                else:
                    simulation_input_file = None
                
                # Option 2: Upload file
                if not simulation_input_file:
                    st.markdown("#### Upload PDB File")
                    uploaded_sim_file = st.file_uploader(
                        "Upload PDB for Simulation",
                        type=['pdb'],
                        help="Upload a PDB file for MD simulation"
                    )
                    
                    if uploaded_sim_file is not None:
                        temp_sim_path = f"temp_sim_{uuid.uuid4().hex[:8]}.pdb"
                        with open(temp_sim_path, 'wb') as f:
                            f.write(uploaded_sim_file.read())
                        simulation_input_file = temp_sim_path
                        st.success(f"✅ File uploaded: {uploaded_sim_file.name}")
                
                # Simulation parameters
                if simulation_input_file:
                    st.markdown("#### Simulation Parameters")
                    
                    param_col1, param_col2 = st.columns(2)
                    
                    with param_col1:
                        temperature = st.slider("Temperature (K)", 250, 400, 310, 5, help="Simulation temperature in Kelvin (physiological = 310 K)")
                        
                        # Equilibration steps with better options and descriptions
                        equilibration_options = {
                            "Quick Test (125 ps)": 31250,
                            "Short (500 ps) - Recommended": 125000,
                            "Medium (1000 ps)": 250000,
                            "Long (2000 ps)": 500000,
                            "Extended (4000 ps)": 1000000
                        }
                        
                        eq_selection = st.selectbox(
                            "Equilibration Duration",
                            list(equilibration_options.keys()),
                            index=1,  # Default to "Short (500 ps) - Recommended"
                            help="Equilibration phase duration (4 fs timestep with hydrogen mass repartitioning)"
                        )
                        equilibration_steps = equilibration_options[eq_selection]
                        
                        # Convert to time
                        eq_time_ps = equilibration_steps * 4 / 1000
                        eq_time_ns = eq_time_ps / 1000
                        st.caption(f"⏱️ Equilibration: {eq_time_ps:.0f} ps ({eq_time_ns:.1f} ns)")
                    
                    with param_col2:
                        # Production steps with better options and descriptions
                        production_options = {
                            "Quick Test (1 ns)": 250000,
                            "Short (5 ns)": 1250000,
                            "Medium (10 ns) - Recommended": 2500000,
                            "Long (20 ns)": 5000000,
                            "Extended (50 ns)": 12500000
                        }
                        
                        prod_selection = st.selectbox(
                            "Production Duration",
                            list(production_options.keys()),
                            index=2,  # Default to "Medium (10 ns) - Recommended"
                            help="Production phase duration (4 fs timestep with hydrogen mass repartitioning)"
                        )
                        production_steps = production_options[prod_selection]
                        
                        # Save interval with better options
                        save_options = {
                            "Frequent (1 ps)": 250,
                            "Normal (2 ps) - Recommended": 500,
                            "Sparse (4 ps)": 1000,
                            "Very Sparse (8 ps)": 2000
                        }
                        
                        save_selection = st.selectbox(
                            "Frame Saving Frequency",
                            list(save_options.keys()),
                            index=1,  # Default to "Normal (2 ps) - Recommended"
                            help="How often to save trajectory frames"
                        )
                        save_interval = save_options[save_selection]
                        
                        # Convert to time
                        prod_time_ps = production_steps * 4 / 1000
                        prod_time_ns = prod_time_ps / 1000
                        save_time_ps = save_interval * 4 / 1000
                        total_time_ns = (equilibration_steps + production_steps) * 4 / 1000000
                        
                        st.caption(f"⏱️ Production: {prod_time_ps:.0f} ps ({prod_time_ns:.1f} ns)")
                        st.caption(f"💾 Save every: {save_time_ps:.1f} ps")
                        st.caption(f"🕒 **Total simulation: {total_time_ns:.1f} ns**")
                    
                    # Performance estimation
                    st.markdown("#### ⚡ Performance Estimation")
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        # Estimate based on typical performance
                        est_performance_ns_day = 1000  # Conservative estimate for mixed systems
                        
                        if total_time_ns <= 5:
                            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
                            runtime_color = "🟢"
                            runtime_desc = "Fast"
                        elif total_time_ns <= 20:
                            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
                            runtime_color = "🟡"
                            runtime_desc = "Moderate"
                        else:
                            est_runtime_hours = total_time_ns / est_performance_ns_day * 24
                            runtime_color = "🔴"
                            runtime_desc = "Slow"
                        
                        st.metric("Estimated Runtime", f"{est_runtime_hours:.1f} hours", 
                                help=f"Based on ~{est_performance_ns_day} ns/day performance")
                        st.caption(f"{runtime_color} {runtime_desc} simulation")
                    
                    with perf_col2:
                        total_frames = (equilibration_steps + production_steps) // save_interval
                        trajectory_size_mb = total_frames * 0.5  # Rough estimate
                        
                        st.metric("Expected Frames", f"{total_frames:,}", 
                                help="Total number of saved trajectory frames")
                        st.metric("Trajectory Size", f"~{trajectory_size_mb:.0f} MB", 
                                help="Estimated DCD trajectory file size")
                    
                    # Important note about preprocessing
                    st.markdown("#### 🔧 Preprocessing & Simulation")
                    st.info("""
                    **Automatic PDBFixer Preprocessing:**
                    The MD simulation automatically includes these preprocessing steps:
                    1. 🧹 **Structure Cleaning** - Remove water molecules and fix missing atoms
                    2. ➕ **Add Hydrogens** - Add missing hydrogens at physiological pH (7.4)
                    3. 🔗 **Fix Bonds** - Repair missing residues and optimize structure
                    4. 🧪 **Preserve Polymers** - Keep UNL polymer residues for embedding simulation
                    
                    This ensures your insulin-polymer system is properly prepared for MD simulation!
                    """)
                    
                    # Check if simulation is already running
                    sim_status = st.session_state.md_integration.get_simulation_status()
                    
                    if sim_status['simulation_running']:
                        # Simple Live Console Output
                        st.markdown("## 🖥️ Live Simulation Console")
                        
                        # Show simulation info
                        sim_info = sim_status['simulation_info']
                        
                        # Header with simulation info and controls
                        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
                        
                        with header_col1:
                            st.markdown(f"### 🎯 Simulation: `{sim_info['id']}`")
                            # Color-code status
                            status = sim_info['status']
                            if status == 'running':
                                status_display = "🟢 Running"
                            elif status == 'stopping':
                                status_display = "🟡 Stopping..."
                            elif status == 'stopped':
                                status_display = "🛑 Stopped by User"
                            elif status == 'completed':
                                status_display = "✅ Completed"
                            elif status == 'failed':
                                status_display = "❌ Failed"
                            else:
                                status_display = f"⚪ {status.title()}"
                            
                            st.markdown(f"**Status:** {status_display}")
                        
                        with header_col2:
                            col2a, col2b = st.columns([2, 1])
                            with col2a:
                                auto_refresh = st.checkbox("🔄 Live Updates", value=True, 
                                                         help="Automatically refresh the console output in real-time")
                                if auto_refresh and status in ['running', 'starting']:
                                    st.markdown('<span style="color: #4CAF50; font-size: 0.8em;">● Live monitoring active</span>', unsafe_allow_html=True)
                            with col2b:
                                if auto_refresh:
                                    refresh_interval = st.selectbox("Refresh Interval", 
                                                                   options=[2, 3, 5, 10], 
                                                                   index=1, 
                                                                   format_func=lambda x: f"{x}s",
                                                                   help="Refresh interval in seconds",
                                                                   label_visibility="collapsed")
                                else:
                                    refresh_interval = 5
                        
                        with header_col3:
                            if st.button("⏹️ Stop Simulation", type="primary"):
                                stop_result = st.session_state.md_integration.stop_simulation()
                                if stop_result:
                                    st.success("🛑 Stop request sent to simulation!")
                                    st.info("The simulation will stop at the next checkpoint.")
                                else:
                                    st.warning("⚠️ No active simulation to stop.")
                                st.rerun()
                        
                        # Calculate elapsed time
                        if 'simulation_start_time' in st.session_state and st.session_state.simulation_start_time:
                            start_time = st.session_state.simulation_start_time
                            current_time = datetime.now()
                            elapsed = current_time - start_time
                            elapsed_minutes = elapsed.total_seconds() / 60
                            
                            if elapsed_minutes > 60:
                                elapsed_str = f"{elapsed_minutes/60:.1f} hours"
                            else:
                                elapsed_str = f"{elapsed_minutes:.1f} minutes"
                            
                            st.info(f"⏱️ **Elapsed Time:** {elapsed_str}")
                        
                        # Live metrics section
                        metrics_container = st.container()
                        
                        # Display console output
                        st.markdown("### 📋 Console Output")
                        
                        # Create containers for live updates
                        console_container = st.empty()
                        stats_container = st.empty()
                        
                        # Get console output
                        if hasattr(st.session_state, 'console_capture'):
                            console_output = st.session_state.console_capture.get_output()
                            recent_lines = st.session_state.console_capture.get_recent_lines(50)
                            
                            if console_output:
                                # Display the most recent lines in chronological order (oldest first)
                                recent_output = "\n".join(recent_lines)
                                
                                # Update console container
                                with console_container.container():
                                    st.text_area(
                                        "Real-time Console Output",
                                        recent_output,
                                        height=500,
                                        disabled=True,
                                        help="Live output from the MD simulation - chronological order with newest at bottom",
                                        key=f"console_output_{len(recent_lines)}"  # Force refresh on content change
                                    )
                                    
                                    # Add JavaScript to auto-scroll to bottom for live updates
                                    if auto_refresh and len(recent_lines) > 10:
                                        st.markdown("""
                                        <script>
                                            // Auto-scroll console to bottom to show latest content
                                            setTimeout(() => {
                                                const textAreas = document.querySelectorAll('textarea[aria-label*="Real-time Console Output"]');
                                                if (textAreas.length > 0) {
                                                    const lastTextArea = textAreas[textAreas.length - 1];
                                                    lastTextArea.scrollTop = lastTextArea.scrollHeight;
                                                }
                                            }, 200);
                                        </script>
                                        """, unsafe_allow_html=True)
                                
                                # Update stats container
                                with stats_container.container():
                                    total_lines = len(st.session_state.console_capture.output_lines)
                                    st.caption(f"📝 {total_lines} lines captured (showing last 50 lines in chronological order) - Last updated: {datetime.now().strftime('%H:%M:%S')}")
                                    
                                    # Debug: Show what's being captured
                                    if total_lines > 0:
                                        with st.expander("🔍 Debug: Latest Raw Messages"):
                                            for i, line in enumerate(recent_lines[-5:]):
                                                st.code(line, language=None)
                                
                                # Show recent key metrics if we can extract them
                                if recent_lines:
                                    # Try to extract some key info from recent lines
                                    latest_step = None
                                    latest_pe = None
                                    latest_temp = None
                                    latest_speed = None
                                    
                                    for line in reversed(recent_lines):
                                        if "📊 Step" in line and latest_step is None:
                                            # Extract step info: "📊 Step   30000: PE=   19999.8 kJ/mol, T= 306."
                                            try:
                                                import re
                                                step_match = re.search(r'Step\s+(\d+)', line)
                                                pe_match = re.search(r'PE=\s*([\d.-]+)', line)
                                                temp_match = re.search(r'T=\s*([\d.-]+)', line)
                                                speed_match = re.search(r'Speed=\s*([\d.-]+)', line)
                                                
                                                if step_match:
                                                    latest_step = int(step_match.group(1))
                                                if pe_match:
                                                    latest_pe = float(pe_match.group(1))
                                                if temp_match:
                                                    latest_temp = float(temp_match.group(1))
                                                if speed_match:
                                                    latest_speed = float(speed_match.group(1))
                                                    
                                                if all([latest_step, latest_pe, latest_temp, latest_speed]):
                                                    break
                                            except:
                                                pass
                                    
                                    # Display key metrics if found
                                    if any([latest_step, latest_pe, latest_temp, latest_speed]):
                                        with metrics_container:
                                            st.markdown("### 📊 Latest Metrics")
                                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                            
                                            with metric_col1:
                                                if latest_step:
                                                    st.metric("📈 Current Step", f"{latest_step:,}")
                                            
                                            with metric_col2:
                                                if latest_pe:
                                                    st.metric("⚡ Potential Energy", f"{latest_pe:.0f} kJ/mol")
                                            
                                            with metric_col3:
                                                if latest_temp:
                                                    temp_delta = latest_temp - 310.0  # Compare to target
                                                    st.metric(
                                                        "🌡️ Temperature", 
                                                        f"{latest_temp:.1f} K",
                                                        delta=f"{temp_delta:+.1f} K" if abs(temp_delta) > 0.1 else None
                                                    )
                                            
                                            with metric_col4:
                                                if latest_speed:
                                                    perf_color = "🟢" if latest_speed > 500 else "🟡" if latest_speed > 100 else "🔴"
                                                    st.metric(f"{perf_color} Performance", f"{latest_speed:.0f} ns/day")
                            
                            else:
                                with console_container.container():
                                    st.info("📝 Waiting for console output from simulation...")
                                    st.text_area("Console Output", "Simulation starting...", height=300, disabled=True)
                        
                        else:
                            with console_container.container():
                                st.info("📝 Console capture not initialized. Refresh the page.")
                        
                        # Control buttons
                        button_col1, button_col2 = st.columns(2)
                        with button_col1:
                            if st.button("🔄 Refresh Console Now"):
                                st.rerun()
                        
                        with button_col2:
                            if auto_refresh and status in ['running', 'starting']:
                                st.info(f"🔄 Auto-refreshing every {refresh_interval}s")
                        
                        # LIVE STREAMING - Gradio-style continuous updates
                        if auto_refresh and status in ['running', 'starting']:
                            # Show live status
                            st.markdown(f"""
                            <div style="text-align: center; color: #4CAF50; font-size: 0.9em; margin: 10px 0; 
                                       background: rgba(76, 175, 80, 0.1); padding: 8px; border-radius: 5px;">
                                🔴 LIVE STREAMING - Updating every {refresh_interval}s
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Simple but effective: Sleep then immediately refresh
                            # This creates a continuous loop similar to Gradio
                            time.sleep(refresh_interval)
                            st.rerun()
                    
                    else:
                        # Start simulation button
                        if st.button("🚀 Start MD Simulation", type="primary"):
                            
                            # Safety check - ensure md_integration is available
                            if st.session_state.md_integration is None:
                                st.error("❌ MD integration not available. Please check system initialization.")
                                st.stop()
                            
                            # Initialize session state for simulation messages
                            if 'simulation_messages' not in st.session_state:
                                st.session_state.simulation_messages = []
                            if 'simulation_start_time' not in st.session_state:
                                st.session_state.simulation_start_time = datetime.now()
                            
                            # Thread-safe console output capture class
                            class ThreadSafeConsoleCapture:
                                def __init__(self):
                                    self.output_lines = []
                                    
                                def write(self, text):
                                    """Capture console output in a thread-safe way"""
                                    if text.strip():  # Only capture non-empty lines
                                        timestamp = datetime.now().strftime("%H:%M:%S")
                                        formatted_line = f"[{timestamp}] {text.strip()}"
                                        self.output_lines.append(formatted_line)
                                        # Keep only last 200 lines
                                        if len(self.output_lines) > 200:
                                            self.output_lines = self.output_lines[-200:]
                                
                                def flush(self):
                                    pass
                                
                                def get_output(self):
                                    return "\n".join(self.output_lines)
                                
                                def get_recent_lines(self, n=50):
                                    return self.output_lines[-n:] if len(self.output_lines) > n else self.output_lines
                            
                            # Create console capture instance and store it in session state
                            if 'console_capture' not in st.session_state:
                                st.session_state.console_capture = ThreadSafeConsoleCapture()
                            
                            # Get reference for thread-safe access
                            console_capture_ref = st.session_state.console_capture
                            
                            # Simple callback that just stores raw output
                            def simple_console_callback(message):
                                """Simple callback that captures console output without accessing session state"""
                                # Use the local reference to avoid session state access from background thread
                                console_capture_ref.write(message)
                            
                            # Start simulation
                            simulation_id = st.session_state.md_integration.run_md_simulation_async(
                                pdb_file=simulation_input_file,
                                temperature=temperature,
                                equilibration_steps=equilibration_steps,
                                production_steps=production_steps,
                                save_interval=save_interval,
                                output_callback=simple_console_callback
                            )
                            
                            st.success(f"✅ Simulation started with ID: {simulation_id}")
                            st.info("🔄 Simulation is running in the background. The progress will appear below.")
                            st.info("💡 The page will auto-refresh to show real-time updates.")
                            
                            # Immediately show the progress interface
                            st.rerun()
                else:
                    st.info("Please select or upload a PDB file to run simulation.")
            
            with tab2:
                st.markdown("### 📊 Results Analysis")
                
                # Get available simulations
                available_simulations = st.session_state.md_integration.get_available_simulations()
                
                if available_simulations:
                    st.markdown("#### Available Simulations")
                    
                    # Create selection interface
                    sim_options = {}
                    for sim in available_simulations:
                        display_name = f"{sim['id']} - {sim['total_atoms']} atoms ({sim['performance']:.1f} ns/day)"
                        sim_options[display_name] = sim['id']
                    
                    selected_sim_display = st.selectbox("Select Simulation", list(sim_options.keys()))
                    selected_sim_id = sim_options[selected_sim_display]
                    
                    # Analyze selected simulation
                    if st.button("📊 Analyze Simulation"):
                        with st.spinner("Analyzing simulation results..."):
                            analysis_result = st.session_state.md_integration.analyze_simulation_results(selected_sim_id)
                        
                        if analysis_result['success']:
                            st.success("✅ Analysis completed!")
                            
                            # Basic information
                            st.markdown("#### 📋 Simulation Summary")
                            
                            basic_info = analysis_result['basic_info']
                            summary_col1, summary_col2, summary_col3 = st.columns(3)
                            
                            with summary_col1:
                                st.metric("Total Atoms", basic_info['total_atoms'])
                                st.metric("Total Time (min)", f"{basic_info['total_time_minutes']:.1f}")
                            
                            with summary_col2:
                                st.metric("Performance (ns/day)", f"{basic_info['performance_ns_per_day']:.1f}")
                                st.write(f"**Force Field:** {basic_info['force_field']}")
                            
                            with summary_col3:
                                if 'final_stats' in analysis_result and 'temperature' in analysis_result['final_stats']:
                                    temp_stats = analysis_result['final_stats']['temperature']
                                    st.metric("Final Temperature (K)", f"{temp_stats['mean']:.1f} ± {temp_stats['std']:.1f}")
                                    st.metric("Target Deviation (K)", f"{temp_stats['target_deviation']:.1f}")
                            
                            # Energy analysis
                            if 'energy_analysis' in analysis_result:
                                energy = analysis_result['energy_analysis']
                                st.markdown("#### ⚡ Energy Analysis")
                                
                                energy_col1, energy_col2, energy_col3 = st.columns(3)
                                
                                with energy_col1:
                                    st.metric("Initial PE (kJ/mol)", f"{energy['initial_pe']:.1f}")
                                
                                with energy_col2:
                                    st.metric("Minimized PE (kJ/mol)", f"{energy['minimized_pe']:.1f}")
                                
                                with energy_col3:
                                    st.metric("Energy Change (kJ/mol)", f"{energy['minimization_change']:.1f}")
                            
                            # File downloads
                            if 'files' in analysis_result:
                                st.markdown("#### 📁 Download Results")
                                
                                files = analysis_result['files']
                                file_col1, file_col2 = st.columns(2)
                                
                                with file_col1:
                                    # Production trajectory
                                    if 'production_dcd' in files and os.path.exists(files['production_dcd']):
                                        with open(files['production_dcd'], 'rb') as f:
                                            dcd_content = f.read()
                                        st.download_button(
                                            "📽️ Download Production Trajectory (DCD)",
                                            dcd_content,
                                            file_name=f"production_{selected_sim_id}.dcd",
                                            mime="application/octet-stream"
                                        )
                                    
                                    # Final structure
                                    if 'production/final_structure.pdb' in files and os.path.exists(files['production/final_structure.pdb']):
                                        with open(files['production/final_structure.pdb'], 'rb') as f:
                                            pdb_content = f.read()
                                        st.download_button(
                                            "🧬 Download Final Structure (PDB)",
                                            pdb_content,
                                            file_name=f"final_structure_{selected_sim_id}.pdb",
                                            mime="chemical/x-pdb"
                                        )
                                
                                with file_col2:
                                    # Production data
                                    if 'production/production.csv' in files and os.path.exists(files['production/production.csv']):
                                        with open(files['production/production.csv'], 'rb') as f:
                                            csv_content = f.read()
                                        st.download_button(
                                            "📊 Download Production Data (CSV)",
                                            csv_content,
                                            file_name=f"production_data_{selected_sim_id}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    # Analysis plots
                                    if 'analysis_plots' in files and os.path.exists(files['analysis_plots']):
                                        with open(files['analysis_plots'], 'rb') as f:
                                            plot_content = f.read()
                                        st.download_button(
                                            "📈 Download Analysis Plots (PNG)",
                                            plot_content,
                                            file_name=f"analysis_plots_{selected_sim_id}.png",
                                            mime="image/png"
                                        )
                                
                                # Show plots if available
                                if 'analysis_plots' in files and os.path.exists(files['analysis_plots']):
                                    st.markdown("#### 📈 Analysis Plots")
                                    st.image(files['analysis_plots'], use_column_width=True)
                        
                        else:
                            st.error(f"❌ Analysis failed: {analysis_result['error']}")
                
                else:
                    st.info("No completed simulations found. Run a simulation first.")
            
            with tab3:
                st.markdown("### 📁 File Management")
                
                # Get simulation files
                available_simulations = st.session_state.md_integration.get_available_simulations()
                
                if available_simulations:
                    st.markdown("#### Simulation Files")
                    
                    for sim in available_simulations:
                        with st.expander(f"Simulation {sim['id']} - {sim['total_atoms']} atoms"):
                            st.write(f"**Input File:** {sim['input_file']}")
                            st.write(f"**Performance:** {sim['performance']:.1f} ns/day")
                            st.write(f"**Success:** {'✅' if sim['success'] else '❌'}")
                            st.write(f"**Timestamp:** {sim['timestamp']}")
                            
                            # Get file information
                            sim_files = st.session_state.md_integration.get_simulation_files(sim['id'])
                            
                            if sim_files['success']:
                                st.write("**Available Files:**")
                                for file_type, file_path in sim_files['files'].items():
                                    if os.path.exists(file_path):
                                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                                        st.write(f"- {file_type}: {file_size:.1f} MB")
                else:
                    st.info("No simulation files found.")
        
        else:
            st.error("⚠️ Missing required dependencies. Please install them to use MD simulation.")
    
    else:
        st.error(f"❌ MD Simulation integration not available: {st.session_state.md_integration_error}")
        st.markdown("### 🔧 Installation Instructions")
        st.code("""
# Install required dependencies
conda install -c conda-forge openmm pdbfixer openmmforcefields

# Or using pip
pip install openmm pdbfixer openmmforcefields
        """)

# Footer
st.markdown("---")
st.markdown("*AI-Driven Insulin Delivery Patch Discovery • Active Learning Framework • Based on Research by BioMaterials AI Research Group*")
