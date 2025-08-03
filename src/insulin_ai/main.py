#!/usr/bin/env python3
"""
Enhanced Insulin AI App with OpenAI Integration
Comprehensive material discovery platform for insulin delivery patches
MODULARIZED VERSION - Uses UI modules for clean architecture
"""

import os
import tempfile
from typing import Dict, List, Optional, Any, Callable
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
import uuid
import time
import base64
import zipfile
import shutil
from pathlib import Path
from io import BytesIO
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# **NEW: Import real systems**
from insulin_ai import InsulinAIChatbot, MaterialsLiteratureMiner, PSMILESGenerator, PSMILESProcessor

# Import PSMILES auto-corrector
try:
    from insulin_ai.integration.corrections.psmiles_auto_corrector import create_psmiles_auto_corrector
    from insulin_ai.integration.corrections.instant_psmiles_corrector import apply_instant_corrections_ui
    AUTOCORRECTOR_AVAILABLE = True
except ImportError:
    AUTOCORRECTOR_AVAILABLE = False

# Import MD simulation integration - USING NEW SIMPLE WORKING SYSTEM
try:
    # Import our NEW SIMPLE WORKING system based on openmm_test.py
    from insulin_ai.integration.analysis.dual_gaff_amber_integration import DualGaffAmberIntegration
    MD_INTEGRATION_AVAILABLE = True
    print("✅ Dual GAFF/AMBER Integration imported successfully")
except ImportError as e:
    # Fallback to old streamlined system
    try:
        from insulin_ai.integration.analysis.streamlined_md_integration import StreamlinedMDIntegration as SimpleMDIntegration
        from insulin_ai.integration.analysis.streamlined_md_integration import run_streamlined_md
        MD_INTEGRATION_AVAILABLE = True
        print("⚠️ Using fallback streamlined system (simple working system not available)")
    except ImportError as e2:
        MD_INTEGRATION_AVAILABLE = False
        print(f"❌ Both simple working and streamlined MD integration failed: {e}, {e2}")
        print("   This may be due to missing dependencies like OpenMM or pdbfixer")
    
# Import legacy system for compatibility (if needed)
try:
    from insulin_ai.integration.analysis.md_simulation_integration import get_insulin_polymer_pdb_files
    LEGACY_MD_UTILS_AVAILABLE = True
except ImportError:
    LEGACY_MD_UTILS_AVAILABLE = False

# Import comprehensive analysis system
try:
    from insulin_ai.integration.analysis.insulin_delivery_analysis_integration import InsulinDeliveryAnalysisIntegration
    from insulin_ai.integration.analysis.insulin_comprehensive_analyzer import InsulinComprehensiveAnalyzer
    COMPREHENSIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYSIS_AVAILABLE = False

# Import simulation automation system (Updated to use DirectPolymerBuilder)
try:
    from integration.automation.simulation_automation import SimulationAutomation, SimulationAutomationPipeline, run_automated_simulation_pipeline
    SIMULATION_AUTOMATION_AVAILABLE = True
    print("✅ DirectPolymerBuilder simulation automation imported successfully")
except ImportError:
    SIMULATION_AUTOMATION_AVAILABLE = False
    print("❌ Simulation automation not available")

# Import debugging utilities
try:
    from utils.debug_tracer import tracer, enable_runtime_debugging
    DEBUGGING_AVAILABLE = True
except ImportError:
    DEBUGGING_AVAILABLE = False

# Import UI modules
from insulin_ai.app.ui import (
    render_navigation,
    render_framework_overview,
    render_literature_mining_ui,
    render_psmiles_generation,
    render_active_learning,
    render_simulation_ui
)

from insulin_ai.app.utils.session_utils import initialize_session_state, safe_get_session_object

# **NEW: Initialize comprehensive session state matching old app**
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

# **NEW: Simplified Custom CSS - more minimal approach**
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 2rem;
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
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .visualization-container {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #ddd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="🧬 Insulin-AI: AI-Powered Drug Delivery System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

# ==================================================
# VALIDATION AND HELPER FUNCTIONS (matching old app)
# ==================================================

def validate_psmiles_processor(processor) -> bool:
    """Validate that PSMILESProcessor has required methods"""
    if not processor:
        return False
    
    required_methods = [
        'process_psmiles_workflow_with_autorepair',
        'add_random_functional_groups',
        'get_session_psmiles'
    ]
    
    for method in required_methods:
        if not hasattr(processor, method):
            return False
    
    return True

def force_refresh_psmiles_processor():
    """Force refresh PSMILESProcessor with latest functionality"""
    try:
        # Clear cached processor
        if 'psmiles_processor' in st.session_state:
            del st.session_state.psmiles_processor
        
        # Reinitialize
        from insulin_ai import PSMILESProcessor
        processor = PSMILESProcessor()
        st.session_state.psmiles_processor = processor
        
        # Validate
        if validate_psmiles_processor(processor):
            return True, "✅ PSMILESProcessor refreshed with auto-repair functionality!"
        else:
            return False, "❌ PSMILESProcessor still missing required methods"
            
    except Exception as e:
        return False, f"❌ Failed to refresh PSMILESProcessor: {str(e)}"

# ==================================================
# ENHANCED SYSTEM VALIDATION
# ==================================================

def validate_session_state_object(obj_name: str, expected_type = None) -> bool:
    """Validate session state object exists and has correct type"""
    if obj_name not in st.session_state:
        return False
    
    obj = st.session_state[obj_name]
    if obj is None:
        return False
    
    if expected_type and not isinstance(obj, expected_type):
        return False
    
    return True

def ensure_systems_initialized() -> bool:
    """Ensure all required systems are properly initialized"""
    required_systems = [
        'psmiles_generator',
        'psmiles_processor', 
        'literature_miner',
        'chatbot'
    ]
    
    for system in required_systems:
        if not validate_session_state_object(system):
            return False
    
    return True

# ==================================================
# HELPER FUNCTIONS
# ==================================================

# ==================================================
# CORE GENERATION FUNCTION
# ==================================================
# Import the new SMILES storage utility
from insulin_ai.app.utils.psmiles_smiles_storage import enhance_psmiles_generation_with_smiles_storage

def psmiles_generation_with_llm(material_request, conversation_memory=None):
    """
    Sophisticated PSMILES generation using the most advanced pipeline:
    - Multiple diverse candidates with temperature variation
    - Chemical functionalization for true diversity
    - Proper validation and diversity scoring
    - **NEW: Automatic SMILES conversion and storage for efficient MD workflows**
    - No fake/hard-coded properties
    """
    
    def _original_generation_function(material_request, conversation_memory=None):
        """Original generation logic wrapped for enhancement"""
        try:
            print(f"🚀 Starting sophisticated PSMILES generation for: {material_request}")
            
            # Use the most sophisticated generation method with functionalization
            results = st.session_state.psmiles_generator.generate_truly_diverse_candidates(
                base_request=material_request,
                num_candidates=5,  # Generate multiple diverse candidates
                enable_functionalization=True,  # Enable chemical functionalization
                diversity_threshold=0.4,  # Require meaningful diversity
                temperature_range=(0.6, 1.0),  # Use temperature variation
                max_retries=2  # Allow retries for better results
            )
            
            print(f"📊 Generation results: success={results.get('success')}, candidates={results.get('num_generated', 0)}")
            
            # Check if generation was successful
            if results.get('success') and results.get('candidates'):
                candidates = results['candidates']
                best_candidate = results.get('best_candidate')
                
                if best_candidate and best_candidate.count('[*]') == 2:
                    return {
                        'psmiles': best_candidate,
                        'explanation': f"Generated using advanced orchestration system with {results.get('total_generated', results.get('num_generated', 0))} total candidates",
                        'candidates': candidates,  # Selected best candidates
                        'all_candidates': results.get('all_candidates', candidates),  # ALL generated candidates including variants
                        'total_generated': results.get('total_generated', len(candidates)),  # Total before selection
                        'diversity_info': {
                            'num_generated': results.get('num_generated', 0),
                            'num_valid': results.get('num_valid', 0),
                            'total_generated': results.get('total_generated', 0),
                            'diversity_score': results.get('diversity_validation', {}).get('diversity_score', 0),
                            'meets_threshold': results.get('diversity_validation', {}).get('meets_diversity_threshold', False),
                            'functionalization_enabled': True
                        },
                        'method': 'truly_diverse_candidates',
                        'temperature_range': results.get('temperature_range', (0.6, 1.0)),
                        'validation_status': 'chemistry_validated',
                        'generation_details': {
                            'pipeline': results.get('pipeline', 'Advanced-Orchestration'),
                            'model': results.get('model', 'unknown'),
                            'timestamp': results.get('timestamp', ''),
                            'orchestration_system': True,
                            'functionalization_applied': True
                        }
                    }
            
            # If truly diverse generation failed, fallback to standard diverse generation
            print("⚠️ Truly diverse generation failed, falling back to standard diverse generation")
            fallback_results = st.session_state.psmiles_generator.generate_diverse_candidates(
                base_request=material_request,
                num_candidates=6,
                temperature_range=(0.6, 1.0)
            )
            
            if fallback_results.get('success') and fallback_results.get('best_candidate'):
                return {
                    'psmiles': fallback_results['best_candidate'],
                    'explanation': f"Generated using standard diverse pipeline (fallback) with {fallback_results.get('total_generated', fallback_results.get('num_generated', 0))} total candidates",
                    'candidates': fallback_results.get('candidates', []),
                    'all_candidates': fallback_results.get('all_candidates', fallback_results.get('candidates', [])),
                    'total_generated': fallback_results.get('total_generated', len(fallback_results.get('candidates', []))),
                    'diversity_info': {
                        'num_generated': fallback_results.get('num_generated', 0),
                        'num_valid': fallback_results.get('num_valid', 0),
                        'total_generated': fallback_results.get('total_generated', 0),
                        'functionalization_enabled': False
                    },
                    'method': 'diverse_candidates_fallback',
                    'temperature_range': fallback_results.get('temperature_range', (0.6, 1.0)),
                    'validation_status': 'chemistry_validated',
                    'generation_details': {
                        'pipeline': fallback_results.get('pipeline', 'NaturalLanguage→SMILES→PSMILES'),
                        'model': fallback_results.get('model', 'unknown'),
                        'timestamp': fallback_results.get('timestamp', ''),
                        'fallback_used': True
                    }
                }
            
            # If everything fails
            return {
                'error': 'All sophisticated generation methods failed',
                'psmiles': None,
                'method': 'generation_failure',
                'details': {
                    'sophisticated_failed': True,
                    'fallback_failed': True,
                    'original_error': results.get('error', 'Unknown error')
                }
            }
            
        except Exception as e:
            print(f"❌ Error in sophisticated PSMILES generation: {e}")
            return {
                'error': str(e),
                'psmiles': None,
                'method': 'generation_error',
                'details': {'exception': str(e)}
            }
    
    # **NEW: Use the enhanced generation with automatic SMILES storage**
    print(f"🧬 Enhanced PSMILES generation with automatic SMILES storage")
    enhanced_result = enhance_psmiles_generation_with_smiles_storage(
        _original_generation_function,
        material_request,
        conversation_memory
    )
    
    # The enhanced result now contains both PSMILES and SMILES for efficient MD workflows
    if enhanced_result.get('smiles_conversion_success'):
        print(f"✅ PSMILES generated with stored SMILES ready for MD simulation")
        print(f"   PSMILES: {enhanced_result.get('psmiles', 'N/A')}")
        print(f"   SMILES:  {enhanced_result.get('smiles', 'N/A')}")
        print(f"   Method:  {enhanced_result.get('smiles_conversion_method', 'N/A')}")
    else:
        print(f"⚠️ PSMILES generated but SMILES conversion failed: {enhanced_result.get('smiles_conversion_error', 'Unknown error')}")
    
    return enhanced_result

# ==================================================
# SYSTEM INITIALIZATION
# ==================================================

def setup_openai_api():
    """Setup OpenAI API key from user input or environment."""
    st.sidebar.header("🔑 OpenAI Configuration")
    
    # Check if API key exists in environment
    env_api_key = os.environ.get('OPENAI_API_KEY', '')
    
    if env_api_key:
        st.sidebar.success("✅ OpenAI API Key found in environment")
        return env_api_key
    else:
        st.sidebar.warning("⚠️ No OpenAI API Key found in environment")
        
        # Get API key from user input
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key:
            # Set environment variable for this session
            os.environ['OPENAI_API_KEY'] = api_key
            st.sidebar.success("✅ OpenAI API Key configured")
            return api_key
        else:
            st.sidebar.error("❌ OpenAI API Key required to proceed")
            st.error("🔑 Please enter your OpenAI API Key in the sidebar to use the application.")
            st.stop()
            return None

def setup_model_selection():
    """Allow user to select OpenAI model."""
    st.sidebar.header("🤖 Model Configuration")
    
    model_options = {
        "gpt-4o": "GPT-4o (Recommended - Best performance)",
        "gpt-4": "GPT-4 (High quality, slower)",
        "gpt-3.5-turbo": "GPT-3.5 Turbo (Fast, cost-effective)",
        "gpt-4-turbo": "GPT-4 Turbo (Balanced performance)"
    }
    
    selected_model = st.sidebar.selectbox(
        "Select OpenAI Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=2  # Default to gpt-3.5-turbo instead of gpt-4o
    )
    
    # Temperature setting
    temperature = st.sidebar.slider(
        "Model Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )
    
    return selected_model, temperature

@st.cache_resource
def initialize_systems(openai_model='gpt-4o', temperature=0.7):
    """Initialize all AI systems with specified parameters"""
    systems = {}
    
    try:
        # Get OpenAI configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("❌ OpenAI API Key not found in environment variables")
            return {'status': 'error', 'error': 'No OpenAI API Key'}
        
        st.info(f"🚀 Initializing systems with OpenAI {openai_model}...")
        
        # Initialize Chatbot System
        systems['chatbot'] = InsulinAIChatbot(
            model_type="openai",
            openai_model=openai_model,
            temperature=temperature,
            memory_type="buffer_window",
            memory_dir="chat_memory"
        )
        print(f"✅ OpenAI chatbot initialized with {openai_model} (temp: {temperature})")
        
        # Initialize Literature Mining System
        systems['literature_miner'] = MaterialsLiteratureMiner(
            semantic_scholar_api_key=os.environ.get('SEMANTIC_SCHOLAR_API_KEY'),
            model_type="openai",
            openai_model=openai_model,
            temperature=temperature
        )
        print(f"✅ Literature Mining initialized with OpenAI {openai_model}")
        
        # Initialize PSMILES Generator
        systems['psmiles_generator'] = PSMILESGenerator(
            model_type='openai',
            openai_model=openai_model,
            temperature=temperature
        )
        print(f"✅ PSMILES Generator initialized with OpenAI {openai_model} (temp: {temperature})")
        
        # Initialize PSMILES Processor
        systems['psmiles_processor'] = PSMILESProcessor()
        print("✅ PSMILES Processor initialized with full functionality!")
        
        # Initialize auto-corrector if available
        if AUTOCORRECTOR_AVAILABLE:
            systems['psmiles_auto_corrector'] = create_psmiles_auto_corrector()
            print("✅ PSMILES Auto-Corrector initialized")
        
        # Initialize MD simulation integration if available
        if MD_INTEGRATION_AVAILABLE:
            try:
                systems['md_integration'] = DualGaffAmberIntegration()
                print("✅ Dual GAFF/AMBER Integration initialized successfully")
            except Exception as e:
                print(f"⚠️ Dual GAFF/AMBER Integration failed to initialize: {e}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Consider installing missing dependencies or check module compatibility")
                systems['md_integration'] = None
        else:
            systems['md_integration'] = None
            print("⚠️ MD Simulation Integration not available - import failed during startup")
        
        systems['status'] = 'success'
        return systems
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return {'status': 'error', 'error': str(e)}

# ==================================================
# MAIN APP
# ==================================================
def main():
    # Initialize session state (use enhanced version)
    initialize_session_state()
    
    # Additional initialization for model tracking
    if 'current_initialized_model' not in st.session_state:
        st.session_state.current_initialized_model = None
    if 'current_initialized_temp' not in st.session_state:
        st.session_state.current_initialized_temp = None
    
    # PSMILES workflow state (additional to session_utils)
    if 'psmiles_workflow_active' not in st.session_state:
        st.session_state.psmiles_workflow_active = False
    if 'current_psmiles' not in st.session_state:
        st.session_state.current_psmiles = None
    if 'svg_content' not in st.session_state:
        st.session_state.svg_content = None
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # **NEW: Enhanced Header with gradient styling**
    st.markdown('<h1 class="main-header">🧬 Insulin-AI: AI-Powered Drug Delivery System</h1>', unsafe_allow_html=True)
    
    # **NEW: Setup OpenAI configuration in sidebar**
    api_key = setup_openai_api()
    if not api_key:
        st.stop()  # Stop execution if no API key

    # **NEW: Model selection in sidebar**
    selected_model, temperature = setup_model_selection()

    # Store in session state for use in cached function
    st.session_state['openai_model'] = selected_model
    st.session_state['temperature'] = temperature

    # Check if model settings have changed
    current_model = st.session_state.get('current_initialized_model')
    current_temp = st.session_state.get('current_initialized_temp')
    model_changed = (current_model != selected_model or current_temp != temperature)
    
    # Initialize systems if not done or if model changed
    if not st.session_state.systems_initialized or model_changed:
        with st.spinner("🔄 Initializing AI systems..."):
            systems = initialize_systems(selected_model, temperature)
            if systems.get('status') == 'success':
                # Store in session state
                for key, system in systems.items():
                    if key != 'status':  # Don't store 'status' as it's metadata
                        setattr(st.session_state, key, system)
                
                # Set availability flags
                st.session_state.systems_initialized = True
                st.session_state.current_initialized_model = selected_model
                st.session_state.current_initialized_temp = temperature
                st.session_state.md_integration_available = systems.get('md_integration') is not None
                
                st.success("✅ All systems initialized successfully!")
            else:
                st.error(f"❌ Failed to initialize systems: {systems.get('error')}")
                st.stop()
    
    # ==================================================
    # SIDEBAR NAVIGATION - Use modular UI
    # ==================================================
    page = render_navigation()
    
    # ==================================================
    # PAGE ROUTING - Use modular UI components
    # ==================================================
    if page == "Framework Overview":
        render_framework_overview()
        
    elif page == "Literature Mining (LLM)":
        render_literature_mining_ui()
        
    elif page == "PSMILES Generation":
        render_psmiles_generation()
        
    elif page == "Active Learning":
        render_active_learning()
        
    elif page == "MD Simulation":
        render_simulation_ui()

# ==================================================
# RUN APP
# ==================================================

def run_streamlit_app():
    """Entry point for running the Streamlit app via package entry point."""
    import subprocess
    import sys
    import os
    
    # Get the current file path
    current_file = os.path.abspath(__file__)
    
    # Run streamlit with this file
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", current_file,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ] + sys.argv[1:])

if __name__ == "__main__":
    main() 