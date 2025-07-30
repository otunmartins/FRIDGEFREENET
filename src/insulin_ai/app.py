#!/usr/bin/env python3
"""
Enhanced Insulin AI App with OpenAI Integration
Comprehensive material discovery platform for insulin delivery patches
PACKAGE VERSION - Uses proper package imports
"""

import sys
from pathlib import Path
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
from io import BytesIO
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# Package imports
from insulin_ai.core.chatbot_system import InsulinAIChatbot
from insulin_ai.core.literature_mining_system import MaterialsLiteratureMiner
from insulin_ai.core.psmiles_generator import PSMILESGenerator
from insulin_ai.core.psmiles_processor import PSMILESProcessor

# Import PSMILES auto-corrector
try:
    from insulin_ai.integration.corrections.psmiles_auto_corrector import create_psmiles_auto_corrector
    from insulin_ai.integration.corrections.instant_psmiles_corrector import apply_instant_corrections_ui
    AUTOCORRECTOR_AVAILABLE = True
except ImportError:
    AUTOCORRECTOR_AVAILABLE = False

# Import MD simulation integration
try:
    from insulin_ai.integration.analysis.simple_md_integration import SimpleMDIntegration
    from insulin_ai.integration.analysis.test_real_md_simulation import find_test_pdb_file
    MD_INTEGRATION_AVAILABLE = True
    print("✅ Simple Working MD Integration imported successfully (based on openmm_test.py)")
except ImportError as e:
    try:
        from insulin_ai.integration.analysis.streamlined_md_integration import StreamlinedMDIntegration as SimpleMDIntegration
        from insulin_ai.integration.analysis.streamlined_md_integration import run_streamlined_md
        MD_INTEGRATION_AVAILABLE = True
        print("⚠️ Using fallback streamlined system (simple working system not available)")
    except ImportError as e2:
        MD_INTEGRATION_AVAILABLE = False
        print(f"❌ Both simple working and streamlined MD integration failed: {e}, {e2}")
        print("   This may be due to missing dependencies like OpenMM or pdbfixer")

# Import comprehensive analysis system
try:
    from insulin_ai.integration.analysis.insulin_delivery_analysis_integration import InsulinDeliveryAnalysisIntegration
    from insulin_ai.integration.analysis.insulin_comprehensive_analyzer import InsulinComprehensiveAnalyzer
    COMPREHENSIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYSIS_AVAILABLE = False

# Import simulation automation system
try:
    from insulin_ai.integration.automation.simulation_automation import SimulationAutomation, SimulationAutomationPipeline, run_automated_simulation_pipeline
    SIMULATION_AUTOMATION_AVAILABLE = True
    print("✅ DirectPolymerBuilder simulation automation imported successfully")
except ImportError:
    SIMULATION_AUTOMATION_AVAILABLE = False
    print("❌ Simulation automation not available")

# Import debugging utilities
try:
    from insulin_ai.utils.debug_tracer import tracer, enable_runtime_debugging
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
    render_material_evaluation_ui,
    render_simulation_interface,
    render_comprehensive_analysis
)

from insulin_ai.app.utils.session_utils import initialize_session_state, safe_get_session_object

# Initialize comprehensive session state
if 'literature_iterations' not in st.session_state:
    st.session_state.literature_iterations = []
if 'psmiles_candidates' not in st.session_state:
    st.session_state.psmiles_candidates = []
if 'active_learning_queue' not in st.session_state:
    st.session_state.active_learning_queue = []
if 'material_library' not in st.session_state:
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

# Custom CSS
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

def setup_openai_api():
    """Setup OpenAI API key from user input or environment."""
    st.sidebar.header("🔑 OpenAI Configuration")
    
    env_api_key = os.environ.get('OPENAI_API_KEY', '')
    
    if env_api_key:
        st.sidebar.success("✅ OpenAI API Key found in environment")
        return env_api_key
    else:
        st.sidebar.warning("⚠️ No OpenAI API Key found in environment")
        
        api_key = st.sidebar.text_input(
            "Enter your OpenAI API Key:",
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        if api_key:
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
        index=2
    )
    
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
                systems['md_integration'] = SimpleMDIntegration()
                print("✅ Simple Working MD Integration initialized successfully")
            except Exception as e:
                print(f"⚠️ Simple Working MD Integration failed to initialize: {e}")
                systems['md_integration'] = None
        else:
            systems['md_integration'] = None
            print("⚠️ MD Simulation Integration not available - import failed during startup")
        
        systems['status'] = 'success'
        return systems
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Additional initialization for model tracking
    if 'current_initialized_model' not in st.session_state:
        st.session_state.current_initialized_model = None
    if 'current_initialized_temp' not in st.session_state:
        st.session_state.current_initialized_temp = None
    
    # PSMILES workflow state
    if 'psmiles_workflow_active' not in st.session_state:
        st.session_state.psmiles_workflow_active = False
    if 'current_psmiles' not in st.session_state:
        st.session_state.current_psmiles = None
    if 'svg_content' not in st.session_state:
        st.session_state.svg_content = None
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Insulin-AI: AI-Powered Drug Delivery System</h1>', unsafe_allow_html=True)
    
    # Setup OpenAI configuration in sidebar
    api_key = setup_openai_api()
    if not api_key:
        st.stop()

    # Model selection in sidebar
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
                    if key != 'status':
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
    
    # Sidebar navigation
    page = render_navigation()
    
    # Page routing
    if page == "Framework Overview":
        render_framework_overview()
        
    elif page == "Literature Mining (LLM)":
        render_literature_mining_ui()
        
    elif page == "PSMILES Generation":
        render_psmiles_generation()
        
    elif page == "Active Learning":
        render_active_learning()
        
    elif page == "Material Evaluation":
        render_material_evaluation_ui()
        
    elif page == "MD Simulation":
        render_simulation_interface()
        
    elif page == "Comprehensive Analysis":
        render_comprehensive_analysis()

if __name__ == "__main__":
    main() 