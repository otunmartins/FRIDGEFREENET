#!/usr/bin/env python3
"""
Session State Management Utilities for Insulin-AI App

This module contains all session state initialization and management functions
extracted from the monolithic app for better modularity and testability.
"""

import streamlit as st
import pandas as pd
import uuid
import importlib
import sys
import os
from typing import Optional, Any, Dict


def initialize_session_state() -> None:
    """Initialize all session state variables for the active learning framework."""
    
    # Core framework state
    if 'literature_iterations' not in st.session_state:
        st.session_state.literature_iterations = []
    if 'psmiles_candidates' not in st.session_state:
        st.session_state.psmiles_candidates = []
    if 'active_learning_queue' not in st.session_state:
        st.session_state.active_learning_queue = []
    
    # Material library
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
    
    # Feedback and queries
    if 'iteration_feedback' not in st.session_state:
        st.session_state.iteration_feedback = {}
    if 'literature_queries' not in st.session_state:
        st.session_state.literature_queries = []
    
    # System state
    if 'systems_initialized' not in st.session_state:
        st.session_state.systems_initialized = False
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Insulin processing state
    if 'use_processed_insulin' not in st.session_state:
        st.session_state.use_processed_insulin = False
    if 'insulin_preprocessing_result' not in st.session_state:
        st.session_state.insulin_preprocessing_result = None


def check_session_init() -> bool:
    """
    Check if session has been properly initialized.
    
    Returns:
        bool: True if session is properly initialized, False otherwise
    """
    required_keys = [
        'session_id',
        'systems_initialized',
        'literature_iterations',
        'psmiles_candidates',
        'active_learning_queue',
        'material_library',
        'iteration_feedback',
        'literature_queries'
    ]
    
    for key in required_keys:
        if key not in st.session_state:
            return False
    
    return True


def get_session_uuid() -> str:
    """
    Get the session UUID, creating one if it doesn't exist.
    
    Returns:
        str: The session UUID
    """
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    return st.session_state.session_id


def clear_session_cache() -> None:
    """
    Clear session cache and reset to initial state.
    """
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Re-initialize session state
    initialize_session_state()


def setup_openai_config() -> None:
    """
    Setup OpenAI configuration in session state.
    """
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    
    if 'openai_model' not in st.session_state:
        st.session_state.openai_model = 'gpt-4o-mini'
    
    if 'openai_temperature' not in st.session_state:
        st.session_state.openai_temperature = 0.7
    
    if 'openai_max_tokens' not in st.session_state:
        st.session_state.openai_max_tokens = 4000


def setup_model_selection() -> None:
    """
    Setup model selection options in session state.
    """
    if 'available_models' not in st.session_state:
        st.session_state.available_models = [
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4-turbo',
            'gpt-3.5-turbo'
        ]
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'gpt-4o-mini'
    
    if 'model_parameters' not in st.session_state:
        st.session_state.model_parameters = {
            'temperature': 0.7,
            'max_tokens': 4000,
            'top_p': 1.0,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }


def validate_session_state_object(obj_name: str, expected_type=None) -> bool:
    """
    Validate that a session state object exists and is of the expected type.
    
    Args:
        obj_name: Name of the session state object to validate
        expected_type: Optional type to check against
        
    Returns:
        bool: True if object exists and meets criteria, False otherwise
    """
    if obj_name not in st.session_state:
        return False
    
    obj = st.session_state[obj_name]
    
    # Check if it's a string (indicating failed initialization)
    if isinstance(obj, str):
        return False
    
    # Check if it's None
    if obj is None:
        return False
    
    # Check specific type if provided
    if expected_type and not isinstance(obj, expected_type):
        return False
    
    return True


def ensure_systems_initialized() -> bool:
    """
    Ensure all systems are properly initialized, with fallback reinitialization if needed.
    
    Returns:
        bool: True if all systems are initialized, False otherwise
    """
    if not st.session_state.get('systems_initialized', False):
        return False
    
    # Check critical objects
    critical_objects = {
        'psmiles_generator': 'PSMILESGenerator',
        'psmiles_processor': 'PSMILESProcessor', 
        'literature_miner': 'MaterialsLiteratureMiner',
        'chatbot': 'InsulinAIChatbot'
    }
    
    for obj_name, obj_type in critical_objects.items():
        if not validate_session_state_object(obj_name):
            st.error(f"❌ {obj_type} not properly initialized. Please restart the application.")
            return False
    
    return True


def safe_get_session_object(obj_name: str, default=None) -> Any:
    """
    Safely get a session state object with validation.
    
    Args:
        obj_name: Name of the session state object
        default: Default value to return if object is invalid
        
    Returns:
        The session state object or default value
    """
    if validate_session_state_object(obj_name):
        obj = st.session_state[obj_name]
        
        # Special validation for PSMILESProcessor
        if obj_name == 'psmiles_processor' and hasattr(obj, '__class__'):
            # Check if it has the auto-repair method
            if not hasattr(obj, 'process_psmiles_workflow_with_autorepair'):
                print(f"🔄 PSMILESProcessor missing auto-repair method, forcing refresh...")
                success, message = force_refresh_psmiles_processor()
                if success:
                    print(f"✅ PSMILESProcessor refreshed successfully")
                    return st.session_state[obj_name]
                else:
                    print(f"❌ Failed to refresh PSMILESProcessor: {message}")
                    return default
            
            if not validate_psmiles_processor(obj):
                print(f"🔄 PSMILESProcessor validation failed, forcing re-initialization...")
                # Force re-initialization by clearing the cache
                try:
                    # **NOTE: Disabled automatic re-initialization to avoid conflicts**
                    # The main app handles system initialization properly
                    print(f"⚠️ PSMILESProcessor missing - please use system refresh in sidebar")
                    return default
                    
                    # OLD CODE: This creates conflicts with main app's initialize_systems
                    # from services.system_service import initialize_systems, clear_systems
                    # clear_systems()
                    # st.session_state.systems_initialized = False
                    # 
                    # # Re-initialize systems
                    # systems = initialize_systems()
                    # if systems['status'] == 'success':
                    #     st.session_state.systems_initialized = True
                    #     st.session_state.psmiles_processor = systems['psmiles_processor']
                    #     return systems['psmiles_processor']
                except Exception as e:
                    print(f"❌ Failed to re-initialize PSMILESProcessor: {e}")
                    return default
        
        return obj
    return default


def validate_psmiles_processor(processor) -> bool:
    """
    Validate that PSMILESProcessor has all required methods.
    
    Args:
        processor: PSMILESProcessor instance to validate
        
    Returns:
        bool: True if processor has all required methods
    """
    required_methods = [
        '_validate_psmiles_format',
        'process_psmiles_workflow',
        '_fix_connection_points',
        'process_psmiles_workflow_with_autorepair'  # New auto-repair method
    ]
    
    for method_name in required_methods:
        if not hasattr(processor, method_name):
            print(f"❌ PSMILESProcessor missing method: {method_name}")
            return False
    
    return True


def force_refresh_psmiles_processor() -> tuple[bool, str]:
    """
    Force refresh PSMILESProcessor to get latest functionality including auto-repair.
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Clear any cached PSMILESProcessor
        if 'psmiles_processor' in st.session_state:
            del st.session_state.psmiles_processor
        
        # Force reimport of the module to get latest code
        if 'core.psmiles_processor' in sys.modules:
            importlib.reload(sys.modules['core.psmiles_processor'])
        
        # Create new instance with latest functionality
        from core.psmiles_processor import PSMILESProcessor
        new_processor = PSMILESProcessor()
        
        # Verify it has all required methods
        required_methods = [
            '_validate_psmiles_format',
            'process_psmiles_workflow',
            '_fix_connection_points',
            'process_psmiles_workflow_with_autorepair'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(new_processor, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            return False, f"❌ Refreshed processor missing methods: {missing_methods}"
        
        # Verify the processor is available and functional
        if not new_processor.available:
            return False, "❌ PSMILESProcessor not available (missing dependencies)"
        
        st.session_state.psmiles_processor = new_processor
        return True, "✅ PSMILESProcessor refreshed with all required functionality!"
            
    except Exception as e:
        return False, f"❌ Failed to refresh PSMILESProcessor: {str(e)}" 


def get_or_create_session_state(key: str, default_value: Any = None) -> Any:
    """
    Get a value from session state or create it with a default value
    
    Args:
        key: The session state key
        default_value: The default value to set if key doesn't exist
        
    Returns:
        The value from session state
    """
    try:
        import streamlit as st
        
        if key not in st.session_state:
            st.session_state[key] = default_value
        
        return st.session_state[key]
        
    except ImportError:
        # Fallback for testing environments
        return default_value
    except Exception as e:
        return default_value 