#!/usr/bin/env python3
"""
System Service for Insulin-AI App

This module handles initialization and management of all AI systems,
extracted from the monolithic app for better modularity and testability.
"""

import os
import streamlit as st
from typing import Dict, Any, Optional


def initialize_systems() -> Dict[str, Any]:
    """
    Initialize all AI systems with OpenAI models.
    
    Returns:
        Dict[str, Any]: Dictionary containing initialized systems or error information
    """
    try:
        # Import core systems
        from insulin_ai import InsulinAIChatbot, PSMILESGenerator, PSMILESProcessor
        from insulin_ai.integration.rag_literature_mining import RAGLiteratureMiningSystem
        
        # Get OpenAI configuration
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return {
                'status': 'error',
                'error': 'OpenAI API Key not configured. Please set it in the sidebar.'
            }
            
        # Get model selection from session state (set by sidebar)
        openai_model = st.session_state.get('openai_model', 'gpt-3.5-turbo')
        temperature = st.session_state.get('temperature', 0.7)
        
        # Initialize chatbot with OpenAI
        chatbot = InsulinAIChatbot(
            model_type="openai",
            openai_model=openai_model,
            temperature=temperature,
            memory_type="buffer_window",
            memory_dir="chat_memory"
        )
        
        # Initialize RAG literature mining system
        literature_miner = RAGLiteratureMiningSystem(
            openai_model=openai_model,
            temperature=temperature
        )
        
        # Initialize PSMILES systems with OpenAI
        psmiles_generator = PSMILESGenerator(
            model_type='openai',
            openai_model=openai_model,
            temperature=temperature
        )
        
        psmiles_processor = PSMILESProcessor()
        
        # Initialize PSMILES auto-corrector with OpenAI if available
        psmiles_auto_corrector = None
        try:
            from integration.corrections.psmiles_auto_corrector import create_psmiles_auto_corrector
            psmiles_auto_corrector = create_psmiles_auto_corrector(
                model_type="openai",
                openai_model=openai_model,
                temperature=temperature
            )
            print("✅ PSMILES Auto-Corrector initialized")
            print(f"   Type: {type(psmiles_auto_corrector)}")
            print(f"   Has correct_psmiles: {hasattr(psmiles_auto_corrector, 'correct_psmiles')}")
        except Exception as e:
            print(f"⚠️ PSMILES Auto-Corrector not available: {e}")
        
        # Initialize MD integration if available
        md_integration = None
        try:
            from integration.md_integration import initialize_md_systems
            md_integration = initialize_md_systems()
            print("✅ MD integration initialized")
        except ImportError as e:
            missing_deps = []
            if 'openmm' in str(e).lower():
                missing_deps.append('openmm')
            # MM-GBSA dependency removed from system
            print(f"MD integration initialization failed: Missing dependencies: {missing_deps}")
        except Exception as e:
            print(f"⚠️ MD integration not available: {e}")
        
        return {
            'status': 'success',
            'chatbot': chatbot,
            'literature_miner': literature_miner,
            'psmiles_generator': psmiles_generator,
            'psmiles_processor': psmiles_processor,
            'psmiles_auto_corrector': psmiles_auto_corrector,
            'md_integration': md_integration
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Failed to initialize systems: {str(e)}"
        }


def load_systems_into_session_state(selected_model: str, temperature: float) -> bool:
    """
    Load AI systems into Streamlit session state with model validation.
    
    Args:
        selected_model: OpenAI model to use
        temperature: Temperature parameter for models
        
    Returns:
        bool: True if systems loaded successfully, False otherwise
    """
    try:
        # Check if re-initialization is needed
        current_model = st.session_state.get('current_initialized_model')
        current_temp = st.session_state.get('current_initialized_temp')
        model_changed = (current_model != selected_model or current_temp != temperature)
        
        if model_changed:
            st.info(f"🔄 Model changed from {current_model} to {selected_model}. Reinitializing systems...")
        
        with st.spinner("🚀 Initializing AI systems..."):
            systems = initialize_systems()
            
            if systems['status'] != 'success':
                st.error(f"❌ {systems['error']}")
                return False
            
            # Store systems in session state
            st.session_state.systems_initialized = True
            st.session_state.current_initialized_model = selected_model
            st.session_state.current_initialized_temp = temperature
            st.session_state.chatbot = systems['chatbot']
            st.session_state.literature_miner = systems['literature_miner']
            st.session_state.psmiles_generator = systems['psmiles_generator']
            st.session_state.psmiles_processor = systems['psmiles_processor']
            st.session_state.psmiles_auto_corrector = systems['psmiles_auto_corrector']
            st.session_state.md_integration = systems['md_integration']
            st.session_state.md_integration_available = systems.get('md_integration') is not None
            
            st.success(f"✅ All systems initialized successfully with {selected_model}!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to load systems: {str(e)}")
        return False


def get_system_status() -> Dict[str, Any]:
    """
    Get the current status of all initialized systems.
    
    Returns:
        Dict[str, Any]: Status information for each system
    """
    status = {
        'systems_initialized': st.session_state.get('systems_initialized', False),
        'current_model': st.session_state.get('current_initialized_model', 'None'),
        'current_temperature': st.session_state.get('current_initialized_temp', 'None'),
        'chatbot_available': 'chatbot' in st.session_state and st.session_state.chatbot is not None,
        'literature_miner_available': 'literature_miner' in st.session_state and st.session_state.literature_miner is not None,
        'psmiles_generator_available': 'psmiles_generator' in st.session_state and st.session_state.psmiles_generator is not None,
        'psmiles_processor_available': 'psmiles_processor' in st.session_state and st.session_state.psmiles_processor is not None,
        'auto_corrector_available': 'psmiles_auto_corrector' in st.session_state and st.session_state.psmiles_auto_corrector is not None,
        'md_integration_available': st.session_state.get('md_integration_available', False)
    }
    
    return status


def clear_systems() -> None:
    """Clear all systems from session state."""
    system_keys = [
        'chatbot', 'literature_miner', 'psmiles_generator', 
        'psmiles_processor', 'psmiles_auto_corrector', 'md_integration'
    ]
    
    for key in system_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.systems_initialized = False
    st.session_state.current_initialized_model = None
    st.session_state.current_initialized_temp = None
    st.session_state.md_integration_available = False 


def check_system_health() -> Dict[str, Any]:
    """
    Check the health status of all initialized systems
    
    Returns:
        Dictionary containing health status of each system component
    """
    try:
        import streamlit as st
        
        health_status = {
            'overall_status': 'healthy',
            'systems_initialized': st.session_state.get('systems_initialized', False),
            'components': {}
        }
        
        # Check individual components
        components_to_check = [
            'chatbot_system',
            'literature_miner', 
            'psmiles_generator',
            'psmiles_processor'
        ]
        
        unhealthy_count = 0
        
        for component in components_to_check:
            if component in st.session_state and st.session_state[component] is not None:
                health_status['components'][component] = 'healthy'
            else:
                health_status['components'][component] = 'missing'
                unhealthy_count += 1
        
        # Determine overall status
        if unhealthy_count == 0:
            health_status['overall_status'] = 'healthy'
        elif unhealthy_count <= len(components_to_check) // 2:
            health_status['overall_status'] = 'degraded'
        else:
            health_status['overall_status'] = 'unhealthy'
        
        health_status['healthy_components'] = len(components_to_check) - unhealthy_count
        health_status['total_components'] = len(components_to_check)
        
        return health_status
        
    except Exception as e:
        return {
            'overall_status': 'error',
            'error': str(e),
            'systems_initialized': False,
            'components': {}
        }


def check_systems_initialized() -> bool:
    """
    Check if systems are properly initialized
    
    Returns:
        bool: True if systems are initialized, False otherwise
    """
    try:
        return st.session_state.get('systems_initialized', False)
    except:
        return False 