"""
Sidebar UI Module for Insulin-AI Application

This module handles all sidebar functionality including navigation, 
system status monitoring, and debug tools.
"""

import streamlit as st
import os
import sys
from typing import Dict, Any, List

# Add the project root to Python path to ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Use absolute imports with better fallback handling
try:
    from app.utils.session_utils import safe_get_session_object, force_refresh_psmiles_processor, validate_psmiles_processor
    from app.services.system_service import initialize_systems
    IMPORTS_AVAILABLE = True
    print("✅ Sidebar UI: Successfully imported session utils and validation")
except ImportError as e1:
    print(f"⚠️ Sidebar UI: Primary import failed: {e1}")
    # Fallback for different import contexts
    try:
        from utils.session_utils import safe_get_session_object, force_refresh_psmiles_processor, validate_psmiles_processor
        from services.system_service import initialize_systems
        IMPORTS_AVAILABLE = True
        print("✅ Sidebar UI: Successfully imported with fallback paths")
    except ImportError as e2:
        print(f"⚠️ Sidebar UI: Fallback import also failed: {e2}")
        # Try importing from the app context
        try:
            # Direct import from the correct modules
            sys.path.append(os.path.join(project_root, 'app'))
            from utils.session_utils import safe_get_session_object, force_refresh_psmiles_processor, validate_psmiles_processor
            from services.system_service import initialize_systems
            IMPORTS_AVAILABLE = True
            print("✅ Sidebar UI: Successfully imported with app context")
        except ImportError as e3:
            print(f"❌ Sidebar UI: All imports failed: {e1}, {e2}, {e3}")
            IMPORTS_AVAILABLE = False
            
            # Define working dummy functions with better error reporting
            def safe_get_session_object(key):
                print(f"⚠️ Using dummy safe_get_session_object for {key}")
                return st.session_state.get(key) if hasattr(st, 'session_state') else None
                
            def force_refresh_psmiles_processor():
                print("⚠️ Using dummy force_refresh_psmiles_processor")
                return False, "Import error - functions not available"
                
            def validate_psmiles_processor(processor):
                print("⚠️ Using dummy validate_psmiles_processor - checking basic availability")
                if not processor:
                    return False
                # At least check if it's the right type of object
                return hasattr(processor, 'process_psmiles_workflow') and hasattr(processor, 'available')
                
            def initialize_systems():
                print("⚠️ Using dummy initialize_systems")
                return {'status': 'error', 'error': 'Import error - system initialization not available'}

# Check for debugging utilities
try:
    from utils.debug_tracer import tracer, enable_runtime_debugging
    DEBUGGING_AVAILABLE = True
except ImportError:
    DEBUGGING_AVAILABLE = False


def render_openai_configuration():
    """Render OpenAI API configuration section"""
    st.sidebar.header("🔑 OpenAI Configuration")
    
    # Check environment
    api_key_env = os.getenv("OPENAI_API_KEY")
    if api_key_env:
        st.sidebar.success("✅ OpenAI API Key found in environment")
    else:
        st.sidebar.warning("⚠️ No OpenAI API Key found in environment")
    
    # Manual input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("✅ OpenAI API Key configured")
        return True
    elif api_key_env:
        return True
    else:
        st.sidebar.error("❌ OpenAI API Key required to proceed")
        return False


def render_model_configuration():
    """Render model configuration section"""
    st.sidebar.header("🤖 Model Configuration")
    
    # Available models
    models = [
        "gpt-4-turbo-preview", 
        "gpt-4", 
        "gpt-3.5-turbo-16k", 
        "gpt-3.5-turbo"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Model",
        models,
        index=0,
        help="Select the OpenAI model to use"
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 2.0, 0.7,
        help="Controls randomness in responses"
    )
    
    return selected_model, temperature


def render_navigation():
    """Render main navigation and return selected page"""
    st.sidebar.title("Framework Navigation")
    
    # System Status
    render_system_status()
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Select Component",
        [
            "Framework Overview", 
            "Active Learning",
            "Active Learning Results",
            "Literature Mining (LLM)", 
            "PSMILES Generation", 
            "MD Simulation"
        ]
    )
    
    # Debug tools
    render_debug_tools()
    
    return page


def render_system_status():
    """Render system status and cache management section"""
    with st.sidebar.expander("🔧 System Status", expanded=False):
        st.markdown("**Cache Management**")
        
        # Check PSMILESProcessor status
        processor = safe_get_session_object('psmiles_processor')
        if processor and validate_psmiles_processor(processor):
            st.success("✅ PSMILESProcessor: OK")
            if hasattr(processor, 'process_psmiles_workflow_with_autorepair'):
                st.success("✅ Auto-Repair: Available")
            else:
                st.warning("⚠️ Auto-Repair: Missing")
        else:
            st.error("❌ PSMILESProcessor: Missing methods")
        
        # Quick fix button for missing auto-repair
        if st.button("🔧 Fix PSMILESProcessor", help="Refresh PSMILESProcessor with latest auto-repair functionality"):
            with st.spinner("Refreshing PSMILESProcessor..."):
                success, message = force_refresh_psmiles_processor()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        
        # Clear cache button
        if st.button("🔄 Force Refresh Systems", help="Clear cache and reinitialize all systems"):
            try:
                # Clear the cached systems
                initialize_systems.clear()
                st.session_state.systems_initialized = False
                
                # Force re-initialization
                with st.spinner("Refreshing systems..."):
                    systems = initialize_systems()
                    if systems['status'] == 'success':
                        st.session_state.systems_initialized = True
                        for key, value in systems.items():
                            if key != 'status':
                                st.session_state[key] = value
                        st.success("✅ Systems refreshed successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to refresh: {systems.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Refresh failed: {e}")
        
        st.markdown("---")


def render_debug_tools():
    """Render debugging tools section"""
    if DEBUGGING_AVAILABLE:
        with st.sidebar.expander("🔍 Debug Tools"):
            st.markdown("**Runtime Debugging**")
            
            debug_mode = st.selectbox(
                "Debug Mode",
                ["Off", "Signal-based", "Function Tracing", "Periodic Dumps"],
                help="Enable runtime debugging to monitor program execution"
            )
            
            if debug_mode == "Signal-based":
                if st.button("🚀 Enable Signal Debugging"):
                    try:
                        tracer.enable_signal_tracing()
                        st.success(f"✅ Signal debugging enabled!")
                        st.info(f"Send signal: `kill -USR1 {os.getpid()}`")
                    except Exception as e:
                        st.error(f"❌ Failed to enable signal debugging: {e}")
            
            elif debug_mode == "Function Tracing":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Enable Function Tracing"):
                        try:
                            tracer.enable_function_tracing()
                            st.success("✅ Function tracing enabled!")
                            st.warning("⚠️ This will be verbose - check console output")
                        except Exception as e:
                            st.error(f"❌ Failed to enable function tracing: {e}")
                
                with col2:
                    if st.button("🛑 Disable Function Tracing"):
                        try:
                            tracer.disable_function_tracing()
                            st.success("✅ Function tracing disabled!")
                        except Exception as e:
                            st.error(f"❌ Failed to disable function tracing: {e}")
            
            elif debug_mode == "Periodic Dumps":
                interval = st.slider("Dump Interval (seconds)", 10, 300, 30)
                if st.button("⏰ Start Periodic Dumps"):
                    try:
                        tracer.periodic_stack_dump(interval)
                        st.success(f"✅ Periodic stack dumps enabled (every {interval}s)")
                    except Exception as e:
                        st.error(f"❌ Failed to enable periodic dumps: {e}")
            
            # Process info
            if st.button("📊 Show Process Info"):
                st.info(f"**Process ID:** {os.getpid()}")
                st.info(f"**Working Directory:** {os.getcwd()}")
                
            # Manual stack trace
            if st.button("📋 Get Stack Trace Now"):
                try:
                    import traceback
                    import io
                    
                    # Capture current stack trace
                    f = io.StringIO()
                    traceback.print_stack(file=f)
                    stack_trace = f.getvalue()
                    
                    st.text_area("Current Stack Trace", stack_trace, height=300)
                except Exception as e:
                    st.error(f"❌ Failed to get stack trace: {e}")
    else:
        st.sidebar.info("🔍 Debug tools not available (debug_tracer.py not found)")


def render_sidebar() -> Dict[str, Any]:
    """
    Render complete sidebar and return configuration
    
    Returns:
        Dictionary containing sidebar configuration and selected page
    """
    # OpenAI configuration
    openai_configured = render_openai_configuration()
    
    # Model configuration
    if openai_configured:
        model, temperature = render_model_configuration()
    else:
        model, temperature = None, None
    
    # Navigation
    page = render_navigation()
    
    return {
        'openai_configured': openai_configured,
        'selected_model': model,
        'temperature': temperature,
        'selected_page': page
    } 