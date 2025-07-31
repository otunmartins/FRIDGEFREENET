"""
Enhanced Navigation Module for Insulin-AI Application

This module provides comprehensive navigation including system status monitoring,
cache management, and debug tools to match the original app's functionality.
"""

import streamlit as st
from typing import Optional
from app.utils.session_utils import safe_get_session_object

# Import validation functions (avoiding circular imports)
try:
    # Try to get validation functions from session state or define simple versions
    def validate_psmiles_processor(processor) -> bool:
        """Simple validation that PSMILESProcessor has required methods"""
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

except ImportError:
    # Fallback functions if imports fail
    def validate_psmiles_processor(processor):
        return processor is not None
    
    def force_refresh_psmiles_processor():
        return False, "Refresh not available"

# Check for debugging utilities
try:
    from utils.debug_tracer import tracer, enable_runtime_debugging
    DEBUGGING_AVAILABLE = True
except ImportError:
    DEBUGGING_AVAILABLE = False

def render_system_status_expander():
    """Render the system status and cache management section"""
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
                # Clear systems-related session state to force re-initialization
                keys_to_clear = ['psmiles_generator', 'psmiles_processor', 'literature_miner', 'chatbot', 'systems_initialized']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.session_state.systems_initialized = False
                st.success("✅ System cache cleared. Systems will re-initialize on next page load.")
                st.info("🔄 Please refresh the page or navigate to another page to complete the re-initialization.")
                st.rerun()
                     
            except Exception as e:
                st.error(f"❌ Refresh failed: {e}")
        
        st.markdown("---")

def render_debug_tools_expander():
    """Render debug tools section if available"""
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
                        import os
                        tracer.enable_signal_tracing()
                        st.success(f"✅ Signal debugging enabled!")
                        st.info(f"Send signal: `kill -USR1 {os.getpid()}`")
                    except Exception as e:
                        st.error(f"❌ Failed to enable signal debugging: {e}")
            
            elif debug_mode == "Function Tracing":
                if st.button("🔄 Enable Function Tracing"):
                    try:
                        tracer.enable_function_tracing()
                        st.success("✅ Function tracing enabled!")
                        st.warning("⚠️ This will be verbose - check console output")
                    except Exception as e:
                        st.error(f"❌ Failed to enable function tracing: {e}")
                
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
                import os
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

def render_navigation() -> str:
    """
    Render the enhanced sidebar navigation with comprehensive system monitoring
    
    Returns:
        str: The selected page name
    """
    # **NEW: Enhanced Sidebar Navigation matching old app**
    st.sidebar.title("Framework Navigation")

    # System Status & Cache Management
    render_system_status_expander()

    # Main page selection
    page = st.sidebar.selectbox(
        "Select Component",
        ["Framework Overview", "Literature Mining (LLM)", "PSMILES Generation", "Active Learning", "Material Evaluation", "MD Simulation", "Comprehensive Analysis"]
    )

    # Debugging Section
    render_debug_tools_expander()

    return page 