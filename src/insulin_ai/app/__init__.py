"""
Streamlit Web Application for Insulin-AI

This module contains the web application interface components:
- UI components and pages
- Session management utilities  
- Styling and configuration
- Service integrations
"""

# Import UI components if available
try:
    from .ui import *
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False

try:
    from .utils.session_utils import initialize_session_state, safe_get_session_object
    SESSION_UTILS_AVAILABLE = True
except ImportError:
    SESSION_UTILS_AVAILABLE = False

__all__ = [
    "UI_COMPONENTS_AVAILABLE",
    "SESSION_UTILS_AVAILABLE",
]

# Add available components to __all__
if SESSION_UTILS_AVAILABLE:
    __all__.extend(["initialize_session_state", "safe_get_session_object"])

def main():
    """Main entry point for the Streamlit application"""
    from .app import main as app_main
    return app_main() 