"""
Integration systems for Insulin-AI

This module contains integration components for external systems:
- Analysis integration (MD simulations, comprehensive analysis)
- Automation systems (simulation automation)
- Corrections systems (PSMILES auto-correction)
- Data integration utilities
"""

# Import what's available
try:
    from .analysis.simple_md_integration import SimpleMDIntegration
    MD_INTEGRATION_AVAILABLE = True
except ImportError:
    MD_INTEGRATION_AVAILABLE = False

try:
    from .automation.simulation_automation import SimulationAutomation
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

try:
    from .corrections.psmiles_auto_corrector import create_psmiles_auto_corrector
    CORRECTIONS_AVAILABLE = True
except ImportError:
    CORRECTIONS_AVAILABLE = False

__all__ = [
    "MD_INTEGRATION_AVAILABLE",
    "AUTOMATION_AVAILABLE", 
    "CORRECTIONS_AVAILABLE",
]

# Add available systems to __all__
if MD_INTEGRATION_AVAILABLE:
    __all__.append("SimpleMDIntegration")
    
if AUTOMATION_AVAILABLE:
    __all__.append("SimulationAutomation")
    
if CORRECTIONS_AVAILABLE:
    __all__.append("create_psmiles_auto_corrector")
