"""
Integration package for Insulin-AI system.

Contains modules for integrating external tools and services:
- Analysis: MD simulation integration and comprehensive analysis
- Automation: Simulation automation pipelines  
- Corrections: Auto-correction systems for chemical structures
- Data: Shared data files and configurations
"""

# Import analysis components
try:
    from .analysis import DualGaffAmberIntegration, MDSimulationIntegration, InsulinComprehensiveAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Analysis integration not fully available: {e}")
    ANALYSIS_AVAILABLE = False
    DualGaffAmberIntegration = None
    MDSimulationIntegration = None
    InsulinComprehensiveAnalyzer = None

# Import automation components (if available)
try:
    from .automation import AutomationPipeline
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False
    AutomationPipeline = None

# Define what gets exported
__all__ = []

if ANALYSIS_AVAILABLE:
    if DualGaffAmberIntegration:
        __all__.append('DualGaffAmberIntegration')
    if MDSimulationIntegration:
        __all__.append('MDSimulationIntegration')
    if InsulinComprehensiveAnalyzer:
        __all__.append('InsulinComprehensiveAnalyzer')

if AUTOMATION_AVAILABLE:
    __all__.append('AutomationPipeline')
