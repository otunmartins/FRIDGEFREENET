"""
Integration Analysis Package
Contains tools for MD simulation integration and comprehensive analysis.
"""

# Remove MM-GBSA dependency that was causing import failures
# MM-GBSA functionality has been removed from the system

try:
    from .md_simulation_integration import MDSimulationIntegrator
    MD_INTEGRATION_AVAILABLE = True
except ImportError:
    MD_INTEGRATION_AVAILABLE = False

try:
    from .insulin_comprehensive_analyzer import InsulinComprehensiveAnalyzer
    COMPREHENSIVE_ANALYZER_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_ANALYZER_AVAILABLE = False

__all__ = []

if MD_INTEGRATION_AVAILABLE:
    __all__.append('MDSimulationIntegrator')

if COMPREHENSIVE_ANALYZER_AVAILABLE:
    __all__.append('InsulinComprehensiveAnalyzer')
