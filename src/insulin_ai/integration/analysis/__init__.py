"""
Integration Analysis Package
Contains tools for MD simulation integration and comprehensive analysis.
"""

try:
    from .insulin_mmgbsa_calculator import InsulinMMGBSACalculator
    MMGBSA_AVAILABLE = True
except ImportError:
    MMGBSA_AVAILABLE = False

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

if MMGBSA_AVAILABLE:
    __all__.append('InsulinMMGBSACalculator')

if MD_INTEGRATION_AVAILABLE:
    __all__.append('MDSimulationIntegrator')

if COMPREHENSIVE_ANALYZER_AVAILABLE:
    __all__.append('InsulinComprehensiveAnalyzer')
