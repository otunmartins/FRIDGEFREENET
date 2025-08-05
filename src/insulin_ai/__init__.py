"""
Insulin-AI: AI-Powered Drug Delivery System for Insulin Patch Materials Discovery

A comprehensive platform that combines AI, molecular dynamics simulations, and 
materials science for intelligent insulin delivery patch development.
"""

import warnings
# Suppress common warnings that occur during package import
warnings.filterwarnings('ignore', message='.*importing.*simtk.openmm.*deprecated.*')
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')

__version__ = "0.1.0"
__author__ = "Insulin-AI Team"
__email__ = "contact@insulin-ai.org"
__license__ = "MIT"

# Core imports for easy access
from .core.chatbot_system import InsulinAIChatbot
from .core.literature_mining_system import MaterialsLiteratureMiner
from .core.psmiles_generator import PSMILESGenerator
from .core.psmiles_processor import PSMILESProcessor

# Integration modules  
from .integration.analysis.dual_gaff_amber_integration import DualGaffAmberIntegration

# Define public API
__all__ = [
    # Core systems
    "InsulinAIChatbot",
    "MaterialsLiteratureMiner", 
    "PSMILESGenerator",
    "PSMILESProcessor",
    
    # Integration systems (if available)
    "DualGaffAmberIntegration",
    
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Availability flags
    "MD_AVAILABLE",
    "AUTOMATION_AVAILABLE",
    
    # Utility functions
    "get_version",
    "get_package_info",
    "get_insulin_pdb_path",
]

# Package-level configuration
import logging
import os
from pathlib import Path

# Set up package logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Define package paths
PACKAGE_ROOT = Path(__file__).parent
DATA_DIR = PACKAGE_ROOT / "data"
CONFIG_DIR = PACKAGE_ROOT / "config"
TEMPLATES_DIR = PACKAGE_ROOT / "templates"

# Create data directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Check availability of optional components
try:
    from .integration.analysis import DualGaffAmberIntegration
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False

try:
    from .integration.automation import AutomationPipeline
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

# Environment setup
def setup_environment():
    """Setup default environment variables and configurations."""
    # Set default OpenMM platform if not specified
    if "OPENMM_DEFAULT_PLATFORM" not in os.environ:
        os.environ["OPENMM_DEFAULT_PLATFORM"] = "CPU"
    
    # Set RDKit verbosity
    if "RDKIT_ERROR_LOGGING" not in os.environ:
        os.environ["RDKIT_ERROR_LOGGING"] = "ERROR"
    
    # Disable TensorFlow warnings if present
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Utility functions
def get_version():
    """Get the current version of the package."""
    return __version__

def get_insulin_pdb_path(filename="insulin_default.pdb"):
    """Get the path to an insulin PDB file.
    
    Args:
        filename: Name of the insulin PDB file to locate
        
    Returns:
        str: Full path to the insulin PDB file
    """
    insulin_dir = PACKAGE_ROOT / "integration" / "data" / "insulin"
    return str(insulin_dir / filename)

def get_package_info():
    """Get comprehensive package information."""
    return {
        "name": "insulin-ai",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "md_available": MD_AVAILABLE,
        "automation_available": AUTOMATION_AVAILABLE,
        "package_root": str(PACKAGE_ROOT),
        "data_dir": str(DATA_DIR),
        "config_dir": str(CONFIG_DIR),
    }

# Initialize package
setup_environment() 