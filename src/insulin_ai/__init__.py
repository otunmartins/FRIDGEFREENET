"""
Insulin-AI: AI-Powered Drug Delivery System for Insulin Patch Materials Discovery

A comprehensive platform for intelligent material discovery and optimization for 
insulin delivery patches using AI, molecular dynamics simulations, and advanced 
polymer chemistry.
"""

__version__ = "0.1.0"
__author__ = "Insulin-AI Team"
__email__ = "contact@insulin-ai.org"
__license__ = "MIT"

# Core imports for easy access
from .core.chatbot_system import InsulinAIChatbot
from .core.literature_mining_system import MaterialsLiteratureMiner
from .core.psmiles_generator import PSMILESGenerator
from .core.psmiles_processor import PSMILESProcessor

# Integration systems
try:
    from .integration.analysis.simple_md_integration import SimpleMDIntegration
    MD_AVAILABLE = True
except ImportError:
    MD_AVAILABLE = False

try:
    from .integration.automation.simulation_automation import SimulationAutomation
    AUTOMATION_AVAILABLE = True
except ImportError:
    AUTOMATION_AVAILABLE = False

# Define public API
__all__ = [
    # Core systems
    "InsulinAIChatbot",
    "MaterialsLiteratureMiner", 
    "PSMILESGenerator",
    "PSMILESProcessor",
    
    # Integration systems (if available)
    "SimpleMDIntegration",
    "SimulationAutomation",
    
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # Availability flags
    "MD_AVAILABLE",
    "AUTOMATION_AVAILABLE",
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