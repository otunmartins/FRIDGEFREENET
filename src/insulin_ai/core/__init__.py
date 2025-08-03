"""
Core components for the Insulin-AI system.

This package contains the core AI systems including:
- InsulinAIChatbot: Conversational AI for material discovery
- MaterialsLiteratureMiner: AI-powered literature mining and analysis
- PSMILESGenerator: AI-powered polymer SMILES generation
- PSMILESProcessor: Processing and validation of polymer SMILES
"""

# Import core classes for package-level access
try:
    from .chatbot_system import InsulinAIChatbot
except ImportError as e:
    print(f"Warning: Failed to import InsulinAIChatbot: {e}")
    InsulinAIChatbot = None

try:
    from .literature_mining_system import MaterialsLiteratureMiner
except ImportError as e:
    print(f"Warning: Failed to import MaterialsLiteratureMiner: {e}")
    MaterialsLiteratureMiner = None

try:
    from .psmiles_generator import PSMILESGenerator
except ImportError as e:
    print(f"Warning: Failed to import PSMILESGenerator: {e}")
    PSMILESGenerator = None

try:
    from .psmiles_processor import PSMILESProcessor
except ImportError as e:
    print(f"Warning: Failed to import PSMILESProcessor: {e}")
    PSMILESProcessor = None

# Define what gets exported when using "from insulin_ai.core import *"
__all__ = [
    "InsulinAIChatbot",
    "MaterialsLiteratureMiner", 
    "PSMILESGenerator",
    "PSMILESProcessor",
]
