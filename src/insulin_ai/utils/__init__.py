"""
Utility functions and classes for Insulin-AI

This module contains utility functions for:
- Molecular structure building and manipulation
- PSMILES to SMILES conversion
- Debugging and tracing utilities
- Molecular weight calculations
- Validation frameworks
"""

# Import available utilities
try:
    from .molecule_builder_utils import *
    MOLECULE_BUILDER_AVAILABLE = True
except ImportError:
    MOLECULE_BUILDER_AVAILABLE = False

try:
    from .psmiles_to_smiles_converter import PSMILESToSMILESConverter
    PSMILES_CONVERTER_AVAILABLE = True
except ImportError:
    PSMILES_CONVERTER_AVAILABLE = False

try:
    from .debug_tracer import tracer, enable_runtime_debugging
    DEBUG_TRACER_AVAILABLE = True
except ImportError:
    DEBUG_TRACER_AVAILABLE = False

try:
    from .molecular_weight_calculator import calculate_molecular_weight
    MOL_WEIGHT_CALCULATOR_AVAILABLE = True
except ImportError:
    MOL_WEIGHT_CALCULATOR_AVAILABLE = False

__all__ = [
    "MOLECULE_BUILDER_AVAILABLE",
    "PSMILES_CONVERTER_AVAILABLE", 
    "DEBUG_TRACER_AVAILABLE",
    "MOL_WEIGHT_CALCULATOR_AVAILABLE",
]

# Add available utilities to __all__
if PSMILES_CONVERTER_AVAILABLE:
    __all__.append("PSMILESToSMILESConverter")
    
if DEBUG_TRACER_AVAILABLE:
    __all__.extend(["tracer", "enable_runtime_debugging"])
    
if MOL_WEIGHT_CALCULATOR_AVAILABLE:
    __all__.append("calculate_molecular_weight")
