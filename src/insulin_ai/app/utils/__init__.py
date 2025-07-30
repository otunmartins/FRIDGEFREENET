"""
Utils Package

This package contains all utility modules for the Insulin-AI application,
providing session management, validation, workflow processing, and more.
"""

# Session management utilities
from .session_utils import (
    initialize_session_state,
    check_session_init,
    get_session_uuid,
    clear_session_cache,
    setup_openai_config,
    setup_model_selection
)

# Validation utilities  
from .validation_utils import (
    validate_psmiles_format,
    validate_file_upload,
    validate_simulation_parameters,
    validate_input_parameters,
    sanitize_filename,
    check_file_exists
)

# General app utilities
from .app_utils import (
    escape_psmiles_for_markdown,
    validate_session_state_object,
    ensure_systems_initialized,
    validate_psmiles_processor,
    force_refresh_psmiles_processor,
    safe_get_session_object,
    add_to_material_library,
    literature_mining_with_llm,
    psmiles_generation_with_llm,
    perform_real_copolymerization,
    parse_simulation_metrics,
    get_molecule_dimensions,
    format_time_duration,
    get_available_pdb_files,
    cleanup_temporary_files
)

# PSMILES workflow utilities
from .psmiles_workflow_utils import (
    display_psmiles_workflow,
    perform_dimerization,
    perform_copolymerization,
    perform_functional_group_addition,
    generate_fingerprints,
    get_inchi_info
)

# PSP AmorphousBuilder utilities
from .psp_utils import (
    build_amorphous_polymer_structure,
    vasp_to_pdb,
    display_3d_structure
)

# PDB preprocessing utilities
from .pdb_utils import (
    preprocess_pdb_standalone
)

# Export all functions for easy import
__all__ = [
    # Session management
    'initialize_session_state',
    'check_session_init',
    'get_session_uuid',
    'clear_session_cache',
    'setup_openai_config',
    'setup_model_selection',
    
    # Validation
    'validate_psmiles_format',
    'validate_file_upload',
    'validate_simulation_parameters', 
    'validate_input_parameters',
    'sanitize_filename',
    'check_file_exists',
    
    # General app utilities
    'escape_psmiles_for_markdown',
    'validate_session_state_object',
    'ensure_systems_initialized',
    'validate_psmiles_processor',
    'force_refresh_psmiles_processor',
    'safe_get_session_object',
    'add_to_material_library',
    'literature_mining_with_llm',
    'psmiles_generation_with_llm', 
    'perform_real_copolymerization',
    'parse_simulation_metrics',
    'get_molecule_dimensions',
    'format_time_duration',
    'get_available_pdb_files',
    'cleanup_temporary_files',
    
    # PSMILES workflow
    'display_psmiles_workflow',
    'perform_dimerization',
    'perform_copolymerization',
    'perform_functional_group_addition',
    'generate_fingerprints',
    'get_inchi_info',
    
    # PSP utilities
    'build_amorphous_polymer_structure',
    'vasp_to_pdb',
    'display_3d_structure',
    
    # PDB preprocessing
    'preprocess_pdb_standalone'
] 