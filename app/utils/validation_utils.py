#!/usr/bin/env python3
"""
Validation Utilities for Insulin-AI App

This module contains validation functions and utility helpers
extracted from the monolithic app for better modularity and testability.
"""

import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


def escape_psmiles_for_markdown(psmiles: str) -> str:
    """
    Escape asterisks in PSMILES to prevent markdown interpretation.
    
    Args:
        psmiles: PSMILES string to escape
        
    Returns:
        str: Escaped PSMILES string safe for markdown display
    """
    if psmiles is None:
        return "None"
    return str(psmiles).replace('*', r'\*')


def validate_psmiles_format(psmiles: str) -> bool:
    """
    Validate a PSMILES string for correct format.
    
    Args:
        psmiles: PSMILES string to validate
        
    Returns:
        bool: True if valid PSMILES with exactly 2 connection points
    """
    if not psmiles or not isinstance(psmiles, str):
        return False
    
    # Check for exactly 2 connection points [*]
    connection_count = psmiles.count('[*]')
    return connection_count == 2


def validate_psmiles(psmiles: str) -> bool:
    """
    Validate a PSMILES string for correct format.
    
    Args:
        psmiles: PSMILES string to validate
        
    Returns:
        bool: True if valid PSMILES with exactly 2 connection points
    """
    return validate_psmiles_format(psmiles)


def validate_file_upload(uploaded_file) -> Dict[str, Any]:
    """
    Validate an uploaded file for size, type, and content.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Dict containing validation results and any error messages
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'file_info': {}
    }
    
    if uploaded_file is None:
        result['errors'].append("No file uploaded")
        return result
    
    # Get file info
    result['file_info'] = {
        'name': uploaded_file.name,
        'size': uploaded_file.size if hasattr(uploaded_file, 'size') else 0,
        'type': uploaded_file.type if hasattr(uploaded_file, 'type') else 'unknown'
    }
    
    # Size validation (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if result['file_info']['size'] > max_size:
        result['errors'].append(f"File size ({result['file_info']['size']} bytes) exceeds maximum allowed size ({max_size} bytes)")
    
    # File type validation
    allowed_extensions = ['pdb', 'sdf', 'mol', 'txt', 'csv', 'json']
    file_ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
    if file_ext not in allowed_extensions:
        result['errors'].append(f"File type '.{file_ext}' not allowed. Allowed types: {allowed_extensions}")
    
    # Set valid flag
    result['valid'] = len(result['errors']) == 0
    
    return result


def validate_simulation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate molecular dynamics simulation parameters.
    
    Args:
        params: Dictionary of simulation parameters
        
    Returns:
        Dict containing validation results and any error messages
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'sanitized_params': {}
    }
    
    # Default parameters
    default_params = {
        'temperature': 298.15,  # K
        'pressure': 1.0,        # atm
        'timestep': 0.002,      # ps
        'steps': 10000,         # simulation steps
        'equilibration_steps': 1000,
        'output_frequency': 100,
        'cutoff_distance': 12.0  # Angstroms
    }
    
    # Start with defaults
    result['sanitized_params'] = default_params.copy()
    
    # Validate each parameter
    for param_name, value in params.items():
        try:
            if param_name == 'temperature':
                temp = float(value)
                if temp <= 0:
                    result['errors'].append("Temperature must be positive")
                elif temp < 250 or temp > 400:
                    result['warnings'].append(f"Temperature {temp}K is outside typical range (250-400K)")
                else:
                    result['sanitized_params'][param_name] = temp
            
            elif param_name == 'pressure':
                press = float(value)
                if press <= 0:
                    result['errors'].append("Pressure must be positive")
                else:
                    result['sanitized_params'][param_name] = press
            
            elif param_name in ['timestep']:
                ts = float(value)
                if ts <= 0:
                    result['errors'].append(f"{param_name} must be positive")
                elif ts > 0.01:
                    result['warnings'].append(f"Timestep {ts} ps may be too large for stable simulation")
                else:
                    result['sanitized_params'][param_name] = ts
            
            elif param_name in ['steps', 'equilibration_steps', 'output_frequency']:
                steps = int(value)
                if steps <= 0:
                    result['errors'].append(f"{param_name} must be positive integer")
                else:
                    result['sanitized_params'][param_name] = steps
            
            elif param_name == 'cutoff_distance':
                cutoff = float(value)
                if cutoff <= 0:
                    result['errors'].append("Cutoff distance must be positive")
                elif cutoff < 8.0:
                    result['warnings'].append("Cutoff distance < 8.0 Å may be too small")
                else:
                    result['sanitized_params'][param_name] = cutoff
            
            else:
                # Unknown parameter - keep as is with warning
                result['warnings'].append(f"Unknown parameter: {param_name}")
                result['sanitized_params'][param_name] = value
                
        except (ValueError, TypeError) as e:
            result['errors'].append(f"Invalid value for {param_name}: {value} ({str(e)})")
    
    # Set valid flag
    result['valid'] = len(result['errors']) == 0
    
    return result


def validate_input_parameters(params: Dict[str, Any], required_params: List[str] = None) -> Dict[str, Any]:
    """
    Validate general input parameters.
    
    Args:
        params: Dictionary of input parameters
        required_params: List of required parameter names
        
    Returns:
        Dict containing validation results and any error messages
    """
    result = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'missing_params': [],
        'invalid_params': []
    }
    
    if required_params is None:
        required_params = []
    
    # Check for required parameters
    for param_name in required_params:
        if param_name not in params:
            result['missing_params'].append(param_name)
            result['errors'].append(f"Missing required parameter: {param_name}")
        elif params[param_name] is None:
            result['missing_params'].append(param_name)
            result['errors'].append(f"Required parameter '{param_name}' is None")
    
    # Validate parameter types and values
    for param_name, value in params.items():
        if value is None and param_name in required_params:
            continue  # Already handled above
        
        # Basic type validations
        if param_name.endswith('_count') or param_name.endswith('_steps'):
            try:
                int_val = int(value)
                if int_val < 0:
                    result['invalid_params'].append(param_name)
                    result['errors'].append(f"{param_name} must be non-negative integer")
            except (ValueError, TypeError):
                result['invalid_params'].append(param_name)
                result['errors'].append(f"{param_name} must be an integer")
        
        elif param_name.endswith('_rate') or param_name.endswith('_factor'):
            try:
                float_val = float(value)
                if float_val < 0:
                    result['invalid_params'].append(param_name)
                    result['errors'].append(f"{param_name} must be non-negative")
            except (ValueError, TypeError):
                result['invalid_params'].append(param_name)
                result['errors'].append(f"{param_name} must be a number")
        
        elif param_name in ['psmiles']:
            if not validate_psmiles_format(value):
                result['invalid_params'].append(param_name)
                result['errors'].append(f"Invalid PSMILES format: {value}")
    
    # Set valid flag
    result['valid'] = len(result['errors']) == 0
    
    return result


def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file exists and is accessible.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file exists and is accessible
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except (OSError, ValueError):
        return False


def validate_property_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """
    Validate that a property value is within expected range.
    
    Args:
        value: Property value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        bool: True if value is within range
    """
    try:
        float_val = float(value)
        return min_val <= float_val <= max_val
    except (ValueError, TypeError):
        return False


def validate_material_properties(properties: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate material properties dictionary.
    
    Args:
        properties: Dictionary of material properties
        
    Returns:
        Dict[str, bool]: Validation results for each property
    """
    validation_results = {}
    
    # Standard property validations
    property_validators = {
        'thermal_stability': lambda x: validate_property_range(x, 0.0, 1.0),
        'biocompatibility': lambda x: validate_property_range(x, 0.0, 1.0),
        'release_control': lambda x: validate_property_range(x, 0.0, 1.0),
        'insulin_binding': lambda x: validate_property_range(x, 0.0, 1.0),
        'insulin_stability_score': lambda x: validate_property_range(x, 0.0, 1.0),
        'uncertainty_score': lambda x: validate_property_range(x, 0.0, 1.0)
    }
    
    for prop_name, validator in property_validators.items():
        if prop_name in properties:
            validation_results[prop_name] = validator(properties[prop_name])
        else:
            validation_results[prop_name] = False
    
    return validation_results


def validate_simulation_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate molecular dynamics simulation configuration.
    
    Args:
        config: Simulation configuration dictionary
        
    Returns:
        List[str]: List of validation error messages (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = ['temperature', 'steps', 'timestep']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
            continue
        
        # Type and range validations
        if field == 'temperature':
            try:
                temp = float(config[field])
                if temp <= 0:
                    errors.append("Temperature must be positive")
            except (ValueError, TypeError):
                errors.append("Temperature must be a valid number")
        
        elif field == 'steps':
            try:
                steps = int(config[field])
                if steps <= 0:
                    errors.append("Steps must be positive integer")
            except (ValueError, TypeError):
                errors.append("Steps must be a valid integer")
        
        elif field == 'timestep':
            try:
                timestep = float(config[field])
                if timestep <= 0:
                    errors.append("Timestep must be positive")
            except (ValueError, TypeError):
                errors.append("Timestep must be a valid number")
    
    return errors


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Ensure not empty
    if not sanitized:
        sanitized = "unnamed_file"
    
    return sanitized


def validate_api_key(api_key: str, service_name: str) -> bool:
    """
    Basic validation for API keys.
    
    Args:
        api_key: API key to validate
        service_name: Name of the service (for specific validations)
        
    Returns:
        bool: True if API key appears valid
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Remove whitespace
    api_key = api_key.strip()
    
    # Basic length check (most API keys are at least 20 characters)
    if len(api_key) < 20:
        return False
    
    # Service-specific validations
    if service_name.lower() == 'openai':
        return api_key.startswith('sk-')
    
    # For other services, just check that it's non-empty and reasonable length
    return True


def validate_molecule_data(molecule_data: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate molecule data structure.
    
    Args:
        molecule_data: Dictionary containing molecule information
        
    Returns:
        Dict[str, bool]: Validation results for each field
    """
    validation_results = {}
    
    # Required fields
    required_fields = ['psmiles', 'material_id', 'source']
    for field in required_fields:
        validation_results[field] = field in molecule_data and molecule_data[field] is not None
    
    # PSMILES-specific validation
    if 'psmiles' in molecule_data:
        validation_results['psmiles_format'] = validate_psmiles(molecule_data['psmiles'])
    
    # Property validations
    if 'properties' in molecule_data:
        prop_validations = validate_material_properties(molecule_data['properties'])
        validation_results.update(prop_validations)
    
    return validation_results


def check_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if a filename has an allowed extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions (without dots)
        
    Returns:
        bool: True if extension is allowed
    """
    if not filename or not allowed_extensions:
        return False
    
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    return file_ext in [ext.lower() for ext in allowed_extensions] 


def validate_psmiles_processor(processor) -> bool:
    """
    Validate that a PSMILES processor has the required methods
    
    Args:
        processor: The PSMILES processor object to validate
        
    Returns:
        bool: True if processor has all required methods, False otherwise
    """
    if not processor:
        return False
    
    # Check the correct methods that actually exist in PSMILESProcessor
    required_methods = [
        'process_psmiles_workflow',
        '_validate_psmiles_format',
        '_fix_connection_points',
        'process_psmiles_workflow_with_autorepair'
    ]
    
    for method_name in required_methods:
        if not hasattr(processor, method_name):
            print(f"❌ PSMILESProcessor missing method: {method_name}")
            return False
        if not callable(getattr(processor, method_name)):
            print(f"❌ PSMILESProcessor method not callable: {method_name}")
            return False
    
    # Check if processor is available (has psmiles library)
    if hasattr(processor, 'available') and not processor.available:
        print("❌ PSMILESProcessor: psmiles library not available")
        return False
    
    return True 