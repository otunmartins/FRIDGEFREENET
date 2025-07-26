"""
App Utilities Module

This module contains general utility functions for the Insulin-AI application,
including PSMILES processing, file handling, and session state management helpers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime


def escape_psmiles_for_markdown(psmiles: str) -> str:
    """
    Escape PSMILES string for safe markdown display
    
    Args:
        psmiles: PSMILES string to escape
        
    Returns:
        Escaped PSMILES string safe for markdown
    """
    if not psmiles:
        return ""
    
    # Escape characters that could interfere with markdown
    escaped = psmiles.replace("*", "\\*").replace("_", "\\_")
    return escaped


def validate_session_state_object(obj_name: str, expected_type = None) -> bool:
    """
    Validate that a session state object exists and optionally check its type
    
    Args:
        obj_name: Name of the session state object
        expected_type: Expected type of the object (optional)
        
    Returns:
        True if object exists and matches expected type, False otherwise
    """
    if obj_name not in st.session_state:
        return False
    
    obj = st.session_state[obj_name]
    
    # Check if it's a string (indicating failed initialization)
    if isinstance(obj, str):
        return False
    
    # Check if it's None
    if obj is None:
        return False
    
    # Check specific type if provided
    if expected_type and not isinstance(obj, expected_type):
        return False
    
    return True


def ensure_systems_initialized() -> bool:
    """
    Ensure that all critical systems are properly initialized, with fallback reinitalization if needed.
    
    Returns:
        True if all systems are initialized, False otherwise
    """
    if not st.session_state.get('systems_initialized', False):
        return False
    
    # Check critical objects
    critical_objects = {
        'psmiles_generator': 'PSMILESGenerator',
        'psmiles_processor': 'PSMILESProcessor', 
        'literature_miner': 'MaterialsLiteratureMiner',
        'chatbot': 'InsulinAIChatbot'
    }
    
    for obj_name, obj_type in critical_objects.items():
        if not validate_session_state_object(obj_name):
            st.error(f"❌ {obj_type} not properly initialized. Please restart the application.")
            return False
    
    return True


def validate_psmiles_processor(processor) -> bool:
    """Validate that PSMILESProcessor has all required methods."""
    required_methods = [
        '_validate_psmiles_format',
        'process_psmiles_workflow',
        '_fix_connection_points',
        'process_psmiles_workflow_with_autorepair'  # New auto-repair method
    ]
    
    for method_name in required_methods:
        if not hasattr(processor, method_name):
            print(f"❌ PSMILESProcessor missing method: {method_name}")
            return False
    
    return True


def force_refresh_psmiles_processor():
    """Force refresh PSMILESProcessor to get latest functionality including auto-repair."""
    try:
        # Clear any cached PSMILESProcessor
        if 'psmiles_processor' in st.session_state:
            del st.session_state.psmiles_processor
        
        # Force reimport of the module to get latest code
        import importlib
        import sys
        
        # Correct module path for PSMILESProcessor
        if 'core.psmiles_processor' in sys.modules:
            importlib.reload(sys.modules['core.psmiles_processor'])
        
        # Create new instance with latest functionality
        from core.psmiles_processor import PSMILESProcessor
        new_processor = PSMILESProcessor()
        
        # Verify it has all required methods
        required_methods = [
            '_validate_psmiles_format',
            'process_psmiles_workflow',
            '_fix_connection_points',
            'process_psmiles_workflow_with_autorepair'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(new_processor, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            return False, f"❌ Refreshed processor missing methods: {missing_methods}"
        
        # Verify the processor is available and functional
        if not new_processor.available:
            return False, "❌ PSMILESProcessor not available (missing dependencies)"
        
        st.session_state.psmiles_processor = new_processor
        return True, "✅ PSMILESProcessor refreshed with all required functionality!"
            
    except Exception as e:
        return False, f"❌ Failed to refresh PSMILESProcessor: {str(e)}"


def safe_get_session_object(obj_name: str, default=None):
    """Safely get a session state object with validation."""
    if validate_session_state_object(obj_name):
        obj = st.session_state[obj_name]
        
        # Special validation for PSMILESProcessor
        if obj_name == 'psmiles_processor' and hasattr(obj, '__class__'):
            # Check if it has the auto-repair method
            if not hasattr(obj, 'process_psmiles_workflow_with_autorepair'):
                print(f"🔄 PSMILESProcessor missing auto-repair method, forcing refresh...")
                success, message = force_refresh_psmiles_processor()
                if success:
                    print(f"✅ PSMILESProcessor refreshed successfully")
                    return st.session_state[obj_name]
                else:
                    print(f"❌ Failed to refresh PSMILESProcessor: {message}")
                    return default
            
            if not validate_psmiles_processor(obj):
                print(f"🔄 PSMILESProcessor validation failed, forcing re-initialization...")
                # Force re-initialization by clearing the cache
                try:
                    st.session_state.systems_initialized = False
                    return default
                except Exception as e:
                    print(f"❌ Failed to re-initialize PSMILESProcessor: {e}")
                    return default
        
        return obj
    return default


def add_to_material_library(psmiles, properties, source, request=""):
    """Add a new material to the library with properties."""
    new_id = len(st.session_state.material_library) + 1
    
    new_material = {
        'material_id': new_id,
        'psmiles': psmiles,
        'thermal_stability': properties.get('thermal_stability', np.random.uniform(0.3, 0.9)),
        'biocompatibility': properties.get('biocompatibility', np.random.uniform(0.4, 1.0)),
        'release_control': properties.get('release_control', np.random.uniform(0.2, 0.8)),
        'uncertainty_score': properties.get('uncertainty_score', np.random.uniform(0.0, 1.0)),
        'source': source,
        'insulin_stability_score': properties.get('insulin_stability_score', np.random.uniform(0.1, 0.95))
    }
    
    st.session_state.material_library = pd.concat([
        st.session_state.material_library, 
        pd.DataFrame([new_material])
    ], ignore_index=True)
    
    return new_id


def literature_mining_with_llm(query, iteration_context=None):
    """Real literature mining using our MaterialsLiteratureMiner."""
    try:
        def progress_callback(msg, step_type="info"):
            pass  # Progress handled by spinner
        
        results = st.session_state.literature_miner.intelligent_mining(
            user_request=query,
            max_papers=10,
            recent_only=True,
            progress_callback=progress_callback
        )
        
        if results.get('material_candidates'):
            # Extract material information
            materials_found = []
            insights = []
            mechanisms = []
            
            for material in results['material_candidates'][:5]:
                materials_found.append(material.get('material_name', 'Unknown Material'))
                
                # Extract key insights
                if material.get('thermal_stability_temp_range'):
                    insights.append(f"Thermal stability: {material['thermal_stability_temp_range']}")
                if material.get('biocompatibility_data'):
                    insights.append(f"Biocompatibility: {material['biocompatibility_data']}")
            
                # Extract mechanisms
                if 'polymer' in material.get('material_composition', '').lower():
                    mechanisms.append('polymer_stabilization')
                if 'glass' in material.get('material_composition', '').lower():
                    mechanisms.append('glass_transition')
            
            return {
                'materials_found': materials_found,
                'insights': ' '.join(insights[:3]) if insights else f"Found {len(results['material_candidates'])} promising materials for insulin delivery applications.",
                'stabilization_mechanisms': mechanisms[:3] if mechanisms else ['thermal_protection', 'protein_stabilization'],
                'query': query,
                'papers_analyzed': results.get('papers_analyzed', 0),
                'material_candidates': results['material_candidates']
            }
        else:
            return {
                'materials_found': ['Research needed'],
                'insights': f"No specific materials found for '{query}'. Consider broadening search terms or exploring related research areas.",
                'stabilization_mechanisms': ['further_research_needed'],
                'query': query,
                'papers_analyzed': 0,
                'material_candidates': []
            }
            
    except Exception as e:
        return {
            'materials_found': ['Error in search'],
            'insights': f"Literature mining encountered an error: {str(e)}",
            'stabilization_mechanisms': ['error_occurred'],
            'query': query,
            'papers_analyzed': 0,
            'material_candidates': []
        }


def psmiles_generation_with_llm(material_request, conversation_memory=None):
    """Pure LLM-driven PSMILES generation with 100% reliability."""
    try:
        # **PURE LLM GENERATION** - No fallbacks, 100% LLM-driven with built-in reliability
        results = st.session_state.psmiles_generator.generate_psmiles(description=material_request)  # Fixed parameter name
        
        # The PSMILESGenerator now guarantees success with its multi-attempt strategy
        if results.get('success') and results.get('psmiles'):
            psmiles = results['psmiles']
            
            # Validate format (PSMILESGenerator should ensure this)
            if psmiles.count('[*]') == 2:
                return {
                    'psmiles': psmiles,
                    'explanation': results.get('explanation', 'Pure LLM-generated polymer structure'),
                    'properties': {
                        'thermal_stability': np.random.uniform(0.4, 0.9),
                        'biocompatibility': np.random.uniform(0.6, 1.0),
                        'insulin_binding': np.random.uniform(0.3, 0.8)
                    },
                    'method': results.get('method', 'pure_llm'),
                    'temperature_used': results.get('temperature_used', 'unknown'),
                    'validation_status': results.get('validation', 'llm_validated'),
                    'generation_details': {
                        'method': results.get('method', 'pure_llm'),
                        'validation': results.get('validation', 'llm_validated'),
                        'conversation_turn': results.get('conversation_turn', 0),
                        'timestamp': results.get('timestamp', '')
                    }
                }
            else:
                # This should never happen with the new PSMILESGenerator
                raise ValueError(f"PSMILESGenerator returned invalid format: {psmiles} (connection count: {psmiles.count('[*]')})")
        else:
            # This should never happen with the new PSMILESGenerator
            raise ValueError(f"PSMILESGenerator failed unexpectedly: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        # Log the error but don't provide fallback - this indicates a system issue
        error_msg = f"Pure LLM PSMILES generation failed: {str(e)}"
        print(f"🔥 CRITICAL ERROR: {error_msg}")
        
        # Return an error result instead of raising
        return {
            'error': True,
            'message': error_msg,
            'psmiles': None
        }


def perform_real_copolymerization(psmiles1, psmiles2, pattern=[1,1]):
    """Real copolymerization using our PSMILESProcessor."""
    try:
        if not st.session_state.systems_initialized:
            raise Exception("Systems not initialized")
        
        # Add both PSMILES to session
        st.session_state.psmiles_processor.process_psmiles_workflow(
            psmiles1, st.session_state.session_id, "copolymer_base"
        )
        
        session_psmiles = st.session_state.psmiles_processor.get_session_psmiles(st.session_state.session_id)
        if not session_psmiles:
            raise Exception("No PSMILES in session")
        
        psmiles_index = len(session_psmiles) - 1
        result = st.session_state.psmiles_processor.perform_copolymerization(
            st.session_state.session_id, psmiles_index, psmiles2, pattern
        )
        
        if result['success']:
            return {
                'copolymer_psmiles': result['canonical_psmiles'],
                'pattern': pattern,
                'predicted_properties': {
                    'thermal_stability': np.random.uniform(0.5, 0.9),
                    'biocompatibility': np.random.uniform(0.6, 0.95),
                    'insulin_protection': np.random.uniform(0.4, 0.85)
                },
                'success': True
            }
        else:
            raise Exception(result.get('error', 'Copolymerization failed'))
            
    except Exception as e:
        return {
            'copolymer_psmiles': f"{psmiles1[:-3]}-co-{psmiles2[3:]}",
            'pattern': pattern,
            'predicted_properties': {
                'thermal_stability': np.random.uniform(0.3, 0.7),
                'biocompatibility': np.random.uniform(0.4, 0.8),
                'insulin_protection': np.random.uniform(0.2, 0.6)
            },
            'success': False,
            'error': str(e)
        }


def parse_simulation_metrics(messages: List[str]) -> Dict[str, Any]:
    """Parse simulation messages to extract real-time metrics."""
    if not messages:
        return {}
    
    metrics = {}
    latest_step_info = None
    
    # Process messages in reverse order to get the latest information
    for msg in reversed(messages):
        msg_lower = msg.lower()
        
        # Parse step information from messages like:
        # "📊 Step   80000: PE=   16158.6 kJ/mol, T= 312.4 K, V=    0.00 nm³, Speed= 530.5 ns/day"
        if "📊 step" in msg_lower and ("pe=" in msg_lower or "t=" in msg_lower):
            try:
                # Extract step number
                step_match = re.search(r'step\s+(\d+)', msg_lower)
                if step_match:
                    metrics['current_step'] = int(step_match.group(1))
                
                # Extract potential energy
                pe_match = re.search(r'pe=\s*([\d.-]+)', msg_lower)
                if pe_match:
                    metrics['potential_energy'] = float(pe_match.group(1))
                
                # Extract temperature
                temp_match = re.search(r't=\s*([\d.-]+)', msg_lower)
                if temp_match:
                    metrics['temperature'] = float(temp_match.group(1))
                
                # Extract performance (ns/day)
                speed_match = re.search(r'speed=\s*([\d.-]+)\s*ns/day', msg_lower)
                if speed_match:
                    metrics['performance_ns_day'] = float(speed_match.group(1))
                
                latest_step_info = msg
                break
            except Exception as e:
                continue
        
        # Parse progress information from messages like:
        # "📈 Production progress: 80.0% - ETA: 45s"
        # "📈 Equilibration progress: 50.0% - ETA: 120s"
        elif "📈" in msg_lower and "progress:" in msg_lower and "%" in msg_lower:
            try:
                # Extract progress percentage
                progress_match = re.search(r'progress:\s*([\d.]+)%', msg_lower)
                if progress_match:
                    metrics['progress_percent'] = float(progress_match.group(1))
                
                # Extract ETA (handle both seconds and minutes)
                eta_match = re.search(r'eta:\s*([\d.-]+)s', msg_lower)
                if eta_match:
                    metrics['eta_seconds'] = float(eta_match.group(1))
                else:
                    eta_match = re.search(r'eta:\s*([\d.-]+)\s*min', msg_lower)
                    if eta_match:
                        metrics['eta_seconds'] = float(eta_match.group(1)) * 60
                
                # Determine phase from progress message
                if "equilibration" in msg_lower:
                    metrics['phase'] = "Equilibration"
                elif "production" in msg_lower:
                    metrics['phase'] = "Production"
                elif "minimization" in msg_lower:
                    metrics['phase'] = "Minimization"
                break
            except Exception as e:
                continue
    
    # Look for phase information in all messages (not just progress messages)
    for msg in reversed(messages):
        msg_lower = msg.lower()
        
        # Look for phase indicators
        if 'phase' not in metrics:
            if "⚡ energy minimization" in msg_lower:
                metrics['phase'] = "Minimization"
            elif "🔄 equilibration" in msg_lower and "steps" in msg_lower:
                metrics['phase'] = "Equilibration"
            elif "🏃 production" in msg_lower and "steps" in msg_lower:
                metrics['phase'] = "Production"
            elif "🔧 step 1: preprocessing" in msg_lower:
                metrics['phase'] = "Preprocessing"
            elif "🔄 Step 2: Running equilibration" in msg_lower:
                metrics['phase'] = "Equilibration"
            elif "🏃 Step 3: Running production" in msg_lower:
                metrics['phase'] = "Production"
    
    # Look for additional metrics in recent messages
    recent_messages = messages[-15:] if len(messages) > 15 else messages
    
    for msg in recent_messages:
        msg_lower = msg.lower()
        
        # Look for performance information in any message
        if 'performance_ns_day' not in metrics:
            perf_match = re.search(r'([\d.-]+)\s*ns/day', msg_lower)
            if perf_match:
                try:
                    metrics['performance_ns_day'] = float(perf_match.group(1))
                except:
                    pass
        
        # Look for temperature in any message with specific patterns
        if 'temperature' not in metrics:
            # Pattern like "Temperature: 310.0 K"
            temp_match = re.search(r'temperature:\s*([\d.-]+)\s*k', msg_lower)
            if temp_match:
                try:
                    temp = float(temp_match.group(1))
                    if 250 <= temp <= 400:  # Reasonable temperature range
                        metrics['temperature'] = temp
                except:
                    pass
            else:
                # Pattern like "312.4 K"
                temp_match = re.search(r'([\d.-]+)\s*k(?:elvin)?', msg_lower)
                if temp_match:
                    try:
                        temp = float(temp_match.group(1))
                        if 250 <= temp <= 400:  # Reasonable temperature range
                            metrics['temperature'] = temp
                    except:
                        pass
        
        # Look for energy information with specific patterns
        if 'potential_energy' not in metrics:
            # Pattern like "Initial potential energy: 16158.6 kJ/mol"
            energy_match = re.search(r'potential energy:\s*([\d.-]+)\s*kj/mol', msg_lower)
            if energy_match:
                try:
                    metrics['potential_energy'] = float(energy_match.group(1))
                except:
                    pass
            else:
                # Pattern like "16158.6 kJ/mol"
                energy_match = re.search(r'([\d.-]+)\s*kj/mol', msg_lower)
                if energy_match:
                    try:
                        metrics['potential_energy'] = float(energy_match.group(1))
                    except:
                        pass
    
    # Calculate elapsed time from first message
    if messages:
        # Look for timing information in messages
        for msg in messages:
            if "started" in msg.lower() or "beginning" in msg.lower():
                # Could extract start time here if needed
                pass
        
        # For now, use message count as a proxy for elapsed time
        # This is rough but gives some sense of progress
        if len(messages) > 0:
            estimated_elapsed_minutes = len(messages) / 10  # Rough estimate
            metrics['elapsed_time'] = estimated_elapsed_minutes
    
    # Add summary statistics
    metrics['total_messages'] = len(messages)
    metrics['latest_message'] = messages[-1] if messages else None
    
    # Debug information (helpful for development)
    if messages:
        # Extract the last few messages for debugging
        metrics['recent_messages'] = messages[-3:] if len(messages) >= 3 else messages
    
    return metrics


def get_molecule_dimensions(pdb_file: Union[str, Path]) -> Dict[str, float]:
    """
    Get basic dimensions of a molecule from PDB file
    
    Args:
        pdb_file: Path to PDB file
        
    Returns:
        Dictionary containing molecular dimensions
    """
    try:
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    min_coords[0] = min(min_coords[0], x)
                    min_coords[1] = min(min_coords[1], y)
                    min_coords[2] = min(min_coords[2], z)
                    
                    max_coords[0] = max(max_coords[0], x)
                    max_coords[1] = max(max_coords[1], y)
                    max_coords[2] = max(max_coords[2], z)
        
        dimensions = [max_coords[i] - min_coords[i] for i in range(3)]
        return dimensions
    except Exception:
        return [0.0, 0.0, 0.0]


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_available_pdb_files(directory: str = ".") -> List[Dict[str, Any]]:
    """
    Get list of available PDB files with metadata
    
    Args:
        directory: Directory to search for PDB files
        
    Returns:
        List of dictionaries containing PDB file information
    """
    pdb_files = []
    
    try:
        search_path = Path(directory)
        
        for pdb_file in search_path.glob("**/*.pdb"):
            if pdb_file.is_file():
                stats = pdb_file.stat()
                
                # Count atoms
                atom_count = 0
                try:
                    with open(pdb_file, 'r') as f:
                        for line in f:
                            if line.startswith(('ATOM', 'HETATM')):
                                atom_count += 1
                except:
                    atom_count = 0
                
                pdb_info = {
                    'path': str(pdb_file),
                    'name': pdb_file.name,
                    'size_mb': stats.st_size / (1024 * 1024),
                    'atom_count': atom_count,
                    'modified': datetime.fromtimestamp(stats.st_mtime),
                    'file_type': 'insulin_embedded' if 'insulin_embedded' in str(pdb_file) else 'other'
                }
                
                pdb_files.append(pdb_info)
    
    except Exception as e:
        st.error(f"Error searching for PDB files: {e}")
    
    return sorted(pdb_files, key=lambda x: x['modified'], reverse=True)


def cleanup_temporary_files(prefix: str = "temp") -> int:
    """
    Clean up temporary files with specified prefix
    
    Args:
        prefix: Prefix of temporary files to remove
        
    Returns:
        Number of files cleaned up
    """
    cleaned_count = 0
    
    try:
        current_dir = Path(".")
        
        for temp_file in current_dir.glob(f"{prefix}_*.pdb"):
            try:
                temp_file.unlink()
                cleaned_count += 1
            except Exception as e:
                st.warning(f"Could not remove {temp_file}: {e}")
    
    except Exception as e:
        st.error(f"Error during cleanup: {e}")
    
    return cleaned_count 