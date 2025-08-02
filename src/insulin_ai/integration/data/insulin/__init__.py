"""
Insulin structure data package
Contains PDB files for insulin structures used in simulations
"""

from pathlib import Path

# Define insulin data directory
INSULIN_DATA_DIR = Path(__file__).parent

# Available insulin structure files
AVAILABLE_STRUCTURES = {
    'output': 'output.pdb',
    'default': 'insulin_default.pdb',
    '3i40': '3i40.pdb', 
    'human_1mso': 'human_insulin_1mso.pdb',
    'default_backup': 'insulin_default_backup.pdb',
    'default_backup_preprocessed': 'insulin_default_backup_preprocessed.pdb'
}

def get_structure_path(name='output'):
    """Get full path to insulin structure file"""
    if name in AVAILABLE_STRUCTURES:
        return str(INSULIN_DATA_DIR / AVAILABLE_STRUCTURES[name])
    else:
        # Default fallback
        return str(INSULIN_DATA_DIR / 'output.pdb')

def list_available_structures():
    """List all available insulin structure names"""
    return list(AVAILABLE_STRUCTURES.keys())

# Export commonly used paths
OUTPUT_PDB = str(INSULIN_DATA_DIR / 'output.pdb')
DEFAULT_PDB = str(INSULIN_DATA_DIR / 'insulin_default.pdb') 