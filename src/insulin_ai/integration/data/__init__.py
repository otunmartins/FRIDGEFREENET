"""
Data package for insulin-AI integration
Contains insulin structure files and other data resources
"""

__version__ = "0.1.0"

# Make data resources easily accessible
from pathlib import Path

# Define data directory paths
DATA_DIR = Path(__file__).parent
INSULIN_DIR = DATA_DIR / "insulin"

# Insulin structure files
INSULIN_PDB_FILES = {
    'output': 'output.pdb',
    'default': 'insulin_default.pdb', 
    '3i40': '3i40.pdb',
    'human_1mso': 'human_insulin_1mso.pdb'
}

def get_insulin_pdb_path(name='output'):
    """Get path to insulin PDB file by name"""
    if name in INSULIN_PDB_FILES:
        return str(INSULIN_DIR / INSULIN_PDB_FILES[name])
    else:
        # Default to output.pdb
        return str(INSULIN_DIR / 'output.pdb')

def list_insulin_files():
    """List all available insulin PDB files"""
    return list(INSULIN_PDB_FILES.keys()) 