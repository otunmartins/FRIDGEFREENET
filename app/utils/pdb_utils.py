"""
PDB Preprocessing Utilities

This module contains functions for preprocessing PDB files using PDBFixer,
including removal of water molecules, addition of missing atoms and hydrogens,
and other structure preparation tasks.
"""

import os
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable


def _remove_water_only(fixer, log_output):
    """
    Remove only water molecules (HOH, WAT) while preserving polymer (UNL) and other residues.
    
    Args:
        fixer: PDBFixer instance
        log_output: Logging function
    """
    try:
        # Get the current topology
        topology = fixer.topology
        
        # Find water residues to remove
        water_residues = []
        for residue in topology.residues():
            if residue.name in ['HOH', 'WAT']:
                water_residues.append(residue)
        
        if not water_residues:
            log_output("      No water molecules found to remove")
            return
        
        # Create a modeller to selectively remove water
        from openmm.app import Modeller
        modeller = Modeller(fixer.topology, fixer.positions)
        
        # Remove water residues
        modeller.delete(water_residues)
        log_output(f"      Removed {len(water_residues)} water molecules")
        
        # Update the fixer with the new topology and positions
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        
        # Verify polymer is still there
        unl_count = sum(1 for residue in fixer.topology.residues() if residue.name == 'UNL')
        log_output(f"      ✅ Preserved {unl_count} UNL polymer residues")
        
    except Exception as e:
        log_output(f"      ⚠️ Error in selective water removal: {e}")
        log_output("      Falling back to keeping all residues")


def preprocess_pdb_standalone(pdb_path: str, 
                            remove_water: bool = True,
                            remove_heterogens: bool = False,
                            add_missing_residues: bool = True,
                            add_missing_atoms: bool = True,
                            add_missing_hydrogens: bool = True,
                            ph: float = 7.0,
                            output_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Standalone PDB preprocessing function using PDBFixer
    
    Args:
        pdb_path: Path to input PDB file
        remove_water: Remove water molecules (HOH)
        remove_heterogens: Remove heterogens (except water)
        add_missing_residues: Add missing residues
        add_missing_atoms: Add missing atoms
        add_missing_hydrogens: Add missing hydrogens
        ph: pH for protonation state
        output_callback: Callback function for output messages
        
    Returns:
        Dict with preprocessing results
    """
    
    def log_output(message: str):
        if output_callback:
            output_callback(message)
        else:
            print(message)
    
    log_output(f"🔧 Starting PDB preprocessing: {pdb_path}")
    
    try:
        # Check if PDBFixer is available
        try:
            from pdbfixer import PDBFixer
            from openmm.app import PDBFile
        except ImportError:
            error_msg = "❌ PDBFixer not available. Please install with: conda install -c conda-forge pdbfixer"
            log_output(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'input_file': pdb_path,
                'timestamp': datetime.now().isoformat()
            }
        
        # Create output paths
        pdb_name = Path(pdb_path).stem
        preprocess_dir = Path(f"preprocessed_{pdb_name}_{uuid.uuid4().hex[:8]}")
        preprocess_dir.mkdir(exist_ok=True)
        
        output_path = preprocess_dir / f"{pdb_name}_processed.pdb"
        
        # Initialize PDBFixer
        log_output("   🔍 Loading PDB file...")
        fixer = PDBFixer(filename=pdb_path)
        
        # Analyze initial structure
        initial_atoms = len(list(fixer.topology.atoms()))
        initial_residues = len(list(fixer.topology.residues()))
        
        log_output(f"   📊 Initial structure: {initial_atoms} atoms, {initial_residues} residues")
        
        # Remove water molecules if requested
        if remove_water:
            log_output("   💧 Removing water molecules...")
            water_residues = []
            for residue in fixer.topology.residues():
                if residue.name in ['HOH', 'WAT']:
                    water_residues.append(residue)
            
            if water_residues:
                # Check if we have polymer (UNL residues) before removing heterogens
                has_polymer = any(residue.name == 'UNL' for residue in fixer.topology.residues())
                
                if has_polymer:
                    log_output("      🧬 Detected polymer (UNL) - removing water selectively...")
                    # Custom water removal that preserves polymer
                    _remove_water_only(fixer, log_output)
                    log_output(f"      Selectively removed {len(water_residues)} water molecules (preserved UNL)")
                else:
                    fixer.removeHeterogens(keepWater=False)
                    log_output(f"      Removed {len(water_residues)} water molecules")
            else:
                log_output("      No water molecules found")
        
        # Remove other heterogens if requested
        if remove_heterogens:
            # Check if we have polymer (UNL residues) - if so, preserve them
            has_polymer = any(residue.name == 'UNL' for residue in fixer.topology.residues())
            
            if has_polymer:
                log_output("   🧬 Detected polymer residues (UNL) - preserving them...")
                log_output("   ⚠️  Skipping heterogen removal to preserve polymer components")
                # Don't remove heterogens when polymer is present
            else:
                log_output("   🧪 Removing heterogens (no polymer detected)...")
                fixer.removeHeterogens(keepWater=not remove_water)
        
        # Find missing residues
        if add_missing_residues:
            log_output("   🔍 Finding missing residues...")
            fixer.findMissingResidues()
            
            missing_residues = len(fixer.missingResidues)
            if missing_residues > 0:
                log_output(f"      Found {missing_residues} missing residues")
            else:
                log_output("      No missing residues found")
        
        # Find missing atoms
        if add_missing_atoms:
            log_output("   ⚛️  Finding missing atoms...")
            fixer.findMissingAtoms()
            
            missing_atoms = len(fixer.missingAtoms)
            if missing_atoms > 0:
                log_output(f"      Found {missing_atoms} missing atoms")
            else:
                log_output("      No missing atoms found")
        
        # Add missing atoms (PDBFixer doesn't have addMissingResidues method)
        if add_missing_atoms and len(fixer.missingAtoms) > 0:
            log_output("   ➕ Adding missing atoms...")
            fixer.addMissingAtoms()
        
        # Add missing hydrogens
        if add_missing_hydrogens:
            log_output(f"   🎈 Adding missing hydrogens (pH {ph})...")
            initial_h_count = sum(1 for atom in fixer.topology.atoms() if atom.element.symbol == 'H')
            
            fixer.addMissingHydrogens(pH=ph)
            
            final_h_count = sum(1 for atom in fixer.topology.atoms() if atom.element.symbol == 'H')
            hydrogens_added = final_h_count - initial_h_count
            
            log_output(f"      Added {hydrogens_added} hydrogen atoms")
        
        # Final structure analysis
        final_atoms = len(list(fixer.topology.atoms()))
        final_residues = len(list(fixer.topology.residues()))
        
        log_output(f"   📊 Final structure: {final_atoms} atoms, {final_residues} residues")
        
        # Save processed structure
        log_output("   💾 Saving processed structure...")
        
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        log_output(f"   ✅ Preprocessing completed: {output_path}")
        
        # Create summary
        summary = {
            'success': True,
            'input_file': pdb_path,
            'output_file': str(output_path),
            'output_directory': str(preprocess_dir),
            'initial_atoms': initial_atoms,
            'final_atoms': final_atoms,
            'initial_residues': initial_residues,
            'final_residues': final_residues,
            'atoms_added': final_atoms - initial_atoms,
            'residues_added': final_residues - initial_residues,
            'settings': {
                'remove_water': remove_water,
                'remove_heterogens': remove_heterogens,
                'add_missing_residues': add_missing_residues,
                'add_missing_atoms': add_missing_atoms,
                'add_missing_hydrogens': add_missing_hydrogens,
                'ph': ph
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = preprocess_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
        
    except Exception as e:
        error_msg = f"❌ PDB preprocessing failed: {str(e)}"
        log_output(error_msg)
        
        return {
            'success': False,
            'error': str(e),
            'input_file': pdb_path,
            'timestamp': datetime.now().isoformat()
        } 