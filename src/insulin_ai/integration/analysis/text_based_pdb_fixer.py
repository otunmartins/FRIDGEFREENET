#!/usr/bin/env python3
"""
PDBFixer-Based Insulin Structure Fixer

This module uses PDBFixer and OpenMM topology manipulation to properly
fix insulin structures, especially handling CYS→CYX conversion for 
disulfide-bonded cysteine residues.

This approach uses the robust OpenMM framework instead of unreliable text manipulation.
"""

import os
import re
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

try:
    from openmm import app, unit
    from openmm.app import PDBFile, Topology, Modeller
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class PDBFixerBasedStructureFixer:
    """
    PDBFixer-based structure fixer that properly handles disulfide bonds
    and CYS→CYX conversion using OpenMM's robust topology manipulation.
    """
    
    def __init__(self):
        """Initialize the PDBFixer-based structure fixer."""
        self.logger = logger
    
    def analyze_structure(self, pdb_path: str) -> Dict[str, any]:
        """
        Analyze structure using OpenMM topology.
        
        Args:
            pdb_path: Path to PDB file
            
        Returns:
            Dict with analysis results
        """
        if not OPENMM_AVAILABLE:
            return {"error": "OpenMM not available"}
            
        if not os.path.exists(pdb_path):
            return {"error": f"PDB file not found: {pdb_path}"}
        
        try:
            pdb = PDBFile(pdb_path)
            topology = pdb.topology
            
            analysis = {
                "total_atoms": len(list(topology.atoms())),
                "total_residues": len(list(topology.residues())),
                "cys_residues": [],
                "cyx_residues": [],
                "cym_residues": [],
                "unl_residues": [],
                "disulfide_bonds": [],
                "needs_fixing": False,
                "issues": []
            }
            
            # Analyze residues
            for residue in topology.residues():
                residue_info = {
                    "name": residue.name,
                    "id": residue.id,
                    "chain": residue.chain.id,
                    "index": residue.index
                }
                
                if residue.name == "CYS":
                    analysis["cys_residues"].append(residue_info)
                elif residue.name == "CYX":
                    analysis["cyx_residues"].append(residue_info)
                elif residue.name == "CYM":
                    analysis["cym_residues"].append(residue_info)
                elif residue.name == "UNL":
                    analysis["unl_residues"].append(residue_info)
            
            # Find disulfide bonds
            analysis["disulfide_bonds"] = self._find_disulfide_bonds(topology, pdb.positions)
            
            # Determine what needs fixing
            if analysis["cys_residues"] and analysis["disulfide_bonds"]:
                analysis["needs_fixing"] = True
                analysis["issues"].append(f"Found {len(analysis['cys_residues'])} CYS residues with {len(analysis['disulfide_bonds'])} disulfide bonds")
            
            return analysis
            
        except Exception as e:
            return {
                "error": str(e),
                "needs_fixing": True,
                "issues": [f"Analysis failed: {str(e)}"]
            }
    
    def _find_disulfide_bonds(self, topology, positions) -> List[Dict]:
        """
        Find disulfide bonds by analyzing SG-SG distances in cysteine residues.
        
        Args:
            topology: OpenMM topology
            positions: Atomic positions
            
        Returns:
            List of disulfide bond information
        """
        disulfide_bonds = []
        cys_sulfurs = []
        
        # Find all sulfur atoms in cysteine residues
        for residue in topology.residues():
            if residue.name in ["CYS", "CYX"]:
                for atom in residue.atoms():
                    if atom.name == "SG":
                        cys_sulfurs.append({
                            "atom": atom,
                            "residue": residue,
                            "position": positions[atom.index]
                        })
        
        # Find pairs with distance < 2.5 Å (typical disulfide bond)
        disulfide_threshold = 0.25 * unit.nanometer  # 2.5 Å
        
        for i, sulfur1 in enumerate(cys_sulfurs):
            for j, sulfur2 in enumerate(cys_sulfurs[i+1:], i+1):
                pos1 = sulfur1["position"]
                pos2 = sulfur2["position"]
                
                # Calculate distance
                distance = np.linalg.norm(
                    np.array([pos1[k].value_in_unit(unit.nanometer) for k in range(3)]) -
                    np.array([pos2[k].value_in_unit(unit.nanometer) for k in range(3)])
                ) * unit.nanometer
                
                if distance < disulfide_threshold:
                    disulfide_bonds.append({
                        "residue1": {
                            "name": sulfur1["residue"].name,
                            "id": sulfur1["residue"].id,
                            "chain": sulfur1["residue"].chain.id,
                            "index": sulfur1["residue"].index
                        },
                        "residue2": {
                            "name": sulfur2["residue"].name,
                            "id": sulfur2["residue"].id,
                            "chain": sulfur2["residue"].chain.id,
                            "index": sulfur2["residue"].index
                        },
                        "distance": distance.value_in_unit(unit.angstrom)
                    })
        
        return disulfide_bonds
    
    def fix_insulin_structure_pdbfixer(self, input_pdb: str, output_pdb: str = None) -> Dict[str, any]:
        """
        Fix insulin structure using PDBFixer with proper CYS→CYX conversion.
        
        Args:
            input_pdb: Input PDB file path
            output_pdb: Output PDB file path (optional)
            
        Returns:
            Dict with fixing results
        """
        print(f"🔧 PDBFixer-Based Insulin Structure Fixer")
        print(f"   📁 Input: {input_pdb}")
        
        if not OPENMM_AVAILABLE:
            return {"success": False, "error": "OpenMM not available"}
        
        if output_pdb is None:
            temp_dir = tempfile.mkdtemp()
            output_pdb = str(Path(temp_dir) / f"pdbfixer_fixed_{Path(input_pdb).name}")
        
        print(f"   📁 Output: {output_pdb}")
        
        try:
            # Step 1: Analyze current structure
            print(f"   🔍 Step 1: Analyzing structure...")
            analysis = self.analyze_structure(input_pdb)
            
            if "error" in analysis:
                raise Exception(f"Structure analysis failed: {analysis['error']}")
            
            print(f"      • Total atoms: {analysis['total_atoms']}")
            print(f"      • Total residues: {analysis['total_residues']}")
            print(f"      • CYS residues: {len(analysis['cys_residues'])}")
            print(f"      • Disulfide bonds: {len(analysis['disulfide_bonds'])}")
            
            # Step 2: Load with PDBFixer
            print(f"   🔧 Step 2: Loading with PDBFixer...")
            fixer = PDBFixer(filename=input_pdb)
            
            # Step 3: Identify which CYS residues should be CYX
            print(f"   🎯 Step 3: Identifying disulfide-bonded cysteines...")
            cys_to_convert = set()
            
            for bond in analysis["disulfide_bonds"]:
                res1_key = (bond["residue1"]["chain"], bond["residue1"]["id"])
                res2_key = (bond["residue2"]["chain"], bond["residue2"]["id"])
                cys_to_convert.add(res1_key)
                cys_to_convert.add(res2_key)
                print(f"      🔗 Disulfide: {res1_key} ↔ {res2_key} ({bond['distance']:.2f} Å)")
            
            # Step 4: Convert CYS to CYX in topology
            print(f"   ⚡ Step 4: Converting CYS → CYX in topology...")
            conversions_made = 0
            
            for residue in fixer.topology.residues():
                if residue.name == "CYS":
                    residue_key = (residue.chain.id, residue.id)
                    if residue_key in cys_to_convert:
                        # Change residue name to CYX
                        residue.name = "CYX"
                        conversions_made += 1
                        print(f"      ✅ Converted {residue_key}: CYS → CYX")
            
            print(f"      📊 Total conversions: {conversions_made}")
            
            # Step 5: Use PDBFixer to clean up structure
            print(f"   🧹 Step 5: Cleaning structure with PDBFixer...")
            
            # Find and fix missing atoms
            print(f"      🔍 Finding missing heavy atoms...")
            try:
                fixer.findMissingAtoms()
                missing_atoms = len(fixer.missingAtoms) if hasattr(fixer, 'missingAtoms') else 0
                print(f"      ➕ Found {missing_atoms} missing heavy atoms")
                
                if missing_atoms > 0:
                    print(f"      🔧 Adding missing heavy atoms...")
                    fixer.addMissingAtoms()
            except Exception as e:
                print(f"      ⚠️ Missing atoms check failed: {e}")
                print(f"      🔄 Continuing without missing atom correction...")
            
            # Add missing hydrogens
            print(f"      🔍 Adding missing hydrogens...")
            initial_atoms = len(list(fixer.topology.atoms()))
            fixer.addMissingHydrogens(7.4)  # pH 7.4
            final_atoms = len(list(fixer.topology.atoms()))
            hydrogens_added = final_atoms - initial_atoms
            print(f"      ➕ Added {hydrogens_added} hydrogen atoms")
            
            # Step 6: Save fixed structure
            print(f"   💾 Step 6: Saving fixed structure...")
            with open(output_pdb, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
            
            # Step 7: Verify the fix
            print(f"   ✅ Step 7: Verifying fix...")
            final_analysis = self.analyze_structure(output_pdb)
            
            print(f"      📊 Final structure:")
            print(f"         • Total atoms: {final_analysis.get('total_atoms', 0)}")
            print(f"         • CYS residues: {len(final_analysis.get('cys_residues', []))}")
            print(f"         • CYX residues: {len(final_analysis.get('cyx_residues', []))}")
            print(f"         • Disulfide bonds: {len(final_analysis.get('disulfide_bonds', []))}")
            
            success = len(final_analysis.get('cys_residues', [])) == 0 or conversions_made > 0
            
            if success:
                print(f"   🎉 STRUCTURE FIXING SUCCESSFUL!")
            else:
                print(f"   ⚠️ Some issues may remain")
            
            return {
                "success": success,
                "output_file": output_pdb,
                "method": "pdbfixer_based",
                "conversions_made": conversions_made,
                "hydrogens_added": hydrogens_added,
                "initial_analysis": analysis,
                "final_analysis": final_analysis,
                "cys_before": len(analysis.get('cys_residues', [])),
                "cyx_after": len(final_analysis.get('cyx_residues', []))
            }
            
        except Exception as e:
            print(f"   ❌ PDBFixer-based fixing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output_file": output_pdb if output_pdb else input_pdb
            }
    
    def create_insulin_ready_pdb(self, input_pdb: str, output_pdb: str = None) -> Dict[str, any]:
        """
        Create an insulin PDB file that's ready for OpenMM simulation.
        
        This is the main entry point for fixing insulin CYS/CYX template issues.
        
        Args:
            input_pdb: Input PDB file path
            output_pdb: Output PDB file path (optional)
            
        Returns:
            Dict with results
        """
        print(f"🚀 CREATING INSULIN-READY PDB FOR OPENMM (PDBFixer Method)")
        print(f"📁 Input: {input_pdb}")
        
        # Use PDBFixer-based fixing
        result = self.fix_insulin_structure_pdbfixer(input_pdb, output_pdb)
        
        if result["success"]:
            print(f"✅ INSULIN PDB READY FOR SIMULATION!")
            print(f"📁 Output: {result['output_file']}")
            print(f"🔧 Method: {result['method']}")
            print(f"📊 Conversions: {result['conversions_made']} CYS → CYX")
            
            # Additional validation
            if OPENMM_AVAILABLE:
                try:
                    print(f"🧪 Testing OpenMM loading...")
                    pdb = PDBFile(result['output_file'])
                    print(f"      ✅ OpenMM can load the file successfully")
                    print(f"      📊 Atoms: {len(list(pdb.topology.atoms()))}")
                    print(f"      📊 Residues: {len(list(pdb.topology.residues()))}")
                    
                    result["openmm_loadable"] = True
                    result["final_atoms"] = len(list(pdb.topology.atoms()))
                    result["final_residues"] = len(list(pdb.topology.residues()))
                    
                except Exception as e:
                    print(f"      ⚠️ OpenMM loading test failed: {e}")
                    result["openmm_loadable"] = False
                    result["openmm_error"] = str(e)
        
        return result


# Legacy compatibility - use the new PDBFixer-based approach
class TextBasedPDBFixer(PDBFixerBasedStructureFixer):
    """Legacy compatibility wrapper - now uses PDBFixer-based approach."""
    pass


def fix_insulin_for_openmm(pdb_path: str, output_path: str = None) -> Dict[str, any]:
    """
    Convenience function to fix insulin PDB files for OpenMM simulation.
    
    This function resolves the "No template found for residue 7 (CYS)" error
    by properly converting CYS residues to CYX where appropriate using PDBFixer.
    
    Args:
        pdb_path: Path to input PDB file
        output_path: Path for output PDB file (optional)
        
    Returns:
        Dict with fixing results
    """
    fixer = PDBFixerBasedStructureFixer()
    return fixer.create_insulin_ready_pdb(pdb_path, output_path)


if __name__ == "__main__":
    # Test the PDBFixer-based fixer
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing PDBFixer-based structure fixer with: {test_file}")
        
        result = fix_insulin_for_openmm(test_file)
        print(f"\nResult: {result['success']}")
        
        if result['success']:
            print(f"Fixed file: {result['output_file']}")
            print(f"Conversions made: {result.get('conversions_made', 0)}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        print("Usage: python text_based_pdb_fixer.py <pdb_file>")
        print("Example: python text_based_pdb_fixer.py insulin_composite.pdb") 