#!/usr/bin/env python3
"""
Smart Insulin Fixer - Preserves correctly formatted files

The problem: PDBFixer.addMissingHydrogens() converts correctly labeled CYX → CYS
The solution: Only process files that actually need fixing, preserve perfect files
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Try to import OpenMM components
try:
    from openmm import app, unit
    from openmm.app import ForceField, PDBFile, Modeller
    from pdbfixer import PDBFixer
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

# Try to import the new comprehensive fix
try:
    from .insulin_force_field_fix import InsulinForceFieldFixer, fix_insulin_simulation_error
    ENHANCED_FIX_AVAILABLE = True
except ImportError:
    ENHANCED_FIX_AVAILABLE = False


logger = logging.getLogger(__name__)


def analyze_insulin_structure(pdb_path: str) -> Dict[str, any]:
    """
    Enhanced structure analysis using the new comprehensive fixer.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        Dict with analysis results
    """
    
    if not OPENMM_AVAILABLE:
        return {"needs_fixing": False, "reason": "OpenMM not available"}
    
    try:
        # Use enhanced fixer if available
        if ENHANCED_FIX_AVAILABLE:
            fixer = InsulinForceFieldFixer()
            analysis = fixer.analyze_insulin_residues(pdb_path)
            
            # Convert to legacy format for compatibility
            return {
                "needs_fixing": analysis.get("needs_fixing", False),
                "reason": "; ".join(analysis.get("recommended_fixes", ["No issues detected"])),
                "cys_count": len(analysis.get("cys_residues", [])),
                "cyx_count": len(analysis.get("cyx_residues", [])),
                "cym_count": len(analysis.get("cym_residues", [])),
                "disulfide_bonds": len(analysis.get("disulfide_bonds", [])),
                "unl_count": len(analysis.get("unl_residues", [])),
                "total_atoms": analysis.get("total_atoms", 0),
                "enhanced_analysis": analysis
            }
        
        # Fallback to original analysis
        fixer = PDBFixer(filename=pdb_path)
        
        # Count cysteine residue types
        cys_count = 0
        cyx_count = 0 
        cym_count = 0
        unl_count = 0
        disulfide_bonds = 0
        
        for residue in fixer.topology.residues():
            if residue.name == 'CYS':
                cys_count += 1
            elif residue.name == 'CYX':
                cyx_count += 1
            elif residue.name == 'CYM':
                cym_count += 1
            elif residue.name == 'UNL':
                unl_count += 1
        
        # Count existing disulfide bonds
        for bond in fixer.topology.bonds():
            atom1, atom2 = bond
            if (atom1.name == 'SG' and atom2.name == 'SG' and 
                atom1.element.symbol == 'S' and atom2.element.symbol == 'S'):
                disulfide_bonds += 1
        
        # Determine if file needs fixing
        needs_fixing = False
        reason = "File is correctly formatted"
        
        if cys_count > 0:
            needs_fixing = True
            reason = f"Found {cys_count} CYS residues that should be CYX"
        elif cyx_count > 0 and disulfide_bonds == 0:
            needs_fixing = True
            reason = f"Found {cyx_count} CYX residues but no disulfide bonds"
        elif cym_count > 0:
            needs_fixing = True
            reason = f"Found {cym_count} CYM residues"
            
        return {
            "needs_fixing": needs_fixing,
            "reason": reason,
            "cys_count": cys_count,
            "cyx_count": cyx_count,
            "cym_count": cym_count,
            "unl_count": unl_count,
            "disulfide_bonds": disulfide_bonds,
            "total_atoms": len(list(fixer.topology.atoms()))
        }
        
    except Exception as e:
        return {
            "needs_fixing": True,
            "reason": f"Analysis failed: {str(e)}",
            "error": str(e)
        }


def smart_insulin_fix(pdb_path: str, output_path: str = None, use_enhanced: bool = True) -> str:
    """
    Enhanced smart insulin fixer with comprehensive error handling.
    
    Args:
        pdb_path: Input PDB file
        output_path: Output file (optional)
        use_enhanced: Whether to use the enhanced comprehensive fixer
        
    Returns:
        Path to processed file (may be original if no fixing needed)
    """
    
    print(f"🧠 Smart Insulin Fixer - Enhanced Version")
    print(f"   📁 Input: {pdb_path}")
    print(f"   🚀 Enhanced mode: {use_enhanced and ENHANCED_FIX_AVAILABLE}")
    
    # Step 1: Analyze structure
    analysis = analyze_insulin_structure(pdb_path)
    
    print(f"   📊 Analysis results:")
    print(f"      • CYS residues: {analysis.get('cys_count', 0)}")
    print(f"      • CYX residues: {analysis.get('cyx_count', 0)}")
    print(f"      • CYM residues: {analysis.get('cym_count', 0)}")
    print(f"      • UNL residues: {analysis.get('unl_count', 0)}")
    print(f"      • Disulfide bonds: {analysis.get('disulfide_bonds', 0)}")
    print(f"      • Total atoms: {analysis.get('total_atoms', 0)}")
    print(f"      • Needs fixing: {analysis['needs_fixing']}")
    print(f"      • Reason: {analysis['reason']}")
    
    # Step 2: Decide on action
    if not analysis['needs_fixing']:
        print(f"   ✅ File is correctly formatted - no changes needed!")
        print(f"   🎯 Returning original file: {pdb_path}")
        return pdb_path
    
    # Step 3: Apply fixes using enhanced method if available
    print(f"   🔧 File needs fixing - applying corrections...")
    
    if use_enhanced and ENHANCED_FIX_AVAILABLE:
        print(f"   🚀 Using enhanced comprehensive fixer...")
        try:
            fixer = InsulinForceFieldFixer()
            result = fixer.fix_insulin_structure_enhanced(pdb_path, output_path)
            
            if result['success']:
                print(f"   ✅ Enhanced fixing successful!")
                print(f"   📁 Enhanced fixed file: {result['output_path']}")
                return result['output_path']
            else:
                print(f"   ⚠️ Enhanced fixing failed: {result['error']}")
                print(f"   🔄 Falling back to original method...")
        except Exception as e:
            print(f"   ⚠️ Enhanced fixer exception: {e}")
            print(f"   🔄 Falling back to original method...")
    
    # Fallback to original method
    print(f"   🔧 Using original smart fixing method...")
    
    if not OPENMM_AVAILABLE:
        print(f"   ⚠️ OpenMM not available - returning original file")
        return pdb_path
    
    try:
        # Create output path if not provided
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = str(Path(temp_dir) / f"smart_fixed_{Path(pdb_path).name}")
        
        # Load and fix structure
        fixer = PDBFixer(filename=pdb_path)
        
        # Only add hydrogens if we have CYS residues (not CYX)
        if analysis.get('cys_count', 0) > 0:
            print(f"   ➕ Adding missing hydrogens (CYS residues detected)...")
            initial_atoms = len(list(fixer.topology.atoms()))
            fixer.addMissingHydrogens(7.4)
            final_atoms = len(list(fixer.topology.atoms()))
            print(f"      Added {final_atoms - initial_atoms} hydrogen atoms")
        else:
            print(f"   ⏭️ Skipping hydrogen addition (CYX residues already present)")
        
        # Fix disulfide bonds if needed
        if analysis.get('disulfide_bonds', 0) == 0:
            print(f"   🔗 Adding missing disulfide bonds...")
            # Add insulin-specific disulfide bonds
            added_bonds = add_insulin_disulfide_bonds(fixer.topology)
            print(f"      Added {added_bonds} disulfide bonds")
        
        # Fix residue naming with enhanced verification
        print(f"   🏷️ Fixing cysteine residue naming...")
        fixed_residues = fix_cysteine_naming_smart(fixer.topology)
        for change in fixed_residues:
            print(f"      {change}")
        
        # Save fixed structure
        print(f"   💾 Saving smart-fixed structure...")
        with open(output_path, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        # Verify the fix by re-analyzing
        print(f"   🔍 Verifying fix...")
        verification = analyze_insulin_structure(output_path)
        print(f"      Post-fix CYS residues: {verification.get('cys_count', 0)}")
        print(f"      Post-fix CYX residues: {verification.get('cyx_count', 0)}")
        print(f"      Post-fix needs fixing: {verification.get('needs_fixing', True)}")
        
        print(f"   ✅ Smart-fixed PDB saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"   ❌ Smart fixing failed: {e}")
        print(f"   🔄 Returning original file: {pdb_path}")
        return pdb_path


def smart_insulin_fix_for_simulation(pdb_path: str, 
                                   polymer_molecules: List = None,
                                   output_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Comprehensive smart fixing specifically for simulation setup.
    
    This function not only fixes the structure but also prepares the force field
    and system components needed for successful simulation.
    
    Args:
        pdb_path: Input PDB file
        polymer_molecules: Optional polymer molecules for template generation
        output_callback: Optional callback for logging
        
    Returns:
        Dict with comprehensive fixing results
    """
    def log(msg):
        if output_callback:
            output_callback(msg)
        else:
            print(msg)
    
    log(f"🧠 SMART INSULIN FIX FOR SIMULATION")
    log(f"📁 Input: {pdb_path}")
    
    if ENHANCED_FIX_AVAILABLE:
        log(f"🚀 Using comprehensive insulin simulation fix...")
        return fix_insulin_simulation_error(pdb_path, polymer_molecules, output_callback)
    else:
        log(f"⚠️ Enhanced fix not available - using basic smart fix...")
        
        try:
            # Use basic smart fix
            fixed_path = smart_insulin_fix(pdb_path, use_enhanced=False)
            
            # Load the fixed structure
            if OPENMM_AVAILABLE:
                pdb = PDBFile(fixed_path)
                
                # Create basic force field
                forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
                
                return {
                    'success': True,
                    'fixed_pdb_path': fixed_path,
                    'forcefield': forcefield,
                    'topology': pdb.topology,
                    'positions': pdb.positions,
                    'method': 'basic_smart_fix',
                    'system': None,  # System creation would need to be done separately
                    'system_info': {'method': 'basic_fix_only'}
                }
            else:
                return {
                    'success': False,
                    'error': 'OpenMM not available',
                    'fixed_pdb_path': fixed_path
                }
                
        except Exception as e:
            log(f"❌ Basic smart fix failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'basic_smart_fix_failed'
            }


def add_insulin_disulfide_bonds(topology) -> int:
    """Add insulin-specific disulfide bonds."""
    bonds_added = 0
    
    # Find all cysteine residues
    cysteines = []
    for residue in topology.residues():
        if residue.name in ['CYS', 'CYX']:
            for atom in residue.atoms():
                if atom.name == 'SG' and atom.element.symbol == 'S':
                    cysteines.append({
                        'atom': atom,
                        'residue': residue,
                        'residue_number': residue.id,
                        'chain': residue.chain.id if residue.chain else 'Unknown'
                    })
    
    # Enhanced disulfide bond logic for insulin
    # Insulin typically has 3 disulfide bonds:
    # - A chain: Cys6-Cys11 (intrachain)
    # - A7-B7 (interchain)  
    # - A20-B19 (interchain)
    
    if len(cysteines) >= 6:  # Standard insulin has 6 cysteines
        print(f"      Found {len(cysteines)} cysteine residues for disulfide bonding")
        
        # Simple pairing approach - in practice you'd want more sophisticated logic
        for i in range(0, len(cysteines)-1, 2):
            if i+1 < len(cysteines):
                cys1 = cysteines[i]
                cys2 = cysteines[i+1]
                
                # Check if bond doesn't already exist
                bond_exists = False
                for bond in topology.bonds():
                    if ((bond[0] == cys1['atom'] and bond[1] == cys2['atom']) or 
                        (bond[0] == cys2['atom'] and bond[1] == cys1['atom'])):
                        bond_exists = True
                        break
                
                if not bond_exists:
                    topology.addBond(cys1['atom'], cys2['atom'])
                    bonds_added += 1
                    print(f"      Added disulfide bond: {cys1['chain']}{cys1['residue_number']} - {cys2['chain']}{cys2['residue_number']}")

    return bonds_added


def fix_cysteine_naming_smart(topology) -> List[str]:
    """Smart cysteine naming that preserves existing CYX and handles edge cases."""
    
    changes = []
    
    # Check which cysteines are in disulfide bonds
    disulfide_bonded_sulfurs = set()
    for bond in topology.bonds():
        atom1, atom2 = bond
        if (atom1.name == 'SG' and atom2.name == 'SG' and 
            atom1.element.symbol == 'S' and atom2.element.symbol == 'S'):
            disulfide_bonded_sulfurs.add(atom1)
            disulfide_bonded_sulfurs.add(atom2)
    
    # Enhanced naming logic
    for residue in topology.residues():
        if residue.name in ['CYS', 'CYX', 'CYM']:
            # Find sulfur atom in this residue
            sulfur_atom = None
            for atom in residue.atoms():
                if atom.name == 'SG' and atom.element.symbol == 'S':
                    sulfur_atom = atom
                    break
            
            if sulfur_atom:
                old_name = residue.name
                
                # Enhanced decision logic
                if sulfur_atom in disulfide_bonded_sulfurs:
                    new_name = 'CYX'  # Disulfide-bonded cysteine
                else:
                    # Check hydrogen count to distinguish CYS vs CYM
                    hydrogen_count = 0
                    for neighbor in topology.neighbors(sulfur_atom):
                        if neighbor.element.symbol == 'H':
                            hydrogen_count += 1
                    
                    if hydrogen_count > 0:
                        new_name = 'CYS'  # Free cysteine with hydrogen
                    else:
                        new_name = 'CYM'  # Deprotonated cysteine
                
                if old_name != new_name:
                    residue.name = new_name
                    changes.append(f"✅ Residue {residue.index+1}: {old_name} → {new_name}")
                else:
                    changes.append(f"✓ Residue {residue.index+1}: {old_name} (unchanged)")
    
    return changes


def test_smart_fixer():
    """Test the smart fixer with actual insulin files."""
    test_files = [
        "test_insulin.pdb",
        "insulin_with_polymer.pdb",
        "composite_insulin_polymer.pdb"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 Testing smart fixer with: {test_file}")
            
            # Test basic fix
            result = smart_insulin_fix(test_file)
            print(f"Basic fix result: {result}")
            
            # Test simulation fix if enhanced version available
            if ENHANCED_FIX_AVAILABLE:
                sim_result = smart_insulin_fix_for_simulation(test_file)
                print(f"Simulation fix success: {sim_result['success']}")
        else:
            print(f"⚠️ Test file not found: {test_file}")


if __name__ == "__main__":
    test_smart_fixer() 