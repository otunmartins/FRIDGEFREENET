#!/usr/bin/env python3
"""
Comprehensive PDB Water Removal Tool using PDBFixer

This tool provides robust water removal capabilities for MD simulation preparation,
with options for selective removal that preserves important heterogens like polymer residues.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse

def remove_water_comprehensive(input_pdb: str, 
                             output_pdb: str = None,
                             preserve_heterogens: List[str] = None,
                             method: str = "selective",
                             ph: float = 7.4,
                             verbose: bool = True) -> Dict[str, any]:
    """
    Comprehensive water removal from PDB files using PDBFixer
    
    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path to output PDB file (if None, will use input name with suffix)
        preserve_heterogens: List of heterogen names to preserve (e.g., ['UNL', 'HEM'])
        method: Removal method ('selective', 'all_heterogens', 'waters_only')
        ph: pH for hydrogen addition (default: 7.4)
        verbose: Print detailed information
        
    Returns:
        Dict with removal statistics and results
    """
    
    try:
        # Import required modules
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile, Modeller
        from openmm import unit
        
        if verbose:
            print(f"🔧 Water Removal Tool")
            print(f"📁 Input PDB: {input_pdb}")
            print(f"🧪 Method: {method}")
            print(f"🌡️  pH: {ph}")
        
        # Generate output filename if not provided
        if output_pdb is None:
            input_path = Path(input_pdb)
            output_pdb = str(input_path.parent / f"{input_path.stem}_no_water{input_path.suffix}")
        
        if verbose:
            print(f"💾 Output PDB: {output_pdb}")
        
        # Initialize PDBFixer
        if verbose:
            print(f"\n🔍 Loading PDB file...")
        fixer = PDBFixer(filename=input_pdb)
        
        # Analyze initial structure
        initial_stats = analyze_pdb_composition(fixer.topology, verbose)
        
        # Default preserve list
        if preserve_heterogens is None:
            preserve_heterogens = ['UNL', 'HEM', 'FAD', 'NAD', 'ATP', 'ADP', 'GTP', 'GDP']
        
        if verbose:
            print(f"🛡️  Preserving heterogens: {preserve_heterogens}")
        
        # Apply water removal based on method
        if method == "selective":
            removal_stats = _remove_water_selective(fixer, preserve_heterogens, verbose)
        elif method == "all_heterogens":
            removal_stats = _remove_all_heterogens(fixer, verbose)
        elif method == "waters_only":
            removal_stats = _remove_waters_only(fixer, verbose)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Fix structure and add hydrogens
        if verbose:
            print(f"\n🔧 Fixing structure and adding hydrogens...")
        
        # Find and fix missing components
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        
        missing_residues = len(fixer.missingResidues)
        missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
        
        if verbose:
            if missing_residues > 0:
                print(f"   🔍 Found {missing_residues} missing residues")
            if missing_atoms > 0:
                print(f"   🔍 Found {missing_atoms} missing atoms")
        
        # Replace non-standard residues
        if missing_residues > 0:
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            if verbose:
                print(f"   ✅ Replaced non-standard residues")
        
        # Add missing atoms
        if missing_atoms > 0:
            fixer.addMissingAtoms()
            if verbose:
                print(f"   ✅ Added missing atoms")
        
        # Add hydrogens
        atoms_before = len(list(fixer.topology.atoms()))
        fixer.addMissingHydrogens(ph)
        atoms_after = len(list(fixer.topology.atoms()))
        hydrogens_added = atoms_after - atoms_before
        
        if verbose:
            print(f"   ➕ Added {hydrogens_added} hydrogen atoms at pH {ph}")
        
        # Analyze final structure
        final_stats = analyze_pdb_composition(fixer.topology, verbose)
        
        # Save the processed PDB
        if verbose:
            print(f"\n💾 Saving processed PDB...")
        
        with open(output_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        if verbose:
            print(f"✅ Water removal completed successfully!")
        
        # Compile results
        results = {
            'success': True,
            'input_file': input_pdb,
            'output_file': output_pdb,
            'method': method,
            'ph': ph,
            'preserve_heterogens': preserve_heterogens,
            'initial_stats': initial_stats,
            'final_stats': final_stats,
            'removal_stats': removal_stats,
            'hydrogens_added': hydrogens_added,
            'atoms_change': final_stats['total_atoms'] - initial_stats['total_atoms']
        }
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'input_file': input_pdb,
            'output_file': output_pdb
        }

def _remove_water_selective(fixer, preserve_heterogens: List[str], verbose: bool) -> Dict[str, int]:
    """Remove water while preserving specified heterogens"""
    
    if verbose:
        print(f"\n💧 Selective water removal...")
    
    # Analyze what we have
    water_residues = []
    preserved_heterogens = []
    other_heterogens = []
    
    standard_amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL'
    }
    
    for residue in fixer.topology.residues():
        if residue.name in ['HOH', 'WAT', 'TIP', 'TIP3', 'SPC']:
            water_residues.append(residue)
        elif residue.name in preserve_heterogens:
            preserved_heterogens.append(residue)
        elif residue.name not in standard_amino_acids:
            other_heterogens.append(residue)
    
    if verbose:
        print(f"   🌊 Water molecules to remove: {len(water_residues)}")
        print(f"   🛡️  Heterogens to preserve: {len(preserved_heterogens)}")
        print(f"   ❓ Other heterogens: {len(other_heterogens)}")
        
        if other_heterogens:
            other_names = list(set(res.name for res in other_heterogens))
            print(f"      Other heterogen types: {other_names}")
    
    # Remove only water using Modeller for precision
    if water_residues:
        from openmm.app import Modeller
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(water_residues)
        
        # Update fixer
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        
        if verbose:
            print(f"   ✅ Removed {len(water_residues)} water molecules")
            print(f"   ✅ Preserved {len(preserved_heterogens)} important heterogens")
    else:
        if verbose:
            print(f"   ℹ️  No water molecules found to remove")
    
    return {
        'water_removed': len(water_residues),
        'heterogens_preserved': len(preserved_heterogens),
        'other_heterogens': len(other_heterogens)
    }

def _remove_all_heterogens(fixer, verbose: bool) -> Dict[str, int]:
    """Remove all heterogens including water"""
    
    if verbose:
        print(f"\n🧪 Removing all heterogens...")
    
    # Count heterogens before removal
    heterogen_count = 0
    water_count = 0
    
    standard_amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL'
    }
    
    for residue in fixer.topology.residues():
        if residue.name in ['HOH', 'WAT', 'TIP', 'TIP3', 'SPC']:
            water_count += 1
        elif residue.name not in standard_amino_acids:
            heterogen_count += 1
    
    # Use PDBFixer's built-in method
    fixer.removeHeterogens(keepWater=False)
    
    if verbose:
        print(f"   🌊 Removed {water_count} water molecules")
        print(f"   🧪 Removed {heterogen_count} other heterogens")
    
    return {
        'water_removed': water_count,
        'heterogens_removed': heterogen_count,
        'total_removed': water_count + heterogen_count
    }

def _remove_waters_only(fixer, verbose: bool) -> Dict[str, int]:
    """Remove only water molecules, keep all other heterogens"""
    
    if verbose:
        print(f"\n💧 Water-only removal...")
    
    # Use PDBFixer's method with keepWater=False but preserve other heterogens
    initial_residues = list(fixer.topology.residues())
    water_residues = [res for res in initial_residues if res.name in ['HOH', 'WAT', 'TIP', 'TIP3', 'SPC']]
    
    if water_residues:
        from openmm.app import Modeller
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.delete(water_residues)
        
        # Update fixer
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions
        
        if verbose:
            print(f"   ✅ Removed {len(water_residues)} water molecules")
            print(f"   ✅ Preserved all other heterogens")
    else:
        if verbose:
            print(f"   ℹ️  No water molecules found")
    
    return {
        'water_removed': len(water_residues),
        'heterogens_preserved': len(initial_residues) - len(water_residues)
    }

def analyze_pdb_composition(topology, verbose: bool = True) -> Dict[str, int]:
    """Analyze the composition of a PDB topology"""
    
    composition = {
        'total_atoms': 0,
        'total_residues': 0,
        'protein_residues': 0,
        'water_residues': 0,
        'heterogens': 0,
        'chains': 0
    }
    
    standard_amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL'
    }
    
    heterogen_types = set()
    
    # Count atoms
    composition['total_atoms'] = len(list(topology.atoms()))
    
    # Count chains
    composition['chains'] = len(list(topology.chains()))
    
    # Analyze residues
    for residue in topology.residues():
        composition['total_residues'] += 1
        
        if residue.name in standard_amino_acids:
            composition['protein_residues'] += 1
        elif residue.name in ['HOH', 'WAT', 'TIP', 'TIP3', 'SPC']:
            composition['water_residues'] += 1
        else:
            composition['heterogens'] += 1
            heterogen_types.add(residue.name)
    
    if verbose:
        print(f"\n📊 PDB Composition Analysis:")
        print(f"   • Total atoms: {composition['total_atoms']}")
        print(f"   • Total residues: {composition['total_residues']}")
        print(f"   • Chains: {composition['chains']}")
        print(f"   • Protein residues: {composition['protein_residues']}")
        print(f"   • Water residues: {composition['water_residues']}")
        print(f"   • Heterogens: {composition['heterogens']}")
        
        if heterogen_types:
            print(f"   • Heterogen types: {sorted(list(heterogen_types))}")
    
    composition['heterogen_types'] = list(heterogen_types)
    
    return composition

def create_water_removal_report(results: Dict[str, any], output_dir: str = ".") -> str:
    """Create a detailed report of the water removal process"""
    
    if not results['success']:
        return f"Water removal failed: {results.get('error', 'Unknown error')}"
    
    report_lines = [
        "🔧 PDB Water Removal Report",
        "=" * 40,
        f"📁 Input file: {results['input_file']}",
        f"💾 Output file: {results['output_file']}",
        f"🧪 Method: {results['method']}",
        f"🌡️  pH: {results['ph']}",
        "",
        "📊 Before Removal:",
        f"   • Total atoms: {results['initial_stats']['total_atoms']}",
        f"   • Total residues: {results['initial_stats']['total_residues']}",
        f"   • Protein residues: {results['initial_stats']['protein_residues']}",
        f"   • Water residues: {results['initial_stats']['water_residues']}",
        f"   • Heterogens: {results['initial_stats']['heterogens']}",
        "",
        "📊 After Removal:",
        f"   • Total atoms: {results['final_stats']['total_atoms']}",
        f"   • Total residues: {results['final_stats']['total_residues']}",
        f"   • Protein residues: {results['final_stats']['protein_residues']}",
        f"   • Water residues: {results['final_stats']['water_residues']}",
        f"   • Heterogens: {results['final_stats']['heterogens']}",
        "",
        "🔄 Changes:",
        f"   • Water removed: {results['removal_stats'].get('water_removed', 0)}",
        f"   • Hydrogens added: {results['hydrogens_added']}",
        f"   • Net atom change: {results['atoms_change']}",
        "",
        "✅ Status: SUCCESS"
    ]
    
    # Save report
    report_path = os.path.join(output_dir, "water_removal_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return '\n'.join(report_lines)

def main():
    """Command-line interface for water removal tool"""
    
    parser = argparse.ArgumentParser(description="Remove water from PDB files using PDBFixer")
    parser.add_argument("input_pdb", help="Input PDB file")
    parser.add_argument("-o", "--output", help="Output PDB file")
    parser.add_argument("-m", "--method", choices=["selective", "all_heterogens", "waters_only"], 
                       default="selective", help="Removal method")
    parser.add_argument("-p", "--preserve", nargs="*", default=["UNL"], 
                       help="Heterogens to preserve (space-separated)")
    parser.add_argument("--ph", type=float, default=7.4, help="pH for hydrogen addition")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-r", "--report", help="Save report to directory")
    
    args = parser.parse_args()
    
    # Run water removal
    results = remove_water_comprehensive(
        input_pdb=args.input_pdb,
        output_pdb=args.output,
        preserve_heterogens=args.preserve,
        method=args.method,
        ph=args.ph,
        verbose=args.verbose
    )
    
    # Generate report if requested
    if args.report:
        report = create_water_removal_report(results, args.report)
        if args.verbose:
            print(f"\n📄 Report saved to: {args.report}")
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)

if __name__ == "__main__":
    main() 