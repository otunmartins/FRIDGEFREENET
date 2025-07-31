#!/usr/bin/env python3
"""
Script to download and clean up PDB file 3I40 using PDBFixer
Removes water molecules and fixes CYS residue issues
"""

import urllib.request
import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

def download_pdb(pdb_id, output_path=None):
    """Download PDB file from RCSB"""
    if output_path is None:
        output_path = f"{pdb_id.lower()}.pdb"
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading {pdb_id} from RCSB...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded successfully: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading PDB file: {e}")
        return None

def cleanup_pdb_with_fixer(input_pdb, output_pdb=None):
    """Clean up PDB file using PDBFixer"""
    if output_pdb is None:
        base_name = os.path.splitext(input_pdb)[0]
        output_pdb = f"{base_name}_fixed.pdb"
    
    print(f"Loading PDB file: {input_pdb}")
    fixer = PDBFixer(filename=input_pdb)
    
    # Find and report issues
    print("Analyzing structure...")
    
    # Find missing residues
    fixer.findMissingResidues()
    if fixer.missingResidues:
        print(f"Found {len(fixer.missingResidues)} missing residues")
    
    # Find missing atoms
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        print(f"Found missing atoms in {len(fixer.missingAtoms)} residues")
    
    # Find non-standard residues
    fixer.findNonstandardResidues()
    if fixer.nonstandardResidues:
        print(f"Found {len(fixer.nonstandardResidues)} non-standard residues")
    
    # Replace non-standard residues with standard ones
    print("Replacing non-standard residues...")
    fixer.replaceNonstandardResidues()
    
    # Remove heterogens (including water)
    print("Removing water molecules and other heterogens...")
    fixer.removeHeterogens(keepWater=False)
    
    # Add missing residues
    if fixer.missingResidues:
        print("Adding missing residues...")
        fixer.addMissingResidues()
    
    # Add missing atoms
    if fixer.missingAtoms:
        print("Adding missing atoms...")
        fixer.addMissingAtoms()
    
    # Add missing hydrogens (optional - uncomment if needed)
    # print("Adding missing hydrogens...")
    # fixer.addMissingHydrogens(7.0)  # pH 7.0
    
    # Save the fixed structure
    print(f"Saving cleaned structure: {output_pdb}")
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))
    
    print("Cleanup completed successfully!")
    return output_pdb

def main():
    pdb_id = "3I40"
    
    # Download the PDB file
    downloaded_file = download_pdb(pdb_id)
    if not downloaded_file:
        return
    
    # Clean up the PDB file
    fixed_file = cleanup_pdb_with_fixer(downloaded_file)
    
    print(f"\nOriginal file: {downloaded_file}")
    print(f"Cleaned file: {fixed_file}")
    
    # Optional: Show file sizes for comparison
    if os.path.exists(downloaded_file) and os.path.exists(fixed_file):
        orig_size = os.path.getsize(downloaded_file)
        fixed_size = os.path.getsize(fixed_file)
        print(f"\nFile sizes:")
        print(f"Original: {orig_size:,} bytes")
        print(f"Fixed: {fixed_size:,} bytes")
        print(f"Reduction: {((orig_size - fixed_size) / orig_size * 100):.1f}%")

if __name__ == "__main__":
    main()
