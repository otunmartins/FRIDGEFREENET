#!/usr/bin/env python3
"""
Enhanced Polymer Builder with Insulin Integration
"""

import os
import uuid
import pandas as pd
import subprocess
import shutil
import random

from datetime import datetime
from typing import Dict, Optional

def build_insulin_polymer_composite(
    psmiles: str,
    insulin_pdb_path: str,
    polymer_length: int = 10,
    num_polymer_molecules: int = 20,
    num_insulin_molecules: int = 5,
    density: float = 0.8,
    box_size_nm: float = 3.0,
    insulin_distribution: str = "random"
) -> Dict:
    """Build composite structure with polymer and insulin molecules using large box with random edges approach."""
    
    print(f"🧬 Building insulin-polymer composite...")
    print(f"   Polymer: {psmiles}")
    print(f"   Insulin molecules: {num_insulin_molecules}")
    print(f"   Polymer molecules: {num_polymer_molecules}")
    print(f"   Approach: Large box with random edges")
    
    # Create output directory
    output_dir = f'insulin_polymer_output_{uuid.uuid4().hex[:8]}'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Build polymer structure using PSP
        print("🔧 Step 1: Building polymer structure with PSP...")
        
        # Import PSP
        import psp.AmorphousBuilder as ab
        
        # Create polymer input
        input_data = {
            'ID': ['polymer'],
            'smiles': [psmiles],
            'Len': [polymer_length],
            'Num': [num_polymer_molecules],
            'NumConf': [1],
            'LeftCap': ['H'],
            'RightCap': ['H'],
            'Loop': [False]
        }
        
        input_df = pd.DataFrame(input_data)
        
        builder = ab.Builder(
            input_df,
            ID_col="ID",
            SMILES_col="smiles",
            NumMole="Num",
            Length='Len',
            NumConf='NumConf',
            OutFile='polymer_structure',
            OutDir=output_dir,
            density=density,
            box_type='c',
            tol_dis=2.0
        )
        
        builder.Build()
        
        # Step 2: Find the generated polymer PDB file
        print("🔍 Step 2: Locating polymer structure files...")
        
        molecules_dir = os.path.join(output_dir, 'molecules')
        polymer_pdb = None
        
        if os.path.exists(molecules_dir):
            for file in os.listdir(molecules_dir):
                if file.endswith('.pdb') and 'polymer' in file.lower():
                    polymer_pdb = os.path.join(molecules_dir, file)
                    break
        
        if not polymer_pdb or not os.path.exists(polymer_pdb):
            return {
                'success': False,
                'error': 'Polymer PDB file not found after PSP build'
            }
        
        print(f"   Found polymer PDB: {polymer_pdb}")
        
        # Step 3: Create large box configuration with random edges using PACKMOL
        print("🏗️ Step 3: Creating large box with random edges using PACKMOL...")
        
        composite_pdb = os.path.join(output_dir, 'insulin_polymer_composite.pdb')
        
        success = create_large_box_with_random_edges(
            polymer_pdb_path=polymer_pdb,
            insulin_pdb_path=insulin_pdb_path,
            output_pdb_path=composite_pdb,
            num_insulin_molecules=num_insulin_molecules,
            num_polymer_duplicates=num_polymer_molecules,
            box_size_nm=box_size_nm,
            output_dir=output_dir
        )
        
        if not success:
            return {
                'success': False,
                'error': 'Failed to create large box composite with PACKMOL'
            }
        
        # Step 4: Calculate final statistics
        print("📊 Step 4: Calculating final statistics...")
        
        total_atoms = count_atoms_in_pdb(composite_pdb)
        
        result = {
            'success': True,
            'output_directory': output_dir,
            'composite_pdb': composite_pdb,
            'polymer_pdb': polymer_pdb,
            'total_atoms': total_atoms,
            'num_polymer_molecules': num_polymer_molecules,
            'num_insulin_molecules': num_insulin_molecules,
            'build_timestamp': datetime.now().isoformat(),
            'approach': 'large_box_random_edges'
        }
        
        print("✅ Successfully built insulin-polymer composite!")
        return result
        
    except ImportError:
        return {
            'success': False,
            'error': 'PSP package not available. Please install PSP.'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Composite building failed: {str(e)}'
        }

def create_large_box_with_random_edges(
    polymer_pdb_path: str,
    insulin_pdb_path: str,
    output_pdb_path: str,
    num_insulin_molecules: int = 1,
    num_polymer_duplicates: int = 20,
    box_size_nm: float = 3.0,
    output_dir: str = ""
) -> bool:
    """Create large box with polymer duplicates using sphere exclusion around insulin (PackmolEmbedder approach)."""
    
    try:
        # Check if PACKMOL is available
        if not shutil.which('packmol'):
            print("   PACKMOL not found. Please install PACKMOL.")
            return False
        
        # Get molecule dimensions for dynamic sizing
        insulin_dims = get_molecule_dimensions(insulin_pdb_path)
        polymer_dims = get_molecule_dimensions(polymer_pdb_path)
        
        print(f"   Insulin dimensions: {insulin_dims}")
        print(f"   Polymer dimensions: {polymer_dims}")
        
        # Calculate dynamic box size based on molecule dimensions and counts
        # Rule: Box should be large enough to accommodate all molecules with proper spacing
        
        # Calculate minimum space needed for insulin
        insulin_max_dim = max(insulin_dims)
        insulin_volume_estimate = insulin_max_dim ** 3
        
        # Calculate minimum space needed for polymers
        polymer_max_dim = max(polymer_dims)
        polymer_volume_estimate = polymer_max_dim ** 3 * num_polymer_duplicates
        
        # Total volume estimate with packing efficiency factor (typically 0.4-0.6 for random packing)
        packing_efficiency = 0.5
        total_volume_needed = (insulin_volume_estimate + polymer_volume_estimate) / packing_efficiency
        
        # Calculate base box size from volume
        base_box_size = total_volume_needed ** (1/3)
        
        # Apply minimum size constraints and user preferences
        min_box_size = max(80.0, insulin_max_dim * 3, polymer_max_dim * 5)  # Minimum practical size
        user_preference_size = box_size_nm * 10.0  # Convert nm to Å
        
        # Final box size is the maximum of all constraints
        box_size = max(min_box_size, base_box_size, user_preference_size)
        
        # Buffer distance based on molecule sizes
        buffer_distance = max(15.0, insulin_max_dim * 0.5, polymer_max_dim * 2.0)
        
        # Calculate insulin exclusion radius
        insulin_radius = insulin_max_dim / 2 + buffer_distance
        
        # Set insulin center position (box center)
        insulin_center = [box_size/2, box_size/2, box_size/2]
        
        print(f"   Dynamic box size calculation:")
        print(f"     - Insulin volume estimate: {insulin_volume_estimate:.1f} Å³")
        print(f"     - Polymer volume estimate: {polymer_volume_estimate:.1f} Å³")
        print(f"     - Total volume needed: {total_volume_needed:.1f} Å³")
        print(f"     - Minimum box size: {min_box_size:.1f} Å")
        print(f"     - User preference size: {user_preference_size:.1f} Å")
        print(f"   Final box size: {box_size:.1f} Å")
        print(f"   Insulin center: {insulin_center}")
        print(f"   Insulin exclusion radius: {insulin_radius:.1f} Å")
        print(f"   Dynamic buffer distance: {buffer_distance:.1f} Å")
        
        # Create PACKMOL input file
        packmol_dir = os.path.join(output_dir, 'packmol')
        os.makedirs(packmol_dir, exist_ok=True)
        
        input_file = os.path.join(packmol_dir, 'sphere_exclusion.inp')
        
        # Copy input files to packmol directory
        polymer_local = os.path.join(packmol_dir, 'polymer.pdb')
        insulin_local = os.path.join(packmol_dir, 'insulin.pdb')
        shutil.copy(polymer_pdb_path, polymer_local)
        shutil.copy(insulin_pdb_path, insulin_local)
        
        with open(input_file, 'w') as f:
            f.write("tolerance 2.0\n")
            f.write("filetype pdb\n")
            f.write(f"output {os.path.abspath(output_pdb_path)}\n")
            f.write("\n")
            
            # Place insulin at the center using fixed position
            f.write(f"structure {os.path.abspath(insulin_local)}\n")
            f.write(f"  number {num_insulin_molecules}\n")
            f.write("  center\n")
            f.write(f"  fixed {insulin_center[0]:.1f} {insulin_center[1]:.1f} {insulin_center[2]:.1f} 0. 0. 0.\n")
            f.write("end structure\n\n")
            
            # Distribute polymers around insulin, excluding the central sphere
            f.write(f"structure {os.path.abspath(polymer_local)}\n")
            f.write(f"  number {num_polymer_duplicates}\n")
            f.write(f"  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}\n")
            f.write(f"  outside sphere {insulin_center[0]:.1f} {insulin_center[1]:.1f} {insulin_center[2]:.1f} {insulin_radius:.1f}\n")
            f.write("end structure\n")
        
        print(f"   Created PACKMOL input: {input_file}")
        
        # Run PACKMOL
        print("   Running PACKMOL...")
        
        with open(input_file, 'r') as f:
            result = subprocess.run(
                ['packmol'],
                stdin=f,
                capture_output=True,
                text=True
            )
        
        if result.returncode != 0:
            print(f"   PACKMOL failed with return code {result.returncode}")
            if result.stderr:
                print(f"   PACKMOL error: {result.stderr}")
            if result.stdout:
                print(f"   PACKMOL output: {result.stdout}")
            return False
        
        if not os.path.exists(output_pdb_path):
            print(f"   Output file not created: {output_pdb_path}")
            return False
        
        print(f"   Successfully created composite structure: {output_pdb_path}")
        return True
        
    except Exception as e:
        print(f"   Error creating large box composite: {e}")
        return False

def get_molecule_dimensions(pdb_file):
    """Extract approximate dimensions from PDB file"""
    import numpy as np
    
    coords = []
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        
        if not coords:
            return [10.0, 10.0, 10.0]  # Default dimensions
        
        coords = np.array(coords)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        dimensions = max_coords - min_coords
        
        return dimensions.tolist()
    except:
        return [10.0, 10.0, 10.0]  # Default dimensions

def count_atoms_in_pdb(pdb_path: str) -> int:
    """Count atoms in PDB file."""
    count = 0
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                count += 1
    return count

if __name__ == "__main__":
    # Example usage
    example_psmiles = "[*]OCC[*]"  # PEG
    example_insulin_pdb = "insulin.pdb"
    
    result = build_insulin_polymer_composite(
        psmiles=example_psmiles,
        insulin_pdb_path=example_insulin_pdb,
        num_polymer_molecules=20,
        num_insulin_molecules=5,
        box_size_nm=3.0,
        insulin_distribution="random"
    )
    
    if result['success']:
        print(f"✅ Success! Composite: {result['composite_pdb']}")
    else:
        print(f"❌ Failed: {result['error']}") 