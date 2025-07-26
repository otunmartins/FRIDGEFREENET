"""
PSP AmorphousBuilder Utilities

This module contains functions for building amorphous polymer structures using
PSP AmorphousBuilder, converting between file formats, and displaying 3D structures.
"""

import streamlit as st
import pandas as pd
import os
import uuid
import streamlit.components.v1 as components
from typing import Dict, Any, Optional


def get_molecule_dimensions(pdb_file):
    """Get molecule dimensions from PDB file."""
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


def vasp_to_pdb(vasp_file_path, pdb_file_path):
    """Convert VASP POSCAR file to PDB format."""
    try:
        # Read VASP file
        with open(vasp_file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse VASP file
        comment = lines[0].strip()
        scale = float(lines[1].strip())
        
        # Lattice vectors
        lattice = []
        for i in range(2, 5):
            lattice.append([float(x) * scale for x in lines[i].strip().split()])
        
        # Element types and counts
        elements = lines[5].strip().split()
        counts = [int(x) for x in lines[6].strip().split()]
        
        # Coordinate type
        coord_type = lines[7].strip().lower()
        
        # Read coordinates
        coordinates = []
        start_line = 8
        for i in range(start_line, start_line + sum(counts)):
            coords = [float(x) for x in lines[i].strip().split()[:3]]
            coordinates.append(coords)
        
        # Convert to Cartesian if needed
        if coord_type.startswith('d'):  # Direct coordinates
            cart_coords = []
            for coord in coordinates:
                cart_coord = [
                    coord[0] * lattice[0][0] + coord[1] * lattice[1][0] + coord[2] * lattice[2][0],
                    coord[0] * lattice[0][1] + coord[1] * lattice[1][1] + coord[2] * lattice[2][1],
                    coord[0] * lattice[0][2] + coord[1] * lattice[1][2] + coord[2] * lattice[2][2]
                ]
                cart_coords.append(cart_coord)
            coordinates = cart_coords
        
        # Write PDB file
        with open(pdb_file_path, 'w') as f:
            f.write(f"HEADER    {comment}\n")
            f.write(f"CRYST1{lattice[0][0]:9.3f}{lattice[1][1]:9.3f}{lattice[2][2]:9.3f}  90.00  90.00  90.00 P 1           1\n")
            
            atom_id = 1
            for elem_idx, element in enumerate(elements):
                for i in range(counts[elem_idx]):
                    coord = coordinates[atom_id - 1]
                    f.write(f"ATOM  {atom_id:5d}  {element:2s}   MOL A   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00          {element:2s}\n")
                    atom_id += 1
            
            f.write("END\n")
        
        return True
        
    except Exception as e:
        return False


def build_amorphous_polymer_structure(psmiles, length=10, num_molecules=20, density=0.8, box_size_nm=1.0):
    """Build amorphous polymer structure using PSP AmorphousBuilder."""
    try:
        # Import PSP
        import psp.AmorphousBuilder as ab
        
        # Create input DataFrame
        input_data = {
            'ID': ['polymer'],
            'smiles': [psmiles],
            'Len': [length],
            'Num': [num_molecules],
            'NumConf': [1],
            'LeftCap': ['H'],
            'RightCap': ['H'],
            'Loop': [False]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Create unique output directory
        output_dir = f'insulin_polymer_output_{uuid.uuid4().hex[:8]}'
        output_file = f'insulin_polymer_{uuid.uuid4().hex[:8]}'
        
        # Calculate box size - PSP uses Angstroms, so convert nm to Angstroms
        box_size_angstrom = box_size_nm * 10.0
        
        # Create AmorphousBuilder
        amor = ab.Builder(
            input_df,
            ID_col="ID",
            SMILES_col="smiles",
            NumMole="Num",
            Length='Len',
            NumConf='NumConf',
            OutFile=output_file,
            OutDir=output_dir,
            density=density,
            box_type='c',  # cubic box
            tol_dis=2.0,
            # Note: PSP calculates box size from density and mass, but we can influence through density
        )
        
        # Build structure
        print(f"\n🔍 DEBUG: Starting PSP AmorphousBuilder")
        print(f"📥 Input PSMILES: '{psmiles}'")
        print(f"📥 Parameters: length={length}, num_molecules={num_molecules}, density={density}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📁 Output file prefix: {output_file}")
        
        amor.Build()
        print(f"✅ DEBUG: PSP Build() completed")
        
        # Check if files were created - PSP creates files in subdirectories
        print(f"🔍 DEBUG: Checking for output files...")
        
        # Look for files in molecules directory (where PSP typically creates them)
        molecules_dir = os.path.join(output_dir, 'molecules')
        packmol_dir = os.path.join(output_dir, 'packmol')
        
        vasp_file = None
        data_file = None
        pdb_file = None
        
        # Search in molecules directory
        if os.path.exists(molecules_dir):
            print(f"📁 DEBUG: Molecules directory exists: {molecules_dir}")
            files_in_molecules = os.listdir(molecules_dir)
            print(f"📁 DEBUG: Files in molecules: {files_in_molecules}")
            
            for file in files_in_molecules:
                full_path = os.path.join(molecules_dir, file)
                if file.endswith('.vasp') and not vasp_file:
                    vasp_file = full_path
                    print(f"✅ DEBUG: Found VASP file: {vasp_file}")
                elif file.endswith('.data') and not data_file:
                    data_file = full_path
                    print(f"✅ DEBUG: Found DATA file: {data_file}")
                elif file.endswith('.pdb') and not pdb_file:
                    pdb_file = full_path
                    print(f"✅ DEBUG: Found PDB file: {pdb_file}")
        else:
            print(f"❌ DEBUG: Molecules directory does not exist")
        
        # Search in packmol directory as backup
        if os.path.exists(packmol_dir) and not vasp_file:
            print(f"📁 DEBUG: Packmol directory exists: {packmol_dir}")
            files_in_packmol = os.listdir(packmol_dir)
            print(f"📁 DEBUG: Files in packmol: {files_in_packmol}")
            
            for file in files_in_packmol:
                full_path = os.path.join(packmol_dir, file)
                if file.endswith('.vasp') and not vasp_file:
                    vasp_file = full_path
                    print(f"✅ DEBUG: Found VASP file in packmol: {vasp_file}")
                elif file.endswith('.pdb') and not pdb_file:
                    pdb_file = full_path
                    print(f"✅ DEBUG: Found PDB file in packmol: {pdb_file}")
        
        # Fallback: check root directory with original naming
        if not vasp_file:
            root_vasp = os.path.join(output_dir, f'{output_file}.vasp')
            root_data = os.path.join(output_dir, f'{output_file}.data')
            root_pdb = os.path.join(output_dir, f'{output_file}.pdb')
            
            print(f"🔍 DEBUG: Checking root directory...")
            print(f"   - Looking for: {root_vasp}")
            print(f"   - Looking for: {root_data}")
            print(f"   - Looking for: {root_pdb}")
            
            if os.path.exists(root_vasp):
                vasp_file = root_vasp
                print(f"✅ DEBUG: Found VASP in root: {vasp_file}")
            if os.path.exists(root_data):
                data_file = root_data
                print(f"✅ DEBUG: Found DATA in root: {data_file}")
            if os.path.exists(root_pdb):
                pdb_file = root_pdb
                print(f"✅ DEBUG: Found PDB in root: {pdb_file}")
        
        print(f"📊 DEBUG: Final file status:")
        print(f"   - VASP file: {vasp_file}")
        print(f"   - DATA file: {data_file}")
        print(f"   - PDB file: {pdb_file}")
        
        # Convert VASP to PDB if VASP exists
        pdb_converted = False
        if vasp_file and os.path.exists(vasp_file):
            pdb_converted = vasp_to_pdb(vasp_file, pdb_file)
        
        # Calculate actual box size from VASP file
        actual_box_size = None
        num_atoms = 0
        if vasp_file and os.path.exists(vasp_file):
            with open(vasp_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 2:
                    lattice_line = lines[2].strip().split()
                    if len(lattice_line) >= 3:
                        actual_box_size = float(lattice_line[0])  # Assuming cubic box
                if len(lines) > 6:
                    counts = [int(x) for x in lines[6].strip().split()]
                    num_atoms = sum(counts)
        
        if vasp_file or data_file or pdb_file:
            print(f"✅ DEBUG: SUCCESS! Found output files")
            return {
                'success': True,
                'output_dir': output_dir,
                'vasp_file': vasp_file,
                'data_file': data_file,
                'pdb_file': pdb_file if pdb_converted else pdb_file,
                'output_file': output_file,
                'actual_box_size_angstrom': actual_box_size,
                'actual_box_size_nm': actual_box_size / 10.0 if actual_box_size else None,
                'num_atoms': num_atoms,
                'target_density': density,
                'pdb_converted': pdb_converted
            }
        else:
            print(f"❌ DEBUG: FAILURE! No output files found")
            # List all files in output directory for debugging
            all_files = []
            if os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        all_files.append(os.path.join(root, file))
            print(f"📁 DEBUG: All files created: {all_files}")
            
            return {
                'success': False,
                'error': f'No output files were created. PSP may have failed silently. Files found: {all_files}'
            }
            
    except ImportError as e:
        print(f"❌ DEBUG: ImportError - PSP not available")
        print(f"   Error: {str(e)}")
        return {
            'success': False,
            'error': 'PSP package not installed. Install with: pip install psp'
        }
    except Exception as e:
        print(f"❌ DEBUG: Unexpected exception in build_amorphous_polymer_structure")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Build failed with error: {str(e)}'
        }


def display_3d_structure(pdb_file_path):
    """Display 3D structure using 3DMol.js."""
    try:
        with open(pdb_file_path, 'r') as f:
            pdb_content = f.read()
        
        # Create unique container ID to avoid conflicts
        container_id = f"mol3d_{uuid.uuid4().hex[:8]}"
        
        # Create 3DMol.js viewer with robust implementation
        html_content = f"""
        <div id="{container_id}" style="height: 600px; width: 100%; border: 1px solid #ddd; border-radius: 10px; background: #f8f9fa; margin-bottom: 20px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.1/3Dmol-min.js"></script>
        <script>
            // Wait for the script to load
            function init3DMol_{container_id.replace('-', '_')}() {{
                try {{
                    let element = document.getElementById('{container_id}');
                    if (!element) {{
                        console.error('3DMol container not found');
                        return;
                    }}
                    
                    let viewer = $3Dmol.createViewer(element, {{
                        defaultcolors: $3Dmol.rasmolElementColors
                    }});
                    
                    let pdb_data = `{pdb_content}`;
                    viewer.addModel(pdb_data, "pdb");
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15}}}});
                    viewer.addStyle({{elem: 'H'}}, {{stick: {{radius: 0.1, hidden: false}}}});
                    viewer.zoomTo();
                    viewer.render();
                    viewer.zoom(0.8);
                    
                    // Add controls info with better positioning
                    let info = document.createElement('div');
                    info.innerHTML = '<div style="text-align: center; margin-top: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;"><small style="color: #666;">💡 Click and drag to rotate • Scroll to zoom • Right-click to pan</small></div>';
                    element.parentNode.insertBefore(info, element.nextSibling);
                    
                }} catch (error) {{
                    console.error('3DMol initialization error:', error);
                    let element = document.getElementById('{container_id}');
                    if (element) {{
                        element.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;"><p>❌ 3D visualization failed to load</p></div>';
                    }}
                }}
            }}
            
            // Initialize when DOM is ready
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', init3DMol_{container_id.replace('-', '_')});
            }} else {{
                init3DMol_{container_id.replace('-', '_')}();
            }}
        </script>
        """
        
        return html_content
        
    except Exception as e:
        return f"""
        <div style="height: 500px; width: 100%; border: 1px solid #ddd; border-radius: 10px; background: #f8f9fa; 
                    display: flex; align-items: center; justify-content: center; color: #666;">
            <div style="text-align: center;">
                <p>❌ Error loading 3D structure</p>
                <p><small>{str(e)}</small></p>
                <p><small>💡 Try downloading the PDB file and opening it in PyMOL, VMD, or ChimeraX</small></p>
            </div>
        </div>
        """ 