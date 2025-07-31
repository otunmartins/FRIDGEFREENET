#!/usr/bin/env python3
"""
Fixed Packmol Builder - Prevents duplicate atoms and ensures clean combination.

Key Fixes:
1. Input validation and cleaning 
2. Proper atom serial number management
3. Single, unified Packmol approach
4. Tolerance optimization
5. Comprehensive error checking
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np


class FixedPackmolBuilder:
    """
    Robust Packmol builder that prevents duplicate atoms and ensures clean combination.
    """
    
    def __init__(self, working_dir: str = "fixed_packmol_output"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Check Packmol availability
        if not shutil.which('packmol'):
            raise RuntimeError("Packmol not found. Please install Packmol first.")
            
        print("🔧 FixedPackmolBuilder initialized")
        print(f"📁 Working directory: {self.working_dir}")
    
    def clean_input_pdb(self, input_pdb: str, output_pdb: str) -> Dict:
        """
        Clean input PDB file to remove duplicates and fix common issues.
        """
        print(f"🧹 Cleaning input PDB: {input_pdb}")
        
        seen_atoms = set()
        clean_lines = []
        atom_count = 0
        duplicate_count = 0
        
        with open(input_pdb, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        # Parse atom record
                        chain_id = line[21:22].strip() or 'A'  # Default to chain A if empty
                        residue_number = int(line[22:26].strip())
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        
                        # Create unique atom identifier
                        atom_key = (chain_id, residue_number, atom_name, residue_name)
                        
                        if atom_key not in seen_atoms:
                            # Renumber atom serial for consistency
                            atom_count += 1
                            new_line = (
                                line[:6] +  # Record type
                                f"{atom_count:5d}" +  # New serial number
                                line[11:]  # Rest of the line
                            )
                            clean_lines.append(new_line)
                            seen_atoms.add(atom_key)
                        else:
                            duplicate_count += 1
                            print(f"   ⚠️  Removed duplicate: {atom_key} (line {line_num})")
                            
                    except (ValueError, IndexError) as e:
                        print(f"   ❌ Skipping malformed line {line_num}: {e}")
                        continue
                        
                elif line.startswith(('TITLE', 'HEADER', 'REMARK')):
                    clean_lines.append(line)
                elif line.startswith('END'):
                    clean_lines.append(line)
                    break
                # Skip CONECT records to avoid connectivity issues
        
        # Write cleaned file
        with open(output_pdb, 'w') as f:
            f.writelines(clean_lines)
        
        result = {
            'original_file': input_pdb,
            'cleaned_file': output_pdb,
            'total_atoms': atom_count,
            'duplicates_removed': duplicate_count,
            'success': True
        }
        
        print(f"   ✅ Cleaned PDB: {atom_count} atoms, {duplicate_count} duplicates removed")
        return result
    
    def get_molecule_dimensions(self, pdb_file: str) -> Dict:
        """Get molecule dimensions for box sizing."""
        coords = []
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip()) 
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except ValueError:
                        continue
        
        if not coords:
            return {'width': 10.0, 'height': 10.0, 'depth': 10.0, 'max_dim': 10.0}
        
        coords = np.array(coords)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        dimensions = max_coords - min_coords
        
        return {
            'width': float(dimensions[0]),
            'height': float(dimensions[1]), 
            'depth': float(dimensions[2]),
            'max_dim': float(np.max(dimensions)),
            'center': ((min_coords + max_coords) / 2).tolist()
        }
    
    def calculate_optimal_box_size(self, 
                                 insulin_dims: Dict, 
                                 polymer_dims: Dict,
                                 num_polymers: int) -> float:
        """Calculate optimal box size based on molecule dimensions."""
        
        # Estimate volume requirements
        insulin_volume = insulin_dims['width'] * insulin_dims['height'] * insulin_dims['depth']
        polymer_volume = polymer_dims['width'] * polymer_dims['height'] * polymer_dims['depth']
        
        # Total volume with packing efficiency (40% efficiency for random packing)
        total_volume = (insulin_volume + polymer_volume * num_polymers) / 0.4
        
        # Calculate cube root for box size
        box_size = total_volume ** (1/3)
        
        # Apply minimum constraints
        min_size = max(
            insulin_dims['max_dim'] * 3,  # At least 3x insulin size
            polymer_dims['max_dim'] * 2,   # At least 2x polymer size  
            50.0  # Absolute minimum 50 Å
        )
        
        optimal_size = max(box_size, min_size)
        
        print(f"📏 Box size calculation:")
        print(f"   - Insulin volume: {insulin_volume:.1f} Å³")
        print(f"   - Polymer volume: {polymer_volume:.1f} Å³ × {num_polymers}")
        print(f"   - Total volume needed: {total_volume:.1f} Å³")
        print(f"   - Optimal box size: {optimal_size:.1f} Å")
        
        return optimal_size
    
    def create_packmol_input(self,
                           insulin_pdb: str,
                           polymer_pdb: str,
                           output_pdb: str,
                           num_polymers: int = 20,
                           box_size: Optional[float] = None,
                           exclusion_radius: Optional[float] = None) -> str:
        """
        Create optimized Packmol input file.
        """
        
        # Get molecule dimensions
        insulin_dims = self.get_molecule_dimensions(insulin_pdb)
        polymer_dims = self.get_molecule_dimensions(polymer_pdb)
        
        # Calculate optimal box size if not provided
        if box_size is None:
            box_size = self.calculate_optimal_box_size(insulin_dims, polymer_dims, num_polymers)
        
        # Set insulin center
        insulin_center = [box_size/2, box_size/2, box_size/2]
        
        # Create Packmol input
        packmol_input = f"""#
# Fixed Packmol Input - Prevents Duplicate Atoms
# Generated by FixedPackmolBuilder
#
tolerance 3.0
filetype pdb
output {output_pdb}
seed 12345

# Place single insulin molecule at center
structure {insulin_pdb}
  number 1
  center
  fixed {insulin_center[0]:.1f} {insulin_center[1]:.1f} {insulin_center[2]:.1f} 0. 0. 0.
end structure

# Distribute polymer molecules"""
        
        if exclusion_radius:
            # With exclusion radius
            packmol_input += f"""
structure {polymer_pdb}
  number {num_polymers}
  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}
  outside sphere {insulin_center[0]:.1f} {insulin_center[1]:.1f} {insulin_center[2]:.1f} {exclusion_radius:.1f}
end structure"""
            print(f"   🚧 Using exclusion radius: {exclusion_radius:.1f} Å")
        else:
            # Without exclusion radius (close packing)
            packmol_input += f"""
structure {polymer_pdb}
  number {num_polymers}
  inside box 0. 0. 0. {box_size:.1f} {box_size:.1f} {box_size:.1f}
end structure"""
            print("   🎯 Close packing enabled")
        
        return packmol_input
    
    def run_packmol(self, packmol_input: str, input_file: str) -> Tuple[bool, str]:
        """
        Run Packmol with proper error handling.
        """
        print("🔄 Running Packmol...")
        
        # Write input file
        with open(input_file, 'w') as f:
            f.write(packmol_input)
        
        try:
            # Run Packmol
            with open(input_file, 'r') as f:
                result = subprocess.run(
                    ['packmol'],
                    stdin=f,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
            
            if result.returncode == 0:
                print("   ✅ Packmol completed successfully")
                return True, result.stdout
            else:
                error_msg = f"Packmol failed (code {result.returncode})"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nOutput: {result.stdout}"
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            return False, "Packmol timed out after 5 minutes"
        except Exception as e:
            return False, f"Packmol execution failed: {str(e)}"
    
    def build_insulin_polymer_system(self,
                                   insulin_pdb: str,
                                   polymer_pdb: str,
                                   output_name: str = "insulin_polymer_system",
                                   num_polymers: int = 20,
                                   box_size: Optional[float] = None,
                                   use_exclusion: bool = False,
                                   exclusion_buffer: float = 15.0) -> Dict:
        """
        Main function to build insulin-polymer system with duplicate prevention.
        """
        
        print("🧬 Building insulin-polymer system (duplicate-free)")
        print(f"   - Insulin: {insulin_pdb}")
        print(f"   - Polymer: {polymer_pdb}")
        print(f"   - Polymers: {num_polymers}")
        print(f"   - Exclusion: {use_exclusion}")
        
        try:
            # Create temporary working directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Step 1: Clean input files
                print("\n📋 Step 1: Cleaning input files...")
                clean_insulin = temp_path / "insulin_clean.pdb"
                clean_polymer = temp_path / "polymer_clean.pdb"
                
                insulin_clean_result = self.clean_input_pdb(insulin_pdb, str(clean_insulin))
                polymer_clean_result = self.clean_input_pdb(polymer_pdb, str(clean_polymer))
                
                if not insulin_clean_result['success'] or not polymer_clean_result['success']:
                    return {'success': False, 'error': 'Failed to clean input files'}
                
                # Step 2: Calculate exclusion radius if needed
                exclusion_radius = None
                if use_exclusion:
                    insulin_dims = self.get_molecule_dimensions(str(clean_insulin))
                    exclusion_radius = insulin_dims['max_dim'] / 2 + exclusion_buffer
                
                # Step 3: Create Packmol input
                print("\n⚙️  Step 2: Creating Packmol input...")
                output_pdb = temp_path / f"{output_name}.pdb"
                
                packmol_input = self.create_packmol_input(
                    str(clean_insulin),
                    str(clean_polymer),
                    str(output_pdb),
                    num_polymers=num_polymers,
                    box_size=box_size,
                    exclusion_radius=exclusion_radius
                )
                
                # Step 4: Run Packmol
                print("\n🔄 Step 3: Running Packmol...")
                input_file = temp_path / "packmol.inp"
                success, output = self.run_packmol(packmol_input, str(input_file))
                
                if not success:
                    return {'success': False, 'error': f'Packmol failed: {output}'}
                
                if not output_pdb.exists():
                    return {'success': False, 'error': 'Output file was not created'}
                
                # Step 5: Copy to final location
                final_output = self.working_dir / f"{output_name}.pdb"
                shutil.copy2(str(output_pdb), str(final_output))
                
                # Step 6: Validate output
                print("\n✅ Step 4: Validating output...")
                final_clean_result = self.clean_input_pdb(str(final_output), str(final_output))
                
                result = {
                    'success': True,
                    'output_file': str(final_output),
                    'insulin_atoms': insulin_clean_result['total_atoms'],
                    'polymer_atoms': polymer_clean_result['total_atoms'],
                    'final_atoms': final_clean_result['total_atoms'],
                    'num_polymers': num_polymers,
                    'duplicates_removed': final_clean_result['duplicates_removed'],
                    'packmol_output': output
                }
                
                print(f"\n🎉 Successfully created system!")
                print(f"   📄 Output: {final_output}")
                print(f"   🧬 Final atoms: {result['final_atoms']}")
                print(f"   🗑️  Duplicates removed: {result['duplicates_removed']}")
                
                return result
                
        except Exception as e:
            return {'success': False, 'error': f'Build failed: {str(e)}'}


# Convenience function for easy use
def build_clean_insulin_polymer_system(insulin_pdb: str,
                                     polymer_pdb: str,
                                     num_polymers: int = 20,
                                     output_name: str = "clean_insulin_polymer_system",
                                     working_dir: str = "fixed_packmol_output") -> Dict:
    """
    Convenience function to build a clean insulin-polymer system.
    """
    builder = FixedPackmolBuilder(working_dir)
    
    return builder.build_insulin_polymer_system(
        insulin_pdb=insulin_pdb,
        polymer_pdb=polymer_pdb,
        output_name=output_name,
        num_polymers=num_polymers,
        use_exclusion=False  # Allow close packing by default
    )


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing FixedPackmolBuilder...")
    
    # Test with your problematic file
    insulin_file = "path/to/insulin.pdb" 
    polymer_file = "path/to/polymer.pdb"
    
    result = build_clean_insulin_polymer_system(
        insulin_pdb=insulin_file,
        polymer_pdb=polymer_file,
        num_polymers=10,
        output_name="test_system"
    )
    
    if result['success']:
        print("✅ Test successful!")
        print(f"Output: {result['output_file']}")
    else:
        print(f"❌ Test failed: {result['error']}") 