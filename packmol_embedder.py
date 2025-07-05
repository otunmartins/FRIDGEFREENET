#!/usr/bin/env python3

import subprocess
import os
import numpy as np
from pathlib import Path

class PackmolEmbedder:
    def __init__(self, insulin_pdb, polymer_pdb, output_pdb="embedded_system.pdb"):
        self.insulin_pdb = insulin_pdb
        self.polymer_pdb = polymer_pdb
        self.output_pdb = output_pdb
        self.packmol_input = "packmol_input.inp"
        
    def count_atoms_in_pdb(self, pdb_file):
        """Count the number of atoms in a PDB file"""
        if not os.path.exists(pdb_file):
            return 0
        
        atom_count = 0
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
        return atom_count
    
    def estimate_total_atoms(self, num_polymers=50, num_insulin=1):
        """Estimate total atoms in the final system"""
        insulin_atoms = self.count_atoms_in_pdb(self.insulin_pdb)
        polymer_atoms = self.count_atoms_in_pdb(self.polymer_pdb)
        
        total_atoms = (insulin_atoms * num_insulin) + (polymer_atoms * num_polymers)
        
        return {
            'insulin_atoms': insulin_atoms,
            'polymer_atoms': polymer_atoms,
            'total_atoms': total_atoms,
            'num_insulin': num_insulin,
            'num_polymers': num_polymers
        }
    
    def calculate_max_polymers_for_limit(self, max_atoms=15000, num_insulin=1):
        """Calculate maximum number of polymers to stay under atom limit"""
        insulin_atoms = self.count_atoms_in_pdb(self.insulin_pdb)
        polymer_atoms = self.count_atoms_in_pdb(self.polymer_pdb)
        
        if polymer_atoms == 0:
            return 0
        
        # Reserve atoms for insulin molecules
        available_atoms = max_atoms - (insulin_atoms * num_insulin)
        
        if available_atoms <= 0:
            return 0
        
        max_polymers = int(available_atoms / polymer_atoms)
        return max(0, max_polymers)
    
    def get_molecule_dimensions(self, pdb_file):
        """Extract approximate dimensions from PDB file"""
        coords = []
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
        
        if not coords:
            raise ValueError(f"No atomic coordinates found in {pdb_file}")
        
        coords = np.array(coords)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        dimensions = max_coords - min_coords
        center = (min_coords + max_coords) / 2
        
        return dimensions, center, min_coords, max_coords
    
    def create_packmol_input(self, box_size=100.0, num_polymers=50, 
                           insulin_center=None, buffer_distance=15.0, max_atoms=15000):
        """
        Create Packmol input file for embedding insulin in polymer matrix
        
        Parameters:
        - box_size: Size of the simulation box
        - num_polymers: Number of polymer molecules to place
        - insulin_center: Center position for insulin (defaults to box center)
        - buffer_distance: Minimum distance between insulin and polymers
        - max_atoms: Maximum total atoms allowed
        """
        
        # Check atom limit and adjust if necessary
        estimate = self.estimate_total_atoms(num_polymers, 1)
        if estimate['total_atoms'] > max_atoms:
            max_polymers = self.calculate_max_polymers_for_limit(max_atoms, 1)
            if max_polymers < num_polymers:
                print(f"⚠️  Requested {num_polymers} polymers would create {estimate['total_atoms']} atoms")
                print(f"⚠️  Reducing to {max_polymers} polymers to stay under {max_atoms} atom limit")
                num_polymers = max_polymers
                
                # Recalculate estimate
                estimate = self.estimate_total_atoms(num_polymers, 1)
        
        print(f"📊 System size estimate:")
        print(f"   Insulin atoms: {estimate['insulin_atoms']}")
        print(f"   Polymer atoms per molecule: {estimate['polymer_atoms']}")
        print(f"   Number of polymers: {num_polymers}")
        print(f"   Total estimated atoms: {estimate['total_atoms']}")
        
        if estimate['total_atoms'] > max_atoms:
            raise ValueError(f"System too large: {estimate['total_atoms']} atoms exceeds limit of {max_atoms}")
        
        # Get dimensions
        insulin_dims, insulin_center_orig, insulin_min, insulin_max = self.get_molecule_dimensions(self.insulin_pdb)
        polymer_dims, _, _, _ = self.get_molecule_dimensions(self.polymer_pdb)
        
        print(f"Insulin dimensions: {insulin_dims}")
        print(f"Polymer dimensions: {polymer_dims}")
        
        # Set insulin center (default to box center)
        if insulin_center is None:
            insulin_center = [box_size/2, box_size/2, box_size/2]
        
        # Calculate insulin boundaries with buffer
        insulin_radius = max(insulin_dims) / 2 + buffer_distance
        
        # Create Packmol input
        packmol_content = f"""#
# Packmol input for embedding insulin in polymer matrix
# Estimated atoms: {estimate['total_atoms']} (limit: {max_atoms})
#

tolerance 2.0
filetype pdb
output {self.output_pdb}

# Place insulin at the center
structure {self.insulin_pdb}
  number 1
  center
  fixed {insulin_center[0]} {insulin_center[1]} {insulin_center[2]} 0. 0. 0.
end structure

# Distribute polymers around insulin, avoiding the central region
structure {self.polymer_pdb}
  number {num_polymers}
  inside box 0. 0. 0. {box_size} {box_size} {box_size}
  outside sphere {insulin_center[0]} {insulin_center[1]} {insulin_center[2]} {insulin_radius}
end structure
"""
        
        with open(self.packmol_input, 'w') as f:
            f.write(packmol_content)
        
        print(f"Created Packmol input file: {self.packmol_input}")
        print(f"Box size: {box_size} Å")
        print(f"Insulin center: {insulin_center}")
        print(f"Insulin exclusion radius: {insulin_radius} Å")
        print(f"Number of polymers: {num_polymers}")
        
        return estimate
    
    def run_packmol(self, packmol_executable="packmol"):
        """Run Packmol with the generated input file"""
        
        if not os.path.exists(self.packmol_input):
            raise FileNotFoundError(f"Packmol input file {self.packmol_input} not found")
        
        print(f"Running Packmol...")
        
        try:
            # Run Packmol
            with open(self.packmol_input, 'r') as input_file:
                result = subprocess.run(
                    [packmol_executable],
                    stdin=input_file,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            print("Packmol completed successfully!")
            print(f"Output written to: {self.output_pdb}")
            
            if result.stdout:
                print("Packmol output:")
                print(result.stdout)
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Packmol failed with error code {e.returncode}")
            print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"Packmol executable '{packmol_executable}' not found.")
            print("Please ensure Packmol is installed and in your PATH")
            return False
    
    def embed_insulin(self, box_size=100.0, num_polymers=50, 
                     buffer_distance=15.0, packmol_executable="packmol", max_atoms=15000):
        """
        Complete workflow to embed insulin in polymer matrix
        """
        
        print(f"Starting insulin embedding process...")
        print(f"Insulin PDB: {self.insulin_pdb}")
        print(f"Polymer PDB: {self.polymer_pdb}")
        print(f"Maximum atoms allowed: {max_atoms}")
        
        # Check if input files exist
        if not os.path.exists(self.insulin_pdb):
            raise FileNotFoundError(f"Insulin PDB file not found: {self.insulin_pdb}")
        if not os.path.exists(self.polymer_pdb):
            raise FileNotFoundError(f"Polymer PDB file not found: {self.polymer_pdb}")
        
        # Create Packmol input with atom limit checking
        estimate = self.create_packmol_input(
            box_size, num_polymers, 
            buffer_distance=buffer_distance, 
            max_atoms=max_atoms
        )
        
        # Run Packmol
        success = self.run_packmol(packmol_executable)
        
        if success:
            # Verify final atom count
            final_atoms = self.count_atoms_in_pdb(self.output_pdb)
            print(f"\n✅ Embedding completed successfully!")
            print(f"📊 Final structure: {final_atoms} atoms")
            print(f"📁 Final structure saved as: {self.output_pdb}")
        else:
            print(f"\n❌ Embedding failed. Check error messages above.")
        
        return success

# Example usage
if __name__ == "__main__":
    # Initialize the embedder
    embedder = PackmolEmbedder(
        insulin_pdb="insulin.pdb",
        polymer_pdb="polymer.pdb",
        output_pdb="insulin_embedded_system.pdb"
    )
    
    # Run the embedding process with atom limit
    success = embedder.embed_insulin(
        box_size=150.0,           # Large box to spread out polymers
        num_polymers=20,          # Reduced from 100 to keep system manageable
        buffer_distance=20.0,     # Buffer around insulin
        packmol_executable="packmol",
        max_atoms=15000           # Limit total atoms to 15K
    )
    
    if success:
        print("\nYou can now visualize the result with:")
        print("- PyMOL: pymol insulin_embedded_system.pdb")
        print("- VMD: vmd insulin_embedded_system.pdb")
        print("- ChimeraX: chimerax insulin_embedded_system.pdb")
        print("\nFor MD simulations:")
        print("- System size is optimized for CPU-based simulations")
        print("- Should work well with FairChem UMA models") 