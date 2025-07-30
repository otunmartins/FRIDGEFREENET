"""
Direct Polymer Builder using PSMILES Package

This module completely bypasses PSP and uses the psmiles package directly for:
1. Polymer chain generation via alternating_copolymer
2. PSMILES to SMILES conversion with end-capping
3. Direct PDB generation with CONECT entries
4. PACKMOL integration for solvation

Author: AI-Driven Material Discovery Team
"""

import os
import sys
import uuid
import psmiles
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, Optional, Tuple
from pathlib import Path

# Import our PSMILES converter
from utils.psmiles_to_smiles_converter import PSMILESConverter


class DirectPolymerBuilder:
    """
    Direct polymer builder using psmiles package, completely bypassing PSP.
    
    Features:
    - Uses psmiles.alternating_copolymer() for chain generation
    - PSMILES to SMILES conversion with end-capping
    - Direct PDB generation with CONECT entries
    - PACKMOL integration for solvation
    """
    
    def __init__(self):
        self.converter = PSMILESConverter()
    
    def build_polymer_chain(self, 
                          psmiles_str: str, 
                          chain_length: int = 10,
                          output_dir: Optional[str] = None,
                          end_cap_atom: str = 'C') -> Dict:
        """
        Build a complete polymer chain using psmiles package directly.
        
        Args:
            psmiles_str: Input PSMILES string (monomer)
            chain_length: Number of repeat units
            output_dir: Output directory
            end_cap_atom: Atom to cap polymer ends (default 'C')
            
        Returns:
            Dict with polymer information and file paths
        """
        
        # Create output directory
        if output_dir is None:
            output_dir = f'direct_polymer_{uuid.uuid4().hex[:8]}'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🧬 DIRECT POLYMER BUILDER (No PSP)")
        print(f"📥 Input PSMILES: '{psmiles_str}'")
        print(f"📏 Chain length: {chain_length} repeat units")
        print(f"🔚 End cap atom: {end_cap_atom}")
        print(f"📁 Output directory: {output_dir}")
        
        try:
            # Step 1: Create polymer chain using psmiles alternating_copolymer
            print(f"\n🔗 Step 1: Creating polymer chain...")
            polymer_psmiles = self._create_polymer_chain(psmiles_str, chain_length)
            print(f"✅ Polymer PSMILES: {polymer_psmiles}")
            
            # Step 2: Convert PSMILES to SMILES with end-capping
            print(f"\n🔄 Step 2: Converting to SMILES with {end_cap_atom} end-caps...")
            polymer_smiles = self._convert_to_smiles_with_endcaps(polymer_psmiles, end_cap_atom)
            print(f"✅ Polymer SMILES: {polymer_smiles}")
            
            # Step 3: Generate 3D structure and PDB with CONECT entries
            print(f"\n🏗️ Step 3: Generating 3D structure and PDB...")
            pdb_file = self._generate_pdb_with_conect(polymer_smiles, output_dir)
            print(f"✅ PDB file created: {pdb_file}")
            
            return {
                'success': True,
                'method': 'direct_psmiles_no_psp',
                'monomer_psmiles': psmiles_str,
                'polymer_psmiles': polymer_psmiles,
                'polymer_smiles': polymer_smiles,
                'pdb_file': pdb_file,
                'output_dir': output_dir,
                'chain_length': chain_length,
                'end_cap_atom': end_cap_atom
            }
            
        except Exception as e:
            print(f"❌ Direct polymer builder failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'method': 'direct_psmiles_failed'
            }
    
    def _create_polymer_chain(self, psmiles_str: str, chain_length: int) -> str:
        """
        Create polymer chain using psmiles alternating_copolymer method.
        
        For a monomer A, creates: A-A-A-A-... (chain_length times)
        """
        
        # Create PolymerSmiles object for the monomer
        monomer = psmiles.PolymerSmiles(psmiles_str)
        print(f"   📝 Monomer: {monomer}")
        
        # Create the polymer chain by self-copolymerization
        # We use alternating_copolymer with the same monomer to create A-A-A-A...
        current_polymer = monomer
        
        # For chain_length N, we need N-1 copolymerization steps
        # Each step adds one more monomer unit
        for i in range(chain_length - 1):
            # Use alternating_copolymer with pattern [0, 1] to create linear chain
            current_polymer = current_polymer.alternating_copolymer(monomer, [0, 1])
            print(f"   🔗 Step {i+1}: {current_polymer}")
        
        return str(current_polymer)
    
    def _convert_to_smiles_with_endcaps(self, polymer_psmiles: str, end_cap_atom: str) -> str:
        """
        Convert polymer PSMILES to SMILES with specified end-caps.
        """
        
        # Create PolymerSmiles object
        ps = psmiles.PolymerSmiles(polymer_psmiles)
        
        # Replace connection points [*] with end-cap atoms
        capped_ps = ps.replace_stars(end_cap_atom)
        
        # Get RDKit molecule and convert to SMILES
        mol = capped_ps.mol
        if mol is None:
            raise ValueError(f"Failed to create RDKit molecule from capped polymer")
        
        smiles = Chem.MolToSmiles(mol)
        return smiles
    
    def _generate_pdb_with_conect(self, smiles: str, output_dir: str) -> str:
        """
        Generate PDB file with CONECT entries from SMILES.
        """
        
        # Create RDKit molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to create RDKit molecule from SMILES: {smiles}")
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        
        # Try multiple conformer generation strategies
        success = False
        for seed in [42, 123, 456, 789]:
            try:
                result = AllChem.EmbedMolecule(mol, randomSeed=seed)
                if result == 0:  # Success
                    success = True
                    break
            except Exception as e:
                print(f"   ⚠️ Conformer generation failed with seed {seed}: {e}")
        
        if not success:
            print(f"   ⚠️ Standard embedding failed, trying distance geometry...")
            try:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                success = True
            except Exception as e:
                print(f"   ❌ Distance geometry also failed: {e}")
        
        if success:
            # Optimize geometry
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
                print(f"   ✅ Geometry optimization completed")
            except Exception as e:
                print(f"   ⚠️ Geometry optimization failed: {e}")
        
        # Write PDB file with CONECT entries
        pdb_file = os.path.join(output_dir, "polymer.pdb")
        
        with open(pdb_file, 'w') as f:
            # Write header
            f.write("REMARK   Direct polymer generated using psmiles package\n")
            f.write(f"REMARK   No PSP used - bypassed completely\n")
            f.write(f"REMARK   Polymer SMILES: {smiles}\n")
            f.write(f"REMARK   Total atoms: {mol.GetNumAtoms()}\n")
            
            # Write atoms
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                element = atom.GetSymbol()
                atom_id = i + 1
                
                f.write(f"ATOM  {atom_id:5d}  {element:<3s} UNL A   1    "
                       f"{pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {element:>2s}\n")
            
            # Write CONECT entries for all bonds
            f.write("REMARK   CONECT entries for all bonds\n")
            for bond in mol.GetBonds():
                atom1_id = bond.GetBeginAtomIdx() + 1
                atom2_id = bond.GetEndAtomIdx() + 1
                f.write(f"CONECT{atom1_id:5d}{atom2_id:5d}\n")
            
            f.write("END\n")
        
        print(f"   ✅ PDB written with {mol.GetNumAtoms()} atoms and {mol.GetNumBonds()} bonds")
        return pdb_file
    
    def create_packmol_solvation(self, 
                                polymer_pdb: str,
                                protein_pdb: str,
                                output_dir: str,
                                box_size: Tuple[float, float, float] = (50.0, 50.0, 50.0),
                                density: float = 1.0,
                                water_model: str = "tip3p") -> Dict:
        """
        Create PACKMOL solvation system with polymer and protein.
        
        Args:
            polymer_pdb: Path to polymer PDB file
            protein_pdb: Path to protein PDB file  
            output_dir: Output directory
            box_size: Box dimensions (Å)
            density: Target density (g/cm³)
            water_model: Water model to use
            
        Returns:
            Dict with solvation results
        """
        
        print(f"\n💧 PACKMOL SOLVATION SETUP")
        print(f"🧬 Polymer: {polymer_pdb}")
        print(f"🦠 Protein: {protein_pdb}")
        print(f"📦 Box size: {box_size}")
        print(f"🏋️ Density: {density} g/cm³")
        
        try:
            # Calculate number of water molecules needed
            box_volume = box_size[0] * box_size[1] * box_size[2]  # Å³
            box_volume_cm3 = box_volume * 1e-24  # Convert to cm³
            
            # Estimate water molecules needed (rough calculation)
            water_mass_per_molecule = 18.015  # g/mol
            avogadro = 6.022e23
            water_density = 1.0  # g/cm³ for pure water
            
            # Account for protein and polymer volume (rough estimate)
            available_volume_fraction = 0.7  # 70% available for water
            available_volume = box_volume_cm3 * available_volume_fraction
            
            water_mass_needed = available_volume * density * water_density
            n_water = int(water_mass_needed * avogadro / water_mass_per_molecule)
            
            print(f"📊 Estimated water molecules: {n_water}")
            
            # Create PACKMOL input file
            packmol_input = os.path.join(output_dir, "packmol_solvation.inp")
            solvated_pdb = os.path.join(output_dir, "solvated_system.pdb")
            
            with open(packmol_input, 'w') as f:
                f.write("# PACKMOL input for insulin-polymer solvation\n")
                f.write("# Generated by Direct Polymer Builder (No PSP)\n\n")
                
                f.write("tolerance 2.0\n")
                f.write("filetype pdb\n")
                f.write(f"output {solvated_pdb}\n\n")
                
                # Add protein (1 copy, centered)
                f.write("structure " + protein_pdb + "\n")
                f.write("  number 1\n")
                center_x, center_y, center_z = box_size[0]/2, box_size[1]/2, box_size[2]/2
                f.write(f"  center\n")
                f.write(f"  fixed {center_x:.1f} {center_y:.1f} {center_z:.1f} 0. 0. 0.\n")
                f.write("end structure\n\n")
                
                # Add polymer (1 copy, random position)
                f.write("structure " + polymer_pdb + "\n")
                f.write("  number 1\n")
                f.write(f"  inside box 5.0 5.0 5.0 {box_size[0]-5:.1f} {box_size[1]-5:.1f} {box_size[2]-5:.1f}\n")
                f.write("end structure\n\n")
                
                # Add water molecules
                if water_model.lower() == "tip3p":
                    water_pdb = "water_tip3p.pdb"  # Would need to be provided
                    f.write(f"structure {water_pdb}\n")
                    f.write(f"  number {n_water}\n")
                    f.write(f"  inside box 0.0 0.0 0.0 {box_size[0]:.1f} {box_size[1]:.1f} {box_size[2]:.1f}\n")
                    f.write("end structure\n")
            
            print(f"✅ PACKMOL input created: {packmol_input}")
            
            return {
                'success': True,
                'packmol_input': packmol_input,
                'expected_output': solvated_pdb,
                'n_water': n_water,
                'box_size': box_size,
                'density': density
            }
            
        except Exception as e:
            print(f"❌ PACKMOL setup failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def test_direct_builder():
    """Test the direct polymer builder."""
    
    print("🚀 TESTING DIRECT POLYMER BUILDER")
    print("=" * 60)
    
    # Test with our problematic PSMILES
    test_psmiles = '[*]C=CS(=O)(=O)COC([*])=O'
    
    builder = DirectPolymerBuilder()
    result = builder.build_polymer_chain(
        psmiles_str=test_psmiles,
        chain_length=5,
        output_dir='direct_polymer_test',
        end_cap_atom='C'
    )
    
    print(f"\n📊 RESULTS:")
    print(f"✅ Success: {result['success']}")
    if result['success']:
        print(f"🔧 Method: {result['method']}")
        print(f"🧬 Polymer SMILES: {result['polymer_smiles']}")
        print(f"📁 PDB file: {result['pdb_file']}")
        print(f"📏 Chain length: {result['chain_length']}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    test_direct_builder() 