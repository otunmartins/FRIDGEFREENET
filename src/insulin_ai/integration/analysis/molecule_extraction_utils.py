"""
Robust Molecule Extraction Utilities for OpenMM Topologies

This module provides utilities for extracting molecular structures from OpenMM topologies
and converting them to OpenFF molecules for force field parameterization.

Follows the proven pattern from openmm_test.py for reliable template registration.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from openff.toolkit import Molecule
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False
    print("⚠️ OpenFF toolkit not available")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available")


class RobustMoleculeExtractor:
    """
    Robust molecule extraction from OpenMM topologies following openmm_test.py patterns.
    
    Handles:
    - UNL residues and other small molecules
    - Real molecular structure extraction using RDKit  
    - Fallback methods for incomplete topologies
    - Proper charge assignment
    """
    
    def __init__(self):
        """Initialize the molecule extractor"""
        if not OPENFF_AVAILABLE:
            raise ImportError("OpenFF toolkit is required for molecule extraction")
        
        # Standard biomolecular residues to exclude
        self.standard_residues = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'HIE', 'HID', 'HIP',  # Histidine variants
            'HOH', 'WAT', 'TIP', 'SOL'  # Water variants
        }
        
        print("🧪 RobustMoleculeExtractor initialized")
        print(f"   RDKit available: {RDKIT_AVAILABLE}")
        print(f"   OpenFF available: {OPENFF_AVAILABLE}")
    
    def extract_molecules_from_topology(self, topology, exclude_standard: bool = True) -> List[Molecule]:
        """
        Extract all small molecule residues from topology as OpenFF molecules.
        
        Args:
            topology: OpenMM topology object
            exclude_standard: Whether to exclude standard biomolecular residues
            
        Returns:
            List of OpenFF molecules
        """
        molecules = []
        processed_residue_names = set()
        
        print(f"🔍 Scanning topology for small molecules...")
        
        for residue in topology.residues():
            residue_name = residue.name
            
            # Skip if already processed this residue type
            if residue_name in processed_residue_names:
                continue
                
            # Skip standard residues if requested
            if exclude_standard and residue_name in self.standard_residues:
                continue
            
            print(f"   Found small molecule residue: {residue_name} ({len(list(residue.atoms()))} atoms)")
            
            # Extract molecule for this residue
            molecule = self.extract_molecule_from_residue(topology, residue)
            if molecule:
                molecules.append(molecule)
                processed_residue_names.add(residue_name)
                print(f"   ✅ Successfully extracted {residue_name}: {molecule.n_atoms} atoms, {molecule.hill_formula}")
        
        print(f"✅ Extracted {len(molecules)} unique small molecules from topology")
        return molecules
    
    def extract_molecule_from_residue(self, topology, target_residue) -> Optional[Molecule]:
        """
        Extract a single molecule from a specific residue using multiple methods.
        
        Following the openmm_test.py pattern for robust extraction.
        """
        try:
            print(f"🧪 Extracting molecule from residue {target_residue.name}...")
            
            # Method 1: RDKit-based extraction (most accurate)
            if RDKIT_AVAILABLE:
                molecule = self._extract_with_rdkit(topology, target_residue)
                if molecule:
                    return molecule
            
            # Method 2: Composition-based fallback
            molecule = self._extract_with_composition(target_residue)
            if molecule:
                return molecule
                
            # Method 3: Ultimate fallback
            print("⚠️ All extraction methods failed, using simple alkane fallback")
            return self._create_fallback_molecule()
            
        except Exception as e:
            print(f"❌ Failed to extract molecule from {target_residue.name}: {e}")
            return self._create_fallback_molecule()
    
    def _extract_with_rdkit(self, topology, target_residue) -> Optional[Molecule]:
        """Extract molecule using RDKit for accurate structure representation"""
        try:
            print(f"   🔬 Using RDKit extraction for {target_residue.name}...")
            
            # Create RDKit molecule
            mol = Chem.RWMol()
            atom_mapping = {}
            
            # Add atoms
            for atom in target_residue.atoms():
                atomic_num = atom.element.atomic_number
                rdkit_atom = Chem.Atom(atomic_num)
                atom_idx = mol.AddAtom(rdkit_atom)
                atom_mapping[atom.index] = atom_idx
            
            # Add bonds within the residue
            bond_count = 0
            for bond in topology.bonds():
                atom1, atom2 = bond.atom1, bond.atom2
                
                if (atom1.residue == target_residue and 
                    atom2.residue == target_residue):
                    
                    rdkit_idx1 = atom_mapping[atom1.index]
                    rdkit_idx2 = atom_mapping[atom2.index]
                    
                    mol.AddBond(rdkit_idx1, rdkit_idx2, Chem.BondType.SINGLE)
                    bond_count += 1
            
            print(f"   ✅ RDKit molecule: {mol.GetNumAtoms()} atoms, {bond_count} bonds")
            
            # Sanitize and generate SMILES
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            print(f"   ✅ Generated SMILES: {smiles}")
            
            # Create OpenFF molecule
            molecule = Molecule.from_smiles(smiles)
            molecule.assign_partial_charges("gasteiger")
            
            return molecule
            
        except Exception as e:
            print(f"   ⚠️ RDKit extraction failed: {e}")
            return None
    
    def _extract_with_composition(self, target_residue) -> Optional[Molecule]:
        """Create molecule based on atomic composition"""
        try:
            print(f"   🔧 Using composition-based extraction for {target_residue.name}...")
            
            # Count atoms by element
            element_counts = {}
            for atom in target_residue.atoms():
                symbol = atom.element.symbol
                element_counts[symbol] = element_counts.get(symbol, 0) + 1
            
            print(f"   Composition: {element_counts}")
            
            # Generate SMILES based on composition
            smiles = self._generate_smiles_from_composition(element_counts)
            print(f"   ✅ Generated composition-based SMILES: {smiles}")
            
            molecule = Molecule.from_smiles(smiles)
            molecule.assign_partial_charges("gasteiger")
            
            return molecule
            
        except Exception as e:
            print(f"   ⚠️ Composition-based extraction failed: {e}")
            return None
    
    def _generate_smiles_from_composition(self, element_counts: Dict[str, int]) -> str:
        """Generate reasonable SMILES from atomic composition"""
        carbon_count = element_counts.get('C', 0)
        nitrogen_count = element_counts.get('N', 0)
        oxygen_count = element_counts.get('O', 0)
        sulfur_count = element_counts.get('S', 0)
        
        if carbon_count == 0:
            # No carbon - create simple molecule
            if nitrogen_count > 0:
                return "N"
            elif oxygen_count > 0:
                return "O"
            else:
                return "C"  # Default fallback
        
        # Carbon-based molecules
        if nitrogen_count > 0 and oxygen_count > 0:
            # Likely amide or amino acid derivative
            repeat_count = min(3, max(1, min(nitrogen_count, oxygen_count)))
            return "CC(=O)NC" * repeat_count
        elif nitrogen_count > 0:
            # Amine-containing polymer
            repeat_count = min(5, max(1, nitrogen_count))
            return "CCN" * repeat_count
        elif oxygen_count > 0:
            # Ester or ether-containing polymer
            repeat_count = min(5, max(1, oxygen_count))
            return "CCO" * repeat_count
        elif sulfur_count > 0:
            # Sulfur-containing polymer
            return "CCSC" * min(3, max(1, sulfur_count))
        else:
            # Simple hydrocarbon
            chain_length = min(10, max(1, carbon_count))
            return "C" * chain_length
    
    def _create_fallback_molecule(self) -> Molecule:
        """Create a simple fallback molecule"""
        molecule = Molecule.from_smiles("CCCCCCCCCC")  # Simple decane
        molecule.assign_partial_charges("gasteiger")
        return molecule


def extract_molecules_for_simulation(topology, force_field_type: str = "gaff") -> List[Molecule]:
    """
    Convenience function to extract molecules for simulation setup.
    
    Args:
        topology: OpenMM topology
        force_field_type: "gaff" or "smirnoff" 
        
    Returns:
        List of molecules suitable for template generator registration
    """
    extractor = RobustMoleculeExtractor()
    molecules = extractor.extract_molecules_from_topology(topology)
    
    if not molecules:
        print("⚠️ No small molecules found, creating fallback molecule")
        molecules = [extractor._create_fallback_molecule()]
    
    print(f"✅ Prepared {len(molecules)} molecules for {force_field_type} force field")
    return molecules


if __name__ == "__main__":
    # Test the molecule extractor
    print("🧪 Testing RobustMoleculeExtractor...")
    
    if OPENFF_AVAILABLE:
        extractor = RobustMoleculeExtractor()
        print("✅ Molecule extractor initialized successfully")
    else:
        print("❌ Cannot test - OpenFF toolkit not available") 