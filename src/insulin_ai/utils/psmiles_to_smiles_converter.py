"""
PSMILES to SMILES Converter

This module provides robust conversion from PSMILES to regular SMILES using
the psmiles package capabilities discovered through systematic inspection.

Author: AI-Driven Material Discovery Team
"""

import psmiles
from rdkit import Chem
from typing import Dict, List, Optional, Tuple


class PSMILESConverter:
    """
    Advanced PSMILES to SMILES converter using multiple strategies.
    
    This class leverages the psmiles package's built-in methods:
    - .mol property (preserves * as dummy atoms)
    - .replace_stars() method (substitutes * with actual atoms)
    - .periodic property (creates cyclic structures)
    - .canonicalize property (canonical form)
    """
    
    def __init__(self):
        self.replacement_atoms = ['C', 'F', 'Cl', 'Br', 'I', 'N', 'O', 'S']
        
    def convert_psmiles_to_smiles(self, psmiles_str: str) -> Dict:
        """
        Convert PSMILES to regular SMILES using multiple strategies.
        
        Args:
            psmiles_str: Input PSMILES string
            
        Returns:
            Dict containing conversion results and metadata
        """
        
        results = {
            'input_psmiles': psmiles_str,
            'success': False,
            'conversions': {},
            'best_smiles': None,
            'best_method': None,
            'error': None
        }
        
        try:
            # Create PolymerSmiles object
            ps = psmiles.PolymerSmiles(psmiles_str)
            
            # Strategy 1: Direct .mol property (preserves dummy atoms)
            try:
                mol = ps.mol
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    results['conversions']['mol_property'] = {
                        'smiles': smiles,
                        'atoms': mol.GetNumAtoms(),
                        'valid': True,
                        'description': 'Direct RDKit mol with dummy atoms preserved'
                    }
                    
                    # This is often the best approach for polymer chemistry
                    if not results['best_smiles']:
                        results['best_smiles'] = smiles
                        results['best_method'] = 'mol_property'
                        results['success'] = True
                        
            except Exception as e:
                results['conversions']['mol_property'] = {
                    'error': str(e),
                    'valid': False
                }
            
            # Strategy 2: Replace stars with different atoms
            for atom in self.replacement_atoms:
                try:
                    replaced_ps = ps.replace_stars(atom)
                    replaced_mol = replaced_ps.mol
                    
                    if replaced_mol is not None:
                        smiles = Chem.MolToSmiles(replaced_mol)
                        results['conversions'][f'replace_stars_{atom}'] = {
                            'smiles': smiles,
                            'atoms': replaced_mol.GetNumAtoms(),
                            'valid': True,
                            'description': f'Replaced [*] with {atom} atoms'
                        }
                        
                        # Use as backup if mol_property failed
                        if not results['success']:
                            results['best_smiles'] = smiles
                            results['best_method'] = f'replace_stars_{atom}'
                            results['success'] = True
                            
                except Exception as e:
                    results['conversions'][f'replace_stars_{atom}'] = {
                        'error': str(e),
                        'valid': False
                    }
            
            # Strategy 3: Periodic structure (experimental)
            try:
                periodic_ps = ps.periodic
                periodic_mol = periodic_ps.mol
                
                if periodic_mol is not None:
                    smiles = Chem.MolToSmiles(periodic_mol)
                    results['conversions']['periodic'] = {
                        'smiles': smiles,
                        'atoms': periodic_mol.GetNumAtoms(),
                        'valid': True,
                        'description': 'Periodic (cyclic) structure by connecting stars'
                    }
                    
            except Exception as e:
                results['conversions']['periodic'] = {
                    'error': str(e),
                    'valid': False
                }
            
            # Strategy 4: Canonicalized form
            try:
                canonical_ps = ps.canonicalize
                canonical_str = str(canonical_ps)
                results['conversions']['canonicalize'] = {
                    'psmiles': canonical_str,
                    'valid': True,
                    'description': 'Canonicalized PSMILES form'
                }
                
            except Exception as e:
                results['conversions']['canonicalize'] = {
                    'error': str(e),
                    'valid': False
                }
                
        except Exception as e:
            results['error'] = f'Failed to create PolymerSmiles object: {str(e)}'
            return results
        
        return results
    
    def get_best_smiles_for_rdkit(self, psmiles_str: str) -> Optional[str]:
        """
        Get the best SMILES string for RDKit processing.
        
        Prioritizes:
        1. mol_property (dummy atoms preserved) - best for polymer chemistry
        2. replace_stars with C - simple and reliable
        3. replace_stars with F - alternative halogen
        
        Args:
            psmiles_str: Input PSMILES string
            
        Returns:
            Best SMILES string or None if all methods fail
        """
        
        results = self.convert_psmiles_to_smiles(psmiles_str)
        
        if results['success']:
            return results['best_smiles']
        
        return None
    
    def get_multiple_smiles_options(self, psmiles_str: str) -> List[Tuple[str, str, str]]:
        """
        Get multiple SMILES options for fallback strategies.
        
        Returns:
            List of (method, smiles, description) tuples
        """
        
        results = self.convert_psmiles_to_smiles(psmiles_str)
        options = []
        
        for method, data in results['conversions'].items():
            if data.get('valid', False) and 'smiles' in data:
                options.append((
                    method,
                    data['smiles'],
                    data['description']
                ))
        
        return options


def quick_psmiles_to_smiles(psmiles_str: str) -> Optional[str]:
    """
    Quick conversion function for simple use cases.
    
    Args:
        psmiles_str: Input PSMILES string
        
    Returns:
        Best SMILES string or None if conversion fails
    """
    
    converter = PSMILESConverter()
    return converter.get_best_smiles_for_rdkit(psmiles_str)


def test_converter():
    """Test the converter with various PSMILES strings."""
    
    print("🧪 TESTING PSMILES TO SMILES CONVERTER")
    print("=" * 60)
    
    test_cases = [
        '[*]C=CS(=O)(=O)COC([*])=O',  # Original problematic PSMILES
        '[*]CC[*]',                    # Simple polymer
        '[*]CCC[*]',                   # Another simple case
    ]
    
    converter = PSMILESConverter()
    
    for psmiles in test_cases:
        print(f"\n📥 Testing: {psmiles}")
        print("-" * 40)
        
        results = converter.convert_psmiles_to_smiles(psmiles)
        
        print(f"✅ Success: {results['success']}")
        print(f"🎯 Best method: {results['best_method']}")
        print(f"🧬 Best SMILES: {results['best_smiles']}")
        
        print(f"\n📊 All conversion results:")
        for method, data in results['conversions'].items():
            if data.get('valid', False):
                smiles = data.get('smiles', 'N/A')
                atoms = data.get('atoms', 'N/A')
                print(f"   ✅ {method}: {smiles} ({atoms} atoms)")
            else:
                error = data.get('error', 'Unknown error')
                print(f"   ❌ {method}: {error}")


if __name__ == "__main__":
    test_converter() 