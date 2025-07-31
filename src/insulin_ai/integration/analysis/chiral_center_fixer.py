#!/usr/bin/env python3
"""
Enhanced Chiral Center Fixer - Multiple Strategies

Fixes undefined stereochemistry in SMILES strings using multiple RDKit approaches.
Perfect for racemic mixtures where specific stereochemistry doesn't matter.
"""

from typing import Optional, Tuple, List
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


def fix_chiral_centers_enhanced(smiles: str, verbose: bool = True) -> Tuple[str, bool]:
    """
    Enhanced chiral center fixing with multiple strategies.
    
    Tries multiple approaches:
    1. Simple auto-assignment
    2. 3D embedding + assignment
    3. Stereoisomer enumeration (pick first)
    4. Explicit stereochemistry removal
    
    Args:
        smiles: Input SMILES string with potential undefined stereocenters
        verbose: Whether to print diagnostic information
        
    Returns:
        Tuple of (fixed_smiles, success_flag)
    """
    
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - returning original SMILES")
        return smiles, False
    
    if verbose:
        print(f"🔧 Enhanced chiral fixing for: {smiles[:60]}...")
    
    # Strategy 1: Simple auto-assignment (original approach)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Failed to parse SMILES")
        
        # Check for undefined stereocenters
        undefined_centers = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_ChiralityPossible') and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                undefined_centers.append(atom.GetIdx())
        
        if not undefined_centers:
            if verbose:
                print("✅ No undefined chiral centers found")
            return smiles, True
        
        if verbose:
            print(f"🎯 Found {len(undefined_centers)} undefined chiral centers")
        
        # Try simple assignment
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        fixed_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        # Verify the fix worked
        test_mol = Chem.MolFromSmiles(fixed_smiles)
        if test_mol:
            remaining_undefined = []
            for atom in test_mol.GetAtoms():
                if atom.HasProp('_ChiralityPossible') and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                    remaining_undefined.append(atom.GetIdx())
            
            if not remaining_undefined:
                if verbose:
                    print(f"✅ Strategy 1 Success: {fixed_smiles[:60]}...")
                return fixed_smiles, True
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Strategy 1 failed: {e}")
    
    # Strategy 2: 3D embedding + assignment
    try:
        if verbose:
            print("🔄 Trying Strategy 2: 3D embedding...")
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        
        fixed_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        
        # Verify the fix
        test_mol = Chem.MolFromSmiles(fixed_smiles)
        if test_mol:
            remaining_undefined = []
            for atom in test_mol.GetAtoms():
                if atom.HasProp('_ChiralityPossible') and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                    remaining_undefined.append(atom.GetIdx())
            
            if not remaining_undefined:
                if verbose:
                    print(f"✅ Strategy 2 Success: {fixed_smiles[:60]}...")
                return fixed_smiles, True
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Strategy 2 failed: {e}")
    
    # Strategy 3: Stereoisomer enumeration
    try:
        if verbose:
            print("🔄 Trying Strategy 3: Stereoisomer enumeration...")
        
        mol = Chem.MolFromSmiles(smiles)
        isomers = list(EnumerateStereoisomers(mol))
        
        if isomers:
            # Pick the first stereoisomer
            first_isomer = isomers[0]
            fixed_smiles = Chem.MolToSmiles(first_isomer, isomericSmiles=True)
            
            if verbose:
                print(f"✅ Strategy 3 Success: {fixed_smiles[:60]}...")
                print(f"   Generated {len(isomers)} stereoisomers, using first")
            return fixed_smiles, True
    
    except Exception as e:
        if verbose:
            print(f"⚠️ Strategy 3 failed: {e}")
    
    # Strategy 4: Remove all stereochemistry (last resort)
    try:
        if verbose:
            print("🔄 Trying Strategy 4: Remove stereochemistry...")
        
        mol = Chem.MolFromSmiles(smiles)
        Chem.RemoveStereochemistry(mol)
        simplified_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        if verbose:
            print(f"✅ Strategy 4 Success (simplified): {simplified_smiles[:60]}...")
        return simplified_smiles, True
    
    except Exception as e:
        if verbose:
            print(f"❌ All strategies failed: {e}")
    
    return smiles, False


def fix_chiral_centers_simple(smiles: str, verbose: bool = True) -> Tuple[str, bool]:
    """
    Simple chiral center fixing (original Option 1 method).
    Kept for backward compatibility.
    """
    return fix_chiral_centers_enhanced(smiles, verbose)


def create_openff_molecule_with_chiral_fix(smiles: str, verbose: bool = True):
    """
    Create OpenFF molecule from SMILES with enhanced chiral center fixing.
    
    This is a drop-in replacement for Molecule.from_smiles() that handles
    undefined stereochemistry automatically using multiple strategies.
    
    Args:
        smiles: Input SMILES string
        verbose: Whether to print diagnostic information
        
    Returns:
        OpenFF Molecule object
        
    Raises:
        RuntimeError: If molecule creation fails even after all fixing strategies
    """
    
    try:
        from openff.toolkit import Molecule
    except ImportError:
        raise ImportError("OpenFF toolkit not available")
    
    if verbose:
        print(f"🧪 Creating OpenFF molecule with enhanced chiral fixing...")
    
    # Try original SMILES first
    try:
        molecule = Molecule.from_smiles(smiles)
        if verbose:
            print("✅ Original SMILES worked without fixing")
        return molecule
    except Exception as original_error:
        if "unspecified stereochemistry" in str(original_error).lower() or "undefined chiral" in str(original_error).lower():
            if verbose:
                print("🔧 Detected stereochemistry issue - applying enhanced fix...")
            
            # Apply enhanced chiral center fix
            fixed_smiles, success = fix_chiral_centers_enhanced(smiles, verbose=verbose)
            
            if not success:
                raise RuntimeError(f"Enhanced chiral center fixing failed: {original_error}")
            
            # Try with fixed SMILES
            try:
                molecule = Molecule.from_smiles(fixed_smiles)
                if verbose:
                    print("✅ Enhanced fixed SMILES worked!")
                return molecule
            except Exception as fixed_error:
                if verbose:
                    print("⚠️ Enhanced fix failed, trying with allow_undefined_stereo...")
                
                # Last resort: use allow_undefined_stereo
                try:
                    molecule = Molecule.from_smiles(fixed_smiles, allow_undefined_stereo=True)
                    if verbose:
                        print("✅ Success with allow_undefined_stereo!")
                    return molecule
                except Exception as final_error:
                    raise RuntimeError(f"All strategies failed. Original: {original_error}, Enhanced: {fixed_error}, Final: {final_error}")
        else:
            # Re-raise non-stereochemistry errors
            raise original_error


def test_enhanced_fixer():
    """Test the enhanced chiral center fixer with problematic molecules."""
    
    test_cases = [
        # Your specific problematic SMILES
        "CNc1ccccc1CO[C@@H](Cc1ccccc1)NC(O)Nc1ccccc1CO[CH](Cc1ccccc1)NC(O)Nc1ccccc1CO[CH](Cc1ccccc1)NC(O)O",
        
        # Simple undefined stereochemistry
        "C[CH](N)C(=O)O",  # Alanine
        
        # Multiple undefined centers
        "C[CH](O)[CH](N)C(=O)O",  # Threonine-like
    ]
    
    print("🧪 Testing Enhanced Chiral Center Fixer")
    print("=" * 60)
    
    for i, smiles in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"Input:  {smiles[:80]}...")
        
        fixed_smiles, success = fix_chiral_centers_enhanced(smiles, verbose=True)
        
        print(f"Result: {'✅ Success' if success else '❌ Failed'}")
        print(f"Output: {fixed_smiles[:80]}...")
        
        # Test OpenFF molecule creation
        try:
            molecule = create_openff_molecule_with_chiral_fix(smiles, verbose=False)
            print(f"OpenFF: ✅ Created molecule with {molecule.n_atoms} atoms")
        except Exception as e:
            print(f"OpenFF: ❌ Failed - {str(e)[:80]}...")


if __name__ == "__main__":
    test_enhanced_fixer() 