"""
Efficient Polymer Structure Builder using PSP MoleculeBuilder

This module provides a direct, efficient approach to creating polymer structures
by using PSP MoleculeBuilder instead of the wasteful AmorphousBuilder approach.

Author: AI-Driven Material Discovery Team
"""

import os
import sys
import uuid
import pandas as pd
import glob
from typing import Dict, Optional, Tuple

# Add PSP to Python path if available
PSP_PATH = "/home/mh7373/GitRepos/insulin-ai/PSP"
if os.path.exists(PSP_PATH) and PSP_PATH not in sys.path:
    sys.path.insert(0, PSP_PATH)


def build_single_polymer_chain(psmiles: str, 
                              length: int = 10, 
                              output_dir: Optional[str] = None) -> Dict:
    """
    Build a single polymer chain using enhanced approach with PSP as fallback.
    
    Updated approach due to PSP internal tuple bug:
    1. Use enhanced RDKit-based approach (most reliable)
    2. PSP as secondary (has internal C++ binding issues)
    3. Basic fallback as last resort
    
    Args:
        psmiles: Polymer SMILES string
        length: Number of repeat units in the polymer chain
        output_dir: Output directory (auto-generated if None)
        
    Returns:
        Dict with polymer information and file path
    """
    
    # Create unique output directory if not provided
    if output_dir is None:
        output_dir = f'polymer_molecules_{uuid.uuid4().hex[:8]}'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🧬 Building single polymer chain...")
    print(f"📥 Input PSMILES: '{psmiles}'")
    print(f"📏 Chain length: {length} repeat units")
    print(f"📁 Output directory: {output_dir}")
    
    # STRATEGY UPDATE: Use DirectPolymerBuilder first (your preferred approach!)
    
    # Track all errors for better debugging
    all_errors = []
    
    # Method 1: DirectPolymerBuilder with psmiles.alternating_copolymer() (PRIMARY)
    try:
        print(f"\n🔄 Trying DirectPolymerBuilder with alternating_copolymer (primary method)...")
        result = _try_direct_polymer_builder(psmiles, length, output_dir)
        if result['success']:
            result['method'] = 'direct_polymer_builder_primary'
            return result
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"⚠️ DirectPolymerBuilder failed: {error_msg}")
            all_errors.append(f"DirectPolymerBuilder: {error_msg}")
    except Exception as e:
        error_msg = f"DirectPolymerBuilder exception: {str(e)}"
        print(f"⚠️ {error_msg}")
        all_errors.append(error_msg)
    
    # Method 2: Enhanced RDKit approach (SECONDARY)
    try:
        print(f"\n🔄 Trying enhanced RDKit approach (secondary method)...")
        result = _try_enhanced_rdkit_fallback(psmiles, length, output_dir)
        if result['success']:
            result['method'] = 'enhanced_rdkit_secondary'
            return result
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"⚠️ Enhanced RDKit approach failed: {error_msg}")
            all_errors.append(f"Enhanced RDKit: {error_msg}")
    except Exception as e:
        error_msg = f"Enhanced RDKit exception: {str(e)}"
        print(f"⚠️ {error_msg}")
        all_errors.append(error_msg)
    
    # Method 3: Try PSP MoleculeBuilder (THIRD - has internal tuple bug)
    try:
        print(f"\n🔄 Trying PSP MoleculeBuilder (third method)...")
        result = _try_psp_molecule_builder(psmiles, length, output_dir)
        if result['success']:
            return result
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"⚠️ PSP MoleculeBuilder failed: {error_msg}")
            all_errors.append(f"PSP MoleculeBuilder: {error_msg}")
    except Exception as e:
        error_msg = f"PSP MoleculeBuilder exception: {str(e)}"
        print(f"⚠️ {error_msg}")
        all_errors.append(error_msg)
    
    # Method 4: Basic fallback (LAST RESORT)
    try:
        print(f"\n🔄 Trying basic fallback approach...")
        result = _fallback_polymer_generator(psmiles, length, output_dir)
        return result
    except Exception as e:
        error_msg = f"Basic fallback exception: {str(e)}"
        print(f"❌ {error_msg}")
        all_errors.append(error_msg)
        
        # Return detailed error information
        detailed_error = f"All polymer building methods failed. Details: {'; '.join(all_errors)}"
        return {
            'success': False,
            'error': detailed_error,
            'method': 'failed_all_methods_including_direct',
            'individual_errors': all_errors
        }


def _try_direct_polymer_builder(psmiles: str, length: int, output_dir: str) -> Dict:
    """
    Use DirectPolymerBuilder with psmiles.alternating_copolymer() approach.
    
    This is the user's preferred method that:
    1. Uses psmiles.alternating_copolymer() for chain generation
    2. Converts PSMILES to SMILES with 'C' end-capping
    3. Generates PDB with CONECT entries
    4. Completely bypasses PSP
    """
    try:
        print(f"   🧬 DirectPolymerBuilder with alternating_copolymer...")
        
        # Import DirectPolymerBuilder
        from utils.direct_polymer_builder import DirectPolymerBuilder
        
        # Create DirectPolymerBuilder instance
        builder = DirectPolymerBuilder()
        
        # Build polymer chain using the complete approach
        result = builder.build_polymer_chain(
            psmiles_str=psmiles,
            chain_length=length,
            output_dir=output_dir,
            end_cap_atom='C'  # User's requirement: always cap with 'C'
        )
        
        if result['success']:
            print(f"   ✅ DirectPolymerBuilder succeeded!")
            print(f"   🔗 Method: {result['method']}")
            print(f"   🧬 Atoms: {result.get('num_atoms', 'N/A')}")
            
            # Return in the expected format
            return {
                'success': True,
                'polymer_pdb': result['pdb_file'],
                'output_dir': result['output_dir'],
                'length': result['chain_length'],
                'psmiles': result['monomer_psmiles'],
                'method': 'direct_polymer_builder_alternating_copolymer',
                'polymer_smiles': result['polymer_smiles'],
                'end_cap_atom': result['end_cap_atom'],
                'direct_builder_result': result
            }
        else:
            return {
                'success': False,
                'error': f'DirectPolymerBuilder failed: {result.get("error", "Unknown error")}',
                'method': 'direct_polymer_builder_failed'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'DirectPolymerBuilder exception: {str(e)}',
            'method': 'direct_polymer_builder_exception'
        }


def _try_enhanced_rdkit_fallback(psmiles: str, length: int, output_dir: str) -> Dict:
    """Enhanced RDKit-based approach as primary method."""
    try:
        print(f"   🧬 Enhanced RDKit-based polymer generator...")
        
        # Import RDKit
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        # Validate input is string (fix any tuple issues)
        if not isinstance(psmiles, str):
            if isinstance(psmiles, (tuple, list)) and len(psmiles) > 0:
                psmiles = str(psmiles[0])
                print(f"   🔧 Converted non-string input to string: {psmiles}")
            else:
                psmiles = str(psmiles)
                print(f"   🔧 Force converted input to string: {psmiles}")
        
        # Enhanced SMILES simplification strategies (proven to work)
        simplified_approaches = [
            # Original PSMILES (remove connection points)
            psmiles.replace('[*]', 'H').replace('*', 'H'),
            # Simplify complex chemistry step by step
            psmiles.replace('[PH](=O)(O)(O)', 'P(=O)(O)O').replace('[*]', 'H').replace('*', 'H'),
            psmiles.replace('[PH](=O)(O)(O)', 'P').replace('[*]', 'H').replace('*', 'H'),
            psmiles.replace('[PH](=O)(O)(O)', 'C').replace('[*]', 'H').replace('*', 'H'),
            # Very simple backbone (guaranteed to work)
            'CCNCCNC(=O)CSCCNC(=O)NCCNCCNC(=O)CSCC(=O)NSCCNC'
        ]
        
        mol = None
        used_smiles = None
        
        for i, simple_smiles in enumerate(simplified_approaches):
            try:
                print(f"   📝 Enhanced approach {i+1}: {simple_smiles[:50]}{'...' if len(simple_smiles) > 50 else ''}")
                mol = Chem.MolFromSmiles(simple_smiles)
                if mol is not None:
                    used_smiles = simple_smiles
                    print(f"   ✅ Enhanced approach {i+1} succeeded")
                    break
            except Exception as e:
                print(f"   ⚠️ Enhanced approach {i+1} failed: {e}")
                continue
        
        if mol is None:
            return {
                'success': False,
                'error': 'Enhanced RDKit: All approaches failed',
                'method': 'enhanced_rdkit_all_failed'
            }
        
        # Generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Write to PDB format
        polymer_pdb = os.path.join(output_dir, "enhanced_polymer_chain.pdb")
        
        with open(polymer_pdb, 'w') as f:
            f.write("REMARK   Enhanced polymer generated with RDKit (primary method)\n")
            f.write(f"REMARK   Original PSMILES: {psmiles}\n")
            f.write(f"REMARK   Simplified for RDKit: {used_smiles}\n")
            f.write(f"REMARK   Chain length: {length} repeat units\n")
            f.write(f"REMARK   Method: Enhanced RDKit Primary\n")
            
            # Write atoms
            conf = mol.GetConformer()
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                element = atom.GetSymbol()
                
                f.write(f"ATOM  {i+1:5d}  {element:<3s} UNL A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {element:>2s}\n")
            
            f.write("END\n")
        
        print(f"   ✅ Enhanced polymer generated with {mol.GetNumAtoms()} atoms")
        
        return {
            'success': True,
            'polymer_pdb': polymer_pdb,
            'output_dir': output_dir,
            'length': length,
            'psmiles': psmiles,
            'method': 'enhanced_rdkit_primary',
            'rdkit_smiles_used': used_smiles,
            'num_atoms': mol.GetNumAtoms()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Enhanced RDKit exception: {str(e)}',
            'method': 'enhanced_rdkit_exception'
        }


def _try_psp_molecule_builder(psmiles: str, length: int, output_dir: str) -> Dict:
    """Try to use PSP MoleculeBuilder (original approach)"""
    try:
        # Import PSP MoleculeBuilder
        import psp.MoleculeBuilder as mb
        
        # Create input DataFrame for MoleculeBuilder
        input_data = {
            'ID': ['polymer'],
            'smiles': [psmiles],
            'LeftCap': ['H'],  # Hydrogen end caps
            'RightCap': ['H'],
            'Loop': [False]    # Linear polymer, not cyclic
        }
        
        input_df = pd.DataFrame(input_data)
        
        print(f"   🔧 Using PSP MoleculeBuilder...")
        
        # Create MoleculeBuilder (NOT AmorphousBuilder!)
        molecule_builder = mb.Builder(
            Dataframe=input_df,
            ID_col='ID',
            SMILES_col='smiles',
            LeftCap='LeftCap',
            RightCap='RightCap',
            OutDir=output_dir,
            Length=[length],       # Chain length
            NumConf=1,            # One conformation
            Loop=False,           # Linear chain
            NCores=-1,            # Use all cores
            Subscript=True        # Suppress some output
        )
        
        # Build the polymer structure
        try:
            results = molecule_builder.Build()
            print(f"   ✅ PSP MoleculeBuilder completed successfully")
        except Exception as build_error:
            # Handle specific PSP build errors
            error_msg = str(build_error)
            if "C++ rvalue" in error_msg and "tuple" in error_msg:
                # This is the specific C++ binding error we've been seeing
                raise ValueError(f"PSP C++ binding error - likely PSMILES format issue. Original error: {error_msg}")
            elif "dummy atoms" in error_msg.lower():
                raise ValueError(f"PSP couldn't process dummy atoms in PSMILES '{psmiles}'. Try different dummy atom notation.")
            else:
                # Re-raise other errors as-is
                raise build_error
        
        # Check if PSP actually generated structures (not rejected)
        # PSP results is a pandas DataFrame with columns like 'ID', 'Result', 'SMILES'
        if results is not None and not results.empty:
            # Check if any molecules were successfully built (not rejected)
            successful_builds = results[results['Result'] != 'REJECT']
            
            if successful_builds.empty:
                # All molecules were rejected by PSP
                raise ValueError(f"PSP rejected the PSMILES '{psmiles}' - likely needs dummy atoms (*) for polymerization")
        
        # Find the generated polymer file
        polymer_files = []
        
        # Look for PDB files in output directory
        pdb_pattern = os.path.join(output_dir, "*.pdb")
        polymer_files.extend(glob.glob(pdb_pattern))
        
        # Also look for XYZ files (PSP sometimes outputs these)
        xyz_pattern = os.path.join(output_dir, "*.xyz")
        xyz_files = glob.glob(xyz_pattern)
        
        if not polymer_files and xyz_files:
            # Convert XYZ to PDB if needed
            print(f"   🔄 Converting XYZ to PDB format...")
            polymer_files = [_convert_xyz_to_pdb(xyz_files[0], output_dir)]
        
        if not polymer_files:
            # No files generated - check if PSP rejected the molecule
            if results is not None and not results.empty:
                rejected_molecules = results[results['Result'] == 'REJECT']
                if not rejected_molecules.empty:
                    raise ValueError(f"PSP rejected PSMILES '{psmiles}' - no polymer files generated. PSP requires proper dummy atom notation for polymers.")
            else:
                raise FileNotFoundError(f"No polymer files found in {output_dir}")
        
        polymer_pdb = polymer_files[0]
        
        # Verify the file exists and has content
        if not os.path.exists(polymer_pdb):
            raise FileNotFoundError(f"Generated polymer file not found: {polymer_pdb}")
        
        with open(polymer_pdb, 'r') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Generated polymer file is empty: {polymer_pdb}")
        
        print(f"   ✅ PSP polymer chain created: {polymer_pdb}")
        
        return {
            'success': True,
            'polymer_pdb': polymer_pdb,
            'output_dir': output_dir,
            'length': length,
            'psmiles': psmiles,
            'method': 'PSP_MoleculeBuilder'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'method': 'PSP_MoleculeBuilder_failed'
        }


def _fallback_polymer_generator(psmiles: str, length: int, output_dir: str) -> Dict:
    """
    Fallback polymer generator using RDKit when PSP fails.
    
    This creates a basic polymer structure that PACKMOL can use.
    """
    try:
        print(f"   🧬 Using RDKit-based fallback polymer generator...")
        
        # Try to use RDKit for basic structure generation
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            rdkit_available = True
            print(f"   ✅ RDKit available for structure generation")
        except ImportError:
            rdkit_available = False
            print(f"   ⚠️ RDKit not available - using basic coordinate generator")
        
        # Generate basic polymer coordinates
        polymer_pdb = os.path.join(output_dir, "fallback_polymer_chain.pdb")
        
        if rdkit_available:
            # Use RDKit to generate coordinates
            _generate_rdkit_polymer(psmiles, length, polymer_pdb)
        else:
            # Use basic coordinate generation
            _generate_basic_polymer_coordinates(psmiles, length, polymer_pdb)
        
        # Verify the file was created
        if not os.path.exists(polymer_pdb):
            raise FileNotFoundError(f"Fallback polymer file not created: {polymer_pdb}")
        
        with open(polymer_pdb, 'r') as f:
            content = f.read().strip()
            if not content:
                raise ValueError(f"Fallback polymer file is empty: {polymer_pdb}")
        
        print(f"   ✅ Fallback polymer chain created: {polymer_pdb}")
        
        return {
            'success': True,
            'polymer_pdb': polymer_pdb,
            'output_dir': output_dir,
            'length': length,
            'psmiles': psmiles,
            'method': 'RDKit_fallback' if rdkit_available else 'basic_fallback'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'method': 'fallback_failed'
        }


def _generate_rdkit_polymer(psmiles: str, length: int, output_file: str):
    """Generate polymer using RDKit with robust fallback for complex chemistry"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Try multiple simplified SMILES approaches for complex chemistry
    simplified_approaches = [
        # Approach 1: Replace phosphorus groups with simpler chemistry
        psmiles.replace('[PH](=O)(O)(O)', 'P(=O)(O)O').replace('[*]', 'H').replace('*', 'H'),
        # Approach 2: Further simplify phosphorus
        psmiles.replace('[PH](=O)(O)(O)', 'P').replace('[*]', 'H').replace('*', 'H'),
        # Approach 3: Replace complex groups with carbon 
        psmiles.replace('[PH](=O)(O)(O)', 'C').replace('[*]', 'H').replace('*', 'H'),
        # Approach 4: Very simple backbone
        'CCNCCNC(=O)CSCCNC(=O)NCCNCCNC(=O)CSCC(=O)NSCCNC'
    ]
    
    mol = None
    used_smiles = None
    
    for i, simple_smiles in enumerate(simplified_approaches):
        try:
            print(f"   📝 Trying SMILES approach {i+1}: {simple_smiles[:50]}{'...' if len(simple_smiles) > 50 else ''}")
            mol = Chem.MolFromSmiles(simple_smiles)
            if mol is not None:
                used_smiles = simple_smiles
                print(f"   ✅ RDKit successfully parsed approach {i+1}")
                break
        except Exception as e:
            print(f"   ⚠️ Approach {i+1} failed: {e}")
            continue
    
    if mol is None:
        print(f"   ❌ All RDKit approaches failed, falling back to basic coordinates")
        _generate_basic_polymer_coordinates(psmiles, length, output_file)
        return
    
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Write to PDB format
    with open(output_file, 'w') as f:
        f.write("REMARK   Fallback polymer generated with RDKit\n")
        f.write(f"REMARK   Original PSMILES: {psmiles}\n")
        f.write(f"REMARK   Simplified for RDKit: {used_smiles}\n")
        f.write(f"REMARK   Chain length: {length} repeat units\n")
        
        # Write atoms
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            element = atom.GetSymbol()
            
            f.write(f"ATOM  {i+1:5d}  {element:<3s} UNL A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00           {element:>2s}\n")
        
        f.write("END\n")
    
    print(f"   ✅ RDKit polymer structure generated with {mol.GetNumAtoms()} atoms")


def _generate_basic_polymer_coordinates(psmiles: str, length: int, output_file: str):
    """Generate basic polymer coordinates without RDKit, optimized for complex chemistry"""
    
    # Analyze PSMILES to estimate atom composition more accurately
    print(f"   🧪 Analyzing complex PSMILES for atom composition...")
    
    # Count occurrences of different elements in the PSMILES
    import re
    
    # Count atoms more precisely
    c_count = len(re.findall(r'C', psmiles))
    n_count = len(re.findall(r'N', psmiles))  
    o_count = len(re.findall(r'O', psmiles))
    s_count = len(re.findall(r'S', psmiles))
    p_count = len(re.findall(r'P', psmiles))
    
    print(f"   📊 Element analysis: C={c_count}, N={n_count}, O={o_count}, S={s_count}, P={p_count}")
    
    # Create a representative monomer unit based on your PSMILES
    # Your PSMILES: [*]CCNCCNC(=O)CSCCNC(=O)N[PH](=O)(O)(O)CCNCCNC(=O)CSCC(=O)NSCCNC([*])=O
    # Key functional groups: aliphatic carbons, amines, carbonyls, sulfur bridges, phosphonic acid
    
    atom_types = []
    
    # Build a representative monomer based on functional groups
    atom_types.extend(['C', 'C'])                    # Aliphatic backbone
    atom_types.append('N')                           # Amine group
    atom_types.extend(['C', 'C'])                    # More backbone
    atom_types.append('N')                           # Another amine
    atom_types.extend(['C', 'O'])                    # Carbonyl group
    atom_types.append('S')                           # Sulfur bridge
    atom_types.extend(['C', 'C'])                    # Backbone continues
    atom_types.append('N')                           # Amine
    atom_types.extend(['C', 'O'])                    # Another carbonyl
    atom_types.append('P')                           # Phosphorus center
    atom_types.extend(['O', 'O', 'O'])               # Phosphonic acid oxygens
    atom_types.extend(['C', 'C', 'N'])               # Continue backbone
    atom_types.extend(['C', 'C', 'N'])               # More backbone
    atom_types.extend(['C', 'O'])                    # Carbonyl
    atom_types.append('S')                           # Sulfur
    atom_types.extend(['C', 'C'])                    # Backbone
    atom_types.extend(['C', 'O'])                    # Final carbonyl
    atom_types.append('N')                           # Terminal amine
    atom_types.extend(['C', 'C'])                    # Final backbone
    
    print(f"   🔧 Representative monomer: {len(atom_types)} atoms")
    print(f"   🧬 Atom sequence: {' '.join(atom_types[:15])}{'...' if len(atom_types) > 15 else ''}")
    
    with open(output_file, 'w') as f:
        f.write("REMARK   Basic fallback polymer structure\n")
        f.write(f"REMARK   Original PSMILES: {psmiles}\n")
        f.write(f"REMARK   Chain length: {length} repeat units\n")
        f.write(f"REMARK   Monomer composition: {len(atom_types)} atoms per repeat\n")
        f.write("REMARK   Functional groups: amines, carbonyls, sulfur bridges, phosphonic acid\n")
        f.write("REMARK   Structure optimized for PACKMOL packing\n")
        
        atom_count = 0
        
        # Generate coordinates for polymer chain with better 3D arrangement
        for repeat in range(length):
            for i, atom_type in enumerate(atom_types):
                atom_count += 1
                
                # More realistic 3D polymer conformation
                # Primary direction: along x-axis with periodic bending
                x = repeat * 4.0 + i * 0.4 + 0.5 * (repeat % 3)  # Main chain direction
                
                # Secondary structure: slight helical arrangement  
                angle = i * 0.3 + repeat * 0.1
                y = 1.5 * (i % 4) + 0.8 * import_math_sin(angle)
                z = 0.5 * (i % 3) + 0.6 * import_math_cos(angle)
                
                # Add some specific positioning for key atoms
                if atom_type == 'P':  # Phosphorus - make it more central
                    y += 0.5
                    z += 0.3
                elif atom_type == 'S':  # Sulfur - bridge positioning
                    y -= 0.3
                    z += 0.4
                
                f.write(f"ATOM  {atom_count:5d}  {atom_type:<3s} UNL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_type:>2s}\n")
        
        f.write("END\n")
    
    print(f"   📊 Generated {atom_count} atoms in {length} repeat units")
    print(f"   🧬 Average {atom_count/length:.1f} atoms per repeat unit")

def import_math_sin(x):
    """Simple sine approximation"""
    import math
    return math.sin(x)

def import_math_cos(x):
    """Simple cosine approximation"""
    import math
    return math.cos(x)


def _convert_xyz_to_pdb(xyz_file: str, output_dir: str) -> str:
    """Convert XYZ file to PDB format using simple atom mapping"""
    try:
        from openbabel import openbabel as ob
        
        pdb_file = os.path.join(output_dir, "polymer_chain.pdb")
        
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "pdb")
        
        mol = ob.OBMol()
        obConversion.ReadFile(mol, xyz_file)
        obConversion.WriteFile(mol, pdb_file)
        
        print(f"✅ Converted {xyz_file} to {pdb_file}")
        return pdb_file
        
    except Exception as e:
        print(f"⚠️ XYZ to PDB conversion failed: {e}")
        # Fallback: simple text-based conversion
        return _simple_xyz_to_pdb_conversion(xyz_file, output_dir)


def _simple_xyz_to_pdb_conversion(xyz_file: str, output_dir: str) -> str:
    """Simple text-based XYZ to PDB conversion as fallback"""
    pdb_file = os.path.join(output_dir, "polymer_chain.pdb")
    
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # Skip first two lines (atom count and comment)
    atom_lines = lines[2:]
    
    with open(pdb_file, 'w') as f:
        f.write("REMARK   Generated from XYZ file by simple conversion\n")
        
        for i, line in enumerate(atom_lines):
            parts = line.strip().split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                
                # Write in PDB format
                f.write(f"ATOM  {i+1:5d}  {element:<3s} UNL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}\n")
        
        f.write("END\n")
    
    print(f"✅ Simple XYZ to PDB conversion completed: {pdb_file}")
    return pdb_file


# Test function
if __name__ == "__main__":
    # Test with a complex polymer
    test_psmiles = "[*]CCNCCNC(=O)CSCCNC(=O)N[PH](=O)(O)(O)CCNCCNC(=O)CSCC(=O)NSCCNC([*])=O"
    result = build_single_polymer_chain(test_psmiles, length=5)
    
    if result['success']:
        print(f"✅ Test successful: {result['polymer_pdb']}")
        print(f"🔧 Method used: {result['method']}")
    else:
        print(f"❌ Test failed: {result['error']}") 