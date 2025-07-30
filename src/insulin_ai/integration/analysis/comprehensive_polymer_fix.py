#!/usr/bin/env python3
"""
Comprehensive Polymer Simulation Fix

This solution addresses the root issue: PDB files lack CONECT records for proper bonding.
The fix rebuilds connectivity from atomic coordinates and creates proper molecule representations.

Key approach:
1. Load PDB and extract UNL residue coordinates
2. Use RDKit to infer connectivity from 3D coordinates
3. Create OpenFF molecule from the inferred structure
4. Use template generator with the reconstructed molecule
"""

import tempfile
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, ForceField
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from openff.toolkit import Molecule, Topology
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
    OPENFF_AVAILABLE = True
except ImportError:
    OPENFF_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def extract_unl_residue_info(pdb_file: str) -> dict:
    """Extract UNL residue atomic information from PDB file"""
    
    unl_atoms = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')) and 'UNL' in line:
                # Parse PDB line
                atom_name = line[12:16].strip()
                element = line[76:78].strip()
                if not element:
                    # Infer element from atom name
                    element = atom_name[0]
                
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                unl_atoms.append({
                    'name': atom_name,
                    'element': element,
                    'coords': [x, y, z]
                })
    
    return {
        'atoms': unl_atoms,
        'atom_count': len(unl_atoms)
    }

def create_rdkit_mol_from_coords(unl_info: dict) -> Optional[Chem.Mol]:
    """Create RDKit molecule from UNL coordinates and infer bonds"""
    
    if not RDKIT_AVAILABLE:
        print("❌ RDKit not available for connectivity inference")
        return None
    
    try:
        # Create RDKit molecule
        mol = Chem.EditableMol(Chem.Mol())
        
        # Add atoms
        for atom_info in unl_info['atoms']:
            element = atom_info['element']
            
            # Create atom
            if element == 'C':
                atom = Chem.Atom(6)  # Carbon
            elif element == 'O':
                atom = Chem.Atom(8)  # Oxygen
            elif element == 'N':
                atom = Chem.Atom(7)  # Nitrogen
            elif element == 'S':
                atom = Chem.Atom(16)  # Sulfur
            elif element == 'H':
                atom = Chem.Atom(1)  # Hydrogen
            else:
                print(f"⚠️ Unknown element: {element}, using carbon")
                atom = Chem.Atom(6)
            
            mol.AddAtom(atom)
        
        # Convert to molecule
        mol = mol.GetMol()
        
        # Set 3D coordinates
        conf = Chem.Conformer(len(unl_info['atoms']))
        for i, atom_info in enumerate(unl_info['atoms']):
            x, y, z = atom_info['coords']
            conf.SetAtomPosition(i, (x, y, z))
        
        mol.AddConformer(conf)
        
        # Determine bonds from 3D coordinates
        print("🔗 Determining bonds from 3D coordinates...")
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        
        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol)
            print(f"✅ Created RDKit molecule: {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
            return mol
        except Exception as e:
            print(f"⚠️ Sanitization failed, trying without: {e}")
            return mol
            
    except Exception as e:
        print(f"❌ Failed to create RDKit molecule: {e}")
        return None

def rdkit_to_openff_molecule(rdkit_mol: Chem.Mol) -> Optional[Molecule]:
    """Convert RDKit molecule to OpenFF molecule"""
    
    if not OPENFF_AVAILABLE:
        print("❌ OpenFF toolkit not available")
        return None
    
    try:
        # Convert RDKit molecule to OpenFF molecule
        openff_mol = Molecule.from_rdkit(rdkit_mol)
        print(f"✅ Converted to OpenFF molecule: {openff_mol.n_atoms} atoms, {openff_mol.n_bonds} bonds")
        return openff_mol
    except Exception as e:
        print(f"❌ Failed to convert RDKit to OpenFF: {e}")
        return None

def create_molecule_with_connectivity_inference(pdb_file: str) -> Optional[Molecule]:
    """Create OpenFF molecule by inferring connectivity from PDB coordinates"""
    
    print("🧬 Extracting UNL residue information...")
    unl_info = extract_unl_residue_info(pdb_file)
    
    if unl_info['atom_count'] == 0:
        print("❌ No UNL residue found in PDB file")
        return None
    
    print(f"📊 Found UNL residue with {unl_info['atom_count']} atoms")
    
    # Create RDKit molecule with inferred bonds
    print("🔬 Creating RDKit molecule with bond inference...")
    rdkit_mol = create_rdkit_mol_from_coords(unl_info)
    
    if rdkit_mol is None:
        return None
    
    # Convert to OpenFF molecule
    print("🔄 Converting to OpenFF molecule...")
    openff_mol = rdkit_to_openff_molecule(rdkit_mol)
    
    return openff_mol

def setup_polymer_simulation_with_inference(pdb_file: str, 
                                           temperature: float = 310.0,
                                           force_field_type: str = 'smirnoff') -> Optional[Tuple]:
    """
    Set up polymer simulation by inferring connectivity from PDB coordinates
    
    Args:
        pdb_file: Path to PDB file with UNL residue
        temperature: Simulation temperature in Kelvin
        force_field_type: 'smirnoff' or 'gaff'
    
    Returns:
        Tuple of (simulation, system, topology) if successful, None if failed
    """
    
    print(f"🚀 Setting up polymer simulation with connectivity inference")
    print(f"📁 PDB file: {pdb_file}")
    print(f"🌡️ Temperature: {temperature} K")
    print(f"🧪 Force field: {force_field_type}")
    
    # Check dependencies
    if not OPENMM_AVAILABLE:
        print("❌ OpenMM not available")
        return None
    
    if not OPENFF_AVAILABLE:
        print("❌ OpenFF toolkit not available")
        return None
    
    if not RDKIT_AVAILABLE:
        print("❌ RDKit not available for connectivity inference")
        return None
    
    try:
        # Load PDB file
        print("📖 Loading PDB file...")
        pdb = PDBFile(pdb_file)
        print(f"✅ Loaded PDB: {len(list(pdb.topology.atoms()))} atoms")
        
        # Check for UNL residues
        unl_residues = [r for r in pdb.topology.residues() if r.name == 'UNL']
        print(f"📊 Found {len(unl_residues)} UNL residues")
        
        if not unl_residues:
            print("❌ No UNL residues found")
            return None
        
        # Create molecule with connectivity inference
        print("🧬 Creating molecule with connectivity inference...")
        molecule = create_molecule_with_connectivity_inference(pdb_file)
        
        if molecule is None:
            print("❌ Failed to create molecule")
            return None
        
        # Create template generator
        print("🔧 Setting up template generator...")
        
        if force_field_type == 'smirnoff':
            template_generator = SMIRNOFFTemplateGenerator(molecules=molecule)
            forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        else:  # gaff
            template_generator = GAFFTemplateGenerator(molecules=molecule)
            forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        
        # Register template generator
        forcefield.registerTemplateGenerator(template_generator.generator)
        print("✅ Registered template generator with force field")
        
        # Create system
        print("⚡ Creating OpenMM system...")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1*unit.nanometer,
            constraints=app.HBonds
        )
        print("✅ Successfully created OpenMM system!")
        
        # Set up integrator
        integrator = mm.LangevinMiddleIntegrator(
            temperature*unit.kelvin,
            1/unit.picosecond,
            0.004*unit.picoseconds
        )
        
        # Create simulation
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        
        print("🎉 Polymer simulation setup successful!")
        print(f"🔢 System atoms: {system.getNumParticles()}")
        print(f"🔗 System forces: {system.getNumForces()}")
        
        return simulation, system, pdb.topology
        
    except Exception as e:
        print(f"❌ Simulation setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def run_short_test_simulation(simulation, steps: int = 1000):
    """Run a short test simulation to verify everything works"""
    
    try:
        print(f"🧪 Running short test simulation ({steps} steps)...")
        
        # Minimize energy
        print("⚡ Minimizing energy...")
        simulation.minimizeEnergy()
        
        # Get initial state
        initial_state = simulation.context.getState(getEnergy=True, getPositions=True)
        initial_energy = initial_state.getPotentialEnergy()
        print(f"🔋 Initial energy: {initial_energy}")
        
        # Run short simulation
        print(f"🏃 Running {steps} MD steps...")
        simulation.step(steps)
        
        # Get final state
        final_state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = final_state.getPotentialEnergy()
        print(f"🔋 Final energy: {final_energy}")
        
        energy_change = final_energy - initial_energy
        print(f"📊 Energy change: {energy_change}")
        
        print("✅ Test simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test simulation failed: {str(e)}")
        return False

def main():
    """Test the comprehensive polymer fix"""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_polymer_fix.py <pdb_file>")
        return
    
    pdb_file = sys.argv[1]
    
    print("🧪 Testing Comprehensive Polymer Fix")
    print("=" * 50)
    
    # Test setup
    result = setup_polymer_simulation_with_inference(pdb_file)
    
    if result is None:
        print("❌ Setup failed")
        return
    
    simulation, system, topology = result
    
    # Run test simulation
    success = run_short_test_simulation(simulation)
    
    if success:
        print("\n🎉 Comprehensive polymer fix successful!")
        print("💡 You can now run full MD simulations with this approach")
    else:
        print("\n❌ Test simulation failed")

if __name__ == "__main__":
    main() 