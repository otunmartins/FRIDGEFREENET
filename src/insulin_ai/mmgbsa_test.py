#!/usr/bin/env python3
"""
Minimal MMGBSA calculation script for insulin-polymer composite.
Computes binding energy as: ΔG_bind = E_complex - (E_insulin + E_polymer)
"""

import numpy as np
from openmm.app import PDBFile, ForceField
from openmm import LangevinIntegrator, Platform
from openmm.app import Simulation
from openmm.unit import kelvin, picosecond, femtosecond
from openff.toolkit import Molecule
from rdkit import Chem
from openmmforcefields.generators import GAFFTemplateGenerator

def setup_forcefield_with_gaff(polymer_pdb_path):
    """Setup forcefield with GAFF parameters for the polymer."""
    # Load polymer and generate GAFF parameters
    your_mol = Chem.MolFromPDBFile(polymer_pdb_path)
    smiles = Chem.MolToSmiles(your_mol)
    molecule = Molecule.from_smiles(smiles)
    molecule.assign_partial_charges("gasteiger")
    
    # Create GAFF template generator
    gaff = GAFFTemplateGenerator(molecules=molecule)
    
    # Setup forcefield with implicit solvent
    forcefield = ForceField(
        "amber/protein.ff14SB.xml",
        "implicit/gbn2.xml"
    )
    forcefield.registerTemplateGenerator(gaff.generator)
    
    return forcefield

def calculate_energy(pdb_file, forcefield, chain_ids=None):
    """Calculate potential energy for a given structure."""
    pdbfile = PDBFile(pdb_file)
    
    # Filter topology by chain IDs if specified
    if chain_ids:
        # Create a new topology with only specified chains
        topology = pdbfile.topology
        positions = pdbfile.positions
        
        # Filter atoms and positions by chain
        filtered_atoms = []
        filtered_positions = []
        
        for atom in topology.atoms():
            if atom.residue.chain.id in chain_ids:
                filtered_atoms.append(atom)
                filtered_positions.append(positions[atom.index])
        
        # Create new topology with filtered atoms
        from openmm.app.topology import Topology
        new_topology = Topology()
        chain_map = {}
        residue_map = {}
        
        for atom in filtered_atoms:
            if atom.residue.chain.id not in chain_map:
                new_chain = new_topology.addChain(atom.residue.chain.id)
                chain_map[atom.residue.chain.id] = new_chain
            
            chain = chain_map[atom.residue.chain.id]
            residue_key = (atom.residue.chain.id, atom.residue.index)
            
            if residue_key not in residue_map:
                new_residue = new_topology.addResidue(atom.residue.name, chain)
                residue_map[residue_key] = new_residue
            
            residue = residue_map[residue_key]
            new_topology.addAtom(atom.name, atom.element, residue)
        
        topology = new_topology
        positions = filtered_positions
    else:
        topology = pdbfile.topology
        positions = pdbfile.positions
    
    # Create system
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=None,  # No cutoff for implicit solvent
        solventDielectric=78.5,
        soluteDielectric=1.0,
        constraints=None,
        removeCMMotion=False
    )
    
    # Setup simulation
    integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 2.0*femtosecond)
    platform = Platform.getPlatformByName('CPU')  # Use CPU for energy calculations
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    
    # Get energy
    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    
    return energy

def mmgbsa_calculation(trajectory_pdb, polymer_pdb_path, insulin_chains=['A', 'B'], polymer_chains=['C']):
    """
    Perform MMGBSA calculation for insulin-polymer binding.
    
    Parameters:
    - trajectory_pdb: PDB file with trajectory frame
    - polymer_pdb_path: PDB file of the polymer for GAFF parameterization
    - insulin_chains: Chain IDs for insulin (default: A, B)
    - polymer_chains: Chain IDs for polymer (default: C)
    """
    
    print("Setting up forcefield with GAFF parameters...")
    forcefield = setup_forcefield_with_gaff(polymer_pdb_path)
    
    print("Calculating energies...")
    
    # Calculate energy of the complex
    print("- Complex energy...")
    E_complex = calculate_energy(trajectory_pdb, forcefield)
    
    # Calculate energy of isolated insulin
    print("- Insulin energy...")
    E_insulin = calculate_energy(trajectory_pdb, forcefield, chain_ids=insulin_chains)
    
    # Calculate energy of isolated polymer
    print("- Polymer energy...")
    E_polymer = calculate_energy(trajectory_pdb, forcefield, chain_ids=polymer_chains)
    
    # Calculate binding energy
    delta_G_bind = E_complex - (E_insulin + E_polymer)
    
    print("\n" + "="*50)
    print("MMGBSA RESULTS")
    print("="*50)
    print(f"Complex energy:    {E_complex:.3f}")
    print(f"Insulin energy:    {E_insulin:.3f}")
    print(f"Polymer energy:    {E_polymer:.3f}")
    print(f"Binding energy:    {delta_G_bind:.3f}")
    print("="*50)
    
    return {
        'E_complex': E_complex,
        'E_insulin': E_insulin,
        'E_polymer': E_polymer,
        'delta_G_bind': delta_G_bind
    }

def process_trajectory(trajectory_pdb, polymer_pdb_path, output_file="mmgbsa_results.txt"):
    """Process multiple frames from a trajectory and save results."""
    
    # For this minimal version, we'll process just one frame
    # You can extend this to loop through multiple frames if needed
    
    results = mmgbsa_calculation(trajectory_pdb, polymer_pdb_path)
    
    # Save results
    with open(output_file, 'w') as f:
        f.write("Frame\tE_complex\tE_insulin\tE_polymer\tDelta_G_bind\n")
        f.write(f"1\t{results['E_complex']:.6f}\t{results['E_insulin']:.6f}\t"
                f"{results['E_polymer']:.6f}\t{results['delta_G_bind']:.6f}\n")
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == "__main__":
    # Example usage
    trajectory_file = "trajectory_implicit.pdb"  # Your trajectory PDB
    polymer_file = "./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb"
    
    # Run MMGBSA calculation
    results = process_trajectory(trajectory_file, polymer_file)
