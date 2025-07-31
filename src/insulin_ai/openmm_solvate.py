# Install openmoltools first: conda install -c conda-forge openmoltools

import openmoltools
from openmoltools import packmol
from openff.toolkit import Molecule
from rdkit import Chem
from openmm.app import PDBFile, ForceField
from openmm import unit
import numpy as np

# Step 1: Prepare your GAFF polymer molecule (as you already have)
your_mol = Chem.MolFromPDBFile('./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb')
smiles = Chem.MolToSmiles(your_mol)
polymer_molecule = Molecule.from_smiles(smiles)
polymer_molecule.assign_partial_charges("gasteiger")

# Step 2: Load the insulin PDB
insulin_pdb = PDBFile("automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/processed_insulin/insulin_processed_001_e36e15.pdb")  # Replace with your insulin PDB path

# Step 3: Convert polymer molecule to PDB format for packmol
# Save the polymer as a temporary PDB file
polymer_molecule.to_file("polymer_temp.pdb", file_format="pdb")

# Step 4: Set up the packing parameters
box_size = 50.0  # Angstroms - adjust based on your system size
num_polymers = 100  # Number of polymer molecules to add around insulin

# Step 5: Create packmol input to mix insulin + polymer molecules
# Note: This requires both molecules to be in PDB format

# Method 1: Using openmoltools pack_box function
# Prepare molecule list with counts
molecules_list = [
    ('./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/processed_insulin/insulin_processed_001_e36e15.pdb', 1),      # 1 insulin molecule
    ('polymer_temp.pdb', num_polymers)  # Multiple polymer molecules
]

# Pack the box using packmol via openmoltools
packed_pdb = packmol.pack_box(
    molecules_list,
    n_molecules_list=[1, num_polymers],
    box_size=[box_size, box_size, box_size]
)

packed_pdb.save_pdb('insulin_polymer_mixture.pdb')
print(f"Successfully packed system with {num_polymers} polymer molecules around insulin")
    

# Step 6: Load the packed system
mixed_system = PDBFile('insulin_polymer_mixture.pdb')
print(f"Loaded mixed system with {mixed_system.topology.getNumAtoms()} atoms")

# Step 7: Set up force field for the mixed system
from openmmforcefields.generators import GAFFTemplateGenerator

# Create GAFF template generator for the polymer
gaff = GAFFTemplateGenerator(molecules=polymer_molecule)

# Create forcefield combining protein and GAFF
forcefield = ForceField(
    "amber/protein.ff14SB.xml",  # For insulin
)

# Register the GAFF template generator for polymer
forcefield.registerTemplateGenerator(gaff.generator)

# Step 8: Create the system
from openmm import app
system = forcefield.createSystem(
    mixed_system.topology,
    nonbondedMethod=app.NoCutoff,     # No cutoff for now
    constraints=app.HBonds,           # Constrain hydrogen bonds
    removeCMMotion=True               # Remove center of mass motion
)

# Step 9: Set up MD simulation (similar to your previous script)
from openmm import LangevinIntegrator, Platform
from openmm.app import Simulation, PDBReporter, StateDataReporter

temperature = 300 * unit.kelvin
friction = 1.0 / unit.picosecond
timestep = 2.0 * unit.femtosecond
integrator = LangevinIntegrator(temperature, friction, timestep)

# Create simulation
platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(mixed_system.topology, system, integrator, platform)
simulation.context.setPositions(mixed_system.positions)

# Energy minimization
print("Starting energy minimization...")
simulation.minimizeEnergy(maxIterations=2000)
print("Energy minimization completed.")

# Set up reporters
pdb_reporter = PDBReporter('insulin_polymer_trajectory.pdb', 1000)
simulation.reporters.append(pdb_reporter)

state_reporter = StateDataReporter(
    'insulin_polymer_log.txt', 
    100,
    step=True,
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    temperature=True,
    separator='\t'
)
simulation.reporters.append(state_reporter)

console_reporter = StateDataReporter(
    None, 100,
    step=True,
    time=True,
    potentialEnergy=True,
    temperature=True,
    separator='\t'
)
simulation.reporters.append(console_reporter)

# Run simulation
print("Starting MD simulation of insulin in polymer environment...")
num_steps = 50000
simulation.step(num_steps)

print("Simulation completed!")
print("Output files:")
print("- insulin_polymer_mixture.pdb: Initial packed structure")
print("- insulin_polymer_trajectory.pdb: MD trajectory") 
print("- insulin_polymer_log.txt: Energy and thermodynamic data")

# Clean up temporary files
import os
try:
    os.remove('polymer_temp.pdb')
    os.remove('packmol_input.inp')
except:
    pass
