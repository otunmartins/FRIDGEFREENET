# Create an OpenFF Molecule object for benzene from SMILES
from openff.toolkit import Molecule
from rdkit import Chem
your_mol = Chem.MolFromPDBFile('./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb')
smiles = Chem.MolToSmiles(your_mol)
molecule = Molecule.from_smiles(smiles)
molecule.assign_partial_charges("gasteiger")

# Create the GAFF template generator
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
)
gaff = GAFFTemplateGenerator(molecules=molecule)

# Create an OpenMM ForceField object with AMBER ff14SB for implicit solvent
# Remove explicit water models - not needed for implicit solvent
from openmm.app import ForceField
forcefield = ForceField(
    "amber/protein.ff14SB.xml",
    "implicit/gbn2.xml",  # Generalized Born implicit solvent model
)

# Register the GAFF template generator
forcefield.registerTemplateGenerator(gaff.generator)

# You can now parameterize an OpenMM Topology object that contains the specified molecule.
# forcefield will load the appropriate GAFF parameters when needed, and antechamber
# will be used to generate small molecule parameters on the fly.
from openmm.app import PDBFile
pdbfile = PDBFile("./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/insulin_polymer_composite_001_e36e15_preprocessed.pdb")

# Create system with implicit solvent parameters
from openmm import app
system = forcefield.createSystem(
    pdbfile.topology,
    nonbondedMethod=app.NoCutoff,  # No cutoff needed for implicit solvent
    solventDielectric=78.5,        # Water dielectric constant
    soluteDielectric=1.0,          # Protein/solute dielectric constant
    constraints=app.HBonds,        # Constrain bonds involving hydrogen
    rigidWater=False,              # Not applicable for implicit solvent
    removeCMMotion=True            # Remove center of mass motion
)

# Import additional OpenMM modules
from openmm import LangevinIntegrator, Platform
from openmm.app import Simulation, PDBReporter, StateDataReporter
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole, molar
import numpy as np

# Set up the integrator for NVT simulation
temperature = 300 * kelvin
friction = 1.0 / picosecond
timestep = 2.0 * femtosecond
integrator = LangevinIntegrator(temperature, friction, timestep)

# Create the simulation object
platform = Platform.getPlatformByName('CUDA')  # Use 'CPU' if CUDA not available
simulation = Simulation(pdbfile.topology, system, integrator, platform)
simulation.context.setPositions(pdbfile.positions)

# Energy minimization
print("Starting energy minimization...")
simulation.minimizeEnergy(maxIterations=1000)
print("Energy minimization completed.")

# Get minimized energy
state = simulation.context.getState(getEnergy=True)
minimized_energy = state.getPotentialEnergy()
print(f"Minimized potential energy: {minimized_energy}")

# Set up reporters for trajectory and energy output
# PDB trajectory reporter (save every 1000 steps)
pdb_reporter = PDBReporter('trajectory_implicit.pdb', 1000)
simulation.reporters.append(pdb_reporter)

# State data reporter for energy output (every 100 steps)
# Note: volume/density not meaningful for implicit solvent
state_reporter = StateDataReporter(
    'simulation_log_implicit.txt', 
    100,  # Report every 100 steps
    step=True,
    time=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    temperature=True,
    separator='\t'
)
simulation.reporters.append(state_reporter)

# Console reporter to see PE every 100 steps
console_reporter = StateDataReporter(
    None,  # Output to console
    100,   # Report every 100 steps  
    step=True,
    time=True,
    potentialEnergy=True,
    temperature=True,
    separator='\t'
)
simulation.reporters.append(console_reporter)

# Run MD simulation
print("Starting NVT MD simulation with implicit solvent...")
num_steps = 50000  # Adjust as needed (100 ps with 2 fs timestep)
simulation.step(num_steps)
print("MD simulation completed.")

# Final state information
final_state = simulation.context.getState(getEnergy=True, getPositions=True)
final_pe = final_state.getPotentialEnergy()
final_positions = final_state.getPositions()

print(f"Final potential energy: {final_pe}")
print(f"Simulation completed successfully!")
print(f"Trajectory saved to: trajectory_implicit.pdb")
print(f"Energy log saved to: simulation_log_implicit.txt")
print("Implicit solvent parameters:")
print("- Model: Generalized Born with neck correction (GBn2)")
print("- Solvent dielectric: 78.5 (water)")
print("- Salt concentration: 150 mM")
print("- No periodic boundary conditions")