# Create an OpenFF Molecule object for polymer from SMILES
from openff.toolkit import Molecule
from rdkit import Chem

# Load polymer molecule from PDB and convert to SMILES
your_mol = Chem.MolFromPDBFile('./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb')
smiles = Chem.MolToSmiles(your_mol)
molecule = Molecule.from_smiles(smiles)
molecule.assign_partial_charges("gasteiger")

# Create the GAFF template generator for polymer
from openmmforcefields.generators import GAFFTemplateGenerator
gaff = GAFFTemplateGenerator(molecules=molecule)

# Create an OpenMM ForceField object with AMBER ff14SB for protein
# NOTE: Removed TIP3P water models - not needed for implicit solvent!
from openmm.app import ForceField
from openmm import app  # Need this for the implicit solvent constants
forcefield = ForceField(
    "amber/protein.ff14SB.xml",
    # No water models needed for implicit solvent
)

# Register the GAFF template generator for polymer
forcefield.registerTemplateGenerator(gaff.generator)

# Load the system PDB file
from openmm.app import PDBFile
pdbfile = PDBFile("./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/insulin_polymer_composite_001_e36e15_preprocessed.pdb")

# Create system with GBN2 implicit solvent
print("Creating system with GBN2 implicit solvent...")
system = forcefield.createSystem(
    pdbfile.topology,
    nonbondedMethod=app.NoCutoff,        # No cutoff for implicit solvent
    implicitSolvent=app.GBn2,            # GBN2 implicit solvent model
    constraints=app.HBonds,              # Constrain hydrogen bonds
    rigidWater=False,                    # No explicit water molecules
    removeCMMotion=True                  # Remove center of mass motion
)
print("✅ System created successfully with GBN2 implicit solvent!")

# Import additional OpenMM modules
from openmm import LangevinIntegrator, Platform
from openmm.app import Simulation, PDBReporter, StateDataReporter
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole
import numpy as np

# Set up the integrator for NVT simulation
# Using slightly lower friction for implicit solvent
temperature = 300 * kelvin
friction = 1.0 / picosecond  # Good for implicit solvent
timestep = 2.0 * femtosecond
integrator = LangevinIntegrator(temperature, friction, timestep)

# Create the simulation object
platform = Platform.getPlatformByName('CUDA')  # Use 'CPU' if CUDA not available
simulation = Simulation(pdbfile.topology, system, integrator, platform)
simulation.context.setPositions(pdbfile.positions)

# Energy minimization
print("Starting energy minimization...")
simulation.minimizeEnergy(maxIterations=2000)  # More iterations for implicit solvent
print("Energy minimization completed.")

# Get minimized energy
state = simulation.context.getState(getEnergy=True)
minimized_energy = state.getPotentialEnergy()
print(f"Minimized potential energy: {minimized_energy}")

# Set up reporters for trajectory and energy output
# PDB trajectory reporter (save every 1000 steps)
pdb_reporter = PDBReporter('trajectory_implicit_gbn2.pdb', 1000)
simulation.reporters.append(pdb_reporter)

# State data reporter for energy output (every 100 steps)
state_reporter = StateDataReporter(
    'simulation_log_implicit_gbn2.txt', 
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

# Console reporter to monitor progress
console_reporter = StateDataReporter(
    None,  # Output to console
    500,   # Report every 500 steps for implicit solvent
    step=True,
    time=True,
    potentialEnergy=True,
    temperature=True,
    separator='\t'
)
simulation.reporters.append(console_reporter)

# Run MD simulation
print("Starting NVT MD simulation with GBN2 implicit solvent...")
print("📊 Simulation details:")
print(f"   - Temperature: {temperature}")
print(f"   - Friction: {friction}")
print(f"   - Timestep: {timestep}")
print(f"   - Implicit solvent: GBN2")
print(f"   - Nonbonded method: NoCutoff")
print(f"   - Platform: {platform.getName()}")

num_steps = 100000  # 200 ps with 2 fs timestep - good for implicit solvent
simulation.step(num_steps)
print("MD simulation completed.")

# Final state information
final_state = simulation.context.getState(getEnergy=True, getPositions=True)
final_pe = final_state.getPotentialEnergy()
final_positions = final_state.getPositions()

print(f"\n🎉 SIMULATION COMPLETED SUCCESSFULLY!")
print(f"📊 Final Results:")
print(f"   - Final potential energy: {final_pe}")
print(f"   - Total simulation time: {num_steps * timestep}")
print(f"   - Trajectory saved to: trajectory_implicit_gbn2.pdb")
print(f"   - Energy log saved to: simulation_log_implicit_gbn2.txt")
print(f"   - Implicit solvent model: GBN2")
print(f"   - Total atoms simulated: {system.getNumParticles()}")

# Save final structure
final_pdb = PDBFile("final_structure_implicit_gbn2.pdb")
with open("final_structure_implicit_gbn2.pdb", 'w') as f:
    PDBFile.writeFile(simulation.topology, final_positions, f)
print(f"   - Final structure saved to: final_structure_implicit_gbn2.pdb") 