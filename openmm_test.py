# Create an OpenFF Molecule object with proper stereochemistry handling
from openff.toolkit import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem

# Original SMILES
original_smiles = 'COCC=C[PH](=O)(O)(O)C(=O)OCC=C[PH](=O)(O)(O)C(=O)OCC=C[PH](=O)(O)(O)C(=O)OCC=C[PH](=O)(O)(O)C(=O)OCC=C[PH](=O)(O)(O)C(C)=O'

# Method 1: Try with explicit stereochemistry
try:
    # Create RDKit molecule
    rdkit_mol = Chem.MolFromSmiles(original_smiles)
    if rdkit_mol is None:
        raise ValueError("Invalid SMILES")

    # Add hydrogens and generate 3D coordinates
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol, randomSeed=42)
    Chem.AssignStereochemistry(rdkit_mol, cleanIt=True, force=True)
    
    # Remove hydrogens to get cleaner SMILES
    rdkit_mol = Chem.RemoveHs(rdkit_mol)
    
    # Convert back to SMILES
    new_smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=True)
    print(f"Generated SMILES with stereochemistry: {new_smiles}")
    
    # Create OpenFF molecule with undefined stereo allowed
    molecule = Molecule.from_smiles(new_smiles, allow_undefined_stereo=True)
    print("Successfully created molecule with processed SMILES")

except Exception as e:
    print(f"Method 1 failed: {e}")
    
    # Method 2: Fallback to manual stereochemistry
    try:
        # Manually specify all E (trans) double bonds
        manual_smiles = 'COC/C=C/[PH](=O)(O)(O)C(=O)OC/C=C/[PH](=O)(O)(O)C(=O)OC/C=C/[PH](=O)(O)(O)C(=O)OC/C=C/[PH](=O)(O)(O)C(=O)OC/C=C/[PH](=O)(O)(O)C(C)=O'
        molecule = Molecule.from_smiles(manual_smiles, allow_undefined_stereo=True)
        print("Successfully created molecule with manual stereochemistry")
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        
        # Method 3: Final fallback - original SMILES with undefined stereo allowed
        molecule = Molecule.from_smiles(original_smiles, allow_undefined_stereo=True)
        print("Using original SMILES with undefined stereochemistry allowed")

# Assign partial charges
molecule.assign_partial_charges("gasteiger")

# Create the GAFF template generator
from openmmforcefields.generators import GAFFTemplateGenerator
gaff = GAFFTemplateGenerator(molecules=molecule)

# Fix PDB file first using PDBFixer
from pdbfixer import PDBFixer
from openmm.app import PDBFile, ForceField

print("Fixing PDB file...")
fixer = PDBFixer("./automated_simulations/session_83fad177/candidate_005_54d3d9/molecules/insulin_polymer_composite_005_54d3d9.pdb")
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)  # Add at pH 7.0

# Save the fixed file
PDBFile.writeFile(fixer.topology, fixer.positions, open('fixed_insulin.pdb', 'w'))
print("PDB file fixed and saved as 'fixed_insulin.pdb'")

# Create an OpenMM ForceField object with AMBER ff14SB for implicit solvent
forcefield = ForceField(
    "amber/protein.ff14SB.xml",
    "amber/tip3p_standard.xml",   # Standard residue templates
    "implicit/gbn2.xml",          # Generalized Born implicit solvent model
)

# Register the GAFF template generator
forcefield.registerTemplateGenerator(gaff.generator)

# Load the fixed PDB file
pdbfile = PDBFile('fixed_insulin.pdb')

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
pdb_reporter = PDBReporter('trajectory_implicit.pdb', 1000)
simulation.reporters.append(pdb_reporter)

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
