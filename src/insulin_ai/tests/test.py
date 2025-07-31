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
# Create an OpenMM ForceField object with AMBER ff14SB and TIP3P with compatible ions
from openmm.app import ForceField

forcefield = ForceField(
    "amber/protein.ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
)
# Register the GAFF template generator
forcefield.registerTemplateGenerator(gaff.generator)
# You can now parameterize an OpenMM Topology object that contains the specified molecule.
# forcefield will load the appropriate GAFF parameters when needed, and antechamber
# will be used to generate small molecule parameters on the fly.
from openmm.app import PDBFile

pdbfile = PDBFile("./automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/insulin_polymer_composite_001_e36e15_preprocessed.pdb")
system = forcefield.createSystem(pdbfile.topology)
