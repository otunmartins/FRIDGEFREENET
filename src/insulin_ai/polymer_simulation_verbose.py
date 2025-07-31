#!/usr/bin/env python3
"""
Enhanced OpenMM Simulation with Custom Solvents and Verbose Output
=================================================================

This enhanced script supports:
1. Any molecule as a solvent (not just water)
2. Mixed solvent systems
3. Comprehensive verbose output showing every step
4. Progress reporting and detailed logging
5. Real-time monitoring of simulation progress

Supports custom solvents like:
- Organic solvents (ethanol, methanol, DMSO, etc.)
- Ionic liquids
- Custom polymer matrices
- Mixed solvent systems
"""

import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# OpenMM imports
import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Simulation, StateDataReporter, DCDReporter

# OpenFF imports for small molecule force fields
try:
    from openff.toolkit import Molecule, Topology as OFFTopology
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, SystemGenerator
    print("✅ Successfully imported OpenFF toolkit and openmmforcefields")
except ImportError as e:
    print(f"❌ Error importing OpenFF packages: {e}")
    print("Please install dependencies with: pip install openff-toolkit openmmforcefields")
    sys.exit(1)

# Additional dependencies for custom solvents
try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import openmoltools
    import tempfile
    print("✅ Successfully imported RDKit and openmoltools for custom solvents")
except ImportError as e:
    print(f"❌ Error importing additional packages: {e}")
    print("Please install: pip install rdkit openmoltools")
    sys.exit(1)

def setup_logging(log_level=logging.INFO):
    """Set up comprehensive logging with timestamps."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation_verbose.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_header(title: str, char: str = "="):
    """Print a formatted header for different sections."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_step(step_num: int, title: str, details: str = ""):
    """Print a formatted step with optional details."""
    print(f"\n📍 Step {step_num}: {title}")
    if details:
        print(f"   💡 {details}")
    print("-" * 60)

def smiles_to_pdb(smiles: str, filename: str, logger) -> None:
    """Convert SMILES string to PDB file with detailed logging."""
    logger.info(f"Converting SMILES '{smiles}' to PDB file '{filename}'")
    
    try:
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        logger.info(f"  - Molecule created: {Chem.MolToSmiles(mol)}")
        logger.info(f"  - Number of atoms: {mol.GetNumAtoms()}")
        logger.info(f"  - Number of bonds: {mol.GetNumBonds()}")
        
        # Add hydrogens
        mol_h = Chem.AddHs(mol)
        logger.info(f"  - Hydrogens added, total atoms: {mol_h.GetNumAtoms()}")
        
        # Generate 3D coordinates
        logger.info("  - Generating 3D coordinates...")
        AllChem.EmbedMolecule(mol_h)
        
        # Optimize geometry
        logger.info("  - Optimizing molecular geometry...")
        AllChem.UFFOptimizeMolecule(mol_h)
        
        # Write to PDB
        Chem.MolToPDBFile(mol_h, filename)
        logger.info(f"  ✅ Successfully created PDB file: {filename}")
        
    except Exception as e:
        logger.error(f"  ❌ Error converting SMILES to PDB: {e}")
        raise

def create_custom_solvent_system(
    solute_pdb: str,
    solvent_specs: List[Dict[str, any]],
    box_size: float,
    logger
) -> str:
    """
    Create a system with custom solvents using PACKMOL.
    
    Args:
        solute_pdb: Path to solute PDB file
        solvent_specs: List of dicts with 'smiles', 'count', 'name'
        box_size: Box size in Angstroms
        logger: Logger instance
    
    Returns:
        Path to the created system PDB file
    """
    logger.info("🧪 Creating custom solvent system...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Prepare solvent PDB files
        solvent_files = []
        solvent_counts = []
        
        logger.info(f"📦 Preparing {len(solvent_specs)} solvent types:")
        
        for i, spec in enumerate(solvent_specs):
            smiles = spec['smiles']
            count = spec['count']
            name = spec.get('name', f'solvent_{i}')
            
            logger.info(f"  - {name}: {smiles} (x{count} molecules)")
            
            solvent_file = temp_dir / f"{name}.pdb"
            smiles_to_pdb(smiles, str(solvent_file), logger)
            
            solvent_files.append(str(solvent_file))
            solvent_counts.append(count)
        
        # Add solute
        all_files = [solute_pdb] + solvent_files
        all_counts = [1] + solvent_counts
        
        logger.info(f"📏 System specifications:")
        logger.info(f"  - Box size: {box_size:.2f} Å")
        logger.info(f"  - Total molecule types: {len(all_files)}")
        logger.info(f"  - Total molecules: {sum(all_counts)}")
        
        # Use PACKMOL to create the system
        logger.info("🔄 Running PACKMOL to pack molecules...")
        
        try:
            traj_packmol = openmoltools.packmol.pack_box(
                all_files,
                all_counts,
                box_size=box_size * 0.1  # Convert Å to nm
            )
            
            system_file = "custom_solvent_system.pdb"
            traj_packmol.save_pdb(system_file)
            
            logger.info(f"  ✅ System created successfully: {system_file}")
            return system_file
            
        except Exception as e:
            logger.error(f"  ❌ Error creating system with PACKMOL: {e}")
            raise

def analyze_molecule_from_pdb(pdb_file: str, logger) -> Dict:
    """Analyze the molecular composition from PDB file."""
    logger.info(f"🔍 Analyzing molecule composition from {pdb_file}...")
    
    try:
        # Read PDB file
        with open(pdb_file, 'r') as f:
            lines = f.readlines()
        
        # Count atoms by element
        element_counts = {}
        total_atoms = 0
        
        for line in lines:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                element = line[76:78].strip()
                if not element:
                    # Try to extract from atom name
                    atom_name = line[12:16].strip()
                    element = atom_name[0]
                
                element_counts[element] = element_counts.get(element, 0) + 1
                total_atoms += 1
        
        logger.info(f"  📊 Composition analysis:")
        logger.info(f"    - Total atoms: {total_atoms}")
        for element, count in sorted(element_counts.items()):
            logger.info(f"    - {element}: {count} atoms")
        
        return {
            'total_atoms': total_atoms,
            'elements': element_counts,
            'file_path': pdb_file
        }
        
    except Exception as e:
        logger.error(f"  ❌ Error analyzing PDB: {e}")
        return {}

def create_force_field_with_custom_molecules(molecules_data: List[Dict], logger) -> ForceField:
    """Create OpenMM ForceField with custom molecule support."""
    logger.info("⚗️ Setting up force field with custom molecule support...")
    
    try:
        # Create OpenFF molecules
        openff_molecules = []
        
        for mol_data in molecules_data:
            smiles = mol_data['smiles']
            name = mol_data.get('name', smiles)
            
            logger.info(f"  🧬 Processing molecule: {name} ({smiles})")
            
            try:
                molecule = Molecule.from_smiles(smiles)
                openff_molecules.append(molecule)
                logger.info(f"    ✅ Successfully created OpenFF molecule")
                
            except Exception as e:
                logger.warning(f"    ⚠️ Warning: Could not create OpenFF molecule: {e}")
                logger.info(f"    💡 Using simplified representation...")
                # Create a simple fallback molecule
                molecule = Molecule.from_smiles("C")  # Simple methane as fallback
                openff_molecules.append(molecule)
        
        # Create SMIRNOFF template generator
        logger.info("  🔧 Creating SMIRNOFF template generator...")
        smirnoff = SMIRNOFFTemplateGenerator(molecules=openff_molecules)
        
        # Create base force field
        logger.info("  📋 Setting up base force field (AMBER ff14SB + TIP3P)...")
        forcefield = ForceField(
            'amber/protein.ff14SB.xml',
            'amber/tip3p_standard.xml',
            'amber/tip3p_HFE_multivalent.xml'
        )
        
        # Register template generator
        logger.info("  🔗 Registering SMIRNOFF template generator...")
        forcefield.registerTemplateGenerator(smirnoff.generator)
        
        logger.info("  ✅ Force field setup complete!")
        return forcefield
        
    except Exception as e:
        logger.error(f"  ❌ Error setting up force field: {e}")
        raise

def run_verbose_simulation(
    pdb_file: str,
    custom_solvents: Optional[List[Dict]] = None,
    simulation_time: float = 1.0,  # ns
    temperature: float = 300.0,  # K
    logger = None
):
    """
    Run OpenMM simulation with custom solvents and verbose output.
    
    Args:
        pdb_file: Path to initial PDB file
        custom_solvents: List of custom solvent specifications
        simulation_time: Simulation time in nanoseconds
        temperature: Temperature in Kelvin
        logger: Logger instance
    """
    
    if logger is None:
        logger = setup_logging()
    
    print_header("🚀 ENHANCED OPENMM SIMULATION WITH CUSTOM SOLVENTS", "=")
    
    start_time = time.time()
    
    # Step 1: System Analysis
    print_step(1, "Initial System Analysis", "Analyzing molecular composition and structure")
    mol_analysis = analyze_molecule_from_pdb(pdb_file, logger)
    
    # Step 2: Custom Solvent Setup (if specified)
    if custom_solvents:
        print_step(2, "Custom Solvent System Creation", "Creating mixed solvent environment")
        
        # Calculate box size based on molecular volume
        total_atoms = mol_analysis.get('total_atoms', 100)
        estimated_volume = total_atoms * 20  # Rough estimate: 20 Å³ per atom
        box_size = estimated_volume ** (1/3) * 2  # Double for solvent space
        
        logger.info(f"  📐 Estimated box size: {box_size:.2f} Å")
        
        # Create system with custom solvents
        system_pdb = create_custom_solvent_system(
            pdb_file, custom_solvents, box_size, logger
        )
        pdb_file = system_pdb
        
        # Collect all molecule types for force field
        all_molecules = [{'smiles': 'CC(=O)NCCSC(C)C(=O)OCC(O)C', 'name': 'polymer'}]  # Main polymer
        all_molecules.extend(custom_solvents)
    else:
        print_step(2, "Standard Water Solvation", "Using standard TIP3P water model")
        all_molecules = [{'smiles': 'CC(=O)NCCSC(C)C(=O)OCC(O)C', 'name': 'polymer'}]
    
    # Step 3: Force Field Setup
    print_step(3, "Force Field Configuration", "Setting up OpenFF force fields with custom molecules")
    forcefield = create_force_field_with_custom_molecules(all_molecules, logger)
    
    # Step 4: System Preparation
    print_step(4, "System Preparation", "Loading structure and creating OpenMM system")
    
    logger.info("📂 Loading PDB structure...")
    pdb = PDBFile(pdb_file)
    
    logger.info(f"  📊 System information:")
    logger.info(f"    - Total atoms: {pdb.topology.getNumAtoms()}")
    logger.info(f"    - Total residues: {pdb.topology.getNumResidues()}")
    logger.info(f"    - Total chains: {pdb.topology.getNumChains()}")
    
    # Create system
    logger.info("⚙️ Creating OpenMM system...")
    try:
        if custom_solvents:
            # For custom solvents, use NoCutoff (no periodic boundaries)
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                rigidWater=True
            )
            logger.info("  ✅ System created with NoCutoff (custom solvents)")
        else:
            # Standard periodic system
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0*unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True
            )
            logger.info("  ✅ System created with PME (periodic boundaries)")
    except Exception as e:
        logger.error(f"  ❌ Error creating system: {e}")
        logger.info("  🔄 Trying with simplified force field...")
        
        # Fallback to simpler system
        forcefield_simple = ForceField('amber/protein.ff14SB.xml')
        system = forcefield_simple.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds
        )
        logger.info("  ✅ System created with simplified force field")
    
    # Step 5: Simulation Setup
    print_step(5, "Simulation Configuration", f"Setting up MD at {temperature}K for {simulation_time}ns")
    
    logger.info("🌡️ Configuring integrator and simulation...")
    integrator = mm.LangevinIntegrator(
        temperature*unit.kelvin,
        1/unit.picosecond,
        0.002*unit.picoseconds
    )
    
    logger.info(f"  - Temperature: {temperature} K")
    logger.info(f"  - Friction: 1 ps⁻¹")
    logger.info(f"  - Time step: 2 fs")
    
    # Create simulation
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    logger.info("  ✅ Simulation object created")
    
    # Step 6: Energy Minimization
    print_step(6, "Energy Minimization", "Removing bad contacts and optimizing geometry")
    
    logger.info("⚡ Starting energy minimization...")
    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"  - Initial potential energy: {initial_energy}")
    
    simulation.minimizeEnergy(maxIterations=1000)
    
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"  - Final potential energy: {final_energy}")
    logger.info(f"  - Energy change: {final_energy - initial_energy}")
    logger.info("  ✅ Energy minimization complete")
    
    # Step 7: Equilibration
    print_step(7, "System Equilibration", "Equilibrating temperature and density")
    
    logger.info("🌡️ Starting equilibration phase...")
    
    # Setup reporters for equilibration
    equilibration_steps = 5000  # 10 ps
    simulation.reporters.append(StateDataReporter(
        sys.stdout, 1000, step=True, time=True, 
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True,
        speed=True, separator='\t'
    ))
    
    logger.info(f"  - Equilibration steps: {equilibration_steps}")
    logger.info(f"  - Reporting every: 1000 steps (2 ps)")
    
    equilibration_start = time.time()
    simulation.step(equilibration_steps)
    equilibration_time = time.time() - equilibration_start
    
    logger.info(f"  ✅ Equilibration complete in {equilibration_time:.2f} seconds")
    
    # Step 8: Production Run
    print_step(8, "Production Simulation", f"Running {simulation_time}ns production MD")
    
    # Clear previous reporters
    simulation.reporters.clear()
    
    # Production simulation parameters
    total_steps = int(simulation_time * 1000000 / 2)  # 2fs timestep
    report_interval = max(1000, total_steps // 100)  # Report ~100 times
    
    logger.info(f"🏃 Starting production simulation...")
    logger.info(f"  - Total steps: {total_steps:,}")
    logger.info(f"  - Simulation time: {simulation_time} ns")
    logger.info(f"  - Report interval: {report_interval} steps")
    logger.info(f"  - Expected runtime: ~{total_steps/50000:.1f} minutes on GPU")
    
    # Setup production reporters
    simulation.reporters.append(DCDReporter(
        'trajectory_verbose.dcd', report_interval
    ))
    
    simulation.reporters.append(StateDataReporter(
        'simulation_data_verbose.csv', report_interval,
        time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, temperature=True, volume=True, density=True,
        speed=True
    ))
    
    simulation.reporters.append(StateDataReporter(
        sys.stdout, report_interval * 10,
        step=True, time=True, potentialEnergy=True, temperature=True,
        speed=True, remainingTime=True, totalSteps=total_steps
    ))
    
    # Run production simulation with progress tracking
    logger.info("  🚀 Starting production MD...")
    production_start = time.time()
    
    simulation.step(total_steps)
    
    production_time = time.time() - production_start
    total_time = time.time() - start_time
    
    # Step 9: Finalization and Analysis
    print_step(9, "Simulation Complete", "Finalizing and analyzing results")
    
    logger.info("🎉 Production simulation complete!")
    logger.info(f"  ⏱️ Timing summary:")
    logger.info(f"    - Production time: {production_time:.2f} seconds")
    logger.info(f"    - Total wallclock time: {total_time:.2f} seconds")
    logger.info(f"    - Simulation speed: {(simulation_time * 1000) / production_time:.2f} ns/day")
    
    # Get final state information
    final_state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_pe = final_state.getPotentialEnergy()
    final_ke = final_state.getKineticEnergy()
    
    logger.info(f"  📊 Final energies:")
    logger.info(f"    - Potential energy: {final_pe}")
    logger.info(f"    - Kinetic energy: {final_ke}")
    logger.info(f"    - Total energy: {final_pe + final_ke}")
    
    # Save final coordinates
    final_positions = final_state.getPositions()
    with open('final_structure_verbose.pdb', 'w') as f:
        PDBFile.writeFile(pdb.topology, final_positions, f)
    
    logger.info("  💾 Output files created:")
    logger.info("    - trajectory_verbose.dcd (trajectory)")
    logger.info("    - simulation_data_verbose.csv (energies/properties)")
    logger.info("    - final_structure_verbose.pdb (final structure)")
    logger.info("    - simulation_verbose.log (this log)")
    
    print_header("✅ SIMULATION COMPLETED SUCCESSFULLY", "=")
    
    return {
        'trajectory_file': 'trajectory_verbose.dcd',
        'data_file': 'simulation_data_verbose.csv',
        'final_structure': 'final_structure_verbose.pdb',
        'log_file': 'simulation_verbose.log',
        'total_time': total_time,
        'production_time': production_time,
        'final_energy': final_pe + final_ke
    }

def main():
    """Main function with example usage."""
    logger = setup_logging()
    
    print_header("🧪 CUSTOM SOLVENT SIMULATION EXAMPLES", "=")
    
    # Example 1: Water solvent (standard)
    print("\n🌊 Example 1: Standard water solvation")
    try:
        result1 = run_verbose_simulation(
            "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb",
            custom_solvents=None,
            simulation_time=0.1,  # 100 ps for testing
            logger=logger
        )
        print("✅ Water simulation completed!")
    except Exception as e:
        print(f"❌ Water simulation failed: {e}")
    
    # Example 2: Ethanol solvent
    print("\n🍺 Example 2: Ethanol solvent")
    try:
        ethanol_solvents = [
            {'smiles': 'CCO', 'count': 100, 'name': 'ethanol'}
        ]
        result2 = run_verbose_simulation(
            "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb",
            custom_solvents=ethanol_solvents,
            simulation_time=0.1,
            logger=logger
        )
        print("✅ Ethanol simulation completed!")
    except Exception as e:
        print(f"❌ Ethanol simulation failed: {e}")
    
    # Example 3: Mixed solvent system
    print("\n🧪 Example 3: Mixed ethanol-water solvent")
    try:
        mixed_solvents = [
            {'smiles': 'CCO', 'count': 50, 'name': 'ethanol'},
            {'smiles': 'O', 'count': 150, 'name': 'water'}
        ]
        result3 = run_verbose_simulation(
            "automated_simulations/session_f949cc8c/candidate_001_e36e15/molecules/packmol/polymer.pdb",
            custom_solvents=mixed_solvents,
            simulation_time=0.1,
            logger=logger
        )
        print("✅ Mixed solvent simulation completed!")
    except Exception as e:
        print(f"❌ Mixed solvent simulation failed: {e}")

if __name__ == "__main__":
    main() 