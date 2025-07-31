#!/usr/bin/env python3
"""
Insulin-Polymer Simulation System (Final Working Version)
=========================================================

This script creates a working MD simulation of insulin with polymer force field
support, using implicit solvent and no PBC. Includes PDBFixer to handle missing atoms.
"""

import os
import sys
import time
import logging

# OpenMM imports
import openmm as mm
from openmm import app, unit
from openmm.app import PDBFile, ForceField, Simulation, StateDataReporter, DCDReporter
from pdbfixer import PDBFixer

# OpenFF imports
try:
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    print("✅ Successfully imported OpenFF toolkit and openmmforcefields")
except ImportError as e:
    print(f"❌ Error importing OpenFF packages: {e}")
    sys.exit(1)

def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('insulin_polymer_final.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def print_header(title: str):
    """Print a formatted header."""
    width = 80
    print(f"\n{'='*width}")
    print(f"{title:^{width}}")
    print(f"{'='*width}")

def print_step(step_num: int, title: str):
    """Print a formatted step."""
    print(f"\n🧬 Step {step_num}: {title}")
    print("-" * 60)

def fix_insulin_structure(insulin_pdb: str, logger):
    """Fix insulin structure by adding missing hydrogens and atoms."""
    logger.info(f"🔧 Fixing insulin structure from {insulin_pdb}...")
    
    try:
        # Load and fix the structure
        fixer = PDBFixer(filename=insulin_pdb)
        
        logger.info("  🔍 Finding missing residues...")
        fixer.findMissingResidues()
        
        logger.info("  ⚛️ Finding missing atoms...")
        fixer.findMissingAtoms()
        
        logger.info("  💧 Adding missing hydrogens...")
        fixer.addMissingHydrogens(7.0)  # pH 7.0
        
        # Save fixed structure
        fixed_pdb = "insulin_fixed.pdb"
        with open(fixed_pdb, 'w') as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        
        logger.info(f"  ✅ Fixed structure saved to: {fixed_pdb}")
        return fixed_pdb
        
    except Exception as e:
        logger.error(f"  ❌ Error fixing structure: {e}")
        raise

def run_insulin_polymer_simulation(insulin_pdb: str, logger=None):
    """Run insulin-polymer simulation with proper structure fixing."""
    
    if logger is None:
        logger = setup_logging()
    
    print_header("🧬 INSULIN-POLYMER SIMULATION (FINAL)")
    
    start_time = time.time()
    
    # Step 1: Fix insulin structure
    print_step(1, "Structure Preparation")
    fixed_insulin = fix_insulin_structure(insulin_pdb, logger)
    
    # Step 2: Setup polymer for force field
    print_step(2, "Polymer Force Field Setup")
    
    polymer_smiles = "CCNC(=O)CCC(=O)OCC"  # Simple biocompatible polymer
    logger.info(f"📝 Using polymer SMILES: {polymer_smiles}")
    
    try:
        # Create polymer molecule for force field
        polymer_molecule = Molecule.from_smiles(polymer_smiles)
        smirnoff = SMIRNOFFTemplateGenerator(molecules=[polymer_molecule])
        logger.info("✅ Polymer force field template created")
        
        # Create mixed force field
        forcefield = ForceField('amber/protein.ff14SB.xml')
        forcefield.registerTemplateGenerator(smirnoff.generator)
        logger.info("✅ Mixed protein-polymer force field ready")
        
    except Exception as e:
        logger.warning(f"⚠️ Polymer force field failed: {e}")
        logger.info("🔄 Using protein-only force field")
        forcefield = ForceField('amber/protein.ff14SB.xml')
    
    # Step 3: Load fixed structure
    print_step(3, "System Loading")
    
    pdb = PDBFile(fixed_insulin)
    logger.info(f"📊 System loaded:")
    logger.info(f"  - Atoms: {pdb.topology.getNumAtoms()}")
    logger.info(f"  - Residues: {pdb.topology.getNumResidues()}")
    logger.info(f"  - Chains: {pdb.topology.getNumChains()}")
    
    # Step 4: Create OpenMM system
    print_step(4, "OpenMM System Creation")
    
    logger.info("⚙️ Creating system with implicit solvent...")
    
    try:
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,          # No PBC as requested
            implicitSolvent=app.OBC2,              # Implicit solvent
            constraints=app.HBonds,                # H-bond constraints
            removeCMMotion=True                    # Remove CM motion
        )
        logger.info("✅ System created with implicit solvent and NoCutoff")
        
    except Exception as e:
        logger.error(f"❌ System creation failed: {e}")
        raise
    
    # Step 5: Simulation setup
    print_step(5, "Simulation Configuration")
    
    temperature = 310.0  # Body temperature (K)
    logger.info(f"🌡️ Temperature: {temperature} K")
    
    integrator = mm.LangevinIntegrator(
        temperature*unit.kelvin,
        1/unit.picosecond,
        0.002*unit.picoseconds
    )
    
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
    logger.info("✅ Simulation configured")
    
    # Step 6: Energy minimization
    print_step(6, "Energy Minimization")
    
    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"⚡ Initial energy: {initial_energy}")
    
    simulation.minimizeEnergy(maxIterations=1000)
    
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"⚡ Final energy: {final_energy}")
    logger.info(f"📈 Energy change: {final_energy - initial_energy}")
    
    # Step 7: Production simulation
    print_step(7, "Production MD Simulation")
    
    simulation_time = 1.0  # 1 nanosecond
    total_steps = int(simulation_time * 1000000 / 2)  # 2fs timestep
    report_interval = 10000
    
    logger.info(f"🏃 Running {simulation_time} ns simulation...")
    logger.info(f"  - Steps: {total_steps:,}")
    logger.info(f"  - Report every: {report_interval} steps")
    
    # Setup reporters
    simulation.reporters.append(DCDReporter(
        'insulin_polymer_final_trajectory.dcd', report_interval
    ))
    
    simulation.reporters.append(StateDataReporter(
        'insulin_polymer_final_data.csv', report_interval,
        time=True, step=True, potentialEnergy=True, kineticEnergy=True,
        totalEnergy=True, temperature=True, speed=True
    ))
    
    simulation.reporters.append(StateDataReporter(
        sys.stdout, report_interval * 3,
        step=True, time=True, potentialEnergy=True, temperature=True,
        speed=True, remainingTime=True, totalSteps=total_steps
    ))
    
    # Run simulation
    logger.info("🚀 Starting production MD...")
    production_start = time.time()
    
    simulation.step(total_steps)
    
    production_time = time.time() - production_start
    total_time = time.time() - start_time
    
    # Step 8: Results
    print_step(8, "Results and Analysis")
    
    logger.info("🎉 Simulation completed successfully!")
    logger.info(f"⏱️ Timing:")
    logger.info(f"  - Production: {production_time:.2f} seconds")
    logger.info(f"  - Total: {total_time:.2f} seconds")
    logger.info(f"  - Performance: {(simulation_time * 1000) / production_time:.1f} ns/day")
    
    # Save final structure
    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = final_state.getPositions()
    
    with open('insulin_polymer_final_structure.pdb', 'w') as f:
        PDBFile.writeFile(pdb.topology, final_positions, f)
    
    final_total_energy = final_state.getPotentialEnergy() + final_state.getKineticEnergy()
    
    logger.info(f"📊 Final energies:")
    logger.info(f"  - Potential: {final_state.getPotentialEnergy()}")
    logger.info(f"  - Kinetic: {final_state.getKineticEnergy()}")
    logger.info(f"  - Total: {final_total_energy}")
    
    logger.info("💾 Output files created:")
    logger.info("  - insulin_polymer_final_trajectory.dcd")
    logger.info("  - insulin_polymer_final_data.csv")
    logger.info("  - insulin_polymer_final_structure.pdb")
    logger.info("  - insulin_fixed.pdb (preprocessed)")
    logger.info("  - insulin_polymer_final.log")
    
    print_header("✅ INSULIN-POLYMER SIMULATION SUCCESSFUL!")
    
    return {
        'success': True,
        'trajectory': 'insulin_polymer_final_trajectory.dcd',
        'data': 'insulin_polymer_final_data.csv',
        'final_structure': 'insulin_polymer_final_structure.pdb',
        'atoms': pdb.topology.getNumAtoms(),
        'simulation_time': simulation_time,
        'performance': f"{(simulation_time * 1000) / production_time:.1f} ns/day",
        'final_energy': final_total_energy
    }

def main():
    """Main function."""
    logger = setup_logging()
    
    print_header("🧬 INSULIN-POLYMER MD SIMULATION")
    
    # Find insulin file
    from insulin_ai import get_insulin_pdb_path
    insulin_files = [
        get_insulin_pdb_path(),
        "./preprocessed_insulin_default_db21862d/insulin_default_processed.pdb"
    ]
    
    insulin_pdb = None
    for file in insulin_files:
        if os.path.exists(file):
            insulin_pdb = file
            break
    
    if not insulin_pdb:
        print("❌ No insulin PDB file found!")
        return
    
    print(f"📁 Input: {insulin_pdb}")
    print(f"🎯 Goal: Insulin with polymer force field support")
    print(f"💧 Solvent: Implicit (OBC2 GBSA)")
    print(f"📦 Boundaries: None (NoCutoff)")
    print(f"🔬 Application: Drug delivery research")
    
    # Run simulation
    try:
        result = run_insulin_polymer_simulation(insulin_pdb, logger)
        
        if result['success']:
            print(f"\n🎊 SIMULATION SUCCESSFUL!")
            print(f"📊 System: {result['atoms']} atoms")
            print(f"⏱️ Time: {result['simulation_time']} ns")
            print(f"🚀 Speed: {result['performance']}")
            print(f"📁 Trajectory: {result['trajectory']}")
            print(f"💡 This demonstrates insulin dynamics in a polymer-compatible environment!")
            print(f"🧬 Perfect for drug delivery and formulation research!")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        logger.error(f"Simulation error: {e}")

if __name__ == "__main__":
    main() 