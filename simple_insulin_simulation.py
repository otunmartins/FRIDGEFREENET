#!/usr/bin/env python3
"""
Simple insulin simulation using original CYX residues with AMBER force field.

The key insight: CYX residues are CORRECT for disulfide-bonded cysteines.
AMBER force fields already support CYX. No need for complex template generators.
"""

from openmm.app import *
from openmm import *
from openmm.unit import *

def run_simple_insulin_simulation(pdb_file: str):
    """
    Run insulin simulation with original CYX residues using simple AMBER setup.
    
    Args:
        pdb_file: Path to insulin PDB file (with CYX residues)
    """
    print(f"🧬 Loading insulin structure: {pdb_file}")
    
    # Load PDB and clean it
    pdb = PDBFile(pdb_file)
    
    # Remove water and add missing atoms using Modeller
    print("🧹 Cleaning structure (removing water, adding hydrogens)...")
    from openmm.app import Modeller
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()  # Remove crystal water
    modeller.addHydrogens() # Add missing hydrogens
    
    print("📋 Cleaned residues in structure:")
    for i, residue in enumerate(modeller.topology.residues()):
        print(f"  {i+1:2d}. {residue.name} {residue.chain.id}:{residue.id}")
    
    # Simple AMBER force field setup
    print("\n⚙️ Setting up AMBER force field...")
    forcefield = ForceField(
        'amber/protein.ff14SB.xml',    # AMBER protein force field
        'implicit/gbn2.xml'            # Implicit solvent
    )
    
    # Create system with implicit solvent (GB)
    print("🔧 Creating system with implicit solvent...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,      # No cutoff for implicit solvent
        constraints=HBonds             # Constrain bonds to hydrogen
    )
    
    print("✅ System created successfully!")
    print(f"   System has {system.getNumParticles()} particles")
    print(f"   System has {system.getNumForces()} forces")
    
    # Set up integrator
    print("\n🕰️ Setting up Langevin integrator...")
    integrator = LangevinMiddleIntegrator(
        300*kelvin,        # Temperature
        1/picosecond,      # Friction coefficient  
        0.002*picoseconds  # Time step (2 fs)
    )
    
    # Create simulation with CUDA platform
    print("🎬 Creating simulation...")
    platform = Platform.getPlatformByName('OpenCL')#CUDA')
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    # Minimize energy
    print("⚡ Minimizing energy...")
    simulation.minimizeEnergy()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"   Initial energy: {energy}")
    
    # Set up reporters
    simulation.reporters.append(StateDataReporter(
        'insulin_simulation.log', 1000,
        step=True, time=True, potentialEnergy=True, temperature=True
    ))
    
    # Run short simulation
    print("🏃 Running simulation (100 ps)...")
    # Log the chosen platform
    platform_name = simulation.context.getPlatform().getName()
    print(f"💻 Using platform: {platform_name}")
    simulation.step(5000000)  # 100 ps
    
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"✅ Simulation complete! Final energy: {final_energy}")
    
    return simulation

if __name__ == "__main__":
    # Use the proper insulin file with SSBOND records
    #pdb_file = "src/insulin_ai/integration/data/insulin/human_insulin_1mso.pdb"
    pdb_file = "src/insulin_ai/integration/data/insulin/insulin_default.pdb"
    
    try:
        simulation = run_simple_insulin_simulation(pdb_file)
        print("\n🎉 SUCCESS: Simple insulin simulation completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nDEBUG: Let's check what residues are in the file...")
        
        # Debug: simplified error message
        print("   💡 This is likely due to missing atoms or improper connectivity.")
        print("   The Modeller should have fixed this by adding missing atoms.") 
