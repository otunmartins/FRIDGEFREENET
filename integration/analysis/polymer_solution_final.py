#!/usr/bin/env python3
"""
Final Polymer Simulation Solution

Based on the working approach provided - using SMILES-based molecule creation
with SMIRNOFFTemplateGenerator for proper UNL residue parameterization.
"""

from pathlib import Path
from typing import Optional, Tuple

try:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator, GAFFTemplateGenerator
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    from sys import stdout
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

def setup_polymer_simulation(pdb_file: str, 
                           polymer_smiles: Optional[str] = None,
                           use_gaff: bool = False,
                           temperature: float = 300.0,
                           steps: int = 10000):
    """
    Setup polymer simulation using template generators
    
    Args:
        pdb_file: Path to PDB file with UNL residues
        polymer_smiles: SMILES string for the polymer (if None, tries to auto-detect)
        use_gaff: Use GAFF instead of SMIRNOFF for small molecules
        temperature: Simulation temperature in Kelvin
        steps: Number of simulation steps
        
    Returns:
        OpenMM Simulation object
    """
    
    print(f"🚀 Setting up polymer simulation")
    print(f"📁 PDB file: {pdb_file}")
    print(f"🧪 Polymer SMILES: {polymer_smiles}")
    
    # Load PDB
    pdb = PDBFile(pdb_file)
    print(f"✅ Loaded PDB: {len(list(pdb.topology.atoms()))} atoms")
    
    # Check for UNL residues
    unl_residues = [res for res in pdb.topology.residues() if res.name == 'UNL']
    print(f"📊 Found {len(unl_residues)} UNL residues")
    
    # Create force field
    if polymer_smiles and unl_residues:
        print(f"🧬 Creating molecule from SMILES for UNL parameterization...")
        
        # Create molecule from SMILES
        molecule = Molecule.from_smiles(polymer_smiles, allow_undefined_stereo=True)
        print(f"✅ Created molecule: {molecule.n_atoms} atoms, {molecule.n_bonds} bonds")
        
        # Choose template generator
        if use_gaff:
            template_generator = GAFFTemplateGenerator(molecules=molecule, forcefield='gaff-2.11')
            print(f"🔧 Using GAFF template generator")
        else:
            template_generator = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield='openff-2.1.0')
            print(f"🔧 Using SMIRNOFF template generator")
        
        # Create force field and register template generator
        forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
        forcefield.registerTemplateGenerator(template_generator.generator)
        print(f"✅ Registered template generator with force field")
        
    else:
        print(f"🔧 Using standard force field (no UNL residues or no SMILES provided)")
        forcefield = ForceField('amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml')
    
    # Create system
    print(f"⚡ Creating OpenMM system...")
    system = forcefield.createSystem(pdb.topology, 
                                   nonbondedMethod=PME,
                                   nonbondedCutoff=1*nanometer,
                                   constraints=HBonds)
    
    print(f"✅ System created: {system.getNumParticles()} particles, {system.getNumForces()} forces")
    
    # Setup integrator
    integrator = LangevinMiddleIntegrator(temperature*kelvin, 1/picosecond, 0.004*picoseconds)
    print(f"🔧 Created Langevin integrator at {temperature} K")
    
    # Create simulation
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    print(f"✅ Simulation created and positions set")
    
    # Energy minimization
    print(f"🔧 Minimizing energy...")
    simulation.minimizeEnergy()
    
    # Get initial state
    state = simulation.context.getState(getEnergy=True)
    initial_energy = state.getPotentialEnergy()
    print(f"🔋 Initial energy after minimization: {initial_energy}")
    
    return simulation

def run_polymer_simulation_with_smiles(pdb_file: str,
                                     polymer_smiles: str,
                                     output_prefix: str = "polymer_sim",
                                     temperature: float = 300.0,
                                     equilibration_steps: int = 5000,
                                     production_steps: int = 10000,
                                     save_interval: int = 100):
    """
    Complete polymer simulation workflow with SMILES
    
    Args:
        pdb_file: Input PDB file
        polymer_smiles: SMILES string for the polymer
        output_prefix: Prefix for output files
        temperature: Temperature in Kelvin
        equilibration_steps: Equilibration steps
        production_steps: Production MD steps
        save_interval: Save frequency
        
    Returns:
        Dict with simulation results
    """
    
    print(f"🚀 Running Complete Polymer Simulation")
    print(f"=" * 60)
    
    try:
        # Setup simulation
        simulation = setup_polymer_simulation(pdb_file, polymer_smiles, temperature=temperature)
        
        # Add reporters
        dcd_file = f"{output_prefix}.dcd"
        log_file = f"{output_prefix}.log"
        
        simulation.reporters.append(DCDReporter(dcd_file, save_interval))
        simulation.reporters.append(StateDataReporter(log_file, save_interval,
                                                     step=True, 
                                                     potentialEnergy=True, 
                                                     temperature=True,
                                                     speed=True))
        
        print(f"📊 Added reporters: {dcd_file}, {log_file}")
        
        # Equilibration
        if equilibration_steps > 0:
            print(f"🔄 Running equilibration: {equilibration_steps} steps...")
            simulation.step(equilibration_steps)
            
            state = simulation.context.getState(getEnergy=True)
            eq_energy = state.getPotentialEnergy()
            print(f"🔋 Energy after equilibration: {eq_energy}")
        
        # Production
        print(f"🏃 Running production MD: {production_steps} steps...")
        simulation.step(production_steps)
        
        # Final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy()
        
        # Save final structure
        final_pdb = f"{output_prefix}_final.pdb"
        with open(final_pdb, 'w') as f:
            PDBFile.writeFile(simulation.topology, state.getPositions(), f)
        
        print(f"✅ Simulation completed successfully!")
        print(f"🔋 Final energy: {final_energy}")
        print(f"💾 Final structure saved: {final_pdb}")
        
        return {
            'success': True,
            'final_energy': final_energy,
            'trajectory': dcd_file,
            'log_file': log_file,
            'final_structure': final_pdb
        }
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

def test_polymer_solution(pdb_file: str):
    """Test the polymer solution with a PDB file"""
    
    print(f"🧪 Testing Polymer Solution")
    print(f"=" * 50)
    
    if not DEPENDENCIES_AVAILABLE:
        print(f"❌ Dependencies not available")
        return False
    
    # Example polymer SMILES strings to try
    test_smiles = [
        # Simple polymer chain
        "CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC",
        # Polymer with nitrogen
        "CCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC",
        # More complex polymer
        "CC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)O",
        # Generic organic polymer
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    ]
    
    for i, smiles in enumerate(test_smiles):
        try:
            print(f"\n🧪 Test {i+1}: Trying SMILES: {smiles}")
            
            # Try to setup simulation
            simulation = setup_polymer_simulation(pdb_file, smiles, temperature=310.0)
            
            # Run a few test steps
            print(f"🏃 Running 10 test steps...")
            simulation.step(10)
            
            # Get final energy
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy()
            
            print(f"✅ SUCCESS with SMILES {i+1}!")
            print(f"🔋 Energy: {energy}")
            
            return True, smiles
            
        except Exception as e:
            print(f"❌ Failed with SMILES {i+1}: {e}")
            continue
    
    print(f"❌ All test SMILES failed")
    return False, None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <pdb_file> [polymer_smiles]")
        print(f"Example: python {sys.argv[0]} polymer.pdb 'CCOCCOCCOCCOC'")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    polymer_smiles = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(pdb_file).exists():
        print(f"❌ PDB file not found: {pdb_file}")
        sys.exit(1)
    
    if polymer_smiles:
        # Run with provided SMILES
        print(f"🚀 Running with provided SMILES: {polymer_smiles}")
        results = run_polymer_simulation_with_smiles(
            pdb_file, 
            polymer_smiles,
            output_prefix="polymer_test",
            equilibration_steps=1000,
            production_steps=5000
        )
        
        if results['success']:
            print(f"\n🎉 SUCCESS!")
            print(f"Final energy: {results['final_energy']}")
            print(f"Trajectory: {results['trajectory']}")
        else:
            print(f"\n❌ FAILED: {results['error']}")
    
    else:
        # Test with different SMILES
        print(f"🧪 Testing with different SMILES strings...")
        success, working_smiles = test_polymer_solution(pdb_file)
        
        if success:
            print(f"\n🎉 SUCCESS! Working SMILES: {working_smiles}")
            print(f"\n📋 Integration example:")
            print(f"""
# Use this in your code:
from polymer_solution_final import run_polymer_simulation_with_smiles

results = run_polymer_simulation_with_smiles(
    '{pdb_file}',
    '{working_smiles}',
    output_prefix='my_polymer_sim',
    equilibration_steps=5000,
    production_steps=10000
)

if results['success']:
    print(f"Simulation completed! Energy: {{results['final_energy']}}")
""")
        else:
            print(f"\n❌ Could not find working SMILES for this polymer")
            print(f"💡 Try providing a custom SMILES string that matches your polymer structure") 