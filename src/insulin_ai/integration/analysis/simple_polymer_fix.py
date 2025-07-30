#!/usr/bin/env python3
"""
Simple Polymer Simulation Fix

This provides a more direct fix that avoids the OpenFF topology conversion issues
by using a different approach to molecule creation and template generation.
"""

import tempfile
from pathlib import Path
from typing import Tuple, Optional

try:
    import openmm as mm
    from openmm import app, unit
    from openmm.app import PDBFile, ForceField
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from openmmforcefields.generators import SystemGenerator, GAFFTemplateGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False

def simple_polymer_fix(pdb_file: str, 
                      temperature: float = 310.0,
                      force_field: str = "gaff-2.11") -> Tuple[mm.System, app.Topology, list]:
    """
    Simple fix for polymer simulation using direct SystemGenerator approach
    
    This avoids the complex molecule-topology matching by using SystemGenerator
    in a way that's more tolerant of connectivity issues.
    
    Args:
        pdb_file: Path to PDB file with UNL residues
        temperature: Simulation temperature
        force_field: Small molecule force field
        
    Returns:
        Tuple of (system, topology, positions)
    """
    
    print(f"🚀 Simple Polymer Fix")
    print(f"📁 PDB file: {pdb_file}")
    
    # Step 1: Load PDB
    pdb = PDBFile(pdb_file)
    print(f"✅ Loaded PDB: {len(list(pdb.topology.atoms()))} atoms")
    
    # Check for UNL residues
    unl_residues = [res for res in pdb.topology.residues() if res.name == 'UNL']
    print(f"📊 Found {len(unl_residues)} UNL residues")
    
    if not unl_residues:
        print(f"⚠️  No UNL residues - using standard force field")
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        system = forcefield.createSystem(pdb.topology)
        return system, pdb.topology, pdb.positions
    
    # Step 2: Use SystemGenerator without pre-registered molecules
    # This approach lets SystemGenerator handle the molecule creation internally
    print(f"⚡ Creating SystemGenerator...")
    
    try:
        # Method 1: Let SystemGenerator auto-detect and parameterize
        system_generator = SystemGenerator(
            forcefields=['amber/protein.ff14SB.xml', 'implicit/gbn2.xml'],
            small_molecule_forcefield=force_field,
            # Don't pre-register molecules - let it figure them out
            cache='simple_polymer_cache.json'
        )
        
        print(f"🧪 Creating system (auto-detection method)...")
        system = system_generator.create_system(pdb.topology)
        
        print(f"✅ SUCCESS! System created with {system.getNumParticles()} particles")
        return system, pdb.topology, pdb.positions
        
    except Exception as e:
        print(f"⚠️  Auto-detection failed: {e}")
        print(f"🔄 Trying manual template approach...")
        
        # Method 2: Manual template approach with more permissive settings
        return manual_template_approach(pdb, force_field)

def manual_template_approach(pdb, force_field: str):
    """
    Manual template approach that's more permissive about molecule matching
    """
    
    print(f"🔧 Manual template approach...")
    
    try:
        # Create base force field
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Create GAFF template generator with empty molecules list
        # This will prompt it to create molecules from topology when needed
        template_generator = GAFFTemplateGenerator(
            molecules=[],  # Start empty
            forcefield=force_field,
            cache='manual_polymer_cache.json'
        )
        
        # Register template generator
        forcefield.registerTemplateGenerator(template_generator.generator)
        print(f"✅ Registered template generator")
        
        # Try to create system - this should trigger molecule creation
        try:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0*unit.nanometer,
                constraints=app.HBonds,
                removeCMMotion=True
            )
            
            print(f"✅ Manual template approach succeeded!")
            return system, pdb.topology, pdb.positions
            
        except Exception as create_error:
            print(f"❌ System creation failed: {create_error}")
            
            # Method 3: Try with renamed residues
            print(f"🔄 Trying residue renaming approach...")
            return residue_renaming_approach(pdb, force_field)
    
    except Exception as e:
        print(f"❌ Manual template approach failed: {e}")
        raise

def residue_renaming_approach(pdb, force_field: str):
    """
    Try renaming UNL residues to something more generic
    """
    
    print(f"🏷️  Residue renaming approach...")
    
    try:
        from openmm.app import Modeller
        
        # Create modeller to modify topology
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Rename UNL residues to LIG (ligand)
        residue_count = 0
        for residue in modeller.topology.residues():
            if residue.name == 'UNL':
                residue.name = 'LIG'
                residue_count += 1
        
        print(f"🏷️  Renamed {residue_count} UNL residues to LIG")
        
        # Now try with SystemGenerator
        system_generator = SystemGenerator(
            forcefields=['amber/protein.ff14SB.xml', 'implicit/gbn2.xml'],
            small_molecule_forcefield=force_field,
            cache='renamed_polymer_cache.json'
        )
        
        system = system_generator.create_system(modeller.topology)
        
        print(f"✅ Residue renaming approach succeeded!")
        return system, modeller.topology, modeller.positions
        
    except Exception as e:
        print(f"❌ Residue renaming failed: {e}")
        
        # Final fallback: Try with generic force field
        print(f"🔄 Final fallback: generic force field...")
        return generic_force_field_fallback(pdb)

def generic_force_field_fallback(pdb):
    """
    Final fallback using a more generic approach
    """
    
    print(f"🛡️  Generic force field fallback...")
    
    try:
        # Create a very basic force field that might handle unknown residues
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        
        # Try to create system with minimal constraints
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
        
        print(f"✅ Generic fallback succeeded!")
        return system, pdb.topology, pdb.positions
        
    except Exception as e:
        print(f"❌ All approaches failed!")
        raise ValueError(f"Unable to create system with any approach. Final error: {e}")

def test_simple_fix(pdb_file: str):
    """Test the simple fix with a polymer PDB file"""
    
    print(f"🧪 Testing Simple Polymer Fix")
    print(f"=" * 50)
    
    try:
        # Test system creation
        system, topology, positions = simple_polymer_fix(pdb_file)
        
        print(f"\n✅ System Creation Successful!")
        print(f"   Particles: {system.getNumParticles()}")
        print(f"   Forces: {system.getNumForces()}")
        
        # Test basic simulation
        print(f"\n🏃 Testing basic simulation...")
        
        # Create integrator
        integrator = mm.LangevinMiddleIntegrator(
            310 * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtosecond
        )
        
        # Create context (use Reference platform for safety)
        platform = mm.Platform.getPlatformByName('Reference')
        context = mm.Context(system, integrator, platform)
        context.setPositions(positions)
        
        # Energy minimization
        print(f"   🔧 Minimizing energy...")
        mm.LocalEnergyMinimizer.minimize(context)
        
        # Get state
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        
        print(f"   ✅ Energy minimization successful")
        print(f"   🔋 Energy: {energy}")
        
        # Run a few steps
        print(f"   🏃 Running 10 simulation steps...")
        integrator.step(10)
        
        state = context.getState(getEnergy=True)
        final_energy = state.getPotentialEnergy()
        
        print(f"   ✅ Simulation steps successful")
        print(f"   🔋 Final energy: {final_energy}")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Simple fix works for your polymer!")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <pdb_file>")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    
    if not Path(pdb_file).exists():
        print(f"❌ PDB file not found: {pdb_file}")
        sys.exit(1)
    
    success = test_simple_fix(pdb_file)
    
    if success:
        print(f"\n✅ SUCCESS! You can now use this approach in your simulation code.")
        print(f"\n📋 Integration Example:")
        print(f"""
# In your existing code, replace the problematic system creation with:

from simple_polymer_fix import simple_polymer_fix

try:
    system, topology, positions = simple_polymer_fix(pdb_file, temperature=310.0)
    
    # Continue with your simulation setup...
    integrator = mm.LangevinMiddleIntegrator(310*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    
    # Your existing simulation code here...
    
except Exception as e:
    print(f"Polymer simulation failed: {{e}}")
""")
    else:
        print(f"\n❌ Simple fix failed. Please check the error messages above.") 