# Polymer Simulation Fix Guide

## Problem Summary

The polymer MD simulation is failing with the error:
```
Did not recognize residue UNL; did you forget to call .add_molecules() to add it?
System creation failed with template generators: No template found for residue 1 (UNL).
```

## Root Cause Analysis

The issue occurs because the OpenMM template generator cannot match the OpenFF molecule to the UNL residue in the topology:

1. **Atom Count Mismatch**: The OpenFF molecule has 207 atoms (with explicit hydrogens), but the PDB topology has 177 atoms
2. **Template Matching Failure**: The template generator uses connectivity patterns to match molecules to residues, but the atom counts don't align
3. **SMILES vs Topology**: Creating molecules from SMILES adds hydrogens differently than the PDB file represents them

## Solution Approach

Based on the OpenMM forcefields documentation, the proper approach is:

### Method 1: SystemGenerator with Topology-Based Molecules (RECOMMENDED)

```python
from openmmforcefields.generators import SystemGenerator
from openff.toolkit import Molecule, Topology as OFFTopology

# Create molecule directly from topology (matches exactly)
off_topology = OFFTopology.from_openmm(topology, unique_molecules=[])
molecule = off_topology.molecules[0]

# Use SystemGenerator with pre-registered molecules
system_generator = SystemGenerator(
    forcefields=['amber/protein.ff14SB.xml', 'implicit/gbn2.xml'],
    small_molecule_forcefield='gaff-2.11',
    molecules=[molecule],  # Pre-register molecules
    cache='polymer_cache.json'
)

system = system_generator.create_system(topology)
```

### Method 2: Proper Template Generator Registration

```python
from openmmforcefields.generators import GAFFTemplateGenerator

# Create molecule from topology (not SMILES)
molecule = create_molecule_from_topology(topology, positions)

# Create and register template generator
template_generator = GAFFTemplateGenerator(
    molecules=[molecule], 
    forcefield='gaff-2.11'
)

forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
forcefield.registerTemplateGenerator(template_generator.generator)

system = forcefield.createSystem(topology)
```

## Implementation Files

### 1. Main Fix Script: `fix_polymer_simulation.py`
- Complete solution with proper template generator workflow
- Handles topology-based molecule creation
- Includes fallback methods and error handling
- Provides diagnostic tools

### 2. Testing Script: `test_polymer_fix.py`
- Tests the fix with your specific polymer PDB file
- Provides detailed output and error handling
- Can be run in diagnostic mode or full simulation mode

## Usage

### Quick Fix Test
```bash
cd integration/analysis
python fix_polymer_simulation.py path/to/polymer.pdb --diagnose
```

### Full Simulation
```bash
python fix_polymer_simulation.py path/to/polymer.pdb \
    --temperature 300 \
    --equilibration-steps 1000 \
    --production-steps 5000 \
    --output-dir polymer_results
```

### Integration with Existing Code
```python
from fix_polymer_simulation import PolymerSimulationFixer

fixer = PolymerSimulationFixer()

# Diagnose issues
diagnosis = fixer.diagnose_polymer_issues("polymer.pdb")

# Run simulation
results = fixer.run_polymer_simulation(
    "polymer.pdb",
    temperature=310.0,
    equilibration_steps=1000,
    production_steps=5000
)
```

## Key Changes to Existing Code

### 1. In `openmm_md_proper.py`

Replace the current UNL molecule creation with:

```python
def create_molecule_from_topology_proper(self, topology, positions):
    """Create OpenFF molecule directly from topology"""
    try:
        from openff.toolkit import Topology as OFFTopology
        
        # Convert OpenMM topology to OpenFF topology
        off_topology = OFFTopology.from_openmm(topology, unique_molecules=[])
        
        if len(off_topology.molecules) == 0:
            return None
            
        # Return the first molecule
        return off_topology.molecules[0]
        
    except Exception as e:
        print(f"Failed to create molecule from topology: {e}")
        return None
```

### 2. Template Generator Registration

```python
def create_system_with_proper_generators(self, topology, positions):
    """Use SystemGenerator for proper molecule handling"""
    
    # Find UNL residues
    unl_residues = [res for res in topology.residues() if res.name == 'UNL']
    
    if not unl_residues:
        # No UNL residues - use standard force field
        forcefield = ForceField('amber/protein.ff14SB.xml', 'implicit/gbn2.xml')
        return forcefield.createSystem(topology)
    
    # Create molecules from topology
    molecule = self.create_molecule_from_topology_proper(topology, positions)
    
    if not molecule:
        raise ValueError("Failed to create molecule from UNL residues")
    
    # Use SystemGenerator
    system_generator = SystemGenerator(
        forcefields=['amber/protein.ff14SB.xml', 'implicit/gbn2.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=[molecule],
        cache='unl_cache.json'
    )
    
    return system_generator.create_system(topology)
```

## Troubleshooting

### Common Issues

1. **OpenFF Toolkit Import Error**
   ```bash
   conda install -c conda-forge openff-toolkit
   ```

2. **SystemGenerator Import Error**
   ```bash
   conda install -c conda-forge openmmforcefields
   ```

3. **Still Getting Template Errors**
   - Check that molecules are properly created from topology
   - Verify that the PDB file has proper connectivity
   - Try preprocessing with PDBFixer first

### Debug Steps

1. **Check Dependencies**
   ```python
   from fix_polymer_simulation import PolymerSimulationFixer
   fixer = PolymerSimulationFixer()  # Will check dependencies
   ```

2. **Diagnose PDB Issues**
   ```python
   diagnosis = fixer.diagnose_polymer_issues("polymer.pdb")
   print(diagnosis)
   ```

3. **Test Molecule Creation**
   ```python
   from openmm.app import PDBFile
   pdb = PDBFile("polymer.pdb")
   molecule = fixer.create_molecule_from_topology(pdb.topology, pdb.positions)
   print(f"Created molecule: {molecule.n_atoms} atoms")
   ```

## Performance Considerations

1. **Caching**: SystemGenerator supports caching of parameterized molecules
2. **Force Field Choice**: GAFF 2.11 is generally faster than SMIRNOFF for polymers
3. **System Size**: Large polymers may need explicit solvent or GPU acceleration

## Integration with Streamlit App

To integrate this fix into your Streamlit app:

```python
# In your MD simulation integration
from fix_polymer_simulation import PolymerSimulationFixer

class MDSimulationIntegration:
    def __init__(self):
        self.polymer_fixer = PolymerSimulationFixer()
    
    def run_md_simulation_with_polymer_fix(self, pdb_file, **kwargs):
        # First try the fix
        try:
            return self.polymer_fixer.run_polymer_simulation(pdb_file, **kwargs)
        except Exception as e:
            # Fall back to original method
            return self.run_md_simulation_original(pdb_file, **kwargs)
```

## References

1. [OpenMM ForceFields Documentation](https://github.com/openmm/openmmforcefields)
2. [OpenFF Toolkit Documentation](https://docs.openforcefield.org/)
3. [OpenMM Template Generators](http://docs.openmm.org/latest/userguide/application.html#adding-residue-template-generators)
4. [SystemGenerator API](https://openmmforcefields.readthedocs.io/en/latest/api.html#systemgenerator) 