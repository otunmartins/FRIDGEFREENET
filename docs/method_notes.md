# Method Notes: Insulin AI MD Simulation Pipeline

## Scientific Methods and Parameters

### Particle Mesh Ewald (PME)

Long-range electrostatic interactions are computed using the Particle Mesh Ewald method.

- **Cutoff**: 1.0 nm (real space)
- **Method**: `openmm.app.PME`
- **Ewald tolerance**: OpenMM default (~1e-4)

Reference: Darden et al., J. Chem. Phys. (1993).

### Force Field

- **Small molecules / polymer repeat units**: GAFF-2.0 via `openmmforcefields`
- **Backbone**: Amber14-all.xml for standard atoms
- **Parameterization**: AM1-BCC charges when OpenFF/GAFF available

### Simulation Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| Temperature | 298 (or 310 for body temp) | K |
| Pressure | 1.01325 | bar |
| Timestep | 2.0 | fs |
| Integrator | Langevin |
| Friction | 1.0 | 1/ps |
| Constraints | HBonds | - |

### PSMILES Capping

Polymer repeat units with `[*]` connection points are capped with hydrogen for single-molecule screening:

```
[*]OCC[*]  →  [H]OCCO[H]
```

This yields a neutral, finite molecule for gas-phase or implicit solvent screening.

### Units

All energies: kJ/mol  
All distances: nm  
All times: ps (unless noted)

### Reproducibility

- Random seed: 42 (configurable)
- Fixed seeds in RDKit embedding
- Deterministic across CPU platforms
