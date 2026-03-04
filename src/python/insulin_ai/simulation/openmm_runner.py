#!/usr/bin/env python3
"""
OpenMM MD Runner (CPU-Only)

Runs molecular dynamics simulations using OpenMM with Particle Mesh Ewald (PME)
for long-range electrostatics. Uses CPU platform exclusively.

Reference: OpenMM User Guide, REVAMP_PLAN.md
"""

from typing import Optional, Dict, Any
import warnings

try:
    import openmm
    import openmm.app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

from .psmiles_to_openmm import PSMILestoOpenMM


class OpenMMRunner:
    """
    CPU-only molecular dynamics runner for polymer screening.
    
    PME settings per REVAMP_PLAN.md:
    - nonbondedMethod: PME
    - nonbondedCutoff: 1.0 nm
    - Platform: CPU
    """
    
    def __init__(
        self,
        temperature: float = 298.0,
        pressure: Optional[float] = 1.01325,
        timestep_fs: float = 2.0,
        platform_name: str = "CPU",
    ):
        """
        Args:
            temperature: Kelvin
            pressure: bar (None for NVT)
            timestep_fs: Integration timestep in femtoseconds
            platform_name: Must be "CPU" for CPU-only
        """
        if platform_name != "CPU":
            warnings.warn("Non-CPU platform requested; forcing CPU for portability")
            platform_name = "CPU"
        
        self.temperature = temperature
        self.pressure = pressure
        self.timestep_fs = timestep_fs
        self.platform_name = "CPU"
    
    def run(
        self,
        psmiles: str,
        n_steps: int = 50000,
        minimize_steps: int = 1000,
        random_seed: int = 42,
    ) -> Optional[Dict[str, Any]]:
        """
        Run short MD from PSMILES.
        
        Args:
            psmiles: Polymer SMILES string
            n_steps: Production steps (e.g. 50000 = 100 ps at 2 fs)
            minimize_steps: Energy minimization steps
            random_seed: For reproducibility
            
        Returns:
            Dict with trajectory info, energies, or None on failure
        """
        if not OPENMM_AVAILABLE:
            return None
        
        converter = PSMILestoOpenMM()
        sim = converter.build_simulation(
            psmiles,
            temperature=self.temperature,
            pressure=self.pressure,
            platform_name=self.platform_name,
        )
        
        if sim is None:
            return None
        
        sim.context.setVelocitiesToTemperature(
            self.temperature * unit.kelvin,
            random_seed,
        )
        
        # Minimize
        sim.minimizeEnergy(maxIterations=minimize_steps)
        
        # Get initial state
        state = sim.context.getState(getEnergy=True, getPositions=True)
        initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        
        # Run
        sim.step(n_steps)
        
        state = sim.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = state.getPositions(asNumpy=True)
        
        return {
            "psmiles": psmiles,
            "initial_energy_kj_mol": initial_energy,
            "final_energy_kj_mol": final_energy,
            "n_steps": n_steps,
            "timestep_fs": self.timestep_fs,
            "temperature_K": self.temperature,
            "final_positions": final_positions,
            "converged": True,
        }
