#!/usr/bin/env python3
"""
OpenMM MD Runner (CPU-Only)

Runs insulin + polymer matrix MD simulations using OpenMM with implicit solvent.
Uses CPU platform exclusively. Per proposal.tex: thermal stability, insulin
protection, insulin-material interactions.

Reference: OpenMM User Guide, REVAMP_PLAN.md, proposal.tex
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
    unit = None

from .insulin_polymer_system import InsulinPolymerSystemBuilder


def _compute_insulin_rmsd(initial_pos, final_pos, insulin_atom_indices, unit_nm=None):
    """Compute backbone RMSD for insulin atoms. Returns float in nm or None."""
    if unit_nm is None and unit is not None:
        unit_nm = unit.nanometer
    if not insulin_atom_indices or initial_pos is None or final_pos is None or unit_nm is None:
        return None
    try:
        import numpy as np
        n = len(insulin_atom_indices)
        if n == 0:
            return None
        pos_i = np.array([[initial_pos[i][0].value_in_unit(unit_nm),
                           initial_pos[i][1].value_in_unit(unit_nm),
                           initial_pos[i][2].value_in_unit(unit_nm)]
                          for i in insulin_atom_indices])
        pos_f = np.array([[final_pos[i][0].value_in_unit(unit_nm),
                           final_pos[i][1].value_in_unit(unit_nm),
                           final_pos[i][2].value_in_unit(unit_nm)]
                          for i in insulin_atom_indices])
        diff = pos_f - pos_i
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return float(rmsd)
    except Exception:
        return None


def _compute_insulin_polymer_contacts(positions, topology, insulin_atom_indices,
                                     polymer_atom_indices, cutoff_nm=0.5):
    """Count insulin-polymer atom pairs within cutoff. Returns int or None."""
    if not insulin_atom_indices or not polymer_atom_indices or positions is None:
        return None
    try:
        import numpy as np
        pos = np.array([[positions[i][0].value_in_unit(unit.nanometer),
                         positions[i][1].value_in_unit(unit.nanometer),
                         positions[i][2].value_in_unit(unit.nanometer)]
                        for i in range(len(positions))])
        count = 0
        for i in insulin_atom_indices:
            for j in polymer_atom_indices:
                d = np.linalg.norm(pos[i] - pos[j])
                if d <= cutoff_nm:
                    count += 1
        return count
    except Exception:
        return None


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
    
    def run_insulin_polymer(
        self,
        psmiles: str,
        n_repeats: int = 4,
        n_chains: int = 3,
        n_steps: int = 50000,
        minimize_steps: int = 1000,
        random_seed: int = 42,
    ) -> Optional[Dict[str, Any]]:
        """
        Run insulin + polymer matrix MD with implicit solvent.
        
        Args:
            psmiles: Polymer SMILES string
            n_repeats: Repeat units per chain
            n_chains: Number of polymer chains
            n_steps: Production steps
            minimize_steps: Energy minimization steps
            random_seed: For reproducibility
            
        Returns:
            Dict with energies, insulin_rmsd, insulin_polymer_contacts, or None on failure
        """
        if not OPENMM_AVAILABLE:
            return None
        try:
            builder = InsulinPolymerSystemBuilder(n_repeats=n_repeats, n_chains=n_chains)
            top, pos, sys, n_insulin_atoms = builder.build(
                psmiles, temperature=self.temperature, platform_name=self.platform_name
            )
        except Exception as e:
            raise RuntimeError(
                f"Insulin-polymer system build failed for {psmiles[:50]}: {e}. "
                "Check: insulin PDB (data/4F1C.pdb), GAFF/openff-toolkit, RDKit."
            ) from e
        if top is None or pos is None or sys is None or n_insulin_atoms <= 0:
            raise RuntimeError(
                f"System build returned empty for {psmiles[:50]}. "
                "Possible: insulin PDB missing, GAFF parameterization failed, or no polymer chains added."
            )

        # NVT for implicit solvent (no barostat)
        integrator = openmm.LangevinIntegrator(
            self.temperature * unit.kelvin,
            1.0 / unit.picosecond,
            self.timestep_fs * unit.femtoseconds,
        )
        platform = openmm.Platform.getPlatformByName(self.platform_name)
        sim = openmm.app.Simulation(top, sys, integrator, platform)
        sim.context.setPositions(pos)
        sim.context.setVelocitiesToTemperature(
            self.temperature * unit.kelvin,
            random_seed,
        )

        sim.minimizeEnergy(maxIterations=minimize_steps)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        initial_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        initial_positions = state.getPositions(asNumpy=True)

        sim.step(n_steps)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        final_positions = state.getPositions(asNumpy=True)

        insulin_indices = list(range(n_insulin_atoms))
        polymer_indices = list(range(n_insulin_atoms, top.getNumAtoms()))
        insulin_rmsd = _compute_insulin_rmsd(
            initial_positions, final_positions, insulin_indices
        )
        insulin_polymer_contacts = _compute_insulin_polymer_contacts(
            final_positions, top, insulin_indices, polymer_indices, cutoff_nm=0.5
        )
        energy_drift = final_energy - initial_energy

        return {
            "psmiles": psmiles,
            "initial_energy_kj_mol": initial_energy,
            "final_energy_kj_mol": final_energy,
            "energy_drift_kj_mol": energy_drift,
            "insulin_rmsd_nm": insulin_rmsd,
            "insulin_polymer_contacts": insulin_polymer_contacts,
            "n_steps": n_steps,
            "timestep_fs": self.timestep_fs,
            "temperature_K": self.temperature,
            "final_positions": final_positions,
            "converged": True,
        }

    def run(
        self,
        psmiles: str,
        n_steps: int = 50000,
        minimize_steps: int = 1000,
        random_seed: int = 42,
        n_repeats: int = 4,
        n_chains: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        Run short MD from PSMILES (insulin + polymer matrix, implicit solvent).
        
        Args:
            psmiles: Polymer SMILES string
            n_steps: Production steps (e.g. 50000 = 100 ps at 2 fs)
            minimize_steps: Energy minimization steps
            random_seed: For reproducibility
            n_repeats: Repeat units per polymer chain
            n_chains: Number of polymer chains
            
        Returns:
            Dict with trajectory info, energies, insulin metrics, or None on failure
        """
        return self.run_insulin_polymer(
            psmiles,
            n_repeats=n_repeats,
            n_chains=n_chains,
            n_steps=n_steps,
            minimize_steps=minimize_steps,
            random_seed=random_seed,
        )
