#!/usr/bin/env python3
"""
PSMILES to OpenMM System Conversion

Converts PSMILES (Polymer SMILES) strings to OpenMM System + Topology + Positions
for CPU-only molecular dynamics simulation. Uses RDKit for 3D embedding and
OpenFF/GAFF for force field parameterization.

Reference: REVAMP_PLAN.md, OpenMM User Guide
"""

from typing import Optional, Tuple
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import openmm
    import openmm.app
    import openmm.unit as unit
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    from openff.toolkit import Molecule
    from openmmforcefields.generators import GAFFTemplateGenerator
    OPENMMFORCEFIELDS_AVAILABLE = True
except ImportError:
    OPENMMFORCEFIELDS_AVAILABLE = False


class PSMILestoOpenMM:
    """
    Converts PSMILES to OpenMM simulation-ready system.
    
    For polymer repeat units with [*] connection points, we cap with hydrogen
    to obtain a single-molecule representation for screening.
    """
    
    def __init__(self, force_field: str = "gaff-2.0"):
        """
        Args:
            force_field: "gaff-2.0" or "openff-2.0.0" (if openff-toolkit installed)
        """
        self.force_field_name = force_field
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for PSMILES to OpenMM conversion")
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM is required for MD simulation")
    
    def _cap_psmiles(self, psmiles: str) -> str:
        """
        Replace [*] connection points with hydrogen for single-molecule simulation.
        
        A simple approach: replace [*] with [H] for terminal capping.
        For proper capping we might use [H] or [*] -> remove and add H.
        RDKit interprets [*] as wildcard; we replace with [H] for a capped unit.
        """
        if "[*]" not in psmiles:
            return psmiles
        return psmiles.replace("[*]", "[H]")
    
    def psmiles_to_mol(self, psmiles: str) -> Optional["Chem.Mol"]:
        """Convert PSMILES to RDKit Mol (capped for simulation)."""
        capped = self._cap_psmiles(psmiles)
        mol = Chem.MolFromSmiles(capped)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        return mol
    
    def mol_to_3d(self, mol: "Chem.Mol") -> bool:
        """Generate 3D coordinates via RDKit embedding."""
        try:
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            return True
        except Exception:
            return False
    
    def build_openmm_system(
        self,
        psmiles: str,
        box_size: Optional[float] = 4.0,
    ) -> Tuple[Optional["openmm.app.Topology"], Optional[list], Optional["openmm.System"]]:
        """
        Build OpenMM Topology, positions, and System from PSMILES.
        
        Args:
            psmiles: Polymer SMILES string (e.g. "[*]OCC[*]", "[*]CC[*]")
            box_size: Cubic box edge length in nm (for explicit solvent would add water)
            
        Returns:
            (topology, positions, system) or (None, None, None) on failure
        """
        mol = self.psmiles_to_mol(psmiles)
        if mol is None:
            return None, None, None
        
        if not self.mol_to_3d(mol):
            return None, None, None
        
        # Export to PDB for OpenMM (RDKit includes 3D from EmbedMolecule)
        import tempfile
        import os
        pdb_path = None
        try:
            fd, pdb_path = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            writer = Chem.PDBWriter(pdb_path)
            writer.write(mol)
            writer.close()
            pdb = openmm.app.PDBFile(pdb_path)
        except Exception as e:
            try:
                pdb_block = Chem.MolToPDBBlock(mol)
                from io import StringIO
                pdb = openmm.app.PDBFile(StringIO(pdb_block))
            except Exception:
                warnings.warn(f"PDB conversion failed: {e}")
                return None, None, None
        finally:
            if pdb_path and os.path.exists(pdb_path):
                os.unlink(pdb_path)
        
        topology = pdb.topology
        positions = pdb.positions
        
        # Force field: GAFF-2.0 for small molecules/polymer repeat units
        # No solvent - gas-phase/vacuum for screening
        ff = openmm.app.ForceField("amber14-all.xml")
        if OPENMMFORCEFIELDS_AVAILABLE:
            try:
                capped = self._cap_psmiles(psmiles)
                off_mol = Molecule.from_smiles(capped)
                gen = GAFFTemplateGenerator(molecules=off_mol)
                ff.registerTemplateGenerator(gen.generator)
            except Exception:
                pass
        
        try:
            system = ff.createSystem(
                topology,
                nonbondedMethod=openmm.app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=openmm.app.HBonds,
            )
        except Exception as e:
            warnings.warn(f"Force field creation failed: {e}")
            return None, None, None
        
        return topology, positions, system
    
    def build_simulation(
        self,
        psmiles: str,
        temperature: float = 298.0,
        pressure: Optional[float] = 1.01325,
        platform_name: str = "CPU",
    ):
        """
        Build a complete OpenMM Simulation object from PSMILES.
        
        Args:
            psmiles: Polymer SMILES
            temperature: Kelvin
            pressure: bar (None for NVT)
            platform_name: "CPU" (required for CPU-only)
            
        Returns:
            openmm.app.Simulation or None
        """
        top, pos, sys = self.build_openmm_system(psmiles)
        if top is None or pos is None or sys is None:
            return None
        
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtoseconds,
        )
        
        if pressure is not None:
            sys.addForce(openmm.MonteCarloBarostat(
                pressure * unit.bar,
                temperature * unit.kelvin,
                25,
            ))
        
        platform = openmm.Platform.getPlatformByName(platform_name)
        sim = openmm.app.Simulation(top, sys, integrator, platform)
        sim.context.setPositions(pos)
        
        return sim
