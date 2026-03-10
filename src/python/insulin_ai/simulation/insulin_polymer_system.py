#!/usr/bin/env python3
"""
Insulin + Polymer Matrix System Builder

Builds MD-ready systems with insulin protein and polymer oligomers in a minimal
periodic box. Uses implicit solvent (GBSA-OBC). Per proposal.tex Section 6:
thermal stability, insulin protection, insulin-material interactions.

Reference: proposal.tex, REVAMP_PLAN.md
"""

import os
import tempfile
import urllib.request
import warnings
from typing import Optional, Tuple, List

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

# Human insulin PDB (4F1C) - fallback 4INS (pig insulin)
INSULIN_PDB_URL = "https://files.rcsb.org/download/4F1C.pdb"
INSULIN_PDB_ALT = "https://files.rcsb.org/download/4INS.pdb"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INSULIN_PDB_PATH = os.path.join(DATA_DIR, "4F1C.pdb")


def _ensure_insulin_pdb() -> str:
    """Ensure insulin PDB exists; fetch from RCSB if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.isfile(INSULIN_PDB_PATH):
        return INSULIN_PDB_PATH
    for url in (INSULIN_PDB_URL, INSULIN_PDB_ALT):
        try:
            urllib.request.urlretrieve(url, INSULIN_PDB_PATH)
            if os.path.isfile(INSULIN_PDB_PATH) and os.path.getsize(INSULIN_PDB_PATH) > 1000:
                return INSULIN_PDB_PATH
        except Exception as e:
            warnings.warn(f"Could not fetch insulin PDB from {url}: {e}")
    raise FileNotFoundError(
        f"Insulin PDB not found at {INSULIN_PDB_PATH}. "
        f"Download manually from RCSB (4F1C or 4INS) and place in {DATA_DIR}/"
    )


def _build_polymer_oligomer(psmiles: str, n_repeats: int) -> Optional[str]:
    """Build polymer chain of n_repeats from PSMILES. Returns capped SMILES or None."""
    if "[*]" not in psmiles or n_repeats < 1:
        return None
    if n_repeats == 1:
        return psmiles.replace("[*]", "[H]")
    try:
        from psmiles import PolymerSmiles
        chain = psmiles
        for _ in range(n_repeats - 1):
            ps = PolymerSmiles(chain)
            chain = str(ps.dimerize(star_index=0))
        return chain.replace("[*]", "[H]")
    except Exception:
        return psmiles.replace("[*]", "[H]")


def _mol_to_pdb_block(mol: "Chem.Mol") -> str:
    """Convert RDKit mol with 3D coords to PDB block."""
    return Chem.MolToPDBBlock(mol)


def _psmiles_chain_to_mol(psmiles: str, n_repeats: int) -> Optional["Chem.Mol"]:
    """Build polymer oligomer and return RDKit Mol with 3D coords. Uses shared embed_mol_3d."""
    from .psmiles_to_openmm import embed_mol_3d

    capped = _build_polymer_oligomer(psmiles, n_repeats)
    if not capped:
        return None
    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if not embed_mol_3d(mol, random_seed=42):
        return None
    return mol


class InsulinPolymerSystemBuilder:
    """
    Builds insulin + polymer matrix systems for MD simulation.

    Uses implicit solvent (GBSA-OBC), AMBER ff14SB for protein and GAFF for polymer.
    When Packmol is installed (conda-forge), packs insulin + polymers with no overlaps;
    otherwise falls back to naive x-axis placement.
    """

    def __init__(
        self,
        n_repeats: int = 4,
        n_chains: int = 3,
        box_padding_nm: float = 1.0,
    ):
        """
        Args:
            n_repeats: Repeat units per polymer chain
            n_chains: Number of polymer chains
            box_padding_nm: Padding around solute for minimal box
        """
        self.n_repeats = n_repeats
        self.n_chains = n_chains
        self.box_padding_nm = box_padding_nm
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for insulin-polymer system")
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM required for insulin-polymer system")

    def build(
        self,
        psmiles: str,
        temperature: float = 298.0,
        platform_name: str = "CPU",
    ) -> Tuple[Optional["openmm.app.Topology"], Optional[list], Optional["openmm.System"], int]:
        """
        Build insulin + polymer system with implicit solvent.
        
        Returns:
            (topology, positions, system, n_insulin_atoms) or (None, None, None, 0) on failure
        """
        # 1. Load insulin
        pdb_path = _ensure_insulin_pdb()
        insulin_pdb = openmm.app.PDBFile(pdb_path)
        insulin_top = insulin_pdb.topology
        insulin_pos = insulin_pdb.positions
        n_insulin_atoms = insulin_top.getNumAtoms()

        # 2. Add hydrogens to insulin (AMBER ff14SB requires them)
        ff_protein = openmm.app.ForceField("amber14-all.xml")
        modeller = openmm.app.Modeller(insulin_top, insulin_pos)
        try:
            modeller.addHydrogens(forcefield=ff_protein)
        except Exception as e:
            warnings.warn(f"Could not add hydrogens to insulin: {e}")
            modeller = openmm.app.Modeller(insulin_top, insulin_pos)

        n_insulin_atoms = modeller.topology.getNumAtoms()
        oligomer_smiles = _build_polymer_oligomer(psmiles, self.n_repeats)

        # 3. Pack with Packmol (realistic) or fallback to naive placement
        use_packmol = False
        polymer_mol = _psmiles_chain_to_mol(psmiles, self.n_repeats)
        if polymer_mol is not None:
            try:
                from .packmol_packer import pack_insulin_polymers, _packmol_available
                if _packmol_available():
                    with tempfile.TemporaryDirectory() as tmpdir:
                        insulin_tmp = os.path.join(tmpdir, "insulin_h.pdb")
                        polymer_tmp = os.path.join(tmpdir, "polymer.pdb")
                        packed_tmp = os.path.join(tmpdir, "packed.pdb")
                        with open(insulin_tmp, "w") as f:
                            openmm.app.PDBFile.writeFile(
                                modeller.topology, modeller.positions, f
                            )
                        polymer_block = _mol_to_pdb_block(polymer_mol)
                        from io import StringIO
                        poly_pdb = openmm.app.PDBFile(StringIO(polymer_block))
                        with open(polymer_tmp, "w") as f:
                            openmm.app.PDBFile.writeFile(
                                poly_pdb.topology, poly_pdb.positions, f
                            )
                        box_nm = max(6.0, 4.0 + 2 * self.box_padding_nm)
                        if pack_insulin_polymers(
                            insulin_tmp, polymer_tmp, self.n_chains,
                            packed_tmp, box_size_nm=box_nm,
                        ):
                            packed_pdb = openmm.app.PDBFile(packed_tmp)
                            combined_top = packed_pdb.topology
                            combined_pos = packed_pdb.positions
                            use_packmol = True
            except Exception as e:
                warnings.warn(f"Packmol packing failed, using naive placement: {e}")

        if not use_packmol:
            offset_nm = 2.0
            for i in range(self.n_chains):
                mol = _psmiles_chain_to_mol(psmiles, self.n_repeats)
                if mol is None:
                    continue
                pdb_block = _mol_to_pdb_block(mol)
                from io import StringIO
                try:
                    poly_pdb = openmm.app.PDBFile(StringIO(pdb_block))
                except Exception:
                    continue
                pos = list(poly_pdb.positions)
                shift_nm = (i + 1) * offset_nm
                shifted = []
                for p in pos:
                    x = p.x.value_in_unit(unit.nanometer) if hasattr(p.x, 'value_in_unit') else float(p.x)
                    y = p.y.value_in_unit(unit.nanometer) if hasattr(p.y, 'value_in_unit') else float(p.y)
                    z = p.z.value_in_unit(unit.nanometer) if hasattr(p.z, 'value_in_unit') else float(p.z)
                    shifted.append(openmm.Vec3(x + shift_nm, y, z) * unit.nanometer)
                modeller.add(poly_pdb.topology, shifted)
            combined_top = modeller.topology
            combined_pos = modeller.positions

        # 4. Force field: AMBER protein + GAFF for polymer (implicit solvent, no TIP3P)
        from .psmiles_to_openmm import register_gaff_for_smiles

        ff = openmm.app.ForceField("amber14-all.xml")
        register_gaff_for_smiles(ff, oligomer_smiles or psmiles.replace("[*]", "[H]"))

        # 5. Create system with implicit solvent (GBSA-OBC)
        try:
            system = ff.createSystem(
                combined_top,
                nonbondedMethod=openmm.app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=openmm.app.HBonds,
                implicitSolvent=openmm.app.OBC2,
                implicitSolventKappa=0.0,
            )
        except TypeError:
            # Older OpenMM may not have implicitSolvent in createSystem
            try:
                system = ff.createSystem(
                    combined_top,
                    nonbondedMethod=openmm.app.NoCutoff,
                    constraints=openmm.app.HBonds,
                )
            except Exception as e:
                warnings.warn(f"Force field creation failed: {e}")
                return None, None, None, 0
        except Exception as e:
            warnings.warn(f"Force field creation failed: {e}")
            return None, None, None, 0

        return combined_top, combined_pos, system, n_insulin_atoms
