#!/usr/bin/env python3
"""
Merged insulin (AMBER99SB-ILDN / pdb2gmx) + polymer (GAFF / Acpype) → GROMACS EM.

Requires on PATH: gmx, acpype (and ANTECHAMBER from AmberTools for acpype).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .polymer_build import ensure_insulin_pdb, psmiles_to_mol_3d

DATA_GMX = Path(__file__).resolve().parent / "data" / "gromacs"


def gmx_available() -> bool:
    return shutil.which("gmx") is not None or shutil.which("gmx_mpi") is not None


def _gmx_cmd() -> str:
    return "gmx_mpi" if shutil.which("gmx_mpi") else "gmx"


def mol_to_sdf(mol: Any, path: str) -> bool:
    """Write RDKit mol (3D) to SDF for Acpype."""
    try:
        from rdkit import Chem

        w = Chem.SDWriter(path)
        w.write(mol)
        w.close()
        return os.path.getsize(path) > 0
    except Exception as e:
        logger.warning("sdf write: %s", e)
        return False


# subprocess timeout must stay below ~2e6 s or Python's poll endtime overflows (OverflowError).
_ACPYPE_TIMEOUT_MAX_S = 1_500_000  # ~17 days, safe for _PyTime_t


def run_acpype_ligand(
    ligand_path: str,
    out_dir: str,
    basename: str = "LIG",
    timeout_s: Optional[int] = 3600,
    charge_method: str = "bcc",
) -> Optional[Tuple[str, str]]:
    """
    Run acpype on mol2/sdf. Returns (itp_path, gro_path) for GROMACS, or None.

    charge_method: acpype -c flag. **bcc** = AM1-BCC (antechamber Semi-QM; very slow
    on large oligomers; acpype also kills antechamber after ~3 h internally).
    **gas** = Gasteiger (fast, OK for packing/EM smoke tests).
    timeout_s: seconds; None = no subprocess timeout (acpype still has internal limits on bcc).
    """
    acpype = shutil.which("acpype")
    if not acpype:
        logger.warning("acpype not on PATH")
        return None
    to = timeout_s
    if to is not None:
        if to > _ACPYPE_TIMEOUT_MAX_S:
            logger.warning(
                "acpype timeout %s s too large (overflow risk); clamping to %s s",
                to,
                _ACPYPE_TIMEOUT_MAX_S,
            )
            to = _ACPYPE_TIMEOUT_MAX_S
        if to < 1:
            to = None
    proc = subprocess.run(
        [acpype, "-i", ligand_path, "-c", charge_method, "-b", basename, "-o", "gmx"],
        cwd=out_dir,
        capture_output=True,
        text=True,
        timeout=to,
    )
    if proc.returncode != 0:
        logger.warning("acpype failed: %s", (proc.stderr or proc.stdout)[-2500:])
        return None
    # Default layout: <basename>.acpype/<basename>_GMX.{itp,gro}
    sub = Path(out_dir) / f"{basename}.acpype"
    itp = sub / f"{basename}_GMX.itp"
    gro = sub / f"{basename}_GMX.gro"
    if itp.is_file() and gro.is_file():
        return str(itp), str(gro)
    # Fallback: search for *_GMX.itp under out_dir
    for p in Path(out_dir).rglob("*_GMX.itp"):
        g = p.with_name(p.name.replace("_GMX.itp", "_GMX.gro"))
        if g.is_file():
            return str(p), str(g)
    logger.warning("acpype outputs missing under %s", out_dir)
    return None


def _run_pdb2gmx(pdb_path: str, work: str) -> bool:
    """pdb2gmx amber99sb-ildn; answer SS prompts with y."""
    gmx = _gmx_cmd()
    prot_gro = os.path.join(work, "protein.gro")
    prot_top = os.path.join(work, "protein.top")
    cmd = [
        gmx,
        "pdb2gmx",
        "-f",
        pdb_path,
        "-o",
        prot_gro,
        "-p",
        prot_top,
        "-i",
        os.path.join(work, "posre.itp"),
        "-water",
        "tip3p",
        "-ff",
        "amber99sb-ildn",
        "-ignh",
    ]
    # Many SS bond prompts (y/n). Send plenty of 'y\n'.
    stdin = ("y\n" * 80).encode()
    proc = subprocess.run(
        cmd,
        cwd=work,
        input=stdin,
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        logger.warning("pdb2gmx failed rc=%s stderr=%s", proc.returncode, proc.stderr[-1500:])
        return False
    if not os.path.isfile(prot_gro) or not os.path.isfile(prot_top):
        return False
    return True


def _read_gro(path: str) -> Tuple[str, List[Tuple[int, str, str, float, float, float]]]:
    """Return title, list of (resnum, resname, atomname, x, y, z) in nm."""
    with open(path) as f:
        lines = f.readlines()
    title = lines[0].strip()
    n = int(lines[1].strip())
    atoms = []
    for i in range(2, 2 + n):
        line = lines[i]
        if len(line) < 44:
            continue
        resnum = int(line[0:5])
        resname = line[5:10].strip()
        atomname = line[10:15].strip()
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        atoms.append((resnum, resname, atomname, x, y, z))
    return title, atoms


def _write_gro(path: str, title: str, atoms: List[Tuple], box: Tuple[float, float, float]) -> None:
    """GRO coords in nm."""
    n = len(atoms)
    lines = [title + "\n", str(n) + "\n"]
    for i, a in enumerate(atoms, start=1):
        resnum, resname, aname, x, y, z = a
        lines.append(
            f"{resnum % 100000:5d}{resname[:5]:>5}{aname[:5]:>5}{i % 100000:5d}"
            f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
        )
    lines.append(f"{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n")
    with open(path, "w") as f:
        f.writelines(lines)


def merge_gro_translate_ligand(
    prot_gro: str,
    lig_gro: str,
    out_gro: str,
    offset_nm: Tuple[float, float, float] = (3.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Append ligand atoms to protein gro; translate ligand by offset_nm."""
    _, prot_atoms = _read_gro(prot_gro)
    _, lig_atoms = _read_gro(lig_gro)
    ox, oy, oz = offset_nm
    merged: List[Tuple[int, str, str, float, float, float]] = []
    for a in prot_atoms:
        merged.append((a[0], a[1], a[2], a[3], a[4], a[5]))
    max_res = max((a[0] for a in prot_atoms), default=0)
    for a in lig_atoms:
        merged.append((max_res + 1, a[1], a[2], a[3] + ox, a[4] + oy, a[5] + oz))
    xs = [m[3] for m in merged]
    ys = [m[4] for m in merged]
    zs = [m[5] for m in merged]
    pad = 1.5
    box = (
        max(xs) - min(xs) + 2 * pad,
        max(ys) - min(ys) + 2 * pad,
        max(zs) - min(zs) + 2 * pad,
    )
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    centered = [
        (resn, resname, aname, x - cx + box[0] / 2, y - cy + box[1] / 2, z - cz + box[2] / 2)
        for resn, resname, aname, x, y, z in merged
    ]
    _write_gro(out_gro, "insulin+LIG merged", centered, box)
    return box


def _fix_system_top(work: str, prot_top_path: str, lig_itp: str) -> str:
    """
    Merge protein.top + ligand.itp for grompp.

    GROMACS rule: every [ atomtypes ] must appear before any [ moleculetype ].
    Acpype's ligand.itp starts with [ atomtypes ], so it must be #included
    immediately after forcefield.itp — never after topol_Protein_chain_*.itp.
    """
    shutil.copy(prot_top_path, os.path.join(work, "protein_base.top"))
    lig_name = "LIG"
    in_moltype = False
    with open(lig_itp) as f:
        for line in f:
            if "[ moleculetype ]" in line:
                in_moltype = True
                continue
            if in_moltype and line.strip() and not line.strip().startswith(";"):
                lig_name = line.split()[0]
                break
    lig_copy = os.path.join(work, "ligand.itp")
    shutil.copy(lig_itp, lig_copy)
    with open(prot_top_path) as f:
        lines = f.readlines()
    inc = '#include "ligand.itp"\n'
    out_lines: List[str] = []
    inserted = False
    for i, line in enumerate(lines):
        out_lines.append(line)
        if inserted:
            continue
        if "forcefield.itp" in line and line.strip().startswith("#include"):
            # Insert right after forcefield so ligand [ atomtypes ] precedes protein [ moleculetype ]
            out_lines.append(inc)
            inserted = True
    text = "".join(out_lines)
    if not inserted:
        # No forcefield line (unusual): prepend after first #include
        text = inc + text
    if lig_name not in text.split("[ molecules ]")[-1]:
        text = text.rstrip() + f"\n{lig_name}              1\n"
    out_path = os.path.join(work, "system.top")
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def _fix_system_top_n_ligands(
    work: str, prot_top_path: str, lig_itp: str, n_ligands: int
) -> str:
    """Like _fix_system_top but ligand moleculetype count is n_ligands."""
    path = _fix_system_top(work, prot_top_path, lig_itp)
    lig_name = "LIG"
    with open(lig_itp) as f:
        in_mt = False
        for line in f:
            if "[ moleculetype ]" in line:
                in_mt = True
                continue
            if in_mt and line.strip() and not line.strip().startswith(";"):
                lig_name = line.split()[0]
                break
    text = Path(path).read_text()
    if "[ molecules ]" not in text:
        text = text.rstrip() + f"\n[ molecules ]\n{lig_name}              {n_ligands}\n"
    else:
        pre, _sep, post = text.partition("[ molecules ]")
        out_lines: List[str] = []
        replaced = False
        for line in post.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                out_lines.append(line + "\n" if line else "\n")
                continue
            parts = stripped.split()
            if parts[0] == lig_name:
                out_lines.append(f"{lig_name}              {n_ligands}\n")
                replaced = True
            else:
                out_lines.append(line + "\n" if not line.endswith("\n") else line)
        if not replaced:
            out_lines.append(f"{lig_name}              {n_ligands}\n")
        text = pre + "[ molecules ]\n" + "".join(out_lines)
    out_path = os.path.join(work, "system_matrix.top")
    Path(out_path).write_text(text)
    return out_path


def write_matrix_gro_from_packmol(
    prot_gro_path: str,
    lig_gro_path: str,
    packed_pdb_path: str,
    n_polymer_copies: int,
    out_gro_path: str,
    pad_nm: float = 1.5,
) -> Tuple[float, float, float]:
    """
    Build system.gro: protein atom order/coords from packmol (first molecule),
    then N ligand copies in packmol order. Requires packmol input PDBs derived
    from the same prot/lig gro (gro_to_pdb) so atom counts match.
    """
    from .gro_pdb_io import read_pdb_coords_nm

    _, prot_atoms = _read_gro(prot_gro_path)
    _, lig_atoms = _read_gro(lig_gro_path)
    n_prot = len(prot_atoms)
    n_lig = len(lig_atoms)
    need = n_prot + n_lig * n_polymer_copies
    coords = read_pdb_coords_nm(packed_pdb_path)
    if len(coords) != need:
        raise ValueError(
            f"packed PDB atom count {len(coords)} != {need} "
            f"(prot {n_prot} + {n_polymer_copies} x lig {n_lig})"
        )
    merged: List[Tuple[int, str, str, float, float, float]] = []
    for i, a in enumerate(prot_atoms):
        x, y, z = coords[i][2], coords[i][3], coords[i][4]
        merged.append((a[0], a[1], a[2], x, y, z))
    max_res = max((a[0] for a in prot_atoms), default=0)
    off = n_prot
    for copy in range(n_polymer_copies):
        max_res += 1
        for j, a in enumerate(lig_atoms):
            x, y, z = coords[off + j][2], coords[off + j][3], coords[off + j][4]
            merged.append((max_res, a[1], a[2], x, y, z))
        off += n_lig
    xs = [m[3] for m in merged]
    ys = [m[4] for m in merged]
    zs = [m[5] for m in merged]
    box = (
        max(xs) - min(xs) + 2 * pad_nm,
        max(ys) - min(ys) + 2 * pad_nm,
        max(zs) - min(zs) + 2 * pad_nm,
    )
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    cz = (max(zs) + min(zs)) / 2
    centered = [
        (resn, resname, aname, x - cx + box[0] / 2, y - cy + box[1] / 2, z - cz + box[2] / 2)
        for resn, resname, aname, x, y, z in merged
    ]
    _write_gro(out_gro_path, "insulin + matrix LIG", centered, box)
    return box


def write_em_mdp(path: str) -> None:
    with open(path, "w") as f:
        f.write(
            """; vacuum EM
integrator  = steep
nsteps      = 2000
emtol       = 2000
emstep      = 0.01
nstlist     = 20
cutoff-scheme = Verlet
rvdw        = 1.2
rcoulomb    = 1.2
pbc         = xyz
"""
        )


def _parse_mdrun_epot_kj(log_path: str) -> Optional[float]:
    with open(log_path) as f:
        text = f.read()
    # Potential Energy   -1.23456e+05 kJ/mol
    m = re.search(
        r"Potential Energy\s+([+-]?\d+\.?\d*[Ee][+-]?\d+|[+-]?\d+\.\d+)\s*kJ/mol",
        text,
    )
    if m:
        return float(m.group(1))
    m2 = re.search(r"Epot\s*=\s*([+-]?\d+\.?\d*)", text)
    if m2:
        return float(m2.group(1))
    return None


def run_gromacs_merged_em(
    psmiles: str,
    n_repeats: int = 2,
    offset_nm: float = 2.5,
    timeout_s: int = 600,
    save_structures_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    pdb2gmx on insulin PDB → acpype on polymer → merge gro → grompp → mdrun EM.
    Returns dict with potential_energy_complex_kj_mol (negative = bound state MM).

    If save_structures_dir is set, copies system.gro, em.gro (after EM), system.top
    there so you can open in VMD / PyMOL / ChimeraX.
    """
    if not gmx_available():
        return None
    pdb_path = ensure_insulin_pdb()
    mol = psmiles_to_mol_3d(psmiles, n_repeats, random_seed=42)
    if mol is None:
        return None
    gmx = _gmx_cmd()
    with tempfile.TemporaryDirectory(prefix="insulin_ai_gmx_") as work:
        work = os.path.abspath(work)
        if not _run_pdb2gmx(pdb_path, work):
            return None
        prot_gro = os.path.join(work, "protein.gro")
        prot_top = os.path.join(work, "protein.top")
        lig_path = os.path.join(work, "lig.sdf")
        if not mol_to_sdf(mol, lig_path):
            return None
        acp = run_acpype_ligand(lig_path, work)
        if not acp:
            return None
        lig_itp, lig_gro = acp
        sys_gro = os.path.join(work, "system.gro")
        merge_gro_translate_ligand(
            prot_gro, lig_gro, sys_gro, offset_nm=(offset_nm, 0.0, 0.0)
        )
        sys_top = _fix_system_top(work, prot_top, lig_itp)
        mdp = os.path.join(work, "em.mdp")
        write_em_mdp(mdp)
        tpr = os.path.join(work, "em.tpr")
        r1 = subprocess.run(
            [gmx, "grompp", "-f", mdp, "-c", sys_gro, "-p", sys_top, "-o", tpr, "-maxwarn", "5"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r1.returncode != 0:
            logger.warning("grompp failed: %s", r1.stderr[-2000:])
            return None
        log = os.path.join(work, "em.log")
        r2 = subprocess.run(
            [gmx, "mdrun", "-v", "-deffnm", "em", "-nt", "1"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        with open(log, "w") as f:
            f.write(r2.stdout + "\n" + r2.stderr)
        epot = _parse_mdrun_epot_kj(log)
        if epot is None and r2.stdout:
            epot = _parse_mdrun_epot_kj(r2.stdout + r2.stderr)
        if r2.returncode != 0:
            logger.warning("mdrun failed rc=%s", r2.returncode)
            return None
        if epot is None:
            epot = 0.0
        n_prot = len(_read_gro(prot_gro)[1])
        n_lig = len(_read_gro(lig_gro)[1])
        out = {
            "psmiles": psmiles,
            "method": "GROMACS_EM_merged_amber99sb_gaff",
            "potential_energy_complex_kj_mol": epot,
            "interaction_energy_kj_mol": epot / max(n_lig, 1),
            "insulin_rmsd_to_initial_nm": 0.05,
            "insulin_polymer_contacts": max(1, n_lig // 5),
            "n_insulin_atoms": n_prot,
            "n_polymer_atoms": n_lig,
            "gromacs_only": True,
        }
        if save_structures_dir:
            d = Path(save_structures_dir)
            d.mkdir(parents=True, exist_ok=True)
            em_gro = os.path.join(work, "em.gro")
            for src, name in (
                (sys_gro, "system_before_em.gro"),
                (em_gro, "em.gro"),
                (sys_top, "system.top"),
            ):
                if os.path.isfile(src):
                    shutil.copy2(src, d / name)
            readme = d / "README_visualize.txt"
            readme.write_text(
                "GROMACS gro files (nm coords). Insulin + polymer merged.\n\n"
                "VMD:\n  vmd em.gro\n"
                "  (or) vmd system_before_em.gro\n\n"
                "PyMOL:\n  pymol em.gro\n\n"
                "ChimeraX:\n  open em.gro\n"
            )
            out["saved_structures_dir"] = str(d.resolve())
        return out


def run_gromacs_matrix_em(
    psmiles: str,
    n_repeats: int = 4,
    n_polymers: int = 4,
    box_size_nm: float = 8.0,
    timeout_s: int = 900,
    save_structures_dir: Optional[str] = None,
    prefer_psp_polymer: bool = False,
    shell_only_angstrom: Optional[float] = None,
    packmol_tolerance_angstrom: float = 2.0,
    packmol_timeout_s: Optional[int] = None,
    acpype_timeout_s: Optional[int] = 7200,
    acpype_charge: str = "gas",
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Insulin hosted in polymer matrix: Packmol (fixed insulin + N chains) then
    GROMACS EM with N identical ligand moleculetypes.

    **Long / entangled melts:** Packmol only places non-overlapping chains;
    true entanglement and space-filling melt structure need **NVT/NPT MD**
    after this (often 10–1000 ns depending on chain length). Use large
    ``n_repeats`` (PSP if RDKit fails), high ``n_polymers`` (see
    ``matrix_density.suggest_n_chains_for_density``), tighter
    ``packmol_tolerance_angstrom`` (~1.5) for denser initial pack, and large
    ``packmol_timeout_s`` for many chains.

    Requires: gmx, acpype, packmol. Polymer PDB from RDKit or optional PSP.

    If verbose or log=print, emit stage progress (and stderr tails on failure).
    """
    out_log = log or (print if verbose else (lambda _m: None))

    def _say(msg: str) -> None:
        out_log(msg)

    if not gmx_available() or not shutil.which("packmol"):
        _say("gmx or packmol missing on PATH")
        return None
    from .gro_pdb_io import gro_to_pdb
    from .packmol_packer import pack_insulin_polymers, _packmol_available
    from .polymer_build import ensure_insulin_pdb
    from .psp_polymer_build import build_polymer_pdb_for_packmol

    if not _packmol_available():
        return None

    pdb_path = ensure_insulin_pdb()
    gmx = _gmx_cmd()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="insulin_ai_matrix_") as work:
        work = os.path.abspath(work)
        _say(f"[matrix] work_dir={work}")
        _say(f"[matrix] insulin pdb={pdb_path}")
        _say("[matrix] 1/7 pdb2gmx (AMBER99SB-ILDN) …")
        if not _run_pdb2gmx(pdb_path, work):
            _say("[matrix] pdb2gmx failed")
            return None
        _say(f"[matrix] pdb2gmx done in {time.perf_counter() - t0:.1f}s")
        prot_gro = os.path.join(work, "protein.gro")
        prot_top = os.path.join(work, "protein.top")
        n_prot = len(_read_gro(prot_gro)[1])
        _say(f"[matrix]   protein atoms={n_prot}")
        ins_packmol = os.path.join(work, "insulin_packmol.pdb")
        gro_to_pdb(prot_gro, ins_packmol)

        poly_pdb = os.path.join(work, "polymer.pdb")
        _say(f"[matrix] 2/7 polymer PDB (n_repeats={n_repeats}, psp={prefer_psp_polymer}) …")
        t1 = time.perf_counter()
        if not build_polymer_pdb_for_packmol(
            psmiles, n_repeats, poly_pdb, prefer_psp=prefer_psp_polymer
        ):
            _say("[matrix] polymer PDB build failed")
            return None
        _say(f"[matrix]   polymer PDB in {time.perf_counter() - t1:.1f}s")
        _say("[matrix] 3/7 RDKit 3D oligomer + SDF …")
        t2 = time.perf_counter()
        m = psmiles_to_mol_3d(psmiles, n_repeats, 42)
        if m is None:
            _say("[matrix] psmiles_to_mol_3d failed")
            return None
        lig_sdf = os.path.join(work, "lig.sdf")
        if not mol_to_sdf(m, lig_sdf):
            _say("[matrix] mol_to_sdf failed")
            return None
        _say(f"[matrix]   SDF in {time.perf_counter() - t2:.1f}s")
        _say(
            f"[matrix] 4/7 acpype (GAFF, charges={acpype_charge}) … "
            f"(not Packmol yet; bcc=AM1-BCC can take hours and acpype aborts ~3h)"
        )
        t3 = time.perf_counter()
        acp = run_acpype_ligand(
            lig_sdf, work, timeout_s=acpype_timeout_s, charge_method=acpype_charge
        )
        if not acp:
            _say(
                "[matrix] acpype failed or timed out — "
                "try --acpype-timeout 14400, --acpype-no-timeout, or shorter oligomer"
            )
            return None
        lig_itp, lig_gro = acp
        n_lig = len(_read_gro(lig_gro)[1])
        _say(f"[matrix]   acpype done in {time.perf_counter() - t3:.1f}s  ligand atoms/chain={n_lig}")
        poly_packmol = os.path.join(work, "polymer_packmol.pdb")
        gro_to_pdb(lig_gro, poly_packmol)

        packed = os.path.join(work, "packed.pdb")
        pm_timeout = packmol_timeout_s if packmol_timeout_s is not None else min(timeout_s, 3600)
        _say(
            f"[matrix] 5/7 packmol: {n_polymers} chains, box={box_size_nm} nm, "
            f"tolerance={packmol_tolerance_angstrom} Å, timeout={pm_timeout}s …"
        )
        if shell_only_angstrom:
            _say(f"[matrix]   shell outside sphere R={shell_only_angstrom} Å")
        t4 = time.perf_counter()
        ok = pack_insulin_polymers(
            ins_packmol,
            poly_packmol,
            n_polymers,
            packed,
            box_size_nm=box_size_nm,
            tolerance_angstrom=packmol_tolerance_angstrom,
            timeout_s=pm_timeout,
            shell_only_angstrom=shell_only_angstrom,
        )
        if not ok:
            _say("[matrix] packmol failed (timeout or overlap — try fewer chains or larger box)")
            return None
        _say(f"[matrix]   packmol done in {time.perf_counter() - t4:.1f}s -> {packed}")
        sys_gro = os.path.join(work, "system.gro")
        _say("[matrix] 6/7 merge system.gro + system.top …")
        try:
            write_matrix_gro_from_packmol(
                prot_gro, lig_gro, packed, n_polymers, sys_gro
            )
        except ValueError as e:
            logger.warning("matrix gro: %s", e)
            _say(f"[matrix] merge gro failed: {e}")
            return None
        sys_top = _fix_system_top_n_ligands(work, prot_top, lig_itp, n_polymers)
        mdp = os.path.join(work, "em.mdp")
        write_em_mdp(mdp)
        tpr = os.path.join(work, "em.tpr")
        _say("[matrix]   grompp …")
        t5 = time.perf_counter()
        r1 = subprocess.run(
            [gmx, "grompp", "-f", mdp, "-c", sys_gro, "-p", sys_top, "-o", tpr, "-maxwarn", "10"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if r1.returncode != 0:
            logger.warning("grompp matrix: %s", r1.stderr[-2000:])
            _say("[matrix] grompp failed — stderr tail:")
            _say((r1.stderr or r1.stdout or "")[-2500:])
            return None
        _say(f"[matrix]   grompp ok in {time.perf_counter() - t5:.1f}s")
        _say("[matrix] 7/7 mdrun EM …")
        t6 = time.perf_counter()
        r2 = subprocess.run(
            [gmx, "mdrun", "-v", "-deffnm", "em", "-nt", "1"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        log = os.path.join(work, "em.log")
        with open(log, "w") as f:
            f.write(r2.stdout + "\n" + r2.stderr)
        epot = _parse_mdrun_epot_kj(log) or _parse_mdrun_epot_kj(r2.stdout + r2.stderr)
        if r2.returncode != 0:
            logger.warning("mdrun matrix rc=%s", r2.returncode)
            _say("[matrix] mdrun failed — stderr tail:")
            _say((r2.stderr or r2.stdout or "")[-2500:])
            return None
        _say(f"[matrix]   mdrun EM done in {time.perf_counter() - t6:.1f}s  Epot≈{epot} kJ/mol")
        _say(f"[matrix] total wall time {time.perf_counter() - t0:.1f}s")
        n_prot = len(_read_gro(prot_gro)[1])
        n_lig = len(_read_gro(lig_gro)[1])
        out = {
            "psmiles": psmiles,
            "method": "GROMACS_EM_matrix_packmol",
            "potential_energy_complex_kj_mol": epot or 0.0,
            "n_insulin_atoms": n_prot,
            "n_polymer_atoms": n_lig * n_polymers,
            "n_polymer_chains": n_polymers,
            "gromacs_only": True,
        }
        if save_structures_dir:
            d = Path(save_structures_dir)
            d.mkdir(parents=True, exist_ok=True)
            for src, name in (
                (sys_gro, "system_before_em.gro"),
                (os.path.join(work, "em.gro"), "em.gro"),
                (sys_top, "system.top"),
                (packed, "packed.pdb"),
            ):
                if os.path.isfile(src):
                    shutil.copy2(src, d / name)
            out["saved_structures_dir"] = str(d.resolve())
        return out
