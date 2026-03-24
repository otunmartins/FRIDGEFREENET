#!/usr/bin/env python3
"""
Matplotlib 3D "chemically meaningful" previews of insulin–polymer complexes:

- **Protein (default chains A+B):** CA trace smoothed and drawn as a thick ribbon-like tube.
- **Polymer (other chains):** ball-and-stick using ``CONECT`` when present, else distance heuristics.

This avoids an external PyMOL/ChimeraX install; quality is report-grade, not publication ray-traced.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class _Atom:
    serial: int
    chain: str
    resseq: int
    name: str
    element: str
    xyz: np.ndarray  # shape (3,), Angstrom


def _parse_pdb(path: Path) -> Tuple[Dict[int, _Atom], Dict[int, List[int]]]:
    atoms: Dict[int, _Atom] = {}
    conect_lines: List[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 54:
                try:
                    serial = int(line[6:11])
                    name = line[12:16].strip()
                    chain = line[21:22].strip() or "?"
                    resseq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                el = line[76:78].strip() if len(line) >= 78 else ""
                if not el and name:
                    el = re.sub(r"\d", "", name)[:1] or "?"
                atoms[serial] = _Atom(
                    serial, chain, resseq, name, el.upper() or "C", np.array([x, y, z], dtype=float)
                )
            elif line.startswith("CONECT"):
                conect_lines.append(line)

    conect: Dict[int, List[int]] = defaultdict(list)
    for line in conect_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            a0 = int(parts[1])
        except ValueError:
            continue
        for b in parts[2:]:
            try:
                bi = int(b)
            except ValueError:
                continue
            if a0 in atoms and bi in atoms:
                conect[a0].append(bi)
    return atoms, conect


def _smooth_xyz(xyz: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average along the backbone (rows = points)."""
    if len(xyz) < 3:
        return xyz
    w = max(3, min(window, len(xyz) // 2 * 2 + 1))
    if w % 2 == 0:
        w += 1
    k = np.ones(w, dtype=float) / w
    out = np.empty_like(xyz)
    for j in range(3):
        out[:, j] = np.convolve(xyz[:, j], k, mode="same")
    return out


def _ca_ribbon_paths(
    atoms: Dict[int, _Atom],
    chains: Sequence[str],
) -> List[np.ndarray]:
    """Ordered CA coordinates per chain, for ribbon drawing."""
    paths: List[np.ndarray] = []
    for ch in chains:
        cas: List[Tuple[int, _Atom]] = []
        for _s, a in atoms.items():
            if a.chain == ch and a.name.upper() == "CA":
                cas.append((a.resseq, a))
        cas.sort(key=lambda t: t[0])
        if len(cas) < 2:
            continue
        xyz = np.array([a.xyz for _, a in cas], dtype=float)
        paths.append(_smooth_xyz(xyz, window=5))
    return paths


def _element_color(el: str) -> Tuple[float, float, float]:
    el = el.upper()[:1]
    if el == "O":
        return (0.9, 0.2, 0.15)
    if el == "N":
        return (0.25, 0.35, 0.95)
    if el == "S":
        return (0.95, 0.85, 0.2)
    if el == "P":
        return (0.95, 0.55, 0.15)
    return (0.45, 0.45, 0.48)


def _polymer_bonds_and_heavy(
    atoms: Dict[int, _Atom],
    protein_chains: Set[str],
    conect: Dict[int, List[int]],
    max_bond_A: float = 1.9,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Heavy atoms not in protein chains; bond list as index pairs into the heavy array.
    """
    heavy_serials: List[int] = []
    for s, a in atoms.items():
        if a.chain in protein_chains:
            continue
        if a.element.upper() in ("H", "D"):
            continue
        heavy_serials.append(s)
    serial_to_i = {s: i for i, s in enumerate(heavy_serials)}
    xyz_h = np.array([atoms[s].xyz for s in heavy_serials], dtype=float)
    elems = [atoms[s].element for s in heavy_serials]

    bond_pairs: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()

    def add_bond(i: int, j: int) -> None:
        if i == j:
            return
        a, b = (min(i, j), max(i, j))
        if (a, b) in seen:
            return
        seen.add((a, b))
        bond_pairs.append((a, b))

    # CONECT-based
    for a, bs in conect.items():
        if a not in serial_to_i:
            continue
        i = serial_to_i[a]
        for b in bs:
            if b not in serial_to_i:
                continue
            j = serial_to_i[b]
            add_bond(i, j)

    n = len(heavy_serials)
    # Distance fallback only when CONECT did not define polymer connectivity
    if len(bond_pairs) == 0 and n <= 80:
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(xyz_h[i] - xyz_h[j]))
                if 0.45 < d < max_bond_A:
                    add_bond(i, j)

    return xyz_h, np.array(elems), bond_pairs


def write_complex_chemviz_png(
    pdb_path: str,
    output_path: str,
    *,
    protein_chains: Sequence[str] = ("A", "B"),
    dpi: int = 140,
    figsize_inches: Tuple[float, float] = (7.0, 6.0),
) -> Dict[str, Any]:
    """
    Render ribbon (CA) + polymer ball-and-stick to PNG.

    Polymer = any chain not listed in ``protein_chains``.
    """
    path = Path(pdb_path)
    out = Path(output_path)
    if out.suffix.lower() != ".png":
        out = out.with_suffix(".png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        return {"ok": False, "error": f"matplotlib required: {e}"}

    if not path.is_file():
        return {"ok": False, "error": f"PDB not found: {path}"}

    atoms, conect = _parse_pdb(path)
    if not atoms:
        return {"ok": False, "error": "no ATOM/HETATM records parsed"}

    pset = set(protein_chains)
    ribbon_paths = _ca_ribbon_paths(atoms, protein_chains)
    xyz_h, elems, bonds = _polymer_bonds_and_heavy(atoms, pset, conect)

    fig = plt.figure(figsize=figsize_inches, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    for rp in ribbon_paths:
        ax.plot(
            rp[:, 0],
            rp[:, 1],
            rp[:, 2],
            color="#2e6b8a",
            linewidth=5.0,
            solid_capstyle="round",
            alpha=0.92,
            label="_nolegend_",
        )

    for i, j in bonds:
        seg = np.vstack([xyz_h[i], xyz_h[j]])
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            seg[:, 2],
            color="#333333",
            linewidth=1.8,
            alpha=0.85,
        )

    for i in range(len(xyz_h)):
        c = _element_color(str(elems[i]))
        ax.scatter(
            [xyz_h[i, 0]],
            [xyz_h[i, 1]],
            [xyz_h[i, 2]],
            color=c,
            s=55,
            edgecolors="0.15",
            linewidths=0.25,
            alpha=1.0,
        )

    ax.set_axis_off()
    ax.view_init(elev=18, azim=72)
    try:
        chunks: List[np.ndarray] = []
        if len(xyz_h):
            chunks.append(xyz_h)
        for p in ribbon_paths:
            if len(p):
                chunks.append(p)
        if chunks:
            all_pts = np.vstack(chunks)
            ctr = all_pts.mean(axis=0)
            r = float(np.ptp(all_pts, axis=0).max()) / 2.0 + 2.0
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] - r, ctr[1] + r)
            ax.set_zlim(ctr[2] - r, ctr[2] + r)
    except Exception:
        pass

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    if not out.is_file():
        return {"ok": False, "error": f"PNG not written: {out}"}
    return {
        "ok": True,
        "path": str(out.resolve()),
        "n_ca_segments": len(ribbon_paths),
        "n_polymer_heavy": int(xyz_h.shape[0]),
        "n_bonds_drawn": len(bonds),
    }
