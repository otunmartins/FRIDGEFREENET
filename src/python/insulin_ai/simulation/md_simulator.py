#!/usr/bin/env python3
"""Evaluate PSMILES via OpenMM Packmol matrix encapsulation + minimize (AMBER14SB + GAFF + Gasteiger)."""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from insulin_ai.run_paths import repo_root_from_package, session_dir_from_env

from .openmm_compat import openmm_available
from .property_extractor import PropertyExtractor


def _env_int(primary: str, fallback: str, default: str) -> int:
    v = os.environ.get(primary) or os.environ.get(fallback) or default
    return int(v)


def _env_float(primary: str, fallback: str, default: str) -> float:
    v = os.environ.get(primary) or os.environ.get(fallback) or default
    return float(v)


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _matrix_target_density_g_cm3() -> Optional[float]:
    raw = os.environ.get("INSULIN_AI_OPENMM_MATRIX_TARGET_DENSITY_G_CM3", "").strip()
    if not raw:
        return None
    return float(raw)


def _matrix_packing_mode() -> str:
    """INSULIN_AI_OPENMM_MATRIX_PACKING_MODE: ``bulk`` (default) or ``shell``."""
    raw = os.environ.get("INSULIN_AI_OPENMM_MATRIX_PACKING_MODE", "bulk").strip().lower()
    if raw == "bulk":
        return "bulk"
    return "shell"


def _matrix_progressive_pack() -> bool:
    """INSULIN_AI_OPENMM_MATRIX_PROGRESSIVE_PACK: greedily add chains until Packmol effort limits."""
    return _env_bool("INSULIN_AI_OPENMM_MATRIX_PROGRESSIVE_PACK", False)


def _matrix_progressive_per_attempt_timeout_s() -> float:
    return _env_float("INSULIN_AI_OPENMM_MATRIX_PACK_PER_ATTEMPT_TIMEOUT_S", "", "120")


def _matrix_progressive_max_total_s() -> Optional[float]:
    raw = os.environ.get("INSULIN_AI_OPENMM_MATRIX_PACK_MAX_TOTAL_S", "").strip()
    if not raw:
        return None
    return float(raw)


def _matrix_progressive_n_max() -> Optional[int]:
    raw = os.environ.get("INSULIN_AI_OPENMM_MATRIX_PROGRESSIVE_N_MAX", "").strip()
    if not raw:
        return None
    return int(raw)


def _complex_viz_mode() -> str:
    """
    INSULIN_AI_COMPLEX_VIZ: ``pymol`` | ``matplotlib`` | ``auto`` (default).

    ``auto`` tries open-source PyMOL on PATH first (ribbon + DSS for insulin, sticks for
    polymer), then falls back to matplotlib chemviz.
    """
    raw = os.environ.get("INSULIN_AI_COMPLEX_VIZ", "auto").strip().lower()
    if raw in ("pymol", "matplotlib", "auto"):
        return raw
    return "auto"


def _effective_matrix_target_density_g_cm3() -> Optional[float]:
    """
    Explicit ``INSULIN_AI_OPENMM_MATRIX_TARGET_DENSITY_G_CM3`` wins.

    Otherwise, unless ``INSULIN_AI_OPENMM_MATRIX_FIXED_MODE=1``, use default density-driven
    packing (``INSULIN_AI_OPENMM_MATRIX_DEFAULT_DENSITY_G_CM3``, default 0.52 g/cm³).
    """
    explicit = _matrix_target_density_g_cm3()
    if explicit is not None:
        return explicit
    if _env_bool("INSULIN_AI_OPENMM_MATRIX_FIXED_MODE", False):
        return None
    return _env_float("INSULIN_AI_OPENMM_MATRIX_DEFAULT_DENSITY_G_CM3", "", "0.52")


def _packmol_required_error() -> RuntimeError:
    return RuntimeError(
        "Packmol is required for evaluate_psmiles / MDSimulator.evaluate_candidates (matrix encapsulation). "
        "Install the packmol binary on PATH (e.g. conda: conda-forge::packmol, or pip: pip install packmol). "
        "See docs/OPENMM_SCREENING.md and docs/DEPENDENCIES.md."
    )


def _eval_quiet() -> bool:
    """
    Suppress per-candidate progress (JSON + stderr) when user opts out.

    INSULIN_AI_EVAL_QUIET=1, or INSULIN_AI_EVAL_VERBOSE=0/false/no.
    """
    if os.environ.get("INSULIN_AI_EVAL_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return True
    v = os.environ.get("INSULIN_AI_EVAL_VERBOSE", "").strip().lower()
    if v in ("0", "false", "no"):
        return True
    return False


def _progress_log(msg: str) -> None:
    """Visible in MCP server stderr (terminal running the server), not in tool return."""
    print(msg, file=sys.stderr, flush=True)


def resolve_eval_structure_artifacts_dir(artifacts_dir: Optional[str] = None) -> Optional[Path]:
    """
    Directory for monomer PNG (psmiles), minimized complex PDB, and preview PNG.

    Resolution order:

    1. Non-empty ``artifacts_dir`` argument.
    2. Env ``INSULIN_AI_EVAL_ARTIFACTS_DIR`` (absolute or relative path).
    3. If ``INSULIN_AI_SESSION_DIR`` is set and ``INSULIN_AI_EVAL_NO_STRUCTURE_ARTIFACTS``
       is not 1/true/yes: ``<session>/structures``.

    Returns None if no directory should be used (callers skip writing structure files).
    """
    raw = (artifacts_dir or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    env_ad = os.environ.get("INSULIN_AI_EVAL_ARTIFACTS_DIR", "").strip()
    if env_ad:
        p = Path(env_ad).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    if os.environ.get("INSULIN_AI_EVAL_NO_STRUCTURE_ARTIFACTS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None
    sd = session_dir_from_env(repo_root_from_package())
    if sd:
        p = (sd / "structures").resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    return None


class MDSimulator:
    def __init__(
        self,
        n_steps: int = 50000,
        temperature: float = 298.0,
        random_seed: int = 42,
    ):
        if not openmm_available():
            raise RuntimeError(
                "OpenMM screening stack not importable. Install with: "
                "pip install -e '.[openmm]' (or conda: openmm, pip: openmmforcefields, openff-toolkit, pdbfixer, rdkit)."
            )
        self.extractor = PropertyExtractor()
        self.n_steps = n_steps
        self.random_seed = random_seed

    def _get_psmiles(self, candidate: Dict[str, Any]) -> Optional[str]:
        if isinstance(candidate, str):
            return candidate
        p = candidate.get("psmiles") or candidate.get("chemical_structure")
        if p:
            return p
        m = candidate.get("material_name", "")
        return m if m and "[*]" in str(m) else None

    def evaluate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        max_candidates: int = 10,
        verbose: bool = True,
        artifacts_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        from insulin_ai.psmiles_drawing import safe_filename_basename, save_psmiles_png

        from .openmm_complex import run_openmm_matrix_relax_and_energy
        from .packmol_packer import _packmol_available
        from .pdb_preview import write_complex_preview_png
        from .pymol_complex_viz import write_complex_viz_png_auto

        if not _packmol_available():
            raise _packmol_required_error()

        if _eval_quiet():
            verbose = False
        struct_dir = resolve_eval_structure_artifacts_dir(artifacts_dir)
        to_eval = candidates[:max_candidates]
        if not to_eval:
            raise ValueError("empty candidates")
        md_results = []
        material_names = []
        progress: List[Dict[str, Any]] = []
        n_repeats = _env_int("INSULIN_AI_OPENMM_N_REPEATS", "INSULIN_AI_GMX_N_REPEATS", "4")
        n_polymers = _env_int("INSULIN_AI_OPENMM_MATRIX_N_POLYMERS", "", "8")
        box_nm = _env_float("INSULIN_AI_OPENMM_MATRIX_BOX_NM", "", "7.5")
        density_n_min = _env_int("INSULIN_AI_OPENMM_MATRIX_DENSITY_N_MIN", "", "4")
        density_n_max = _env_int("INSULIN_AI_OPENMM_MATRIX_DENSITY_N_MAX", "", "100")
        shell_a = _env_float("INSULIN_AI_OPENMM_MATRIX_SHELL_A", "", "14.0")
        max_minimize = int(os.environ.get("INSULIN_AI_OPENMM_MAX_MINIMIZE_STEPS", "2000"))
        target_density = _effective_matrix_target_density_g_cm3()
        packing_mode = _matrix_packing_mode()
        run_npt = _env_bool("INSULIN_AI_OPENMM_MATRIX_NPT", False)
        npt_ps = _env_float("INSULIN_AI_OPENMM_MATRIX_NPT_PS", "", "0.5")
        wall_s = _env_float("INSULIN_AI_OPENMM_MATRIX_WALL_CLOCK_S", "", "180.0")
        progressive_pack = _matrix_progressive_pack()
        complex_viz_mode = _complex_viz_mode()
        _rs = os.environ.get("INSULIN_AI_OPENMM_MATRIX_RESTRAIN_SHELL")
        if _rs is None or not str(_rs).strip():
            restrain_shell: Optional[bool] = None
        else:
            restrain_shell = _env_bool("INSULIN_AI_OPENMM_MATRIX_RESTRAIN_SHELL", True)
        barostat_fs = _env_float("INSULIN_AI_OPENMM_MATRIX_BAROSTAT_INTERVAL_FS", "", "10.0")
        n_total = len(to_eval)
        geom = "polymer bulk (full cell)" if packing_mode == "bulk" else "polymer shell"
        msg = (
            f"[insulin-ai] OpenMM matrix (Packmol): {n_total} candidate(s) — "
            f"insulin + {geom}, minimize"
            + (" + NPT sampling" if run_npt else "")
            + ", interaction energy (kJ/mol)."
        )
        print(f"  Evaluating {n_total} via OpenMM Packmol matrix...")
        if verbose:
            _progress_log(msg)

        for i, cand in enumerate(to_eval):
            psmiles = self._get_psmiles(cand)
            if not psmiles or "[*]" not in str(psmiles):
                md_results.append(None)
                material_names.append(cand.get("material_name", f"candidate_{i}"))
                if verbose:
                    entry = {
                        "index": i,
                        "total": n_total,
                        "status": "skipped",
                        "reason": "no valid PSMILES with [*]",
                        "material_name": cand.get("material_name", f"candidate_{i}"),
                    }
                    progress.append(entry)
                    _progress_log(f"[insulin-ai] {i + 1}/{n_total} skipped (no valid PSMILES)")
                continue
            name = cand.get("material_name", psmiles)
            material_names.append(name)
            preview = str(psmiles)[:60] + ("…" if len(str(psmiles)) > 60 else "")
            t0 = time.perf_counter()
            if verbose:
                _progress_log(
                    f"[insulin-ai] {i + 1}/{n_total} Packmol+matrix: {preview} "
                    f"(max {max_minimize} minimizer steps)"
                )
            slug = safe_filename_basename(str(name))
            pdb_out: Optional[str] = None
            if struct_dir is not None:
                pdb_out = str(struct_dir / f"{slug}_complex_minimized.pdb")
            matrix_kw: Dict[str, Any] = dict(
                n_repeats=n_repeats,
                random_seed=self.random_seed,
                max_minimize_steps=max_minimize,
                save_minimized_pdb=pdb_out,
                verbose=verbose,
                restrain_shell=restrain_shell,
                run_npt=run_npt,
                barostat_interval_fs=barostat_fs,
                npt_duration_ps=npt_ps,
                wall_clock_limit_s=wall_s,
                packing_mode=packing_mode,
            )
            if progressive_pack:
                matrix_kw["progressive_pack"] = True
                matrix_kw["progressive_per_attempt_timeout_s"] = _matrix_progressive_per_attempt_timeout_s()
                matrix_kw["progressive_max_total_s"] = _matrix_progressive_max_total_s()
                matrix_kw["progressive_n_max"] = _matrix_progressive_n_max()
            if target_density is not None:
                matrix_kw["target_density_g_cm3"] = target_density
                matrix_kw["box_size_nm"] = box_nm
                matrix_kw["density_polymer_n_min"] = density_n_min
                matrix_kw["density_polymer_n_max"] = density_n_max
            else:
                matrix_kw["n_polymers"] = n_polymers
                matrix_kw["box_size_nm"] = box_nm
                if packing_mode != "bulk":
                    matrix_kw["shell_only_angstrom"] = shell_a
            res = run_openmm_matrix_relax_and_energy(psmiles, **matrix_kw)
            elapsed = time.perf_counter() - t0
            if res is None:
                raise RuntimeError(
                    f"OpenMM matrix evaluate failed for {name[:40]} (Packmol or setup error; see logs)."
                )
            if pdb_out:
                res["complex_pdb_path"] = pdb_out
            elif res.get("minimized_pdb"):
                res["complex_pdb_path"] = res["minimized_pdb"]
            npc = res.get("n_polymer_atoms_per_chain")
            nch = res.get("n_polymer_chains")
            if npc is not None and nch is not None:
                res["n_polymer_atoms"] = int(npc) * int(nch)
            cp = res.get("complex_pdb_path")
            nprot = res.get("n_insulin_atoms")
            if cp and nprot is not None:
                try:
                    from .matrix_packing_metrics import compute_matrix_packing_metrics

                    res["packing_metrics"] = compute_matrix_packing_metrics(
                        str(cp), int(nprot)
                    )
                except Exception as ex:
                    res["packing_metrics"] = {"ok": False, "error": str(ex)}
            if struct_dir is not None:
                monomer_png = struct_dir / f"{slug}_monomer.png"
                r_mono = save_psmiles_png(psmiles, monomer_png, overwrite=True)
                res["monomer_png_path"] = r_mono.get("path") if r_mono.get("ok") else None
                res["monomer_png_error"] = r_mono.get("error")
                preview_png = struct_dir / f"{slug}_complex_preview.png"
                if res.get("complex_pdb_path"):
                    r_prev = write_complex_preview_png(
                        res["complex_pdb_path"],
                        str(preview_png),
                    )
                    res["complex_preview_png_path"] = r_prev.get("path") if r_prev.get("ok") else None
                    res["complex_preview_png_error"] = r_prev.get("error")
                    chemviz_png = struct_dir / f"{slug}_complex_chemviz.png"
                    r_cv, cv_backend = write_complex_viz_png_auto(
                        res["complex_pdb_path"],
                        str(chemviz_png),
                        mode=complex_viz_mode,
                        n_protein_atoms=res.get("n_insulin_atoms"),
                    )
                    res["complex_chemviz_png_path"] = r_cv.get("path") if r_cv.get("ok") else None
                    res["complex_chemviz_png_error"] = r_cv.get("error")
                    res["complex_chemviz_backend"] = cv_backend
                else:
                    res["complex_preview_png_path"] = None
                    res["complex_preview_png_error"] = "complex PDB not written"
                    res["complex_chemviz_png_path"] = None
                    res["complex_chemviz_png_error"] = "complex PDB not written"
                    res["complex_chemviz_backend"] = None
            md_results.append(res)
            if verbose:
                entry = {
                    "index": i,
                    "total": n_total,
                    "status": "completed",
                    "material_name": name,
                    "psmiles_preview": preview,
                    "seconds": round(elapsed, 3),
                    "method": res.get("method"),
                    "interaction_energy_kj_mol": res.get("interaction_energy_kj_mol"),
                    "potential_energy_complex_kj_mol": res.get("potential_energy_complex_kj_mol"),
                    "n_insulin_atoms": res.get("n_insulin_atoms"),
                    "n_polymer_atoms": res.get("n_polymer_atoms"),
                    "n_polymer_chains": res.get("n_polymer_chains"),
                    "complex_pdb_path": res.get("complex_pdb_path"),
                    "monomer_png_path": res.get("monomer_png_path"),
                    "complex_preview_png_path": res.get("complex_preview_png_path"),
                    "complex_chemviz_png_path": res.get("complex_chemviz_png_path"),
                    "complex_chemviz_backend": res.get("complex_chemviz_backend"),
                }
                pm = res.get("packing_metrics") or {}
                if pm.get("ok"):
                    entry["min_polymer_protein_distance_nm"] = pm.get(
                        "min_polymer_protein_distance_nm"
                    )
                    entry["fraction_polymer_within_0.80_nm"] = pm.get(
                        "fraction_polymer_within_0.80_nm"
                    )
                progress.append(entry)
                log_tail = f"E_int={res.get('interaction_energy_kj_mol')} kJ/mol"
                if pm.get("ok"):
                    log_tail += (
                        f", d_min(poly-prot)={pm.get('min_polymer_protein_distance_nm'):.3f} nm"
                    )
                _progress_log(f"[insulin-ai] {i + 1}/{n_total} done in {elapsed:.1f}s {log_tail}")

        feedback = self.extractor.extract_feedback(md_results, material_names)
        out: Dict[str, Any] = {
            "high_performers": feedback["high_performers"],
            "effective_mechanisms": feedback["effective_mechanisms"],
            "problematic_features": feedback["problematic_features"],
            "property_analysis": feedback["property_analysis"],
            "successful_materials": feedback["high_performers"],
            "md_results_raw": md_results,
        }
        if struct_dir is not None:
            out["structure_artifacts_dir"] = str(struct_dir)
        if verbose:
            out["evaluation_progress"] = progress
            out["evaluation_note"] = (
                "Each candidate: Packmol-packed polymer shell around insulin (periodic box), "
                "LocalEnergyMinimizer, optional short NPT segment (INSULIN_AI_OPENMM_MATRIX_NPT), "
                "then interaction energy (kJ/mol). Requires packmol on PATH. "
                "By default uses density-driven chain count (INSULIN_AI_OPENMM_MATRIX_DEFAULT_DENSITY_G_CM3); "
                "set INSULIN_AI_OPENMM_MATRIX_FIXED_MODE=1 for fixed N_POLYMERS + shell instead. "
                "packing_metrics reports polymer–protein proximity on the minimized PDB."
            )
        return out
