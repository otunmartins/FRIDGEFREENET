#!/usr/bin/env python3
"""
Pre-compute a PSMILES evaluation cache for reproducible IBM RL training.

Runs ``MDSimulator.evaluate_candidates`` (OpenMM Packmol matrix — same as the
agentic MCP ``evaluate_psmiles`` tool) over a batch of PSMILES and saves the
results to a JSON cache file.

The cache maps PSMILES (canonical form) → per-candidate ``property_analysis``
row so that ``InsulinPSMILESEnv`` can look up results instantly during training
without re-running OpenMM.

Usage
-----
.. code-block:: bash

    # Default: 200 diverse PSMILES from the built-in mutation pool
    python benchmarks/precompute_psmiles_cache.py \\
        --output data/ibm_psmiles_cache.json

    # Custom seed + larger pool
    python benchmarks/precompute_psmiles_cache.py \\
        --seeds "[*]OCC[*]" "[*]OC(=O)C(C)[*]" "[*]CC([*])c1ccccc1" \\
        --n-candidates 500 \\
        --md-steps 3000 \\
        --output data/ibm_psmiles_cache.json

    # Resume / extend existing cache (skip already-computed entries)
    python benchmarks/precompute_psmiles_cache.py \\
        --resume data/ibm_psmiles_cache.json \\
        --n-candidates 100 \\
        --output data/ibm_psmiles_cache.json

Environment
-----------
Must be run with the ``insulin-ai-sim`` conda environment (OpenMM, RDKit,
openff-toolkit, packmol on PATH).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src" / "python") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src" / "python"))


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

# Seed PSMILES spanning diverse polymer chemistries (polyethers, polyesters,
# vinyl, polyamides, polycarbonates, silicones, polysaccharide-like, fluorinated).
_DEFAULT_SEEDS = [
    "[*]OCC[*]",                                    # PEG
    "[*]OC(=O)C(C)[*]",                             # PLA
    "[*]OC(=O)C[*]",                                # PGA
    "[*]OC(=O)CCCCC[*]",                            # PCL
    "[*]CC([*])c1ccccc1",                            # PS
    "[*]CC([*])(C)C(=O)OC",                          # PMMA
    "[*]CC([*])C(=O)O",                              # PAA
    "[*]NC(=O)CCCCC[*]",                             # Nylon-6
    "[*]OC(=O)Oc1ccc(cc1)C(C)(C)c1ccc(cc1)[*]",    # PC (BPA)
    "[*]O[Si](C)(C)[*]",                             # PDMS
    "[*]CC([*])C(=O)NC(C)C",                         # PNIPAM
    "[*]CC([*])N1CCCC1=O",                           # PVP
    "[*]OC(=O)COC(=O)C(C)[*]",                      # PLGA
    "[*]C(F)(F)C(F)(F)[*]",                          # PTFE
    "[*]CC([*])O",                                   # PVA
    "[*]CC([*])Cl",                                  # PVC
    "[*]OC(=O)OCCC[*]",                              # PTMC
    "[*]CC([*])C#N",                                 # PAN
]


def generate_candidate_pool(
    seeds: List[str],
    n_candidates: int,
    random_seed: int = 42,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Generate a diverse pool of PSMILES via mutation from seeds.

    Args:
        seeds: Initial PSMILES to mutate from.
        n_candidates: Target pool size (may be fewer if many fail validation).
        random_seed: RNG seed.
        exclude: PSMILES to skip (e.g. already-cached entries).

    Returns:
        List of unique, validated canonical PSMILES.
    """
    import numpy as np

    from insulin_ai.material_mappings import prescreen_psmiles_for_md, validate_psmiles
    from insulin_ai.mutation import feedback_guided_mutation

    rng = np.random.default_rng(random_seed)
    exclude = exclude or set()

    pool: List[str] = []
    seen: Set[str] = set(exclude)

    # Validate and collect seeds themselves
    for s in seeds:
        vr = validate_psmiles(s)
        if vr.get("valid"):
            canonical = str(vr.get("canonical") or s)
            if canonical not in seen:
                pre = prescreen_psmiles_for_md(canonical)
                if pre.get("ok"):
                    pool.append(canonical)
                    seen.add(canonical)

    # Mutate until we have enough
    max_rounds = 10
    for round_idx in range(max_rounds):
        if len(pool) >= n_candidates:
            break
        fb = {
            "high_performer_psmiles": pool[:10] if pool else seeds[:3],
            "problematic_psmiles": [],
        }
        lib_size = min(n_candidates * 3, 500)
        mutated = feedback_guided_mutation(
            fb,
            library_size=lib_size,
            random_seed=int(rng.integers(0, 1_000_000)),
        )
        added_this_round = 0
        for c in mutated:
            if len(pool) >= n_candidates:
                break
            ps = c.get("chemical_structure") or c.get("psmiles")
            if not ps:
                continue
            vr = validate_psmiles(str(ps))
            if not vr.get("valid"):
                continue
            canonical = str(vr.get("canonical") or ps)
            if canonical in seen:
                continue
            pre = prescreen_psmiles_for_md(canonical)
            if not pre.get("ok"):
                seen.add(canonical)
                continue
            pool.append(canonical)
            seen.add(canonical)
            added_this_round += 1

        logger.info(
            "Round %d: +%d candidates (total %d / %d)",
            round_idx + 1,
            added_this_round,
            len(pool),
            n_candidates,
        )
        if added_this_round == 0:
            logger.warning("No new candidates generated in round %d; stopping.", round_idx + 1)
            break

    logger.info("Candidate pool ready: %d PSMILES", len(pool))
    return pool[:n_candidates]


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    psmiles_list: List[str],
    md_steps: int = 5000,
    batch_size: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a list of PSMILES via MDSimulator, return cache dict.

    Args:
        psmiles_list: Validated canonical PSMILES to evaluate.
        md_steps: Steps for OpenMM minimisation.
        batch_size: How many PSMILES to pass per MDSimulator call (1 = safest).
        verbose: Forward to MDSimulator.

    Returns:
        Dict mapping canonical PSMILES → property_analysis row.
    """
    from insulin_ai.simulation import MDSimulator
    from insulin_ai.simulation.openmm_compat import openmm_available

    if not openmm_available():
        raise RuntimeError(
            "OpenMM stack not available. Install with the insulin-ai-sim environment."
        )

    sim = MDSimulator(n_steps=md_steps)
    cache: Dict[str, Any] = {}
    total = len(psmiles_list)

    for start in range(0, total, batch_size):
        chunk = psmiles_list[start : start + batch_size]
        candidates = [
            {"material_name": f"PS_{i}", "chemical_structure": ps}
            for i, ps in enumerate(chunk)
        ]
        t0 = time.perf_counter()
        try:
            result = sim.evaluate_candidates(
                candidates, max_candidates=len(candidates), verbose=verbose
            )
        except Exception as e:
            logger.warning("Batch %d-%d evaluation error: %s", start, start + len(chunk), e)
            for ps in chunk:
                cache[ps] = {"psmiles": ps, "error": str(e)}
            continue

        elapsed = time.perf_counter() - t0
        pa = result.get("property_analysis") or {}

        for i, ps in enumerate(chunk):
            cand_name = f"PS_{i}"
            row = pa.get(cand_name) or {}
            row["psmiles"] = ps
            cache[ps] = row
            e_int = row.get("interaction_energy_kj_mol")
            rmsd = row.get("insulin_rmsd_to_initial_nm")
            logger.info(
                "[%d/%d] %s | E_int=%.2f kJ/mol | RMSD=%.4f nm | %.1fs",
                start + i + 1,
                total,
                ps[:40],
                e_int if e_int is not None else float("nan"),
                rmsd if rmsd is not None else float("nan"),
                elapsed / len(chunk),
            )

    return cache


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Pre-compute PSMILES evaluation cache for IBM RL training.\n"
            "Runs OpenMM Packmol matrix screening (identical to MCP evaluate_psmiles).\n"
            "Output: JSON dict {psmiles: {interaction_energy_kj_mol, ...}}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=str,
        default="data/ibm_psmiles_cache.json",
        help="Output cache JSON path (default: data/ibm_psmiles_cache.json).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        default=None,
        help="Seed PSMILES (default: 18 diverse polymers).",
    )
    p.add_argument(
        "--n-candidates",
        type=int,
        default=200,
        help=(
            "Number of PSMILES to evaluate (default: 200 = 20 agentic iterations "
            "× 10 evals, matching IBM RL default budget)."
        ),
    )
    p.add_argument(
        "--md-steps",
        type=int,
        default=5000,
        help="OpenMM minimisation steps per candidate (default: 5000).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Candidates per MDSimulator call (default: 1 for safety).",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed for mutation pool generation.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing cache JSON to extend (skips already-evaluated PSMILES).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate candidate pool but skip OpenMM evaluation (for testing).",
    )
    args = p.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache if resuming
    existing_cache: Dict[str, Any] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            with open(resume_path) as f:
                existing_cache = json.load(f)
            logger.info("Resuming with %d cached entries from %s", len(existing_cache), resume_path)

    seeds = args.seeds or _DEFAULT_SEEDS

    # Generate candidates (excluding already-cached)
    already_done: Set[str] = set(existing_cache.keys())
    candidates = generate_candidate_pool(
        seeds=seeds,
        n_candidates=args.n_candidates,
        random_seed=args.random_seed,
        exclude=already_done,
    )

    if args.dry_run:
        logger.info("Dry-run: generated %d candidates (no OpenMM).", len(candidates))
        out = {
            "dry_run": True,
            "n_candidates": len(candidates),
            "candidates": candidates,
        }
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Dry-run output written to {output_path}")
        return

    if not candidates:
        logger.info("No new candidates to evaluate; cache is already complete.")
        return

    logger.info(
        "Evaluating %d PSMILES with OpenMM (md_steps=%d) …", len(candidates), args.md_steps
    )
    t_start = time.perf_counter()

    new_cache = evaluate_batch(
        candidates,
        md_steps=args.md_steps,
        batch_size=args.batch_size,
    )

    elapsed_total = time.perf_counter() - t_start
    logger.info(
        "Evaluation complete: %d entries in %.1fs (avg %.1fs/candidate)",
        len(new_cache),
        elapsed_total,
        elapsed_total / max(len(candidates), 1),
    )

    # Merge with existing
    merged = {**existing_cache, **new_cache}

    # Summary stats
    energies = [
        v["interaction_energy_kj_mol"]
        for v in merged.values()
        if isinstance(v, dict) and v.get("interaction_energy_kj_mol") is not None
    ]
    n_evaluated = len(energies)
    n_targets = sum(1 for e in energies if e <= -5.0)

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)

    summary = {
        "n_total_cached": len(merged),
        "n_evaluated": n_evaluated,
        "n_target_candidates": n_targets,
        "target_fraction": round(n_targets / max(n_evaluated, 1), 4),
        "energy_mean_kj_mol": round(sum(energies) / max(n_evaluated, 1), 4) if energies else None,
        "energy_min_kj_mol": round(min(energies), 4) if energies else None,
        "energy_max_kj_mol": round(max(energies), 4) if energies else None,
        "wall_time_s": round(elapsed_total, 1),
        "output": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
