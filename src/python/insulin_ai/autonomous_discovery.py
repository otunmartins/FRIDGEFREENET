#!/usr/bin/env python3
"""
Autoresearch-style autonomous materials discovery loop.

Runs literature mining + mutation + MD evaluation until a wall-clock budget
expires. Appends one TSV row per iteration (autoresearch-style logging).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root when run as script from repo root
def _ensure_paths(root: Optional[str] = None) -> str:
    r = root or os.environ.get("INSULIN_AI_ROOT", "")
    if not r:
        here = os.path.abspath(__file__)
        # .../src/python/insulin_ai/autonomous_discovery.py -> repo root is 4 levels up
        r = os.path.dirname(here)
        for _ in range(3):
            r = os.path.dirname(r)
        if not os.path.isfile(os.path.join(r, "insulin_ai_mcp_server.py")):
            r = os.path.dirname(r)  # fallback if layout differs
    if r not in sys.path:
        sys.path.insert(0, r)
    sp = os.path.join(r, "src", "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return r


def _append_tsv(
    path: Path,
    run_id: str,
    score: float,
    memory_gb: float,
    status: str,
    description: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        if write_header:
            w.writerow(["run_id", "score", "memory_gb", "status", "description"])
        w.writerow([run_id, f"{score:.4f}", f"{memory_gb:.1f}", status, description[:500]])


def _memory_gb() -> float:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss: KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return round(ru.ru_maxrss / (1024 * 1024 * 1024), 2)  # bytes -> GB
        return round(ru.ru_maxrss / (1024 * 1024), 2)  # KB -> GB on Linux
    except Exception:
        return 0.0


def run_autonomous_discovery_loop(
    budget_minutes: float,
    results_tsv: str,
    root: Optional[str] = None,
    md_steps: int = 5000,
    max_eval_per_iteration: int = 8,
    mutation_size: int = 10,
    log_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run discovery iterations until wall-clock budget is exhausted.

    Returns summary dict: iterations_run, tsv_path, last_score, errors.
    """
    root = _ensure_paths(root)
    os.chdir(root)

    from insulin_ai.simulation import MDSimulator
    from insulin_ai.simulation.scoring import discovery_score

    # iterative_literature_mining lives at repo root
    from iterative_literature_mining import IterativeLiteratureMiner

    tsv_path = Path(results_tsv)
    if not tsv_path.is_absolute():
        tsv_path = Path(root) / tsv_path

    deadline = time.monotonic() + budget_minutes * 60.0
    miner = IterativeLiteratureMiner()
    md_sim = MDSimulator(n_steps=md_steps)
    mutator = None
    try:
        from insulin_ai.mutation import MaterialMutator

        mutator = MaterialMutator(random_seed=42)
    except ImportError:
        pass

    feedback_state: Dict[str, Any] = {
        "top_candidates": [],
        "stability_mechanisms": [],
        "target_properties": {},
        "limitations": [],
        "high_performer_psmiles": [],
        "problematic_psmiles": [],
    }

    iterations_run = 0
    last_score = 0.0
    errors: List[str] = []
    log_lines: List[str] = []

    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        run_id = f"{datetime.now().strftime('%Y%m%d')}_{iteration:04d}"
        desc_parts: List[str] = [f"iter{iteration}"]
        status = "keep"
        score = 0.0

        try:
            mining_results = miner.mine_with_feedback(
                iteration=iteration,
                top_candidates=feedback_state["top_candidates"],
                stability_mechanisms=feedback_state["stability_mechanisms"],
                target_properties=feedback_state["target_properties"],
                limitations=feedback_state["limitations"],
                num_candidates=12,
            )
            candidates = list(mining_results.get("material_candidates", []))

            if mutator:
                try:
                    from insulin_ai.mutation import feedback_guided_mutation

                    if iteration > 1 and feedback_state.get("high_performer_psmiles"):
                        mutated = feedback_guided_mutation(
                            feedback_state,
                            library_size=mutation_size,
                            feedback_fraction=0.7,
                            random_seed=42 + iteration,
                        )
                    else:
                        mutated = mutator.generate_library(library_size=mutation_size)
                    candidates.extend(mutated)
                    desc_parts.append(f"+{len(mutated)}mut")
                except Exception as e:
                    log_lines.append(f"mutation skip: {e}")

            # Only evaluate candidates that have valid PSMILES (speed)
            with_psmiles = [
                c
                for c in candidates
                if "[*]" in str(c.get("chemical_structure") or c.get("psmiles") or "")
            ]
            if not with_psmiles and candidates:
                # No PSMILES from literature — still run mutator-only eval if we had mutated
                with_psmiles = [
                    c
                    for c in candidates
                    if "[*]" in str(c.get("chemical_structure") or "")
                ]
            to_eval = with_psmiles[:max_eval_per_iteration] if with_psmiles else candidates[:max_eval_per_iteration]

            if not to_eval:
                status = "discard"
                desc_parts.append("no_psmiles_candidates")
                score = -1.0
            else:
                md_results = md_sim.evaluate_candidates(to_eval, max_candidates=len(to_eval))
                score = discovery_score(md_results)
                desc_parts.append(f"score={score:.2f}")
                desc_parts.append(f"hp={len(md_results.get('high_performers', []))}")
                feedback_state = miner._update_feedback_state(
                    md_results, feedback_state, to_eval
                )  # type: ignore[attr-defined]

            iterations_run += 1
            last_score = score

            # Persist iteration state for resume
            state_dir = Path(root) / "discovery_state"
            state_dir.mkdir(parents=True, exist_ok=True)
            with open(state_dir / f"autoresearch_iteration_{iteration}.json", "w", encoding="utf-8") as sf:
                json.dump(
                    {
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                        "score": score,
                        "feedback_state": {k: v for k, v in feedback_state.items() if k != "target_properties"},
                        "run_id": run_id,
                    },
                    sf,
                    indent=2,
                    default=str,
                )

        except Exception as e:
            status = "crash"
            score = 0.0
            errors.append(f"{run_id}: {e}\n{traceback.format_exc()}")
            desc_parts.append(str(e)[:200])

        mem = _memory_gb()
        _append_tsv(tsv_path, run_id, score, mem, status, " ".join(desc_parts))
        log_lines.append(f"{run_id}\t{score}\t{status}\t{' '.join(desc_parts)}")

        if time.monotonic() >= deadline:
            break

    summary = {
        "iterations_run": iterations_run,
        "tsv_path": str(tsv_path),
        "last_score": last_score,
        "errors": errors,
        "log": log_lines,
    }
    if log_json_path:
        lp = Path(log_json_path)
        if not lp.is_absolute():
            lp = Path(root) / lp
        lp.parent.mkdir(parents=True, exist_ok=True)
        with open(lp, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Autoresearch-style autonomous materials discovery")
    p.add_argument("--budget-minutes", type=float, default=60.0)
    p.add_argument(
        "--results-tsv",
        default="discovery_state/autoresearch_results.tsv",
        help="TSV log path (relative to repo root unless absolute)",
    )
    p.add_argument("--root", default="", help="Repo root (default: auto)")
    p.add_argument("--md-steps", type=int, default=5000)
    p.add_argument("--max-eval", type=int, default=8)
    p.add_argument("--log-json", default="", help="Write summary JSON here when done")
    args = p.parse_args()
    root = _ensure_paths(args.root or None)
    summary = run_autonomous_discovery_loop(
        budget_minutes=args.budget_minutes,
        results_tsv=args.results_tsv,
        root=root,
        md_steps=args.md_steps,
        max_eval_per_iteration=args.max_eval,
        log_json_path=args.log_json or None,
    )
    print(json.dumps(summary, indent=2))
    sys.exit(0 if not summary.get("errors") else 1)


if __name__ == "__main__":
    main()
