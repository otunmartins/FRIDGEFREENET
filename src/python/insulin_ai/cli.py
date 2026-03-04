#!/usr/bin/env python3
"""
Insulin AI – CLI Materials Discovery Platform

Designed for OpenCode.ai, Claude Code, and terminal-based AI coding agents.
The CLI is the primary interface for materials discovery; no web UI required.

Usage:
    insulin-ai discover --iterations 2              # Full feedback loop
    insulin-ai mine "hydrogels insulin stabilization"  # Literature only
    insulin-ai evaluate "[*]OCC[*]" "[*]CC[*]"      # MD/proxy evaluation
    insulin-ai status                                # System status
"""

import argparse
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "python"))


def cmd_discover(args):
    """Run active learning: literature → evaluate → feedback → iterate."""
    from iterative_literature_mining import IterativeLiteratureMiner

    md_sim = None
    if not args.no_md:
        try:
            from insulin_ai.simulation import MDSimulator
            md_sim = MDSimulator(n_steps=5000)
        except ImportError:
            print("⚠️ MD unavailable, using literature-only mode")

    miner = IterativeLiteratureMiner()
    results = miner.run_active_learning_cycle(
        max_iterations=args.iterations,
        md_simulator=md_sim,
        generative_model=None,
    )
    print("\n" + "=" * 50)
    for i, r in enumerate(results):
        n = len(r.get("material_candidates", []))
        md_ev = r.get("md_evaluation", {})
        perf = len(md_ev.get("high_performers", [])) if md_ev else 0
        print(f"Iteration {i+1}: {n} candidates, {perf} high performers")


def cmd_mine(args):
    """Literature mining only – no MD evaluation."""
    from iterative_literature_mining import IterativeLiteratureMiner

    miner = IterativeLiteratureMiner()
    results = miner.mine_with_feedback(
        iteration=1,
        num_candidates=args.num_candidates,
    )
    n = len(results.get("material_candidates", []))
    print(f"\nFound {n} material candidates")
    for c in results.get("material_candidates", [])[:args.show]:
        name = c.get("material_name", "?")
        comp = c.get("material_composition", "") or ""
        line = f"  - {name}: {comp[:60]}..." if len(comp) > 60 else f"  - {name}: {comp}"
        print(line)
    print(f"\nResults saved to iterative_results/")


def cmd_evaluate(args):
    """Evaluate PSMILES candidates (MD or RDKit proxy)."""
    from insulin_ai.simulation import MDSimulator

    candidates = [{"material_name": f"Candidate_{i}", "chemical_structure": p} for i, p in enumerate(args.psmiles)]
    sim = MDSimulator(n_steps=args.steps)
    result = sim.evaluate_candidates(candidates, max_candidates=len(candidates))
    print("High performers:", result["high_performers"])
    print("Mechanisms:", result["effective_mechanisms"])
    print("Problematic:", result["problematic_features"])
    for name, props in result.get("property_analysis", {}).items():
        print(f"  {name}: {props}")


def cmd_status(args):
    """Print system status (literature, MD, proxy)."""
    print("Insulin AI – Materials Discovery Platform (CLI)")
    print("=" * 50)
    try:
        from insulin_ai.simulation import MDSimulator
        sim = MDSimulator()
        md_ok = sim.runner is not None
        print(f"MD Simulation: {'OpenMM+PME' if md_ok else 'RDKit proxy'} (CPU)")
    except Exception:
        print("MD Simulation: unavailable")
    try:
        from literature_mining_system import MaterialsLiteratureMiner
        print("Literature Mining: available (needs Ollama + Semantic Scholar)")
    except ImportError:
        print("Literature Mining: import error")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        prog="insulin-ai",
        description="Insulin AI Materials Discovery – CLI for OpenCode.ai / Claude Code",
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # discover
    p_discover = sub.add_parser("discover", help="Run active learning feedback loop")
    p_discover.add_argument("--iterations", "-n", type=int, default=2)
    p_discover.add_argument("--no-md", action="store_true", help="Literature only, no MD")
    p_discover.set_defaults(func=cmd_discover)

    # mine
    p_mine = sub.add_parser("mine", help="Literature mining only")
    p_mine.add_argument("--num-candidates", type=int, default=15)
    p_mine.add_argument("--show", type=int, default=10, help="Show first N candidates")
    p_mine.set_defaults(func=cmd_mine)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate PSMILES with MD/proxy")
    p_eval.add_argument("psmiles", nargs="+", help="PSMILES strings")
    p_eval.add_argument("--steps", type=int, default=5000)
    p_eval.set_defaults(func=cmd_evaluate)

    # status
    p_status = sub.add_parser("status", help="System status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
