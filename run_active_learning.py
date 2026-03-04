#!/usr/bin/env python3
"""
Run Active Learning Cycle with CPU-Only MD Feedback

Demonstrates the full feedback loop:
1. Literature mining → material candidates
2. MD simulation (OpenMM + PME, CPU) → evaluation
3. Feedback → dynamic queries for next iteration

Usage:
    python run_active_learning.py [--iterations N] [--no-md]

Without --no-md, requires: openmm, rdkit, openmmforcefields
"""

import argparse
import sys
import os

# Add src to path for insulin_ai package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "python"))

from iterative_literature_mining import IterativeLiteratureMiner


def get_md_simulator():
    """Get MDSimulator if dependencies available."""
    try:
        from insulin_ai.simulation import MDSimulator
        return MDSimulator(n_steps=5000, temperature=298.0)
    except ImportError as e:
        print(f"⚠️ MD simulation unavailable (optional): {e}")
        print("   Install: pip install openmm openmmforcefields")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run active learning with MD feedback")
    parser.add_argument("--iterations", type=int, default=2, help="Number of cycles")
    parser.add_argument("--no-md", action="store_true", help="Skip MD evaluation")
    args = parser.parse_args()

    md_simulator = None if args.no_md else get_md_simulator()

    print("🔄 Insulin AI Active Learning - CPU-Only Revamp")
    print("=" * 60)
    if md_simulator:
        print("✅ MD Simulator: OpenMM + PME (CPU)")
    else:
        print("📋 MD Simulator: disabled (literature-only mode)")

    miner = IterativeLiteratureMiner()
    results = miner.run_active_learning_cycle(
        max_iterations=args.iterations,
        md_simulator=md_simulator,
        generative_model=None,
    )

    print("\n" + "=" * 60)
    print("✅ Active learning cycle complete")
    for i, r in enumerate(results):
        n = len(r.get("material_candidates", []))
        md_ev = r.get("md_evaluation", {})
        perf = md_ev.get("high_performers", []) if md_ev else []
        print(f"   Iteration {i+1}: {n} candidates, {len(perf)} high performers")


if __name__ == "__main__":
    main()
