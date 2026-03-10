#!/usr/bin/env python3
"""
CPU-Only MD Benchmark

Measures OpenMM + PME performance on CPU for polymer screening.
Serves as baseline for optimization and regression testing.
"""

import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

# Benchmark parameters (REVAMP_PLAN.md)
N_STEPS = 10000  # Short run for benchmark
TIMESTEP_FS = 2.0
PSMILES_CANDIDATES = ["[*]OCC[*]", "[*]CC[*]", "[*]CC(C)[*]"]


def benchmark_md_evaluate(n_candidates: int = 3) -> float:
    """Benchmark MD evaluator (insulin + polymer). Returns -1 if MD unavailable."""
    from insulin_ai.simulation import MDSimulator

    sim = MDSimulator()
    if sim.runner is None:
        return -1.0
    candidates = [{"material_name": f"Polymer_{i}", "chemical_structure": "[*]OCC[*]"}
                 for i in range(n_candidates)]

    t0 = time.perf_counter()
    _ = sim.evaluate_candidates(candidates, max_candidates=n_candidates)
    return time.perf_counter() - t0


def benchmark_openmm_single(psmiles: str) -> float:
    """Benchmark single PSMILES OpenMM run (if MD works)."""
    try:
        from insulin_ai.simulation.openmm_runner import OpenMMRunner
        
        runner = OpenMMRunner(platform_name="CPU")
        t0 = time.perf_counter()
        result = runner.run(psmiles, n_steps=N_STEPS, minimize_steps=100)
        elapsed = time.perf_counter() - t0
        return elapsed if result else -1.0
    except Exception:
        return -1.0


def main():
    print("=" * 60)
    print("Insulin AI - CPU-Only MD Benchmark")
    print("=" * 60)
    
    # MD evaluate (requires OpenMM + GAFF)
    t_md = benchmark_md_evaluate(3)
    status = f"{t_md:.3f} s" if t_md >= 0 else "UNAVAILABLE (OpenMM/GAFF required)"
    print(f"\nMD insulin+polymer (3 candidates): {status}")
    
    # OpenMM (may fail without openff-toolkit)
    print("\nOpenMM single runs:")
    for psmiles in PSMILES_CANDIDATES:
        t = benchmark_openmm_single(psmiles)
        status = f"{t:.3f} s" if t >= 0 else "FAILED (GAFF unavailable)"
        print(f"  {psmiles}: {status}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
