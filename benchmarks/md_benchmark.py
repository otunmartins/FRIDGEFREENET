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


def benchmark_rdkit_proxy(n_candidates: int = 10) -> float:
    """Benchmark RDKit proxy evaluator (fallback path)."""
    from insulin_ai.simulation import MDSimulator
    
    sim = MDSimulator()
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
    
    # RDKit proxy (always works)
    t_proxy = benchmark_rdkit_proxy(5)
    print(f"\nRDKit Proxy (5 candidates): {t_proxy:.3f} s")
    
    # OpenMM (may fail without openff-toolkit)
    print("\nOpenMM single runs:")
    for psmiles in PSMILES_CANDIDATES:
        t = benchmark_openmm_single(psmiles)
        status = f"{t:.3f} s" if t >= 0 else "FAILED (GAFF unavailable)"
        print(f"  {psmiles}: {status}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
