"""GROMACS merged EM timing (requires gmx + acpype on PATH)."""

import time


def benchmark_gromacs_merged(psmiles: str) -> float:
    from insulin_ai.simulation.gromacs_complex import gmx_available, run_gromacs_merged_em

    if not gmx_available():
        return -1.0
    t0 = time.perf_counter()
    run_gromacs_merged_em(psmiles, n_repeats=2)
    return time.perf_counter() - t0


if __name__ == "__main__":
    ps = "[*]COC[*]"
    t = benchmark_gromacs_merged(ps)
    print(f"GROMACS merged EM [{ps}]: {t:.3f} s" if t >= 0 else "gmx not on PATH")
