#!/usr/bin/env python3
"""
Insulin AI MCP Server – Materials Discovery Tools for OpenCode

Exposes literature mining, PSMILES evaluation, and active learning cycle
to OpenCode's materials-discovery agent via Model Context Protocol.
"""

import os
import sys
import json

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("insulin-ai-materials", instructions="Tools for insulin patch polymer materials discovery: literature mining, PSMILES evaluation, MD simulation.")


@mcp.tool
def mine_literature(query: str = "hydrogels insulin stabilization transdermal", max_candidates: int = 15) -> str:
    """
    Mine scientific literature for insulin delivery materials.
    Use Semantic Scholar; returns material candidates with composition, stability, references.
    """
    try:
        from iterative_literature_mining import IterativeLiteratureMiner
        miner = IterativeLiteratureMiner()
        results = miner.mine_with_feedback(iteration=1, num_candidates=max_candidates)
        candidates = results.get("material_candidates", [])
        out = [f"Found {len(candidates)} candidates."]
        for i, c in enumerate(candidates[:10]):
            out.append(f"{i+1}. {c.get('material_name', '?')}: {c.get('material_composition', '')[:80]}")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool
def evaluate_psmiles(psmiles_list: str) -> str:
    """
    Evaluate PSMILES polymer structures (MD or RDKit proxy).
    Input: comma-separated PSMILES, e.g. "[*]OCC[*], [*]CC[*]"
    Returns high performers, mechanisms, problematic features.
    """
    try:
        from insulin_ai.simulation import MDSimulator
        parts = [p.strip().strip('"') for p in psmiles_list.split(",")]
        candidates = [{"material_name": f"Candidate_{i}", "chemical_structure": p} for i, p in enumerate(parts)]
        sim = MDSimulator(n_steps=5000)
        result = sim.evaluate_candidates(candidates, max_candidates=len(candidates))
        return json.dumps({
            "high_performers": result["high_performers"],
            "effective_mechanisms": result["effective_mechanisms"],
            "problematic_features": result["problematic_features"],
        }, indent=2)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool
def run_discover_cycle(iterations: int = 2, use_md: bool = True) -> str:
    """
    Run the full active learning cycle: literature -> evaluate -> feedback -> iterate.
    Results saved to cycle_results/, iterative_results/.
    """
    try:
        from iterative_literature_mining import IterativeLiteratureMiner
        from insulin_ai.simulation import MDSimulator
        md_sim = MDSimulator(n_steps=5000) if use_md else None
        miner = IterativeLiteratureMiner()
        results = miner.run_active_learning_cycle(max_iterations=iterations, md_simulator=md_sim, generative_model=None)
        out = [f"Completed {len(results)} iterations."]
        for i, r in enumerate(results):
            n = len(r.get("material_candidates", []))
            md_ev = r.get("md_evaluation", {})
            perf = len(md_ev.get("high_performers", [])) if md_ev else 0
            out.append(f"Iteration {i+1}: {n} candidates, {perf} high performers")
        return "\n".join(out)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool
def get_materials_status() -> str:
    """Get status of materials discovery system (MD simulation, literature mining)."""
    lines = ["Insulin AI Materials Discovery Status"]
    try:
        from insulin_ai.simulation import MDSimulator
        sim = MDSimulator()
        lines.append(f"MD Simulation: {'OpenMM+PME' if sim.runner else 'RDKit proxy'} (CPU)")
    except Exception:
        lines.append("MD Simulation: unavailable")
    try:
        from literature_mining_system import MaterialsLiteratureMiner
        lines.append("Literature Mining: available")
    except ImportError:
        lines.append("Literature Mining: import error (needs Ollama)")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
