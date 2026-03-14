#!/usr/bin/env python3
"""
Insulin AI MCP Server – Materials Discovery Tools for OpenCode

Consolidated: literature mining, PaperQA2 RAG, PSMILES, MD, PubMed, arXiv,
Semantic Scholar, web search. Single MCP server for materials discovery.
"""

import os
import sys
import json
import time
import subprocess
import traceback
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    requests = None

# Project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "insulin-ai-materials",
    instructions="Materials discovery: literature mining (incl. PaperQA2 RAG), PSMILES, MD, run_autonomous_discovery (overnight autoresearch loop), PubMed, arXiv, Semantic Scholar, web search.",
)


# --- PaperQA2 helpers (optional; paper-qa may not be installed) ---
def _paper_qa_available():
    try:
        import paperqa
        return True
    except ImportError:
        return False


def _paper_qa_settings():
    from paper_qa_config import get_paper_qa_settings
    return get_paper_qa_settings()


def _paper_qa_index_status():
    """Check index status. Returns dict with ready, message, etc."""
    if not _paper_qa_available():
        return {"ready": False, "message": "paper-qa not installed. pip install paper-qa"}
    try:
        from pathlib import Path
        settings = _paper_qa_settings()
        paper_dir = Path(settings.agent.index.paper_directory)
        if not paper_dir.is_dir():
            return {"ready": False, "message": f"Paper dir not found: {paper_dir}"}
        total = sum(1 for f in paper_dir.rglob("*") if f.suffix.lower() == ".pdf")
        if total == 0:
            return {"ready": False, "message": f"No PDFs in {paper_dir}. Add papers and run index_papers."}
        # Check for built index (paper-qa stores in index_directory / get_index_name())
        index_dir = Path(settings.agent.index.index_directory)
        index_name = settings.get_index_name()
        manifest_path = index_dir / index_name / "files.zip"
        if not manifest_path.exists():
            return {"ready": total <= 10, "message": f"0/{total} indexed. Run index_papers first."}
        try:
            import pickle
            import zlib
            manifest = pickle.loads(zlib.decompress(manifest_path.read_bytes()))
            errored = sum(1 for v in manifest.values() if v == "ERROR")
            indexed = len(manifest) - errored
            unindexed = max(0, total - len(manifest))
            ready = unindexed <= 10 and errored == 0
            msg = f"{indexed}/{total} indexed"
            if errored:
                msg += f", {errored} errors"
            if unindexed:
                msg += f", {unindexed} unindexed"
            return {"ready": ready, "message": msg}
        except Exception:
            return {"ready": False, "message": f"Index may be incomplete. Try running index_papers."}
    except Exception as e:
        return {"ready": False, "message": str(e)}


@mcp.tool()
def mine_literature(
    query: str = "hydrogels insulin stabilization transdermal",
    max_candidates: int = 15,
    iteration: int = 1,
    top_candidates: str = "",
    stability_mechanisms: str = "",
    limitations: str = "",
    use_paper_qa: bool = True,
) -> str:
    """
    Mine scientific literature for insulin delivery materials.
    Combines API mining (Semantic Scholar) with PaperQA2 deep reading when papers are indexed.
    Returns material NAMES (e.g. chitosan, hydrogel, PEG). Translate to PSMILES before evaluate_psmiles.

    For iteration 2+, pass feedback from the previous iteration:
      top_candidates: comma-separated high performers (e.g. "chitosan,PEG")
      stability_mechanisms: comma-separated mechanisms (e.g. "hydrogen bonding,hydrophobic")
      limitations: comma-separated problems to avoid (e.g. "high_crystallinity")
    use_paper_qa: if True and papers are indexed, appends PaperQA2 synthesis to results.
    """
    out = []
    # Optional: PaperQA2 deep reading first (if indexed)
    if use_paper_qa and _paper_qa_available():
        status = _paper_qa_index_status()
        if status.get("ready"):
            try:
                import asyncio
                from paperqa import agent_query
                pqa_query = f"What polymer materials and stabilization mechanisms are effective for insulin delivery or transdermal patches? Query focus: {query}"
                if top_candidates or stability_mechanisms:
                    pqa_query += f". Prior high performers: {top_candidates or 'none'}. Mechanisms: {stability_mechanisms or 'none'}."
                settings = _paper_qa_settings()
                response = asyncio.run(agent_query(query=pqa_query, settings=settings))
                if response.session.formatted_answer:
                    out.append("--- PaperQA2 synthesis (from your indexed PDFs) ---")
                    out.append(response.session.formatted_answer)
                    out.append("")
            except Exception as e:
                out.append(f"(PaperQA2 skipped: {e})")
                out.append("")
        elif status.get("message"):
            out.append(f"(PaperQA2: {status['message']})")
            out.append("")

    try:
        from iterative_literature_mining import IterativeLiteratureMiner
        miner = IterativeLiteratureMiner()
        top = [s.strip() for s in top_candidates.split(",") if s.strip()] or None
        mechs = [s.strip() for s in stability_mechanisms.split(",") if s.strip()] or None
        lims = [s.strip() for s in limitations.split(",") if s.strip()] or None
        results = miner.mine_with_feedback(
            iteration=iteration,
            top_candidates=top,
            stability_mechanisms=mechs,
            limitations=lims,
            num_candidates=max_candidates,
        )
        candidates = results.get("material_candidates", [])
        names = [c.get("material_name") or c.get("material_composition", "") for c in candidates if c.get("material_name") or c.get("material_composition")]
        out.append(f"Iteration {iteration}: Found {len(names)} material candidates. Translate to PSMILES before evaluation.")
        out.append("")
        out.append("Material names (you translate to PSMILES):")
        for i, n in enumerate(names[:15], 1):
            out.append(f"  {i}. {n}")
        out.append("")
        out.append("Example PSMILES: PEG=[*]OCC[*], polyethylene=[*]CC[*], chitosan=[*]CC([*])OC1OC(C)C(O)C(O)C1O")
        return "\n".join(out)
    except Exception as e:
        return "\n".join(out) + f"\n\nError (mining): {e}" if out else f"Error: {e}"


@mcp.tool()
async def paper_qa(question: str) -> str:
    """
    Deep synthesis across indexed PDFs with citations (PaperQA2 RAG).
    Ask a question; get an answer from your papers/ directory with inline citations.
    Best for: mechanisms, material comparisons, literature-backed validation.
    If index incomplete, run index_papers first. Can take 30–90 seconds.
    """
    if not _paper_qa_available():
        return "paper-qa not installed. pip install paper-qa"
    status = _paper_qa_index_status()
    if not status.get("ready"):
        return f"Index incomplete: {status.get('message', 'Run index_papers first.')}"
    try:
        from paperqa import agent_query
        settings = _paper_qa_settings()
        response = await agent_query(query=question, settings=settings)
        return response.session.formatted_answer or f"PaperQA could not answer (status: {response.status})"
    except Exception as e:
        return f"PaperQA error: {e}"


@mcp.tool()
def paper_qa_index_status() -> str:
    """Check PaperQA2 index status (indexed/unindexed counts). Run index_papers to build."""
    status = _paper_qa_index_status()
    return status.get("message", str(status))


@mcp.tool()
def index_papers() -> str:
    """Build PaperQA2 search index over papers in papers/. Run once before using paper_qa. May take minutes for many PDFs."""
    if not _paper_qa_available():
        return "paper-qa not installed. pip install paper-qa"
    try:
        from paper_qa_config import build_index
        return build_index()
    except Exception as e:
        return f"Index error: {e}"


@mcp.tool()
def lookup_material(material_name: str, max_results: int = 5) -> str:
    """
    Quick lookup for polymer/structure info when translating material names to PSMILES.
    Searches PubMed (API-free) for papers about the material's repeat unit, SMILES, or structure.
    Use when unsure about a material's PSMILES; then validate_psmiles your translation.
    """
    if not material_name or not material_name.strip():
        return "Error: provide a material name (e.g. chitosan, PLGA, PEG)."
    if not requests:
        return "Error: requests library required for lookup."
    query = f"{material_name.strip()} polymer repeat unit SMILES structure"
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        params = {
            "term": query,
            "db": "pubmed",
            "retmax": min(max_results, 10),
            "retmode": "json",
            "tool": "insulin-ai",
            "email": "research@example.com",
        }
        time.sleep(0.35)
        r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=15)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return f"No PubMed hits for '{material_name}'. Try a different query or use your chemistry knowledge."
        params2 = {"id": ",".join(ids), "db": "pubmed", "rettype": "xml", "tool": "insulin-ai", "email": "research@example.com"}
        time.sleep(0.35)
        r2 = requests.get(f"{base}/efetch.fcgi", params=params2, timeout=15)
        if r2.status_code != 200 or not r2.text.strip():
            return "Could not fetch abstracts."
        root = ET.fromstring(r2.content)
        lines = [f"PubMed lookup for '{material_name}' (query: {query})", ""]
        for art in root.findall(".//PubmedArticle")[:max_results]:
            title_el = art.find(".//ArticleTitle")
            abs_el = art.find(".//AbstractText")
            title = (title_el.text or "") if title_el is not None else ""
            abstract = (abs_el.text or "") if abs_el is not None else ""
            lines.append(f"Title: {title[:100]}...")
            lines.append(f"Abstract: {abstract[:500]}..." if len(abstract) > 500 else f"Abstract: {abstract}")
            lines.append("")
        return "\n".join(lines).strip()
    except Exception as e:
        return f"Lookup error: {e}"


@mcp.tool()
def validate_psmiles(psmiles: str) -> str:
    """
    Validate and optionally canonicalize a PSMILES string.
    Returns JSON: {valid: bool, canonical?: str, error?: str}.
    Use before evaluate_psmiles when the agent translated material names to PSMILES.
    """
    try:
        from insulin_ai.material_mappings import validate_psmiles as _validate
        return json.dumps(_validate(psmiles), indent=2)
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


@mcp.tool()
def evaluate_psmiles(psmiles_list: str) -> str:
    """
    Evaluate PSMILES polymer structures (MD / OpenMM).
    Input: comma-separated PSMILES, e.g. "[*]OCC[*], [*]CC[*]"
    Returns high performers, mechanisms, problematic features.
    """
    try:
        from insulin_ai.simulation import MDSimulator
        parts = [p.strip().strip('"') for p in psmiles_list.split(",")]
        candidates = [{"material_name": f"Candidate_{i}", "chemical_structure": p} for i, p in enumerate(parts)]
        sim = MDSimulator(n_steps=5000)
        result = sim.evaluate_candidates(candidates, max_candidates=len(candidates))
        try:
            from insulin_ai.simulation.scoring import discovery_score

            _score = discovery_score(result)
        except Exception:
            _score = None
        out = {
            "high_performers": result["high_performers"],
            "effective_mechanisms": result["effective_mechanisms"],
            "problematic_features": result["problematic_features"],
        }
        if _score is not None:
            out["discovery_score"] = round(_score, 4)
        return json.dumps(out, indent=2)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def mutate_psmiles(library_size: int = 10, feedback_json: str = "") -> str:
    """
    Generate mutated PSMILES candidates via cheminformatics.
    Optionally pass feedback JSON with high_performer_psmiles, problematic_psmiles for feedback-guided mutation.
    Returns JSON list of candidates with material_name, chemical_structure.
    """
    try:
        import json as _json
        from insulin_ai.mutation import MaterialMutator, feedback_guided_mutation
        feedback = {}
        if feedback_json:
            feedback = _json.loads(feedback_json)
        if feedback.get("high_performer_psmiles"):
            cands = feedback_guided_mutation(feedback, library_size=library_size, random_seed=42)
        else:
            mutator = MaterialMutator(random_seed=42)
            cands = mutator.generate_library(library_size=library_size)
        return _json.dumps([
            {"material_name": c["material_name"], "chemical_structure": c["chemical_structure"]}
            for c in cands
        ], indent=2)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def run_autonomous_discovery(
    budget_minutes: float = 60.0,
    run_in_background: bool = True,
    results_tsv_path: str = "discovery_state/autoresearch_results.tsv",
    md_steps: int = 5000,
    max_eval_per_iteration: int = 8,
) -> str:
    """
    Autoresearch-style autonomous loop: literature mine → mutate → MD evaluate → score → TSV log,
    repeated until budget_minutes elapses. Same philosophy as Karpathy autoresearch (overnight runs).

    When run_in_background=True (default), spawns a subprocess so the MCP call returns immediately
    (avoids timeouts). Check discovery_state/autoresearch_subprocess.log and results TSV for progress.

    TSV columns: run_id, score, memory_gb, status (keep|discard|crash), description.
    Foreground mode runs inline (may take hours; use only for short budgets).
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    tsv = results_tsv_path
    if not os.path.isabs(tsv):
        tsv = os.path.join(ROOT, results_tsv_path)
    log_json = os.path.join(STATE_DIR, "autoresearch_summary.json")
    log_out = os.path.join(STATE_DIR, "autoresearch_subprocess.log")
    script = os.path.join(ROOT, "scripts", "run_autonomous_discovery.py")
    env = os.environ.copy()
    env["INSULIN_AI_ROOT"] = ROOT

    if run_in_background:
        if not os.path.isfile(script):
            return json.dumps(
                {"error": f"Script not found: {script}", "hint": "Run from insulin-ai repo root."}
            )
        cmd = [
            sys.executable,
            script,
            "--budget-minutes",
            str(budget_minutes),
            "--results-tsv",
            results_tsv_path,
            "--md-steps",
            str(md_steps),
            "--max-eval",
            str(max_eval_per_iteration),
            "--log-json",
            log_json,
        ]
        try:
            log_f = open(log_out, "a", encoding="utf-8")
            log_f.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} budget={budget_minutes}m ---\n")
            log_f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log_f.close()
            return json.dumps(
                {
                    "status": "started_background",
                    "pid": proc.pid,
                    "budget_minutes": budget_minutes,
                    "results_tsv": tsv,
                    "subprocess_log": log_out,
                    "summary_json_when_done": log_json,
                    "note": "Tail subprocess_log for progress; TSV appends one row per iteration.",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    # Foreground: short runs only
    try:
        sys.path.insert(0, os.path.join(ROOT, "src", "python"))
        from insulin_ai.autonomous_discovery import run_autonomous_discovery_loop

        summary = run_autonomous_discovery_loop(
            budget_minutes=budget_minutes,
            results_tsv=results_tsv_path,
            root=ROOT,
            md_steps=md_steps,
            max_eval_per_iteration=max_eval_per_iteration,
            log_json_path=log_json,
        )
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def get_materials_status() -> str:
    """Get status of materials discovery system (MD, literature, PaperQA2, mutation)."""
    lines = ["Insulin AI Materials Discovery Status"]
    try:
        from insulin_ai.simulation import MDSimulator
        sim = MDSimulator()
        lines.append(f"MD Simulation: {'insulin + polymer (implicit solvent)' if sim.runner else 'unavailable'} (CPU)")
    except Exception:
        lines.append("MD Simulation: unavailable")
    try:
        from insulin_ai.mutation import MaterialMutator
        lines.append("Mutation: available (cheminformatics)")
    except ImportError:
        lines.append("Mutation: unavailable (pip install psmiles)")
    try:
        from literature_mining_system import MaterialsLiteratureMiner
        lines.append("Literature Mining: available")
    except ImportError:
        lines.append("Literature Mining: import error (needs Ollama)")
    pqa = _paper_qa_index_status()
    lines.append(f"PaperQA2: {pqa.get('message', 'unavailable')}")
    return "\n".join(lines)


STATE_DIR = os.path.join(ROOT, "discovery_state")


@mcp.tool()
def save_discovery_state(
    iteration: int,
    feedback_json: str,
    query_used: str = "",
    notes: str = "",
) -> str:
    """
    Persist discovery state after an iteration so you can resume or review later.

    Args:
        iteration: Iteration number (1, 2, 3, ...)
        feedback_json: JSON string with keys: high_performers, effective_mechanisms,
                       problematic_features, high_performer_psmiles, problematic_psmiles
        query_used: The literature query used in this iteration
        notes: Free-text notes (user instructions, observations)

    Writes to discovery_state/iteration_N.json.
    """
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        feedback = json.loads(feedback_json) if feedback_json else {}
    except json.JSONDecodeError as e:
        return f"Error parsing feedback_json: {e}"

    from datetime import datetime

    state = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "query_used": query_used,
        "notes": notes,
        "feedback": feedback,
    }
    path = os.path.join(STATE_DIR, f"iteration_{iteration}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return f"State saved to {path}"


@mcp.tool()
def load_discovery_state(iteration: int = 0) -> str:
    """
    Load discovery state from a previous iteration.

    Args:
        iteration: Iteration number to load. Pass 0 to load the latest.

    Returns JSON with: iteration, timestamp, query_used, notes, feedback.
    """
    if not os.path.isdir(STATE_DIR):
        return "No discovery_state directory found. Run an iteration first."

    if iteration > 0:
        path = os.path.join(STATE_DIR, f"iteration_{iteration}.json")
        if not os.path.isfile(path):
            return f"No state file for iteration {iteration}."
    else:
        files = sorted(
            [f for f in os.listdir(STATE_DIR) if f.startswith("iteration_") and f.endswith(".json")]
        )
        if not files:
            return "No iteration state files found."
        path = os.path.join(STATE_DIR, files[-1])

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Literature search (folded from lit-* servers) ---
@mcp.tool()
def semantic_scholar_search(query: str, max_results: int = 20) -> str:
    """Search Semantic Scholar. No API key required (rate limited). Set SEMANTIC_SCHOLAR_API_KEY for higher limits."""
    try:
        from semantic_scholar_client import SemanticScholarClient
        client = SemanticScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
        results = client.search_papers(query=query, limit=max_results)
        papers = results.get("data", [])
        lines = [f"Found {len(papers)} papers."]
        for i, p in enumerate(papers[:10], 1):
            title = p.get("title", "")[:80]
            year = p.get("year", "")
            lines.append(f"{i}. {title}... ({year})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def _pubmed_esearch(query: str, retmax: int) -> list:
    if not requests:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {"term": query, "db": "pubmed", "retmax": retmax, "retmode": "json", "tool": "insulin-ai", "email": "research@example.com"}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    time.sleep(0.35)
    r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _pubmed_get_abstracts(ids: list) -> list:
    if not requests or not ids:
        return []
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    params = {"id": ",".join(str(i) for i in ids), "db": "pubmed", "rettype": "xml", "tool": "insulin-ai", "email": "research@example.com"}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    time.sleep(0.35)
    r = requests.get(f"{base}/efetch.fcgi", params=params, timeout=15)
    if r.status_code != 200 or not r.text.strip():
        return []
    root = ET.fromstring(r.content)
    out = []
    for art in root.findall(".//PubmedArticle"):
        aid = art.find(".//PMID")
        title = art.find(".//ArticleTitle")
        abstract = art.find(".//AbstractText")
        out.append({
            "pmid": aid.text if aid is not None else "",
            "title": (title.text or "") if title is not None else "",
            "abstract": (abstract.text or "") if abstract is not None else "",
        })
    return out


@mcp.tool()
def pubmed_search(query: str, max_results: int = 20) -> str:
    """Search PubMed for papers. No API key required. Set NCBI_API_KEY for higher rate limit."""
    try:
        ids = _pubmed_esearch(query, retmax=max_results)
        if not ids:
            return "No papers found."
        papers = _pubmed_get_abstracts(ids[:max_results])
        lines = [f"Found {len(papers)} papers."]
        for i, p in enumerate(papers[:10], 1):
            lines.append(f"{i}. {p['title'][:80]}... (PMID {p['pmid']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def arxiv_search(query: str, max_results: int = 20) -> str:
    """Search arXiv for papers. No API key required."""
    if not requests:
        return "Error: requests required"
    try:
        params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}
        r = requests.get("https://export.arxiv.org/api/query", params=params, headers={"User-Agent": "insulin-ai/1.0 (research@example.com)"}, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        lines = [f"Found {len(entries)} papers."]
        for i, entry in enumerate(entries[:10], 1):
            title = entry.find("atom:title", ns)
            aid = entry.find("atom:id", ns)
            t = (title.text or "").replace("\n", " ").strip() if title is not None else ""
            id_text = (aid.text or "").split("/")[-1] if aid is not None else ""
            lines.append(f"{i}. {t[:80]}... ({id_text})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. No API key. Use for material structures, PSMILES, polymer repeat units."""
    try:
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=min(max_results, 10)))
    except ImportError:
        return "Error: pip install duckduckgo-search"
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return f"No results for: {query}"
    lines = [f"Web search: {query}", ""]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "")[:80]
        body = (r.get("body") or r.get("snippet", ""))[:400]
        url = r.get("href") or r.get("link", "")
        lines.append(f"{i}. {title}\n   {body}...\n   {url}")
        lines.append("")
    return "\n".join(lines).strip()


# --- PSMILES (Ramprasad, folded from psmiles-ramprasad) ---
def _psmiles_check():
    try:
        from psmiles import PolymerSmiles
        return None
    except ImportError:
        return "psmiles not installed. Use insulin-ai-sim env or: pip install git+https://github.com/FermiQ/psmiles.git"


@mcp.tool()
def psmiles_canonicalize(psmiles: str) -> str:
    """Canonicalize PSMILES (Ramprasad-Group). Returns unique representation."""
    err = _psmiles_check()
    if err:
        return err
    try:
        from psmiles import PolymerSmiles
        return str(PolymerSmiles(psmiles).canonicalize())
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_dimerize(psmiles: str, star_index: int = 0) -> str:
    """Dimerize PSMILES at connection point. star_index: 0 or 1 for which [*]."""
    err = _psmiles_check()
    if err:
        return err
    try:
        from psmiles import PolymerSmiles
        return str(PolymerSmiles(psmiles).dimerize(star_index=star_index))
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_fingerprint(psmiles: str, fingerprint_type: str = "rdkit") -> str:
    """Get fingerprint for PSMILES. Types: rdkit, mordred, polyBERT, morgan."""
    err = _psmiles_check()
    if err:
        return err
    try:
        from psmiles import PolymerSmiles
        fp = PolymerSmiles(psmiles).descriptor(fingerprint_type)
        if hasattr(fp, "tolist"):
            return json.dumps(fp.tolist()[:20])
        return str(fp)[:500]
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_similarity(psmiles1: str, psmiles2: str) -> str:
    """Compute similarity between two PSMILES (Ramprasad-Group)."""
    err = _psmiles_check()
    if err:
        return err
    try:
        from psmiles import PolymerSmiles
        sim = PolymerSmiles(psmiles1).similarity(PolymerSmiles(psmiles2))
        return f"Similarity: {sim}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
