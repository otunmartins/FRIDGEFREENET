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
import shutil
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

from pathlib import Path
from typing import Any, List, Union

from mcp.server.fastmcp import FastMCP

from insulin_ai.run_paths import ENV_SESSION, new_session_dir, session_dir_from_env


def _normalize_psmiles_list_for_eval(psmiles_list: Union[str, List[Any], None]) -> List[str]:
    """
    Build a list of PSMILES strings from MCP arguments.

    Some clients send a comma-separated string (schema-native); others send a JSON array of
    strings. A few send a single string that is itself a JSON array. All are accepted so tool
    validation does not abort before the handler runs.
    """
    if psmiles_list is None:
        return []
    if isinstance(psmiles_list, list):
        out: List[str] = []
        for p in psmiles_list:
            if p is None:
                continue
            s = str(p).strip().strip('"').strip("'")
            if s:
                out.append(s)
        return out

    s = str(psmiles_list).strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    str(x).strip().strip('"').strip("'")
                    for x in parsed
                    if str(x).strip()
                ]
        except json.JSONDecodeError:
            pass
    return [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]


def _coerce_bool_flag(value: Any, default: bool = True) -> bool:
    """Accept bool or common string/int shapes from MCP clients."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("0", "false", "no", "off", ""):
            return False
        if v in ("1", "true", "yes", "on"):
            return True
    return default


def _coerce_single_psmiles_string(value: Union[str, List[Any], None]) -> str:
    """
    Normalize MCP ``psmiles`` for ``validate_psmiles`` (one repeat unit).

    Hosts sometimes send a JSON array with one element or mangle quoting; this
    avoids schema/type failures and empty tool results.
    """
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0]).strip().strip('"').strip("'")
    s = str(value).strip()
    if s.startswith("[") and "[*]" in s:
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) == 1:
                return str(parsed[0]).strip()
        except json.JSONDecodeError:
            pass
    return s


mcp = FastMCP(
    "insulin-ai-materials",
    instructions=(
        "Materials discovery: literature mining, PSMILES (see repo docs/PSMILES_GUIDE.md), "
        "validate_psmiles (functional_groups + name_consistency + cached PubChem monomer lookup; "
        "optional crosscheck_web+material_name for DuckDuckGo snippets), "
        "evaluate_psmiles (OpenMM merged minimize), render_psmiles_png (2D PNG for reports), "
        "compile_discovery_markdown_to_pdf (agent-written MD → PDF), "
        "write_discovery_summary_report (optional batch from saved JSON), "
        "save_session_transcript / import_chat_transcript_file (required each run: copy chat archive into runs/<session>/ only, never under .cursor/), "
        "PubMed, arXiv, Scholar, web search."
    ),
)


# --- PaperQA2 helpers (optional; paper-qa may not be installed) ---
def _paper_qa_available():
    try:
        import paperqa
        return True
    except ImportError:
        return False


def _paper_qa_settings():
    from insulin_ai.paper_qa_config import get_paper_qa_settings
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
    Literature: **Asta MCP** when ASTA_API_KEY is set (server-side); else Semantic Scholar REST.
    Optional PaperQA2 if indexed. You read abstracts and propose materials + PSMILES; then validate_psmiles / evaluate_psmiles.

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
        import os
        from insulin_ai.literature.literature_scholar_only import (
            format_mine_literature_text,
            run_scholar_mine,
        )

        run_dir = session_dir_from_env(Path(ROOT))
        top = [s.strip() for s in top_candidates.split(",") if s.strip()] or None
        mechs = [s.strip() for s in stability_mechanisms.split(",") if s.strip()] or None
        lims = [s.strip() for s in limitations.split(",") if s.strip()] or None
        asta_key = os.environ.get("ASTA_API_KEY")
        if asta_key:
            from insulin_ai.literature.literature_scholar_only import run_asta_mine

            results = run_asta_mine(
                asta_api_key=asta_key,
                base_query=query,
                iteration=iteration,
                top_candidates=top,
                stability_mechanisms=mechs,
                limitations=lims,
                run_dir=run_dir,
                num_candidates=max_candidates,
            )
        else:
            results = run_scholar_mine(
                api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
                base_query=query,
                iteration=iteration,
                top_candidates=top,
                stability_mechanisms=mechs,
                limitations=lims,
                run_dir=run_dir,
                num_candidates=max_candidates,
            )
        out.append(format_mine_literature_text(results))
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
        from insulin_ai.paper_qa_config import build_index
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
def validate_psmiles(
    psmiles: Union[str, List[Any]],
    material_name: str = "",
    crosscheck_web: bool = False,
) -> str:
    """
    Validate, annotate functional groups, and check name-structure consistency of a PSMILES.

    **Always returned:** ``valid``, ``canonical`` (when valid), and ``functional_groups``
    (RDKit SMARTS-based counts of carboxylic_acid, ester, ether, amine, amide, hydroxyl,
    aldehyde, ketone, aromatic, etc.).

    **When ``material_name`` is set:** ``name_consistency`` checks whether the name's
    implied chemistry (e.g. "acid" expects carboxylic_acid or ester) matches the actual
    functional groups. ``pubchem_lookup`` queries PubChem (cached per monomer in-process,
    bounded HTTP timeouts) for the monomer SMILES and Tanimoto similarity vs the repeat
    unit. If ``name_consistency.consistent`` is false, **fix the PSMILES before evaluating**.

    **When ``crosscheck_web`` is also true:** adds ``name_crosscheck`` with DuckDuckGo
    snippets (heuristic aid for manual comparison).
    """
    try:
        from insulin_ai.material_mappings import (
            annotate_functional_groups,
            check_name_structure_consistency,
            lookup_monomer_pubchem,
            validate_psmiles as _validate,
        )

        psm = _coerce_single_psmiles_string(psmiles)
        out = dict(_validate(psm))

        fg = annotate_functional_groups(psm)
        if fg.get("ok"):
            out["functional_groups"] = fg["groups"]
        else:
            out["functional_groups_error"] = fg.get("error", "unknown")

        name = (material_name or "").strip()
        if name:
            out["name_consistency"] = check_name_structure_consistency(name, psm)
            try:
                out["pubchem_lookup"] = lookup_monomer_pubchem(name, psm, timeout=5.0)
            except Exception as e:
                out["pubchem_lookup"] = {"ok": False, "error": str(e)}

        if crosscheck_web and name:
            q = f"{name} polymer repeat unit SMILES structure"
            raw = _ddg_text_results(q, max_results=5)
            snippets = []
            for r in raw:
                snippets.append(
                    {
                        "title": (r.get("title") or "")[:120],
                        "snippet": (r.get("body") or r.get("snippet", ""))[:500],
                        "url": r.get("href") or r.get("link", ""),
                    }
                )
            out["name_crosscheck"] = {
                "material_name": name,
                "query": q,
                "snippets": snippets,
                "disclaimer": (
                    "Web snippets are for human/agent review only. They do not prove the PSMILES "
                    "matches the material name; compare chemistry carefully."
                ),
                "psmiles_submitted": psm.strip()[:200],
            }
        elif crosscheck_web and not name:
            out["name_crosscheck"] = {
                "error": "crosscheck_web requires a non-empty material_name",
            }
        return json.dumps(out, indent=2)
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


@mcp.tool()
def evaluate_psmiles(
    psmiles_list: Union[str, List[Any]],
    verbose: Union[bool, str, int] = True,
    run_dir: str = "",
    artifacts_dir: str = "",
) -> str:
    """
    Evaluate PSMILES via OpenMM **Packmol matrix**: insulin AMBER14SB + multiple polymer
    chains (GAFF, Gasteiger) packed **bulk-in-cell** by default (or annulus **shell** via env), energy minimization, optional short NPT,
    then interaction energy (screening — not a multi-ns production MD).

    **psmiles_list:** comma-separated string (preferred in docs) **or** a JSON array of strings,
    e.g. ``"[*]CC[*],[*]O[*]"`` or ``["[*]CC[*]", "[*]O[*]"]``. OpenCode and other hosts vary;
    both shapes are accepted.

    **Requires the ``packmol`` binary on PATH** (conda-forge or ``pip install packmol``). If Packmol is
    missing, the tool fails immediately. See ``docs/OPENMM_SCREENING.md`` for matrix parameters
    (``INSULIN_AI_OPENMM_MATRIX_*``, etc.). For a fast **single-oligomer** vacuum test without Packmol,
    use ``scripts/diagnose_openmm_complex.py`` — that path is **not** used here.

    By default (verbose=true) the JSON includes per-candidate timing and energies (evaluation_progress)
    and the MCP server logs progress to stderr. Pass verbose=false or set env INSULIN_AI_EVAL_QUIET=1
    (or INSULIN_AI_EVAL_VERBOSE=0) for a smaller response and no progress lines.

    **Structure artifacts for SUMMARY_REPORT:** When ``run_dir`` is set (or ``INSULIN_AI_SESSION_DIR``
    points at the session folder), minimized matrix complex PDB plus monomer 2D PNG (psmiles ``savefig``),
    preview PNG, and ribbon/chemviz PNG are written under ``<session>/structures/`` unless
    disabled with ``INSULIN_AI_EVAL_NO_STRUCTURE_ARTIFACTS=1``. Override the directory with
    non-empty ``artifacts_dir`` or env ``INSULIN_AI_EVAL_ARTIFACTS_DIR``.
    """
    try:
        parts = _normalize_psmiles_list_for_eval(psmiles_list)
        if not parts:
            return json.dumps(
                {
                    "error": "psmiles_list is empty or could not be parsed",
                    "hint": "Pass a comma-separated string or a JSON array of PSMILES strings (each with two [*]).",
                    "received_type": type(psmiles_list).__name__,
                },
                indent=2,
            )
        from insulin_ai.simulation import MDSimulator

        candidates = [{"material_name": f"Candidate_{i}", "chemical_structure": p} for i, p in enumerate(parts)]
        sim = MDSimulator(n_steps=5000)
        ad = (artifacts_dir or "").strip()
        if not ad and (run_dir or "").strip():
            ad = str(_session_dir_for_mcp(run_dir) / "structures")
        vb = _coerce_bool_flag(verbose, default=True)
        result = sim.evaluate_candidates(
            candidates,
            max_candidates=len(candidates),
            verbose=vb,
            artifacts_dir=ad or None,
        )
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
        if result.get("property_analysis"):
            out["property_analysis"] = result["property_analysis"]
        if _score is not None:
            out["discovery_score"] = round(_score, 4)
        if result.get("evaluation_progress") is not None:
            out["evaluation_progress"] = result["evaluation_progress"]
        if result.get("evaluation_note"):
            out["evaluation_note"] = result["evaluation_note"]
        if result.get("structure_artifacts_dir"):
            out["structure_artifacts_dir"] = result["structure_artifacts_dir"]
        raw = result.get("md_results_raw") or []
        paths = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            paths.append(
                {
                    "psmiles": r.get("psmiles"),
                    "complex_pdb_path": r.get("complex_pdb_path"),
                    "monomer_png_path": r.get("monomer_png_path"),
                    "complex_preview_png_path": r.get("complex_preview_png_path"),
                    "complex_chemviz_png_path": r.get("complex_chemviz_png_path"),
                    "packing_metrics": r.get("packing_metrics"),
                }
            )
        if paths:
            out["structure_artifact_paths"] = paths
        return json.dumps(out, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "ok": False}, indent=2)


@mcp.tool()
def generate_psmiles_from_name(material_name: str) -> str:
    """
    Convert a polymer or monomer **name** to a PSMILES repeat-unit string.

    Resolution order:

    1. **Known polymer table** (~60 common polymers: PEG, PLA, PLGA, PCL, PS,
       PMMA, PVDF, chitosan, ...).  High confidence — no network call.
    2. **PubChem lookup** → monomer SMILES → automated polymerisation-site
       detection (vinyl C=C opening, hydroxy-acid condensation, amino-acid
       amide condensation).  Medium confidence.

    Examples::

        generate_psmiles_from_name("PEG")           → "[*]OCC[*]"
        generate_psmiles_from_name("polystyrene")    → "[*]CC([*])c1ccccc1"
        generate_psmiles_from_name("lactic acid")    → "[*]OC(=O)C(C)[*]"

    Returns JSON with ``ok``, ``psmiles``, ``source``, ``confidence``,
    ``mechanism`` (for PubChem auto), and ``md_compatible`` (prescreen result).
    If conversion fails, ``ok`` is false with ``error`` and the raw PubChem
    SMILES so the caller can attempt manual conversion.
    """
    try:
        from insulin_ai.material_mappings import name_to_psmiles

        result = name_to_psmiles(material_name)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, indent=2)


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
def start_discovery_session(run_name: str = "") -> str:
    """
    Start a new discovery session. All subsequent saves and autonomous runs can use this folder.
    Returns session_dir path; pass it to save_discovery_state(run_dir=...) or set INSULIN_AI_SESSION_DIR in shell.
    """
    try:
        d = new_session_dir(Path(ROOT), name=run_name.strip() or None)
        os.environ[ENV_SESSION] = str(d)
        return json.dumps(
            {"session_dir": str(d), "note": "Server process now uses this session for mine_literature saves and save_discovery_state when run_dir omitted."},
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def run_autonomous_discovery(
    budget_minutes: float = 60.0,
    run_in_background: bool = True,
    run_name: str = "",
    md_steps: int = 5000,
    max_eval_per_iteration: int = 8,
) -> str:
    """
    Autonomous loop; all outputs live in one new folder runs/<session_id>/ (TSV, log, summary, iteration JSON).
    When run_in_background=True, subprocess logs to that folder's autoresearch_subprocess.log.
    """
    session_dir = new_session_dir(Path(ROOT), name=(run_name.strip() or f"autonomous_{time.strftime('%Y%m%d_%H%M%S')}"))
    log_out = session_dir / "autoresearch_subprocess.log"
    script = os.path.join(ROOT, "scripts", "run_autonomous_discovery.py")
    env = os.environ.copy()
    env["INSULIN_AI_ROOT"] = ROOT
    env[ENV_SESSION] = str(session_dir)

    if run_in_background:
        if not os.path.isfile(script):
            return json.dumps({"error": f"Script not found: {script}"})
        cmd = [
            sys.executable,
            script,
            "--budget-minutes",
            str(budget_minutes),
            "--session-dir",
            str(session_dir),
            "--md-steps",
            str(md_steps),
            "--max-eval",
            str(max_eval_per_iteration),
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
                    "session_dir": str(session_dir),
                    "budget_minutes": budget_minutes,
                    "subprocess_log": str(log_out),
                    "results_tsv": str(session_dir / "autoresearch_results.tsv"),
                    "summary_json_when_done": str(session_dir / "autoresearch_summary.json"),
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    try:
        from insulin_ai.autonomous_discovery import run_autonomous_discovery_loop

        summary = run_autonomous_discovery_loop(
            budget_minutes=budget_minutes,
            session_dir=session_dir,
            root=ROOT,
            md_steps=md_steps,
            max_eval_per_iteration=max_eval_per_iteration,
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
    lines.append("Literature Mining: Semantic Scholar + agent extraction (no Ollama)")
    pqa = _paper_qa_index_status()
    lines.append(f"PaperQA2: {pqa.get('message', 'unavailable')}")
    return "\n".join(lines)


def _session_dir_for_mcp(run_dir: str = "") -> Path:
    if run_dir.strip():
        return Path(run_dir.strip()).resolve()
    d = session_dir_from_env(Path(ROOT))
    if d:
        return d
    d = new_session_dir(Path(ROOT), name=None)
    os.environ[ENV_SESSION] = str(d)
    return d


def _allowed_transcript_source(src: Path, repo_root: Path) -> bool:
    """Allow repo files or Cursor/OpenCode agent-transcripts under ~/.cursor."""
    try:
        src = src.resolve()
    except OSError:
        return False
    repo_root = repo_root.resolve()
    if src == repo_root or repo_root in src.parents:
        return True
    cursor_home = (Path.home() / ".cursor").resolve()
    if not cursor_home.is_dir():
        return False
    try:
        src.relative_to(cursor_home)
    except ValueError:
        return False
    return "agent-transcripts" in src.parts


@mcp.tool()
def save_session_transcript(
    content: str,
    filename: str = "SESSION_TRANSCRIPT.md",
    run_dir: str = "",
) -> str:
    """
    Save **text you provide** into the active discovery session. **Default materials-discovery protocol:**
    call this **every iteration** if ``import_chat_transcript_file`` cannot be used (unknown JSONL path
    or copy failure), with a **complete** Markdown recap (tool calls, decisions, results). OpenCode
    does not mirror chat into ``runs/`` automatically.

    Writes UTF-8 to ``<session>/<filename>`` (default ``SESSION_TRANSCRIPT.md``) under the iteration
    output folder only — **not** under ``.cursor/``. For JSONL originals from disk, prefer
    ``import_chat_transcript_file``.
    """
    session = _session_dir_for_mcp(run_dir)
    session.mkdir(parents=True, exist_ok=True)
    fn = (filename or "SESSION_TRANSCRIPT.md").strip()
    if not fn or ".." in fn.replace("\\", "/"):
        return json.dumps({"error": "invalid filename"})
    path = session / fn
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"saved": str(path), "session_dir": str(session)}, indent=2)


@mcp.tool()
def import_chat_transcript_file(
    source_path: str,
    dest_filename: str = "",
    run_dir: str = "",
) -> str:
    """
    Copy a chat transcript **file** from disk **into** ``runs/<session>/`` (same folder as SUMMARY_REPORT
    and other iteration outputs). **Do not** use ``.cursor/`` as the **destination**; it may be the
    **source** path only. Allowed sources only:

    - Any path **under this repository** (insulin-ai), or
    - Files under ``~/.cursor/.../agent-transcripts/`` (OpenCode parent chat JSONL).

    **Materials discovery:** invoke **by default at the end of every iteration** so the session folder
    contains the OpenCode chat snapshot. If this fails, use ``save_session_transcript`` instead.
    See ``docs/OpenCode_PLATFORM.md``.
    """
    src = Path(source_path).expanduser()
    if not src.is_file():
        return json.dumps({"error": f"not a file: {src}"})
    if not _allowed_transcript_source(src, Path(ROOT)):
        return json.dumps(
            {
                "error": "path not allowed (use repo path or ~/.cursor/.../agent-transcripts/...)",
                "hint": "see docs/OpenCode_PLATFORM.md",
            },
            indent=2,
        )
    session = _session_dir_for_mcp(run_dir)
    session.mkdir(parents=True, exist_ok=True)
    dest = (dest_filename or "").strip() or src.name
    if ".." in dest.replace("\\", "/"):
        return json.dumps({"error": "invalid dest_filename"})
    out = session / dest
    try:
        shutil.copy2(src, out)
    except OSError as e:
        return json.dumps({"error": str(e)})
    return json.dumps({"copied_to": str(out), "session_dir": str(session)}, indent=2)


@mcp.tool()
def save_discovery_state(
    iteration: int,
    feedback_json: str,
    query_used: str = "",
    notes: str = "",
    run_dir: str = "",
) -> str:
    """
    Persist discovery state under the session folder (runs/.../iteration_N.json).
    If run_dir omitted, uses active session (start_discovery_session) or creates a new session.
    """
    session = _session_dir_for_mcp(run_dir)
    session.mkdir(parents=True, exist_ok=True)
    os.environ[ENV_SESSION] = str(session)
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
    path = session / f"agent_iteration_{iteration}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return json.dumps({"saved": str(path), "session_dir": str(session)}, indent=2)


@mcp.tool()
def load_discovery_state(iteration: int = 0, run_dir: str = "") -> str:
    """
    Load discovery state from session folder. iteration=0 loads latest agent_iteration_*.json.
    run_dir: session path; if empty, uses active session (same server as start_discovery_session / save).
    """
    if run_dir.strip():
        session = Path(run_dir.strip()).resolve()
    else:
        session = session_dir_from_env(Path(ROOT))
    if not session or not session.is_dir():
        return "No session directory. Call start_discovery_session or pass run_dir= path to runs/.../"

    if iteration > 0:
        path = session / f"agent_iteration_{iteration}.json"
        if not path.is_file():
            return f"No state file for iteration {iteration} in {session}."
    else:
        files = sorted(
            [f for f in os.listdir(session) if f.startswith("agent_iteration_") and f.endswith(".json")]
        )
        if not files:
            return f"No agent_iteration_*.json in {session}."
        path = session / files[-1]

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# --- Literature search (folded from lit-* servers) ---
@mcp.tool()
def semantic_scholar_search(query: str, max_results: int = 20) -> str:
    """Search Semantic Scholar. No API key required (rate limited). Set SEMANTIC_SCHOLAR_API_KEY for higher limits."""
    try:
        from insulin_ai.literature.scholar_client import SemanticScholarClient
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


def _ddg_text_results(query: str, max_results: int = 5) -> list:
    """Return raw DuckDuckGo text results, or empty list on failure."""
    try:
        from duckduckgo_search import DDGS

        return list(DDGS().text(query, max_results=min(max_results, 10)))
    except Exception:
        return []


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web via DuckDuckGo. No API key. Use for material structures, PSMILES, polymer repeat units."""
    try:
        from duckduckgo_search import DDGS  # noqa: F401 — ensure package exists before _ddg_text_results
    except ImportError:
        return "Error: pip install duckduckgo-search"
    results = _ddg_text_results(query, max_results)
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
        ps = PolymerSmiles(psmiles)
        c = ps.canonicalize
        if callable(c):
            c = c()
        return str(c)
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
        ps = PolymerSmiles(psmiles)
        if hasattr(ps, "dimer"):
            return str(ps.dimer(star_index))
        return str(ps.dimerize(star_index=star_index))
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


@mcp.tool()
def render_psmiles_png(
    psmiles: str,
    output_basename: str = "",
    run_dir: str = "",
) -> str:
    """
    Render a 2D depiction of the polymer repeat unit to PNG (psmiles ``PolymerSmiles.savefig``).

    **Reporting workflow:** you (the agent) author ``SUMMARY_REPORT.md`` and embed figures with
    ``![caption](structures/<file>.png)`` after saving PNGs here. Requires **psmiles** (see
    ``docs/DEPENDENCIES.md``).

    Saves under ``<session>/structures/<basename>.png``. Session is the active discovery run
    (``start_discovery_session``) or ``run_dir`` when set. Use after ``validate_psmiles``.
    """
    err = _psmiles_check()
    if err:
        return err
    from insulin_ai.psmiles_drawing import safe_filename_basename, save_psmiles_png

    session = _session_dir_for_mcp(run_dir)
    session.mkdir(parents=True, exist_ok=True)
    struct = session / "structures"
    struct.mkdir(parents=True, exist_ok=True)
    base = (output_basename or "").strip() or safe_filename_basename(psmiles[:80])
    out = struct / f"{base}.png"
    r = save_psmiles_png(psmiles.strip(), out, overwrite=True)
    payload = {**r, "session_dir": str(session), "relative": f"structures/{out.name}"}
    return json.dumps(payload, indent=2)


@mcp.tool()
def compile_discovery_markdown_to_pdf(
    markdown_path: str = "SUMMARY_REPORT.md",
    output_pdf_name: str = "SUMMARY_REPORT.pdf",
    run_dir: str = "",
) -> str:
    """
    Convert **agent-written** Markdown (default ``SUMMARY_REPORT.md``) to a PDF in the session folder.

    You compose the narrative, tables, and interpretation in Markdown; follow ``docs/SUMMARY_REPORT_STYLE.md``
    (research-paper tone, full journal-style references, avoid em-dash/colon AI prose patterns). Call
    ``render_psmiles_png`` for 2D figures, reference them in the MD, then run this tool to produce
    ``SUMMARY_REPORT.pdf``. Uses **markdown** + **fpdf2** + **Pillow** (see ``docs/DEPENDENCIES.md``).
    Local images (e.g. under ``structures/``) are re-encoded to RGB PNG for fpdf2; you do not need
    separate ``*_raster.png`` copies. Relative image paths are resolved against the session directory.
    """
    session = _session_dir_for_mcp(run_dir)
    from insulin_ai.discovery_report import compile_markdown_to_pdf

    r = compile_markdown_to_pdf(
        session,
        markdown_filename=markdown_path.strip() or "SUMMARY_REPORT.md",
        output_pdf_name=output_pdf_name.strip() or "SUMMARY_REPORT.pdf",
    )
    return json.dumps(r, indent=2, default=str)


@mcp.tool()
def write_discovery_summary_report(
    title: str = "Discovery summary",
    run_dir: str = "",
    include_all_iterations: bool = True,
) -> str:
    """
    **Optional batch helper** (not a substitute for an AI-written report): reads ``agent_iteration_*.json``,
    auto-builds a minimal **SUMMARY_REPORT.md** + PNGs + PDF from saved feedback only—use when you need
    a quick skeleton without narrative. Any evaluate_psmiles-style files already in ``structures/``
    (``*_monomer.png``, ``*_complex_preview.png``, ``*_complex_chemviz.png``, optional ``*_complex_minimized_pymol.png``)
    are embedded in the Markdown (and PDF) under each matching candidate slug, or in a **Molecular visualizations**
    section for filenames that do not match feedback labels (e.g. ``Candidate_0_*``). For **normal** scientific
    summaries, the agent should write ``SUMMARY_REPORT.md`` and call ``compile_discovery_markdown_to_pdf`` after
    ``render_psmiles_png`` (same image paths; see ``docs/SUMMARY_REPORT_STYLE.md``).
    Requires **psmiles**, **fpdf2**, **markdown** (see ``docs/DEPENDENCIES.md``).
    """
    session = _session_dir_for_mcp(run_dir)
    from insulin_ai.discovery_report import write_session_summary_reports

    r = write_session_summary_reports(
        session,
        title=title,
        include_all_iterations=include_all_iterations,
    )
    return json.dumps(r, indent=2, default=str)


if __name__ == "__main__":
    mcp.run()
