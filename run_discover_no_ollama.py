#!/usr/bin/env python3
"""
Run 5 feedback iterations WITHOUT Ollama - uses no-API-key literature sources only.
Literature: Semantic Scholar, PubMed, arXiv (direct API, no LLM).
Evaluation: RDKit proxy (no OpenMM/GAFF).
"""

import os
import sys
import json
import re
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

# Material keywords to extract from abstracts
MATERIAL_PATTERNS = [
    r"\b(chitosan|PEG|PLGA|PVA|PMMA|hydrogel|alginate|hyaluronic acid|cellulose|collagen)\b",
    r"\b(poly\w+-based|peg-based|polymer)\b",
    r"\b(copolymer|block polymer|composite)\b",
]


def search_semantic_scholar(query: str, max_results: int = 15) -> list:
    """Semantic Scholar - no API key."""
    from semantic_scholar_client import SemanticScholarClient
    client = SemanticScholarClient(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    results = client.search_papers(query=query, limit=max_results)
    papers = results.get("data", [])
    out = []
    for p in papers:
        if p.get("abstract"):
            out.append({"title": p.get("title", ""), "abstract": p.get("abstract", ""), "source": "semantic_scholar"})
    return out


def search_pubmed(query: str, max_results: int = 10) -> list:
    """PubMed - no API key."""
    import requests
    time.sleep(0.4)
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={"term": query, "db": "pubmed", "retmax": max_results, "retmode": "json", "tool": "insulin-ai", "email": "research@example.com"},
        timeout=15,
    )
    try:
        data = r.json()
    except Exception:
        return []
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    time.sleep(0.4)
    r2 = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"id": ",".join(ids), "db": "pubmed", "rettype": "xml", "tool": "insulin-ai", "email": "research@example.com"},
        timeout=15,
    )
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r2.content)
    out = []
    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        abs_el = art.find(".//AbstractText")
        t = (title_el.text or "") if title_el is not None else ""
        a = (abs_el.text or "") if abs_el is not None else ""
        if t or a:
            out.append({"title": t, "abstract": a, "source": "pubmed"})
    return out


def search_arxiv(query: str, max_results: int = 10) -> list:
    """arXiv - no API key."""
    import requests
    r = requests.get(
        "https://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": max_results},
        headers={"User-Agent": "insulin-ai/1.0"},
        timeout=30,
    )
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        t = (title.text or "").replace("\n", " ").strip() if title is not None else ""
        a = (summary.text or "").replace("\n", " ").strip() if summary is not None else ""
        if t or a:
            out.append({"title": t, "abstract": a, "source": "arxiv"})
    return out


def extract_materials(papers: list) -> list:
    """Extract material candidates from papers using keyword matching."""
    seen = set()
    candidates = []
    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        for pat in MATERIAL_PATTERNS:
            for m in re.finditer(pat, text, re.I):
                name = m.group(1) if m.lastindex else m.group(0)
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    candidates.append({
                        "material_name": name,
                        "material_composition": name,
                        "source": p.get("source", ""),
                    })
    return candidates


def evaluate_candidates(candidates: list) -> dict:
    """RDKit proxy evaluation."""
    from insulin_ai.simulation import MDSimulator
    sim = MDSimulator()
    return sim.evaluate_candidates(candidates, max_candidates=min(10, len(candidates)))


def main():
    queries_cycle = [
        "hydrogel insulin transdermal",
        "polymer protein stabilization thermal",
        "biocompatible drug delivery skin patch",
        "chitosan PEG insulin",
        "smart hydrogel temperature release",
    ]
    all_materials = []
    feedback_top = []

    os.makedirs("cycle_results", exist_ok=True)

    for iteration in range(1, 6):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}")
        print("="*50)

        q = queries_cycle[(iteration - 1) % len(queries_cycle)]
        if feedback_top:
            q = f"{q} {feedback_top[0]}"

        papers = []
        papers.extend(search_semantic_scholar(q, 10))
        time.sleep(1)
        papers.extend(search_pubmed(q, 8))
        time.sleep(1)
        papers.extend(search_arxiv(q, 5))

        candidates = extract_materials(papers)
        print(f"  Papers: {len(papers)}, Materials: {len(candidates)}")

        if not candidates:
            candidates = [{"material_name": "PEG", "material_composition": "polyethylene glycol"},
                         {"material_name": "chitosan", "material_composition": "chitosan"}]

        fb = evaluate_candidates(candidates)
        high = fb.get("high_performers", [])
        feedback_top = high[:2]
        all_materials.extend([c for c in candidates if c["material_name"] not in {m["material_name"] for m in all_materials}])

        print(f"  High performers: {high}")

        with open(f"cycle_results/iteration_{iteration}.json", "w") as f:
            json.dump({"iteration": iteration, "query": q, "candidates": candidates, "feedback": fb, "papers_count": len(papers)}, f, indent=2)

    print("\n" + "="*50)
    print("MATERIALS OBTAINED (5 iterations)")
    print("="*50)
    for i, m in enumerate(all_materials[:30], 1):
        print(f"  {i}. {m['material_name']} ({m.get('source', '')})")
    print(f"\nTotal unique materials: {len(all_materials)}")
    return all_materials


if __name__ == "__main__":
    main()
