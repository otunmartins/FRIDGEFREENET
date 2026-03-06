#!/usr/bin/env python3
"""
PubMed Literature Mining MCP Server

Uses NCBI E-utilities - NO API KEY required (3 req/sec limit).
Set NCBI_API_KEY env for 10 req/sec (free from ncbi.nlm.nih.gov/account).
"""

import os
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("pubmed-lit", instructions="PubMed/NCBI literature search. No API key required.")


def _efetch(ids: list, retmax: int = 10) -> list:
    """Fetch paper details from PubMed."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY", "")
    tool, email = "insulin-ai", "research@example.com"
    ids_str = ",".join(str(i) for i in ids[:retmax])
    params = {"id": ids_str, "db": "pubmed", "rettype": "json", "tool": tool, "email": email}
    if api_key:
        params["api_key"] = api_key
    r = requests.get(f"{base}/esummary.fcgi", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("result", {})


def _esearch(query: str, retmax: int = 20) -> list:
    """Search PubMed."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY", "")
    tool, email = "insulin-ai", "research@example.com"
    params = {"term": query, "db": "pubmed", "retmax": retmax, "retmode": "json", "tool": tool, "email": email}
    if api_key:
        params["api_key"] = api_key
    time.sleep(0.35)  # Rate limit
    r = requests.get(f"{base}/esearch.fcgi", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def _get_abstracts(ids: list) -> list:
    """Fetch abstracts via efetch XML."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    api_key = os.environ.get("NCBI_API_KEY", "")
    params = {"id": ",".join(str(i) for i in ids), "db": "pubmed", "rettype": "xml", "tool": "insulin-ai", "email": "research@example.com"}
    if api_key:
        params["api_key"] = api_key
    time.sleep(0.35)
    r = requests.get(f"{base}/efetch.fcgi", params=params, timeout=15)
    if r.status_code != 200 or not r.text.strip():
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
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
    """
    Search PubMed for papers. No API key required.
    Set NCBI_API_KEY for higher rate limit.
    """
    try:
        ids = _esearch(query, retmax=max_results)
        if not ids:
            return "No papers found."
        papers = _get_abstracts(ids[:max_results])
        lines = [f"Found {len(papers)} papers."]
        for i, p in enumerate(papers[:10], 1):
            lines.append(f"{i}. {p['title'][:80]}... (PMID {p['pmid']})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
