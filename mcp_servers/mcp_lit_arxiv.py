#!/usr/bin/env python3
"""
arXiv Literature Mining MCP Server

Uses arXiv API - NO API KEY required.
"""

import os
import sys
import requests
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arxiv-lit", instructions="arXiv preprint search. No API key required.")


@mcp.tool()
def arxiv_search(query: str, max_results: int = 20) -> str:
    """
    Search arXiv for papers. No API key required.
    """
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        r = requests.get(
            "https://export.arxiv.org/api/query",
            params=params,
            headers={"User-Agent": "insulin-ai/1.0 (research@example.com)"},
            timeout=30,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        lines = [f"Found {len(entries)} papers."]
        for i, entry in enumerate(entries[:10], 1):
            title = entry.find("atom:title", ns)
            aid = entry.find("atom:id", ns)
            summary = entry.find("atom:summary", ns)
            t = (title.text or "").replace("\n", " ").strip() if title is not None else ""
            id_text = (aid.text or "").split("/")[-1] if aid is not None else ""
            abs_text = (summary.text or "")[:200] if summary is not None else ""
            lines.append(f"{i}. {t[:80]}... ({id_text})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
