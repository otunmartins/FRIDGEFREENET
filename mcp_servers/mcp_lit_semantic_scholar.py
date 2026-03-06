#!/usr/bin/env python3
"""
Semantic Scholar Literature Mining MCP Server

Works WITHOUT API key (rate limited). Set SEMANTIC_SCHOLAR_API_KEY for higher limits.
"""

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP
from semantic_scholar_client import SemanticScholarClient

mcp = FastMCP("semantic-scholar-lit", instructions="Semantic Scholar paper search. Optional API key for rate limits.")


@mcp.tool()
def semantic_scholar_search(query: str, max_results: int = 20) -> str:
    """
    Search Semantic Scholar. No API key required (rate limited).
    Set SEMANTIC_SCHOLAR_API_KEY for higher limits.
    """
    try:
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


if __name__ == "__main__":
    mcp.run()
