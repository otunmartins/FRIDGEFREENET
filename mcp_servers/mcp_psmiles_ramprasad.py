#!/usr/bin/env python3
"""
Ramprasad Group PSMILES MCP Server

Uses psmiles package (github.com/Ramprasad-Group/psmiles):
- canonicalize, dimerize, fingerprints (polyBERT, RDKit, Mordred)
- No API key required.
Install: pip install psmiles[mordred,polyBERT]  (or pip install psmiles for basic)
"""

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("psmiles-ramprasad", instructions="Ramprasad-Group PSMILES: canonicalize, dimerize, fingerprints. No API key.")

try:
    from psmiles import PolymerSmiles
    PSMILES_AVAILABLE = True
except ImportError:
    PSMILES_AVAILABLE = False


def _check():
    if not PSMILES_AVAILABLE:
        return "psmiles not installed. Run: pip install 'psmiles[mordred,polyBERT]'"


@mcp.tool()
def psmiles_canonicalize(psmiles: str) -> str:
    """
    Canonicalize PSMILES (Ramprasad-Group). Returns unique representation.
    """
    err = _check()
    if err:
        return err
    try:
        ps = PolymerSmiles(psmiles)
        return str(ps.canonicalize())
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_dimerize(psmiles: str, star_index: int = 0) -> str:
    """
    Dimerize PSMILES at connection point. star_index: 0 or 1 for which [*].
    """
    err = _check()
    if err:
        return err
    try:
        ps = PolymerSmiles(psmiles)
        dimer = ps.dimerize(star_index=star_index)
        return str(dimer)
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_fingerprint(psmiles: str, fingerprint_type: str = "rdkit") -> str:
    """
    Get fingerprint for PSMILES. Types: rdkit, mordred, polyBERT, morgan.
    polyBERT requires sentence-transformers.
    """
    err = _check()
    if err:
        return err
    try:
        ps = PolymerSmiles(psmiles)
        fp = ps.descriptor(fingerprint_type)
        if hasattr(fp, "tolist"):
            return json.dumps(fp.tolist()[:20])  # First 20 values
        return str(fp)[:500]
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def psmiles_similarity(psmiles1: str, psmiles2: str) -> str:
    """
    Compute similarity between two PSMILES (Ramprasad-Group).
    """
    err = _check()
    if err:
        return err
    try:
        ps1 = PolymerSmiles(psmiles1)
        ps2 = PolymerSmiles(psmiles2)
        sim = ps1.similarity(ps2)
        return f"Similarity: {sim}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run()
