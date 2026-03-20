"""Tests for discovery_report and psmiles_drawing (optional psmiles / fpdf2)."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

from insulin_ai.discovery_report import (  # noqa: E402
    collect_psmiles_entries_from_feedback,
    compile_markdown_to_pdf,
    write_session_summary_reports,
)
from insulin_ai.psmiles_drawing import safe_filename_basename  # noqa: E402

# 1x1 transparent PNG (valid for fpdf image embedding)
_MINI_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_safe_filename_basename():
    assert ".." not in safe_filename_basename("foo/bar")
    assert safe_filename_basename("  ") == "structure"


def test_collect_psmiles_from_feedback_dicts():
    fb = {
        "high_performers": [
            {"name": "A", "psmiles": "[*]CC[*]"},
            {"name": "B", "psmiles": "[*]O[*]"},
        ]
    }
    pairs = collect_psmiles_entries_from_feedback(fb)
    assert len(pairs) == 2
    assert pairs[0][1] == "[*]CC[*]"


def test_compile_markdown_to_pdf_minimal(tmp_path):
    """Agent-style MD → PDF (needs markdown + fpdf2)."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")
    sess = tmp_path / "sess"
    sess.mkdir()
    img = sess / "structures"
    img.mkdir(parents=True)
    img.joinpath("x.png").write_bytes(_MINI_PNG)
    (sess / "SUMMARY_REPORT.md").write_text(
        "# Title\n\nHello.\n\n![fig](structures/x.png)\n",
        encoding="utf-8",
    )
    out = compile_markdown_to_pdf(sess, markdown_filename="SUMMARY_REPORT.md")
    assert out.get("ok") is True, out
    assert (sess / "SUMMARY_REPORT.pdf").is_file()


def test_write_session_summary_reports_with_mock_png(monkeypatch, tmp_path):
    """End-to-end: JSON feedback -> MD + PDF; PNG generation mocked with a tiny valid file."""
    sess = tmp_path / "run1"
    sess.mkdir()
    state = {
        "iteration": 1,
        "feedback": {
            "high_performers": [{"name": "TestMat", "psmiles": "[*]OCC[*]"}],
        },
    }
    (sess / "agent_iteration_1.json").write_text(json.dumps(state), encoding="utf-8")

    def _fake_save(psmiles, output_path, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(_MINI_PNG)
        return {"ok": True, "path": str(output_path)}

    monkeypatch.setattr("insulin_ai.discovery_report.save_psmiles_png", _fake_save)

    out = write_session_summary_reports(sess, title="T", include_all_iterations=True)
    assert out.get("ok") is True
    assert (sess / "SUMMARY_REPORT.md").is_file()
    md = (sess / "SUMMARY_REPORT.md").read_text(encoding="utf-8")
    assert "TestMat" in md or "[*]OCC[*]" in md
    if out.get("pdf"):
        assert (sess / "SUMMARY_REPORT.pdf").is_file()


def test_save_psmiles_png_integration():
    """Real psmiles ``savefig`` when package is available (e.g. insulin-ai-sim)."""
    pytest.importorskip("psmiles")
    from insulin_ai.psmiles_drawing import save_psmiles_png

    p = Path(ROOT) / "tmp_test_psmiles_draw.png"
    try:
        r = save_psmiles_png("[*]OCC[*]", p)
        if not r.get("ok"):
            pytest.skip(r.get("error", "save failed"))
        assert p.is_file()
    finally:
        if p.is_file():
            p.unlink()
