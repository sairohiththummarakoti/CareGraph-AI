"""Exercise the installed application with real embeddings and network disabled.

Run after ``python ingest.py``. This uses the same workflow as the CLI and UI;
it never writes reports, rebuilds the reference index, or contacts an LLM.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from caregraph.graph import run_workflow  # noqa: E402
from caregraph.retrieval import DEFAULT_INDEX_DIR, SentenceTransformerEmbedder  # noqa: E402


def index_snapshot() -> dict[str, tuple[str, int]]:
    """Both bytes and modification times must survive a runtime query."""
    return {
        str(path.relative_to(DEFAULT_INDEX_DIR)): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in Path(DEFAULT_INDEX_DIR).rglob("*")
        if path.is_file()
    }


def main() -> None:
    before = index_snapshot()
    assert before, "Run python ingest.py before this acceptance check."
    results = []
    synthetic_report = (
        "SYNTHETIC PATHOLOGY REPORT\n"
        "Tumor site: Breast\n"
        "Histology: Synthetic carcinoma example\n"
        "Grade: 2\n"
        "Stage: Unknown\n"
        "Tumor size: 2 cm\n"
        "Biomarkers: Pending\n"
    ).encode()
    environment = {
        "LANGSMITH_TRACING": "true",
        "LANGCHAIN_TRACING_V2": "true",
        "CAREGRAPH_CLOUD_TRACING": "false",
        "CAREGRAPH_SYNTHETIC_DATA_CONFIRMED": "false",
    }
    with patch.dict(os.environ, environment), patch(
        "socket.socket.connect",
        side_effect=AssertionError("Unexpected network connection during local execution"),
    ) as network_guard:
        embedder = SentenceTransformerEmbedder(offline=True)
        long_question = "Explain educational cancer and heart terminology. " * 60
        contextual_vectors = embedder.encode_queries([
            long_question + "Confirmed finding: estrogen receptor positive.",
            long_question + "Confirmed finding: atrial fibrillation.",
        ])
        assert np.linalg.norm(contextual_vectors[0] - contextual_vectors[1]) > 1e-4, (
            "Late report facts were silently lost from long query embeddings."
        )

        def run(name: str, question: str, **kwargs) -> dict:
            state = run_workflow(question, embedder=embedder, **kwargs)
            assert not state.get("error"), (name, state.get("error"))
            results.append(
                {
                    "case": name,
                    "route": state.get("route"),
                    "decision": state.get("decision"),
                    "chunks": len(state.get("evidence", [])),
                    "nodes": state.get("executed_nodes", []),
                }
            )
            return state

        oncology = run("A oncology", "What is oncology care?")
        assert oncology["route"] == "oncology" and oncology["evidence"]
        cardiology = run("B semantic synonym", "What is a heart attack?")
        assert cardiology["route"] == "cardiology" and cardiology["evidence"]
        assert any("myocardial infarction" in e["text"].lower() for e in cardiology["evidence"])
        both = run("C both domains", "What considerations apply to cancer plus heart disease?")
        assert both["route"] == "both"
        assert {e["domain"] for e in both["evidence"]} == {"oncology", "cardiology"}
        nodes = both["executed_nodes"]
        for once in ("join_evidence", "synthesis", "assurance"):
            assert nodes.count(once) == 1, (once, nodes)
        assert nodes.index("synthesis") > max(
            nodes.index("retrieve_oncology"), nodes.index("retrieve_cardiology")
        )
        fallback = run("D unsupported", "What is the weather?")
        assert fallback["route"] == "fallback" and not fallback["evidence"]
        waiting = run(
            "E report confirmation",
            "Which evidence is relevant to this tumor profile?",
            report_bytes=synthetic_report,
            report_name="synthetic_test.txt",
        )
        assert waiting["awaiting_confirmation"]
        assert not waiting["evidence"]
        report = run(
            "F confirmed report with missing data",
            "Which evidence is relevant to this tumor profile?",
            report_bytes=synthetic_report,
            report_name="synthetic_test.txt",
            report_confirmed=True,
        )
        assert report["route"] == "oncology" and report["evidence"]
        assert {"stage", "biomarkers"} <= set(report["missing_facts"])
        assert "stage" not in report["pathology_facts"]
        assert "clinician review" in report["final_answer"].lower()
        risk = run("G treatment escalation", "Choose my chemotherapy and dose")
        assert risk["decision"] in {"escalate", "block"}
        repeat = run("H repeat query", "What is a heart attack?")
        assert [e["chunk_id"] for e in repeat["evidence"]] == [
            e["chunk_id"] for e in cardiology["evidence"]
        ]
        for state in (oncology, cardiology, both, report, repeat):
            for chunk in state["evidence"]:
                assert chunk["chunk_id"] in state["final_answer"]
                assert chunk["source"] in state["final_answer"]
                assert chunk["section"] in state["final_answer"]
        assert network_guard.call_count == 0, "Local execution attempted a network connection."

    assert before == index_snapshot(), "Runtime changed reference index bytes or timestamps."
    print(json.dumps({
        "status": "passed",
        "actual_model": True,
        "network_blocked": True,
        "ambient_tracing_overridden": True,
        "late_query_context_preserved": True,
        "index_unchanged": True,
        "cases": results,
    }, indent=2))


if __name__ == "__main__":
    main()
