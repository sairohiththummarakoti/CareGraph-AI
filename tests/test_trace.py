import json
from unittest.mock import patch

from caregraph.graph import run_workflow
from caregraph.trace import export_redacted_trace


def test_ambient_cloud_settings_cannot_enable_tracing(monkeypatch, reference_index, embedder, report_bytes):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-test-secret")
    monkeypatch.delenv("CAREGRAPH_CLOUD_TRACING", raising=False)
    with patch("socket.socket.connect", side_effect=AssertionError("Network must stay disabled")) as connection:
        state = run_workflow("Explain this cancer report", report_bytes, "private_report_name.txt", True, index_dir=reference_index, embedder=embedder)
        assert state["evidence"]
        assert connection.call_count == 0
    trace = json.dumps(state["trace"])
    for private in ("private_report_name", "Breast", "Synthetic carcinoma example", "Explain this cancer report", "synthetic-test-secret"):
        assert private not in trace
    assert "source_ids" in trace and "evidence_count" in trace
    assert state["cloud_trace_status"] == "disabled"


def test_cloud_export_requires_both_flags(monkeypatch):
    monkeypatch.setenv("CAREGRAPH_CLOUD_TRACING", "1")
    monkeypatch.delenv("CAREGRAPH_SYNTHETIC_DATA_CONFIRMED", raising=False)
    with patch("langsmith.Client") as client:
        assert "confirmation" in export_redacted_trace({})
        client.assert_not_called()


def test_opted_in_export_contains_only_redacted_summary(monkeypatch):
    monkeypatch.setenv("CAREGRAPH_CLOUD_TRACING", "1")
    monkeypatch.setenv("CAREGRAPH_SYNTHETIC_DATA_CONFIRMED", "1")
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-secret")
    state = {"question": "PRIVATE QUESTION", "pathology_facts": {"histology": "PRIVATE FACT"}, "report_name": "PRIVATE NAME", "final_answer": "PRIVATE ANSWER", "route": "oncology", "executed_nodes": ["coordinator"]}
    with patch("langsmith.Client") as client:
        assert export_redacted_trace(state) == "redacted summary exported"
        payload = json.dumps(client.return_value.create_run.call_args.kwargs, default=str)
        assert "PRIVATE" not in payload
        assert "synthetic-secret" not in payload
        assert "question_characters" in payload
