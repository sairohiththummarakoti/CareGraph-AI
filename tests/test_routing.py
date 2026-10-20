import pytest

from caregraph.graph import run_workflow
from caregraph.routing import classify_domains


@pytest.mark.parametrize("question, expected", [
    ("What is oncology care?", ["oncology"]),
    ("What is a heart attack?", ["cardiology"]),
    ("What considerations apply to cancer plus heart disease?", ["oncology", "cardiology"]),
    ("What is the weather?", []),
    ("", []),
    ("Explain myocardial infarction", ["cardiology"]),
    ("How is a lymphoma report organized?", ["oncology"]),
    ("Explain atrial fibrillation", ["cardiology"]),
    ("What is radiation therapy?", ["oncology"]),
    ("Explain atrial flutter and radiotherapy", ["oncology", "cardiology"]),
    ("What is radiation from the sun?", []),
])
def test_routing(question, expected):
    assert classify_domains(question) == expected


@pytest.mark.parametrize("question", ["", "   ", "What is the weather?"])
def test_unsupported_needs_no_index(tmp_path, question):
    state = run_workflow(question, index_dir=tmp_path / "absent")
    assert state["route"] == "fallback"
    assert state["executed_nodes"] == ["parse_report", "coordinator", "fallback"]
    assert not state["evidence"]
    assert "Ask an educational question" in state["final_answer"]


def test_missing_index_is_actionable(tmp_path):
    state = run_workflow("What is oncology care?", index_dir=tmp_path)
    assert state["decision"] == "block"
    assert "ingest.py" in state["error"]


def test_question_limit():
    assert run_workflow("x" * 4001)["decision"] == "block"
