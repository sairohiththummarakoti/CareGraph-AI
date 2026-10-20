"""Report transitions must invalidate both released output and confirmation."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from caregraph import retrieval


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
REPORT_A = b"SYNTHETIC REPORT\nTumor site: Breast\nHistology: Synthetic carcinoma A\n"
REPORT_B = b"SYNTHETIC REPORT\nTumor site: Lung\nHistology: Synthetic carcinoma B\n"


@pytest.fixture
def workspace(monkeypatch, reference_index, embedder):
    """Exercise the real graph and FAISS with the offline deterministic embedder."""
    monkeypatch.setattr(retrieval, "DEFAULT_INDEX_DIR", reference_index)
    monkeypatch.setattr(retrieval, "SentenceTransformerEmbedder", lambda offline=True: embedder)
    monkeypatch.setenv("CAREGRAPH_CLOUD_TRACING", "0")
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    return app


def button(app, label):
    return next(item for item in app.button if item.label == label)


def release_report_a(app):
    app.session_state["sample_report"] = (REPORT_A, "synthetic-a.txt")
    app.session_state["question"] = "Which evidence is relevant to this tumor profile?"
    app.run()
    assert button(app, "Explore evidence").disabled
    app.checkbox[0].check().run()
    button(app, "Explore evidence").click().run(timeout=30)
    assert not app.exception
    result = app.session_state["result"]
    assert result["report_confirmed"]
    assert result["pathology_facts"]["tumor_site"] == "Breast"
    assert result["evidence"]
    assert app.session_state["result_question"]


def assert_result_cleared(app):
    assert not app.exception
    assert app.session_state["result"] is None
    assert app.session_state["result_question"] == ""
    assert not app.tabs
    assert not any("Synthetic carcinoma A" in str(item.value) for item in app.markdown)


def test_replacing_report_clears_output_and_requires_new_confirmation(workspace):
    release_report_a(workspace)
    workspace.session_state["sample_report"] = (REPORT_B, "synthetic-b.txt")
    workspace.run()
    assert_result_cleared(workspace)
    assert not workspace.checkbox[0].value
    assert button(workspace, "Explore evidence").disabled
    values = workspace.dataframe[0].value["Reported value"].tolist()
    assert "Lung" in values
    assert "Breast" not in values


@pytest.mark.parametrize("remove_with_button", [True, False])
def test_removing_report_clears_output(workspace, remove_with_button):
    release_report_a(workspace)
    if remove_with_button:
        button(workspace, "Remove report").click().run()
    else:
        # Covers the file-uploader's empty transition as well as the explicit
        # Remove report callback, without persisting an uploaded file.
        workspace.session_state["sample_report"] = None
        workspace.run()
    assert_result_cleared(workspace)
    assert workspace.session_state["report_fingerprint"] is None
    assert not workspace.session_state["confirmed_report"]
    assert not workspace.checkbox
    assert not button(workspace, "Explore evidence").disabled


def test_revoking_confirmation_clears_output(workspace):
    release_report_a(workspace)
    workspace.checkbox[0].uncheck().run()
    assert_result_cleared(workspace)
    assert not workspace.checkbox[0].value
    assert button(workspace, "Explore evidence").disabled
    assert workspace.session_state["sample_report"] == (REPORT_A, "synthetic-a.txt")
