import json

import app
from caregraph.graph import run_workflow


def test_cli_json_report_preview_then_confirmation(tmp_path, reference_index, embedder, report_bytes, monkeypatch, capsys):
    report_path = tmp_path / "synthetic.txt"
    report_path.write_bytes(report_bytes)
    monkeypatch.setattr(app, "run_workflow", lambda *args, **kwargs: run_workflow(*args, embedder=embedder, **kwargs))
    arguments = ["--question", "Which evidence is relevant to this tumor profile?", "--report", str(report_path), "--index-dir", str(reference_index), "--json"]
    assert app.main(arguments) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["awaiting_confirmation"]
    assert not preview["evidence"]
    assert "stage" in preview["missing_facts"]
    assert app.main(arguments + ["--confirm-report"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["evidence"]
    assert result["report_confirmed"]
    assert "report_bytes" not in result


def test_cli_missing_index_error_is_valid_json(tmp_path, capsys):
    assert app.main(["--question", "What is oncology care?", "--index-dir", str(tmp_path), "--json"]) == 2
    assert "ingest.py" in json.loads(capsys.readouterr().out)["error"]


def test_cli_interactive_fallback_and_exit(monkeypatch, capsys):
    answers = iter(["What is the weather?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert app.main([]) == 0
    output = capsys.readouterr().out
    assert "NODE = DO | EDGE = GO | STATE = KNOW" in output
    assert "Route: fallback" in output
    assert "before:" in output and "after:" in output
