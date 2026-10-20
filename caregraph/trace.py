"""Redacted local observability and strictly opt-in aggregate cloud tracing."""

import os
from datetime import datetime, timezone
from typing import Callable


def summarize_state(state: dict) -> dict:
    """Never include prompts, report bytes/names, fact values, excerpts, or drafts."""
    return {
        "question_characters": len(state.get("question", "")),
        "route": state.get("route", "pending"),
        "domains": state.get("domains", []),
        "report_present": bool(state.get("report_present") or state.get("report_bytes")),
        "report_confirmed": bool(state.get("report_confirmed")),
        "fact_count": len(state.get("pathology_facts", {})),
        "unknown_count": len(state.get("missing_facts", [])),
        "evidence_count": len(state.get("evidence", [])),
        "source_ids": [c["chunk_id"] for c in state.get("evidence", [])],
        "completed_domains": state.get("completed_domains", []),
        "decision": state.get("decision", "pending"),
    }


def traced_node(name: str, function: Callable) -> Callable:
    def execute(state: dict) -> dict:
        changes = function(state)
        after = dict(state)
        after.update(changes)
        for key in ("evidence", "completed_domains"):
            if key in changes:
                after[key] = state.get(key, []) + changes[key]
        event = {
            "node": name,
            "changed_keys": sorted(k for k, value in changes.items() if state.get(k) != value),
            "before": summarize_state(state),
            "after": summarize_state(after),
        }
        return {**changes, "executed_nodes": [name], "trace": [event]}
    execute.__name__ = name
    return execute


def export_redacted_trace(state: dict) -> str:
    """Automatic LangGraph tracing remains disabled even for opted-in users."""
    if os.getenv("CAREGRAPH_CLOUD_TRACING") != "1":
        return "disabled"
    if os.getenv("CAREGRAPH_SYNTHETIC_DATA_CONFIRMED") != "1":
        return "disabled: synthetic-data confirmation is required"
    if not os.getenv("LANGSMITH_API_KEY"):
        return "disabled: LANGSMITH_API_KEY is required for optional tracing"
    try:
        from langsmith import Client
        client = Client(auto_batch_tracing=False)
        now = datetime.now(timezone.utc)
        client.create_run(
            name="caregraph-redacted-workflow",
            run_type="chain",
            inputs={"route": state.get("route", "pending"), "synthetic_only": True},
            outputs={
                "summary": summarize_state(state),
                "nodes": state.get("executed_nodes", []),
            },
            start_time=now,
            end_time=now,
            project_name="caregraph-synthetic",
        )
        return "redacted summary exported"
    except Exception:
        # Never place client exceptions, credentials, or request bodies in logs.
        return "optional redacted trace export failed; local result is available"
