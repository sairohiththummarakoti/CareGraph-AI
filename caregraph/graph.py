"""A real LangGraph workflow with parallel retrieval and one superstep join."""

from pathlib import Path
from threading import Lock

from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from .assurance import assess_draft, risk_decision
from .parsing import ReportError, parse_report
from .retrieval import DEFAULT_INDEX_DIR, IndexSetupError, ReferenceIndex
from .routing import REPORT_CONTEXT, classify_domains, route_name
from .state import CareState
from .synthesis import DISCLAIMER, build_draft
from .trace import export_redacted_trace, traced_node


def build_graph(index_dir: str | Path = DEFAULT_INDEX_DIR, embedder=None, retriever=None):
    """Build a fresh memory-only graph. No checkpointer, store, or ingestion path."""
    retriever_lock = Lock()
    reference = retriever

    def get_reference():
        nonlocal reference
        with retriever_lock:
            if reference is None:
                reference = ReferenceIndex(index_dir=index_dir, embedder=embedder)
        return reference

    def parse_node(state: CareState) -> dict:
        data = state.get("report_bytes")
        if data is None:
            return {"report_present": False, "pathology_facts": {}, "missing_facts": [], "awaiting_confirmation": False}
        parsed = parse_report(data, state.get("report_name") or "")
        return {
            **parsed, "report_bytes": None, "report_name": None, "report_present": True,
            "awaiting_confirmation": not state.get("report_confirmed", False),
        }

    def coordinator(state: CareState) -> dict:
        confirmed_facts = state.get("pathology_facts") if state.get("report_confirmed") else None
        domains = classify_domains(state["question"], confirmed_facts)
        if state.get("report_present") and state.get("report_confirmed") and REPORT_CONTEXT.search(state["question"]) and "oncology" not in domains:
            domains.insert(0, "oncology")
        _, risk_reason = risk_decision(state["question"])
        return {"domains": domains, "route": route_name(domains), "risk_reason": risk_reason}

    def route(state: CareState):
        if state.get("awaiting_confirmation"):
            return "confirm_report"
        if state.get("risk_reason"):
            return "assurance"
        if not state.get("domains"):
            return "fallback"
        return ["retrieve_" + domain for domain in state["domains"]]

    def confirmation(state: CareState) -> dict:
        return {
            "final_answer": f"{DISCLAIMER}\n\nReview the extracted fields and unknowns, then explicitly confirm the synthetic report before retrieval.",
            "decision": "block",
        }

    def retrieve(domain: str):
        def execute(state: CareState) -> dict:
            query = state["question"]
            if state.get("report_present"):
                if not state.get("report_confirmed"):
                    raise RuntimeError("Unconfirmed report reached retrieval.")
                fields = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in state.get("pathology_facts", {}).items())
                query += "\nConfirmed synthetic report fields: " + fields
            chunks = get_reference().search(query, domain, top_k=3)
            return {"evidence": chunks, "completed_domains": [domain]}
        return execute

    def join(state: CareState) -> dict:
        selected = set(state["domains"])
        completed = state.get("completed_domains", [])
        if set(completed) != selected or len(completed) != len(selected):
            raise RuntimeError("Selected retrieval branches did not complete exactly once.")
        return {"evidence_counts": {domain: sum(c["domain"] == domain for c in state.get("evidence", [])) for domain in state["domains"]}}

    def fallback(state: CareState) -> dict:
        return {
            "decision": "release",
            "final_answer": f"{DISCLAIMER}\n\nAsk an educational question about oncology, cardiology, or both. For example: “What is a heart attack?”",
        }

    builder = StateGraph(CareState)
    for name, function in [
        ("parse_report", parse_node), ("coordinator", coordinator),
        ("confirm_report", confirmation),
        ("retrieve_oncology", retrieve("oncology")),
        ("retrieve_cardiology", retrieve("cardiology")),
        ("join_evidence", join),
        ("synthesis", lambda state: {"draft_answer": build_draft(state)}),
        ("assurance", assess_draft), ("fallback", fallback),
    ]:
        builder.add_node(name, traced_node(name, function))
    builder.add_edge(START, "parse_report")
    builder.add_edge("parse_report", "coordinator")
    builder.add_conditional_edges("coordinator", route, {
        name: name for name in ("confirm_report", "assurance", "fallback", "retrieve_oncology", "retrieve_cardiology")
    })
    # Fan-out branches occupy the same LangGraph superstep. The shared downstream
    # node is scheduled once in the next superstep after every selected branch.
    # A fixed two-node barrier would incorrectly wait on an unselected branch.
    builder.add_edge("retrieve_oncology", "join_evidence")
    builder.add_edge("retrieve_cardiology", "join_evidence")
    builder.add_edge("join_evidence", "synthesis")
    builder.add_edge("synthesis", "assurance")
    for node in ("assurance", "fallback", "confirm_report"):
        builder.add_edge(node, END)
    return builder.compile()


def run_workflow(
    question: str,
    report_bytes: bytes | None = None,
    report_name: str | None = None,
    report_confirmed: bool = False,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedder=None,
    retriever=None,
) -> dict:
    """Run without implicit cloud tracing, storage, ingestion, or report files."""
    if not isinstance(question, str) or len(question) > 4000:
        return {"error": "Enter a question of at most 4,000 characters.", "decision": "block", "final_answer": "The question could not be processed.", "executed_nodes": [], "trace": []}
    initial = {
        "question": question.strip(), "report_bytes": report_bytes, "report_name": report_name,
        "report_confirmed": bool(report_confirmed), "evidence": [], "completed_domains": [],
        "executed_nodes": [], "trace": [], "pathology_facts": {}, "missing_facts": [],
    }
    try:
        # Overrides ambient LANGSMITH_TRACING / LANGCHAIN_TRACING_V2 settings,
        # including context propagated into the parallel LangGraph executor.
        with tracing_context(enabled=False):
            result = build_graph(index_dir, embedder, retriever).invoke(initial, config={"recursion_limit": 20})
    except (IndexSetupError, ReportError) as exc:
        result = {
            "question": initial["question"], "error": str(exc), "decision": "block",
            "final_answer": f"{DISCLAIMER}\n\n{exc}", "executed_nodes": [], "trace": [],
            "evidence": [], "domains": [], "route": "error",
        }
    result.pop("report_bytes", None)
    result.pop("report_name", None)
    result["cloud_trace_status"] = export_redacted_trace(result)
    return result
