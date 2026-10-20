"""Exact extractive synthesis; no external LLM is simulated."""

from html import escape

DISCLAIMER = "Synthetic education only — not medical advice, diagnosis, staging, or a treatment recommendation."
NO_EVIDENCE = (
    f"{DISCLAIMER}\n\n"
    "No sufficiently relevant reference excerpts were found. Rephrase an oncology or cardiology "
    "education question. A clinician must review patient-specific questions."
)


def citation(chunk: dict) -> str:
    return f"[{chunk['source']} · {chunk['section']} · v{chunk['version']} · {chunk['chunk_id']}]"


def _literal(value: str) -> str:
    """Show report fields as data, without permitting Markdown/HTML formatting."""
    return escape(value).replace("\\", "\\\\").replace("`", "\\`").replace("*", "\\*").replace("_", "\\_").replace("[", "\\[").replace("]", "\\]")


def build_draft(state: dict) -> str:
    evidence = state.get("evidence", [])
    if not evidence:
        return NO_EVIDENCE
    lines = [
        DISCLAIMER, "",
        "Retrieved excerpts may not address all details of your question.",
        "", "Educational reference excerpts",
    ]
    for chunk in sorted(evidence, key=lambda c: (c["domain"], -c["score"], c["chunk_id"])):
        lines.extend(["", f"> {chunk['text'].replace(chr(10), chr(10) + '> ')}", "", citation(chunk)])
    if state.get("report_present") and state.get("report_confirmed"):
        lines.extend(["", "Confirmed synthetic report fields (user-confirmed transcription; not independently verified)"])
        for field, value in state.get("pathology_facts", {}).items():
            lines.append(f"- {field.replace('_', ' ').capitalize()}: {_literal(value)}")
        unknown = state.get("missing_facts", [])
        if unknown:
            lines.extend(["", "Unknown / not provided: " + ", ".join(item.replace("_", " ") for item in unknown) + "."])
        lines.extend(["", "Report fields are shown as supplied. Missing findings have not been inferred; clinician review is required."])
    lines.extend(["", "These excerpts come from a small synthetic teaching corpus. Similarity ranks retrieval results; it is not clinical confidence."])
    return "\n".join(lines)


def draft_is_supported(draft: str, state: dict) -> bool:
    """Only the exact evidence template is allowed; added claims cannot pass."""
    return draft == build_draft(state)
