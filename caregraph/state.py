"""Shared workflow values. This state is memory, not the reference index."""

import operator
from typing import Annotated, Literal, TypedDict

Domain = Literal["oncology", "cardiology"]
Decision = Literal["release", "rewrite", "escalate", "block"]


class EvidenceChunk(TypedDict):
    domain: str
    text: str
    source: str
    section: str
    version: str
    chunk_id: str
    score: float


class TraceEvent(TypedDict):
    node: str
    changed_keys: list[str]
    before: dict
    after: dict


class CareState(TypedDict, total=False):
    question: str
    domains: list[str]
    route: str
    report_bytes: bytes | None
    report_name: str | None
    report_present: bool
    report_confirmed: bool
    awaiting_confirmation: bool
    pathology_facts: dict[str, str]
    missing_facts: list[str]
    evidence: Annotated[list[EvidenceChunk], operator.add]
    completed_domains: Annotated[list[str], operator.add]
    draft_answer: str
    final_answer: str
    decision: Decision
    risk_reason: str
    evidence_counts: dict[str, int]
    assurance_checks: list[str]
    executed_nodes: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceEvent], operator.add]
    error: str
    cloud_trace_status: str
