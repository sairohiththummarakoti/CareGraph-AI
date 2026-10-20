# CareGraph AI acceptance checklist

This checklist interprets the supplied requirements and seven diagrams as project source material. The user's request is to build the application end to end; the starter pack's suggestion to stop after every milestone is not a separate user instruction.

## Required application behavior

| Requirement | Observable acceptance evidence |
|---|---|
| FR-01 Input | CLI accepts a question, with or without a synthetic TXT/PDF report. Empty or malformed input produces a useful error. |
| FR-02 Routing | Oncology, cardiology, combined, and unsupported questions deterministically select the expected route. |
| FR-03 Graph execution | A real LangGraph `StateGraph` executes coordinator, selected retrieval branches, join, synthesis, assurance, or fallback. Conditional edges read the coordinator's state. |
| FR-04 Offline ingestion | A separate ingestion command validates approved synthetic references, preserves sections, chunks with overlap, embeds locally, and persists a FAISS index plus metadata. |
| FR-05 Semantic retrieval | Runtime embeds only the current query and confirmed facts, searches the existing index, and returns ranked top-k domain-appropriate evidence. “Heart attack” retrieves relevant “myocardial infarction” evidence. |
| FR-06 Shared state | A typed shared state contains the question, domain decisions, report findings, evidence, draft, final answer, and assurance decision. State is distinct from the vector index. |
| FR-07 Parallel merge | Combined requests execute both retrieval branches. Reducers or separate fields retain both results; join and synthesis run after the selected branches complete. |
| FR-08 Report lifetime | Report input is session-scoped. Queries and report uploads leave reference index file checksums unchanged and do not persist uploaded report text. |
| FR-09 Report confirmation | Display extracted facts and missing fields before using confirmed facts in retrieval. Do not infer missing stage, tumor findings, or biomarkers. |
| FR-10 Grounded output | Deterministic offline synthesis works without paid APIs. Every factual excerpt links to an actual retrieved filename, section, and chunk ID. Unsupported or empty evidence produces a bounded fallback. |
| FR-11 Assurance | Educational responses may release; patient-specific treatment/dose requests escalate; forbidden actions block; unsupported drafts are rewritten and rechecked or safely rejected. |
| FR-12 Tracing | Cloud tracing is off by default, including when unrelated ambient tracing variables are set. Optional tracing requires explicit synthetic-data opt-in. |
| Reproducibility | Python 3.11/3.12 Windows setup works. Dependencies/model revision are pinned, corpus is stable, and first model download is explained. |
| Observability | Terminal output names executed nodes, route, changed state fields, source IDs, and evidence counts, plus `NODE = DO`, `EDGE = GO`, and `STATE = KNOW`. |
| Documentation | README contains Windows venv instructions and activation alternative, ingestion/query/test commands, graph routing, before/after state snapshots, glossary, and clear future-work boundaries. |

## Source acceptance scenarios

- A: “What is oncology care?” → oncology and cited educational evidence.
- B: “What is a heart attack?” → cardiology and semantically relevant myocardial infarction evidence.
- C: “What considerations apply to cancer plus heart disease?” → both domains retained.
- D: “What is the weather?” → scope clarification/fallback.
- E: Report plus “Which evidence is relevant to this tumor profile?” → extracted/confirmed facts and existing reference retrieval.
- F: Missing biomarker or stage → explicit missing-data flags; clinician review without invention.
- G: “Choose my chemotherapy and dose” → escalation without a regimen.
- H: Repeated question → no reference reindexing.
- I: Parallel retrieval → no lost evidence, duplicate synthesis, or state update error.
- J: Tracing disabled → local workflow with no cloud trace.

## High-value edge cases

1. **Reports in the reference folder:** ingestion must explicitly select reference sources, excluding `sample_pathology_report.txt` and uploads.
2. **Unconfirmed findings:** parsing success does not equal human confirmation; report content must not silently become verified facts.
3. **Parallel completion:** test single-domain and combined routes; an unconditional two-branch barrier can deadlock single-domain requests, while separate branch edges can accidentally trigger synthesis twice.
4. **Index compatibility:** detect missing files, corrupt metadata, model/dimension mismatch, and stale content. Give an ingestion command rather than silently rebuilding on a query.
5. **No evidence:** an irrelevant or low-score query must not receive unrelated educational claims solely because FAISS always returns neighbors.
6. **Citation integrity:** validate citations against returned chunk IDs; stale filenames, invented IDs, or uncited added facts must not pass assurance.
7. **Report parser limits:** reject unsupported types, oversized inputs, invalid/encrypted/image-only PDFs, malformed encoding, ambiguous duplicate fields, and unusable empty text with understandable feedback.
8. **Missing means unknown:** blank or “not provided” report fields should remain missing; do not infer stage from tumor size or diagnose from keywords.
9. **Scope and intent:** catch treatment, prescribing, dosing, diagnostic, staging, and instructions to bypass assurance, including requests phrased without exact demonstration keywords.
10. **Untrusted report text:** embedded instructions are data and cannot authorize indexing, logging, cloud transmission, or overriding safety behavior.
11. **Tracing and persistence:** do not accidentally serialize raw reports into logs, feedback, checkpoints, browser state downloads, or cloud traces. Keep `.env`, caches, indexes, and upload artifacts gitignored.
12. **Synthetic labeling:** bundled documents and output must not be represented as authentic NCCN guidance or a validated clinical system.

## Scope interpretation

The required deliverable is the full local educational application, including the CLI, semantic FAISS RAG, pathology handling, assurance, tests, and Windows documentation. The requirements list a Streamlit UI, feedback/evaluations, optional LLM synthesis, optional hybrid search, and cloud tracing as later or optional features. Cloud deployment, real PHI, licensed NCCN content, FHIR, OpenSearch, S3, A2A, and MCP production integrations are explicitly outside the initial scope.

All seven diagrams were inspected through their SVG text and available editable sources. Diagrams 02 and 06 add explicit source/version lineage and reference-currentness expectations; diagram 03 emphasizes confirmation before retrieval; diagram 07 shows rewrite followed by revalidation. The markdown requirements are the source of truth when illustrative diagrams simplify the graph or show an optional LLM.
