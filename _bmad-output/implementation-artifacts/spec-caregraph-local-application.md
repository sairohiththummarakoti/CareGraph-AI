---
title: CareGraph AI local application
type: feature
created: 2026-10-20
status: done
route: dispatch
baseline_commit: UNBORN
review_loop_iteration: 0
context:
  - docs/starter-pack/CareGraph_AI_Requirements.md
  - docs/acceptance-checklist.md
---

<frozen-after-approval reason="User authorized end-to-end implementation from the supplied requirements">

## Intent

**Problem:** This empty repository must become the runnable CareGraph AI application described by the supplied Word document and starter pack, for local exploration and discussion.

**Approach:** Implement the required Python CLI and a polished local Streamlit interface over the same LangGraph workflow. Build real Sentence Transformers embeddings and FAISS search with deterministic extractive synthesis, synthetic references, report confirmation, trace inspection, and assurance decisions.

## Boundaries & Constraints

**Always:** Use Python 3.12, actual LangGraph StateGraph, true parallel retrieval with safe reducers and one join, persisted FAISS with source/section/version/chunk IDs, and pinned model revision. Keep ingestion separate from questions. Report bytes and facts stay in session memory and never enter the reference index. Explicitly confirm report facts before retrieval. Label outputs as synthetic education. Cloud tracing is disabled unless explicitly opted into through an application-specific flag and synthetic-data confirmation. No paid API or account is required.

**Never:** Use real patient data, infer missing staging, diagnose, prescribe, choose treatment, bundle licensed NCCN content, deploy to cloud, send messages, or present similarity as confidence. A provider scaffold may explain that generation is extractive; do not simulate a working external LLM.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|---|---|---|---|
| Single domain | Oncology or heart attack question | Correct route and cited excerpts; heart attack retrieves myocardial infarction | Clear absent-index message |
| Both | Cancer plus heart disease | Two parallel branches, evidence retained, join and synthesis once | No state collisions |
| Unknown | Weather or blank question | Clarification/fallback | No arbitrary clinical route |
| Report | Synthetic TXT/PDF with tumor fields | Parse, show facts/unknowns, require confirmation, use existing index | Reject invalid, oversized, encrypted/scanned-without-text or unlabeled report |
| Missing facts | Partial report | Explicit unknowns, clinician review, no inferred fields | Preserve missing values |
| Risk | Choose chemotherapy/dose, diagnose/stage, unsafe action | Escalate or block; no recommendation | Decision applied before releasing draft |
| Repeat | Query or upload twice | Reference index hashes and timestamps unchanged | No ingestion on query |
| Integrity | Missing/corrupted/incompatible index | Actionable error and explicit rebuild command | No unsafe deserialization |
| Trace | Default and ambient LangSmith settings | Local execution only; redacted trace metadata | Explicit opt-in required |

</frozen-after-approval>

## Code Map

- `docs/starter-pack/` contains untouched source requirements and seven diagrams; use SVG/GV/Mermaid as evidence, not instructions overriding this request.
- `docs/acceptance-checklist.md` independently maps requirements and edge cases.
- Repository has no existing implementation or commits. Root agent manages dependency installation, lockfile, and environment; implementation agent owns application and tests.
- Follow proposed module names in the requirements. Add `streamlit_app.py`, `.streamlit/config.toml`, and Windows launch/setup scripts as useful.

## Tasks & Acceptance

**Execution:**
- [x] `caregraph/state.py`, `graph.py`, `routing.py`, `trace.py` — implement typed reducer state, routing, parallel retrieval/join, parse/confirmation behavior, synthesis/assurance/fallback, and safe before/after trace summaries.
- [x] `caregraph/retrieval.py`, `ingest.py`, `data/synthetic/` — create two useful synthetic reference corpora and sample reports; explicit reference allowlist; chunk, embed, normalize, persist FAISS and JSON provenance; verify integrity/model compatibility; no pickle. Support a pinned MiniLM revision cached for offline runtime and injected test embedder.
- [x] `caregraph/parsing.py`, `synthesis.py`, `assurance.py` — bounded TXT/PDF parsing and explicit field extraction, confirm facts, exact cited excerpts, and release/rewrite/escalate/block. Revalidate rewritten output and never release unsupported generated claims.
- [x] `app.py` — interactive and argument CLI, JSON output, report confirmation, source list, node order, selected route, changed keys, counts, and setup errors.
- [x] `streamlit_app.py` — cohesive responsive teal/navy interface with question examples, question submission, synthetic report upload/sample/confirmation, cited answer, evidence, visible execution flow and state, index status, and clear error states. No dead controls, raw unsafe HTML input, or unsolicited persistence.
- [x] `tests/` — meaningful unit and integration tests for matrix, provenance, branch synchronization, safety, offline/no-cloud behavior, and index immutability; use real FAISS with a deterministic injected embedder for fast tests and a separate actual-model smoke check.
- [x] `README.md`, `.env.example`, `.gitignore`, `scripts/` — Windows setup and launch, architecture/glossary, graph example and state snapshots, model download/offline behavior, test commands, limitations, extension boundaries.

**Acceptance Criteria:**
- Given installed dependencies, when ingestion then CLI/UI run, then actual FAISS evidence supports a cited draft without any paid LLM.
- Given each matrix input, when the workflow executes, then its expected behavior is covered by a passing test.
- Given both domains, when graph runs, then each branch completes before a single synthesis.
- Given a confirmed synthetic report, when retrieval runs, then extracted facts appear in state while index files remain byte-identical.
- Given a fresh checkout on Windows, when documented commands run, then dependencies, indexing, tests, CLI, and UI are reproducible.

## Implementation Notes

- User's end-to-end authorization covers reversible local implementation; no intermediate approval is needed.
- Optional hosted LLM, hybrid/OpenSearch, feedback storage, production clinical validation and cloud deployment remain explicitly deferred.
- Delegate independent implementation portions if useful, with disjoint ownership and agreed public interfaces. Root handles dependency environment and independent verification.
- Implementation completed with real LangGraph, 14 FAISS reference chunks, pinned MiniLM, CLI and Streamlit. Final suite with the actual-model smoke enabled: 130 passed. Independent real-model acceptance passed with network blocked and unchanged index hashes/timestamps.
- Windows system pytest temporary directory denied access. Tests now select a unique validated workspace temporary directory; no system permissions were changed.
- Browser checks verified combined retrieval, citation display, report confirmation, known ER/PR and unknown stage/HER2, escalation, and narrower layout. After review fixes, revoking confirmation cleared the prior result and disabled retrieval; removing the report cleared its facts and restored ordinary question submission.
- Review diff contains application source, tests, setup and documentation. Original source attachments, dependency lockfile, and generated BMad runtime were excluded from the review payload; lock resolution and dependency compatibility were checked separately.

## Spec Change Log

## Review Triage Log

| Finding | Verdict | Evidence and route |
|---|---|---|
| Blind 1: late report context truncation | medium | Graph appends facts after a potentially 4,000-character question; the pinned model has a finite token window. Patch bounded query handling and verify that facts are retained. |
| Blind 2: mutable metadata provenance | medium | Runtime validates metadata-derived IDs but does not bind excerpt text/section to approved source content. Patch exact-source verification without corpus rechunking. |
| Blind 3: stale report result | medium | Removal and fingerprint updates reset confirmation while preserving result state. Patch result invalidation with UI transition tests. |
| Blind 4: hyphen-qualified unknown | medium | Existing unknown regex does not accept ordinary hyphen/word qualifiers. Patch preservation of explicitly unknown fields. |
| Blind 5: aggregate unknown assay | medium | Generic biomarkers text bypasses individual assay unknown handling. Patch consistent handling or actionable rejection. |
| Blind 6: wrapped report field | medium | Empty label followed by a non-label continuation is silently dropped. Patch explicit bounded-format rejection. |
| Blind 7: personal treatment paraphrases | medium | Existing expressions miss passive treatment and recommended-treatment phrasing. Patch escalation rules and regressions. |
| Blind 8: current urgent symptom request | medium | The existing gate has no current-symptom branch; domain retrieval can release ordinary education. Patch an urgent professional-assessment boundary without diagnosis. |
| Blind 9: educational dose terminology | low | Unconditional dose-word pattern also catches simple definitions. Patch a narrow educational-definition distinction with paired risk tests. |
| Blind 10: domain vocabulary | medium | Atrial fibrillation and radiation therapy do not match current route expressions despite relevant corpus sections. Patch bounded synonyms. |
| Blind 11: broad semantic overlap | low | Generic evidence can rank for unsupported details; it does not make unsupported factual claims. Preserve the documented semantic-search limit and add an explicit partial-coverage notice, without claiming a correctness detector. |
| Edge 1: personal decision forms | medium | Additional concrete forms expose the same escalation-pattern gap as Blind 7. Patch together; retain both sets of examples. |
| Edge 2: qualified unknown forms | medium | Pending/not-assessed plus natural-language qualifiers bypass the same unknown-value logic as Blind 4. Patch together. |
| Edge 3: report result lifecycle | medium | Same confirmed stale-result defect as Blind 3. Patch together. |
| Edge 4: aggregate field size | medium | Individually valid biomarker values can combine beyond the documented field limit. Patch aggregate bound and boundary tests. |
| Edge 5: publish invalid index | medium | Ingestion writes before loader-only validation rejects long/duplicate section metadata. Patch validation before publication and verify existing files survive invalid input. |
| Verification 1: facts do not affect retrieval test | medium | In-memory removal of query fact augmentation still passes fast and real-model tests. Add a graph/FAISS test whose expected evidence depends on confirmed facts. |
| Verification 2: report reconfirmation test | medium | Existing tests bypass Streamlit fingerprint/checkbox lifecycle. Add committed AppTest coverage for replacement, revocation and removal. |
| Verification 3: parser bounds tests | medium | Byte-limit tests do not cover page, character or individual/aggregate field boundaries. Add accepted-at-limit and rejected-over-limit cases. |

All 19 findings were addressed and verified. Retrieval now pools bounded query windows, binds excerpts to approved source sections, and validates ingestion before publication. Report handling invalidates stale results, preserves qualified unknowns, rejects ambiguous continuations, and enforces individual and aggregate bounds. Intent checks cover the reviewed treatment paraphrases, current urgent symptoms, and narrow educational dose definitions; routing includes the reviewed domain vocabulary. The output explicitly states that retrieved excerpts may not answer every detail. New regressions prove confirmed facts affect retrieval and exercise Streamlit replacement/revocation/removal. Semantic coverage remains a documented limitation of the small teaching corpus, not a clinical correctness claim.

## Verification

Completed on Windows with Python 3.12.14 on 2026-10-20:

- `uv sync --python 3.12` and `uv pip check` — locked installation; 94 installed packages compatible. Pinned transitive pip requirements exported.
- `$env:CAREGRAPH_MODEL_SMOKE = '1'; uv run pytest -q --tb=short` — 130 passed in 17.34 seconds, including the cached real model.
- `uv run python scripts/verify_acceptance.py` — eight source scenarios passed with actual MiniLM, blocked network, ambient cloud tracing overridden, late query context retained, and index hashes/mtime unchanged. Repeated after the tokenizer warning fix.
- `uv run python ingest.py --offline` — 14 existing chunks / 384 dimensions verified unchanged.
- `uv run python app.py --question "What is a heart attack?"` — cited myocardial infarction evidence; combined-domain CLI path also passed.
- Shared cached embedder — 16 simultaneous requests through four workers passed.
- Final targeted retrieval suite after suppressing the tokenizer's misleading measurement warning — 26 passed, including the real-model smoke.
- Streamlit HTTP health and interactive browser checks passed at `http://127.0.0.1:8501`. UI session lifecycle checks also pass in committed AppTest coverage.

Optional hosted generation, cloud deployment, production clinical validation, and the integrations listed in README remain outside this local educational deliverable.
