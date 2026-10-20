# Paste this into Codex

Build the full runnable Windows/Python project called **CareGraph AI** from the attached `CareGraph_AI_Requirements.md` document. Read and follow that document and all seven diagrams in `diagrams/` before writing code.

Implement it **incrementally**, stopping to show the code and actual terminal output of each milestone:

1. A LangGraph `StateGraph` with coordinator, oncology retrieval, cardiology retrieval, join, synthesis, assurance, and fallback nodes. Print clearly: `NODE = DO`, `EDGE = GO`, `STATE = KNOW`. Coordinator sets the routing decision in state and a conditional edge performs routing.
2. Support multi-domain questions by running **both** retrieval nodes in parallel, correctly merging evidence with a reducer or separate evidence fields. Avoid parallel state-update errors.
3. Implement a real local **offline ingestion pipeline** for synthetic reference documents: load, parse, split into chunks, embed using a local Sentence Transformers model, and persist FAISS vectors with document source metadata.
4. Implement separate **runtime RAG**: embed question, search existing index, retrieve top-k passages, write evidence to LangGraph shared state, synthesize a cited educational response. Do **not** reindex the whole corpus for every question.
5. Support optional synthetic `.txt`/`.pdf` pathology report upload. Parse text and extract tumor fields for confirmation during a run. **Do not persistently index the uploaded pathology report**. Use the extracted facts and question to search existing synthetic guidance files. Display missing information; do not diagnose/stage or independently prescribe treatment.
6. Provide an offline deterministic synthesis mode as the **default** so paid APIs are unnecessary. Optional LLM/Claude backend may be scaffolded behind an environment toggle, but must not block the local project.
7. Add assurance: release an evidence-linked educational answer, rewrite unsupported text, escalate patient-specific treatment decisions, and block forbidden actions. Never claim a synthetic recommendation is authentic NCCN guidance.
8. LangSmith optional and OFF by default; no real PHI or clinical records.
9. Provide tests for oncology, cardiology, both, fallback, state merging, no reindexing on query or one-time upload, citation correctness, and safe escalation.
10. Create `README.md` with Windows VS Code setup (`py -3.11 -m venv .venv`, PowerShell activation alternative, `pip install -r requirements.txt`, `python ingest.py`, `python app.py`, and `pytest -q`). Include a printout of graph routing and before/after shared-state snapshots.

Use only synthetic local records. Favor simple, commented, easy-to-explain functions and files over production complexity. Implement working code; verify execution and tests locally, and report what was implemented versus what remains future work. For any dependency issue, resolve it and continue. Read the diagrams' source `.gv` or `.svg` files if needed.
