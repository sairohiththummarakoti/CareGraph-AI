# CareGraph AI — Product Requirements and Flow Diagrams

**Status:** Proposed learning-project specification, not a deployed clinical system.  
**Audience:** Python/Windows/VS Code/Codex learner and technical interview preparation.  
**First milestone:** Run a LangGraph workflow locally, observe nodes/edges/state, then retrieve synthetic reference passages from FAISS.  
**Later extension:** Optional Claude-based generation, LangSmith tracing, and OpenSearch integration.

## 1. Problem and outcome

A clinician or care manager asks a question about oncology, cardiology, or both. Optionally, a user submits a synthetic pathology report. CareGraph AI routes to the appropriate retrieval node(s), obtains relevant passages from an **already-built knowledge index**, combines retrieved evidence with the question and any confirmed report findings in **LangGraph shared state**, and produces a **cited draft for clinician review**. The project teaches exactly where the RAG ingestion pipeline ends and the runtime RAG pipeline begins.

CareGraph AI **must not diagnose a patient, prescribe medication, or independently select treatment**. Any disease-specific explanation is for educational demonstration only.

## 2. Mental model

- **Node = DO:** a function such as coordinator, report parser, retrieval, synthesis, or assurance.
- **Edge = GO:** a connection that determines the next node.
- **Conditional edge = ROUTE:** chooses oncology, cardiology, both, or fallback by reading state.
- **Shared state = KNOW:** the structured values nodes read and update: question, selected domains, report findings, retrieved chunks, draft answer, and review decision. This is **not** a folder, an agent, or FAISS/OpenSearch.
- **FAISS / OpenSearch = INDEX:** stores chunk embeddings and metadata for similarity search, independent of LangGraph state.
- **LangGraph = BUILD/RUN:** workflow execution, routing, state, and optional checkpoints.
- **LangSmith = TRACE/EVALUATE:** optional observing/debugging/evaluating execution; keep off for unapproved data.

## 3. Two pipelines and the pathology exception

### A. Offline indexing — when the reference library changes

Approved synthetic oncology/cardiology documents → validate and parse → clean and chunk → embed chunks → build/persist FAISS index + source metadata. Run again only for changed or new approved reference content. This is not repeated for every question.

### B. Runtime RAG — for each new question

Question → coordinator and conditional routing → query embedding → search existing FAISS index → retrieve and rank chunks → update LangGraph shared state → synthesis (deterministic or optional LLM) → assurance checkpoint → cited answer, clarification, or escalation.

### C. One-time pathology-report upload — special case

Synthetic report PDF → secure parse → extract tumor-specific facts / highlight missing facts → optional human confirmation → **do not permanently index the report** → search the already-indexed guideline-like knowledge using the question + verified findings → synthesize an evidence-grounded explanation for clinician review. Persistent report indexing requires a separate authorized retention/search use case and is deliberately excluded from this milestone.

**NCCN content:** NCCN guidelines are proprietary/licensed. Do not download, redistribute, or bundle NCCN PDFs without appropriate authorization. Use clearly labeled **synthetic guidance excerpts** for the tutorial; no synthetic content should be represented as genuine NCCN treatment guidance.

## 4. Scope

### Must-have for version 1

1. Local command-line interface on Windows using Python 3.11 or 3.12.
2. LangGraph `StateGraph` with coordinator, oncology retrieval, cardiology retrieval, synthesis, assurance, and fallback nodes.
3. Deterministic coordinator routing for oncology, cardiology, combined questions, and unsupported input.
4. `TypedDict` shared state; distinct evidence fields for both domains **or reducers** for parallel writes.
5. Synthetic oncology and cardiology documents; ingest, chunk, embed, index to FAISS once.
6. Runtime semantic retrieval with citations including source filename, section, and chunk ID.
7. Synthesis that works **offline without any paid LLM API**, using a simple extractive template.
8. Optional local pathology `.txt` and `.pdf` parsing, with synthetic tumor fields and missing-data flags.
9. Assurance checkpoint that never autonomously selects treatment, and distinguishes supported education from clinician-required decisions.
10. Terminal logs/trace showing node execution order, route, state changes, and retrieved chunk count.
11. Unit/integration tests and Windows setup instructions.

### Should-have for version 1.1

1. Optional LLM synthesis behind an environment flag; send only synthetic data.
2. Optional LangSmith tracing; OFF by default. Avoid PHI in traces.
3. Streamlit user interface for question/report upload and execution trace.
4. Keyword + semantic **hybrid search** as optional extension; the local FAISS MVP provides **semantic only**.
5. Feedback capture, case-based evaluation, and simple refusal / escalation metrics.

### Out of scope for initial project

Real patient records or PHI; licensed NCCN content; FHIR production integrations; autonomous treatment recommendations; diagnosis/staging inference; cloud deployment; AWS S3, OpenSearch, A2A, or MCP production servers. These may be later architectural extensions, not prerequisites today.

## 5. Functional requirements

| ID | Requirement | Verification |
|---|---|---|
| FR-01 | Accept a user question and optional synthetic report. | CLI test with and without report |
| FR-02 | Coordinator selects oncology/cardiology/both/fallback based on content. | Four routing unit tests |
| FR-03 | Conditional edges run selected retrieval node(s). | Printed graph execution trace |
| FR-04 | Index synthetic knowledge only during ingestion command. | Index files created once |
| FR-05 | Use the query embedding to retrieve top-k chunks at runtime. | Top-k results and sources displayed |
| FR-06 | Store question, domains, and retrieved evidence in shared state. | State snapshots / unit tests |
| FR-07 | Merge parallel oncology and cardiology evidence safely. | No lost chunks in combined query |
| FR-08 | Parse synthetic pathology report without indexing it by default. | Index checksum unchanged after upload |
| FR-09 | Show extracted report facts and missing elements for confirmation. | Sample report structured output |
| FR-10 | Produce evidence-linked explanatory draft / safe fallback. | Every factual excerpt source-linked |
| FR-11 | Route high-risk treatment requests to clinician review rather than automated prescribing. | Safety test passes |
| FR-12 | Support optional LangSmith tracing without data leakage by default. | Tracing disabled unless opted in |

## 6. Nonfunctional requirements

- **Privacy:** Synthetic-only, no real PHI. Keep local docs and upload data in scoped directories; `.env` in `.gitignore`.
- **Safety:** Never suggest a patient-specific treatment plan autonomously. If clinical prerequisites are missing, label uncertainty and defer.
- **Reproducibility:** Stable test dataset, deterministic routing, dependency installation instructions, and a fixed embedding model version or pinned environment on completion.
- **Observability:** Print each executed node, conditional route selected, retrieved source IDs, and state fields changed; never print private text in production.
- **Usability:** Display the complete evidence flow and a short glossary in README.
- **Performance:** Small reference corpus should run on an ordinary Windows laptop; first-time embedding-model download may take longer.

## 7. State schema (illustrative)

```python
from typing import Annotated, TypedDict, Literal
import operator

class EvidenceChunk(TypedDict):
    domain: str
    text: str
    source: str
    section: str
    chunk_id: str

class CareState(TypedDict, total=False):
    question: str
    domains: list[str]                  # oncology, cardiology, or both
    report_path: str | None             # local, synthetic-only reference
    pathology_facts: dict[str, str]     # verified extracted facts
    missing_facts: list[str]
    evidence: Annotated[list[EvidenceChunk], operator.add]
    draft_answer: str
    final_answer: str
    decision: Literal['release','rewrite','escalate','block']
    executed_nodes: Annotated[list[str], operator.add]
```

**Important parallelism detail:** If both retrieval nodes update the same `evidence` field concurrently, a reducer such as `Annotated[..., operator.add]` is required, or use separate domain-specific fields. Nodes return partial state updates; they do not manually create a new shared-state database.

## 8. Main graph nodes and routing

| Node | Input from state | Action | Partial state output |
|---|---|---|---|
| `coordinator` | `question`, optionally report facts | Deterministic classify domains | `domains` |
| `parse_report` | `report_path` | Extract text / structured facts | `pathology_facts`, `missing_facts` |
| `retrieve_oncology` | `question`, `pathology_facts` | Embed query + search oncology FAISS | oncology `evidence` chunks |
| `retrieve_cardiology` | `question` | Embed query + search cardiology FAISS | cardiology `evidence` chunks |
| `join_evidence` | collected `evidence` | Wait/confirm selected results | no change or summary counts |
| `synthesis` | `question`, `evidence`, `pathology_facts` | Assemble cited educational draft | `draft_answer` |
| `assurance` | `draft_answer`, sources, missing fields | Scope, support, and safety review | `decision`, `final_answer` |
| `fallback` | `question` | Ask clarification or refuse out of scope | `final_answer` |

Routing: 
- Oncology only → `retrieve_oncology` → synthesis.
- Cardiology only → `retrieve_cardiology` → synthesis.
- Both → **parallel** retrieval (proper reducer + join) → synthesis.
- Ambiguous/unsupported → fallback, not unsafe guesswork.

The pathology upload can be parsed as a dedicated node before routing. In version 1, use deterministic rules. Later, move to a constrained classifier or LLM router with explicit fallback and permitted actions.

## 9. Suggested files and modules

```text
caregraph-ai/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ .gitignore
├─ app.py                      # CLI menu
├─ ingest.py                   # offline create/update index
├─ caregraph/
│  ├─ state.py                 # CareState
│  ├─ graph.py                 # nodes + edges + graph.compile()
│  ├─ routing.py               # classification / route mapping
│  ├─ parsing.py               # pathology upload extraction
│  ├─ retrieval.py             # query embedding + FAISS lookup
│  ├─ synthesis.py             # cited answer template; optional LLM
│  ├─ assurance.py             # release / rewrite / escalate / block
│  └─ trace.py                 # local execution prints
├─ data/
│  └─ synthetic/
│     ├─ oncology_reference.txt
│     ├─ cardiology_reference.txt
│     └─ sample_pathology_report.txt
├─ indexes/                    # generated locally, gitignored
└─ tests/
   ├─ test_routing.py
   ├─ test_indexing.py
   ├─ test_parallel_state.py
   ├─ test_pathology.py
   └─ test_safety.py
```

## 10. Acceptance scenarios

| Case | Input | Expected behavior |
|---|---|---|
| A | “What is oncology care?” | Oncology route, cited retrieved educational text |
| B | “What is a heart attack?” | Cardiology route; semantic retrieval supports “myocardial infarction” |
| C | “What considerations apply to cancer plus heart disease?” | Two retrieval nodes; chunks from both present in final state |
| D | “What is the weather?” | Fallback / out-of-scope |
| E | Synthetic pathology report + “Which evidence is relevant to this tumor profile?” | Parse facts; search existing reference index; no persistent report indexing |
| F | Report omits important biomarker or stage | Flag missing detail and request clinician review; do not invent |
| G | “Choose my chemotherapy and dose” | No autonomous recommendation; escalate |
| H | Re-run same question | Reference indexing not triggered again |
| I | Parallel domain retrieval | Evidence from each domain retained; no state merge failure |
| J | LangSmith disabled | Execution succeeds entirely locally, with no cloud trace |

## 11. Suggested dependencies

- `langgraph`
- `langchain-core`
- `langchain-text-splitters`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf` (optional for PDFs; `.txt` works first)
- `python-dotenv`
- `langsmith` (optional tracing)
- `pytest`
- `rich` (optional clear terminal output)

Start with `pip` and a Windows `.venv`. Some models download weights on first run; no paid API is required for the initial version.

### Windows setup

```powershell
mkdir caregraph-ai
cd caregraph-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python ingest.py
python app.py
pytest -q
```

If PowerShell blocks activation, use ` .\.venv\Scripts\python.exe ` instead of activating.

## 12. Implementation stages

1. **Learn nodes/edges/state:** a rule-based coordinator and simple fixed nodes without vector search.
2. **Add real local RAG:** ingest sample knowledge, embeddings, FAISS; inspect retrieved chunks.
3. **Add pathology upload:** parse sample document at runtime and demonstrate no permanent index update.
4. **Add synthesis and assurance:** cite sources; release generic education or escalate risk.
5. **Add LangSmith (optional):** inspect traces with synthetic data only.
6. **Later:** hybrid search, OpenSearch, S3, proper MCP tools, A2A interoperability, clinician UI, certified evidence access, full eval suite.

## 13. Notes for technical accuracy

- Vector similarity helps retrieve *meaning-related* passages, but it does not guarantee the medical relevance or correctness of those passages.
- RAG is the **whole process** of retrieving passages and using them during generation. Retrieval alone is not the full RAG cycle.
- Indexing/chunking of reference documents happens offline; **query embedding is still needed at runtime** for semantic search.
- A one-time uploaded report can be parsed into structured facts without being embedded into the persistent reference vector index.
- LangGraph shared state is **not necessarily durable storage**. Persistence requires checkpoints; tracing/checkpoints must be controlled when dealing with sensitive information.
- A2A = agent-to-agent interoperability; MCP = protocol for access to tools/resources. Neither is required to learn the core `StateGraph` first.
- The **NCCN** acronym means National Comprehensive Cancer Network. The tutorial must not represent its synthetic guidance files as authentic NCCN recommendations.

## 14. Official references

- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- Conditional edges: https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges
- Graph API and reducers: https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/graph-api.mdx
- LangSmith tracing: https://reference.langchain.com/python/langsmith

