"""CareGraph's local, session-scoped educational workspace."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from caregraph.graph import run_workflow
from caregraph.parsing import ReportError, parse_report
from caregraph.retrieval import DEFAULT_INDEX_DIR, IndexSetupError, ReferenceIndex, SentenceTransformerEmbedder


ROOT = Path(__file__).resolve().parent
EXAMPLES = {
    "Oncology care": "What is oncology care?",
    "Heart attack": "What is a heart attack?",
    "Cancer + heart disease": "What considerations apply to cancer plus heart disease?",
}
NODE_LABELS = {
    "parse_report": "Parse report",
    "coordinator": "Coordinate",
    "confirm_report": "Confirm report",
    "retrieve_oncology": "Oncology retrieval",
    "retrieve_cardiology": "Cardiology retrieval",
    "join_evidence": "Join evidence",
    "synthesis": "Synthesize",
    "assurance": "Assurance",
    "fallback": "Clarify scope",
}

st.set_page_config(
    page_title="CareGraph AI · Evidence workspace",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Only this fixed, application-owned markup is rendered as HTML. Input and report
# contents are always rendered by text/dataframe/code components.
st.markdown(
    """
    <style>
    :root { --ink:#142f43; --muted:#647b88; --teal:#087f83; }
    .stApp { background:#f4f7f8; color:var(--ink); }
    [data-testid="stAppViewBlockContainer"] {
        max-width:1510px; padding-top:2.1rem; padding-bottom:3rem;
    }
    [data-testid="stHeader"] { background:rgba(244,247,248,.95); }
    [data-testid="stSidebar"] { background:#102f42; }
    [data-testid="stSidebar"] * { color:#e8f1f4; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color:#abc0ca; }
    [data-testid="stSidebar"] hr { border-color:#385666; }
    [data-testid="stSidebar"] [data-testid="stAlert"] { background:#214557; }
    [data-testid="stSidebar"] button { background:#244a5b; border-color:#4b6d7d; }
    [data-testid="stSidebar"] button:hover { border-color:#65ddd1; }
    h1,h2,h3 { letter-spacing:-.035em; color:var(--ink); }
    h1 { font-size:2.8rem!important; font-weight:720!important; line-height:1.12!important; }
    h3 { font-size:1.18rem!important; }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius:16px!important; border-color:#dce6ea!important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
        background:white; border-radius:15px;
    }
    [data-testid="stTextArea"] textarea {
        background:#f8fafb; font-size:1.03rem; border-color:#d6e2e7; line-height:1.6;
    }
    [data-testid="stButton"] button {
        border-radius:9px; font-weight:600; min-height:2.6rem;
    }
    [data-testid="stButton"] button p {
        white-space:normal; overflow:visible; text-overflow:clip;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background:#087f83; border-color:#087f83; color:white;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:1.7rem; }
    [data-testid="stTabs"] [data-baseweb="tab"] { font-size:.97rem; }
    [data-testid="stMetricValue"] { font-size:1.5rem; color:#087f83; }
    [data-testid="stMetricLabel"] { color:#5b7280; }
    .cg-brand { display:flex; gap:12px; align-items:center; margin:5px 0 24px; }
    .cg-mark {
        width:43px; height:43px; background:#126373; color:#8ce2d3;
        border:1px solid #3c8790; border-radius:12px; display:grid;
        place-items:center; font-size:30px; line-height:1;
    }
    .cg-brand-name { font-size:21px; font-weight:700; letter-spacing:-.5px; }
    .cg-brand-sub { color:#a7c2cd; font-size:11px; letter-spacing:1.8px; }
    .cg-eyebrow { font-size:11px; font-weight:700; letter-spacing:2px; color:#087f83; margin-bottom:12px; }
    .cg-subtitle { font-size:1.05rem; color:#59717f; max-width:760px; line-height:1.65; margin:5px 0 22px; }
    .cg-local { display:inline-block; padding:5px 10px; background:#164d58; color:#b8efe3!important;
        border:1px solid #39727a; border-radius:6px; font-size:11px; letter-spacing:1.2px; }
    .cg-scope { border-left:3px solid #65bfb7; color:#506b79; padding:9px 14px;
        background:#eaf4f3; border-radius:0 7px 7px 0; margin:8px 0 22px; font-size:.89rem; }
    .cg-step { font-size:11px; font-weight:700; letter-spacing:1.6px; color:#087f83; margin-bottom:7px; }
    .cg-flow { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 20px; }
    .cg-flow span { padding:7px 11px; border-radius:7px; background:#e7f3f1;
        border:1px solid #c5e3de; color:#236e6b; font-size:.78rem; }
    .cg-empty { padding:27px 8px 20px; text-align:center; }
    .cg-empty-icon { font-size:31px; color:#218c86; margin-bottom:8px; }
    .cg-empty-title { font-size:1.15rem; font-weight:650; color:#254455; }
    .cg-empty-copy { color:#718792; font-size:.93rem; line-height:1.6; margin-top:7px; }
    .cg-footer { color:#78909c; font-size:.77rem; margin-top:24px; }
    @media (max-width:1100px) {
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
            flex-direction:column; align-items:stretch; gap:.75rem;
        }
        [data-testid="stMain"] [data-testid="stColumn"] {
            width:100%!important; flex:1 1 100%!important; min-width:0!important;
        }
        [data-testid="stAppViewBlockContainer"] {
            padding-left:1.35rem; padding-right:1.35rem;
        }
        h1 { font-size:2.25rem!important; }
        .cg-subtitle { font-size:1rem; margin-bottom:8px; }
    }
    @media (max-width:600px) {
        [data-testid="stAppViewBlockContainer"] {
            padding-left:1rem; padding-right:1rem; padding-top:1.4rem;
        }
        h1 { font-size:2rem!important; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.6rem; }
        [data-testid="stTabs"] [data-baseweb="tab"] { font-size:.85rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformerEmbedder:
    """Cache only public model weights; never cache questions or report data."""
    return SentenceTransformerEmbedder(offline=True)


def init_session() -> None:
    for key, value in {
        "question": "",
        "result": None,
        "result_question": "",
        "sample_report": None,
        "upload_generation": 0,
        "report_fingerprint": None,
        "confirmed_report": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def choose_example(question: str) -> None:
    st.session_state.question = question


def clear_result() -> None:
    st.session_state.result = None
    st.session_state.result_question = ""


def confirmation_changed() -> None:
    if not st.session_state.confirmed_report:
        clear_result()


def clear_workspace() -> None:
    st.session_state.question = ""
    st.session_state.result = None
    st.session_state.result_question = ""
    st.session_state.sample_report = None
    st.session_state.report_fingerprint = None
    st.session_state.confirmed_report = False
    st.session_state.upload_generation += 1


def remove_report() -> None:
    clear_result()
    st.session_state.sample_report = None
    st.session_state.report_fingerprint = None
    st.session_state.confirmed_report = False
    st.session_state.upload_generation += 1


def index_summary() -> tuple[str, str]:
    if not (Path(DEFAULT_INDEX_DIR) / "metadata.json").exists():
        return "Setup needed", "Build the reference library before your first question."
    try:
        status = ReferenceIndex(embedder=get_embedder()).status()
    except IndexSetupError as exc:
        return "Index needs repair", str(exc)
    return "Library ready", f"{status['count']} reference chunks · Integrity verified. The cached model loads when needed."


def render_report() -> tuple[bytes | None, str | None, bool, bool]:
    """Returns in-memory bytes/name, explicit confirmation, and parse validity."""
    with st.container(border=True):
        st.markdown('<div class="cg-step">OPTIONAL CONTEXT</div>', unsafe_allow_html=True)
        st.subheader("Add a synthetic report")
        st.caption("Review extracted facts before they are used in retrieval.")
        uploaded = st.file_uploader(
            "Synthetic TXT or text-based PDF",
            type=["txt", "pdf"],
            key=f"report_upload_{st.session_state.upload_generation}",
            help="Synthetic reports only. Scanned, encrypted, unlabeled, and oversized files are rejected.",
        )
        if st.button("Use sample report", icon=":material/description:", width="stretch"):
            sample_path = ROOT / "data" / "synthetic" / "sample_pathology_report.txt"
            if sample_path.exists():
                st.session_state.sample_report = (
                    sample_path.read_bytes(),
                    sample_path.name,
                )
                st.session_state.question = "Which evidence is relevant to this tumor profile?"
                st.session_state.upload_generation += 1
                st.rerun()
            else:
                st.error("The bundled sample report is missing. Restore data/synthetic.")
        report_bytes, report_name = None, None
        if uploaded is not None:
            report_bytes, report_name = uploaded.getvalue(), uploaded.name
        elif st.session_state.sample_report is not None:
            report_bytes, report_name = st.session_state.sample_report
            st.caption(f"Sample loaded: {report_name}")

        if report_bytes is None:
            if st.session_state.report_fingerprint is not None:
                clear_result()
                st.session_state.report_fingerprint = None
                st.session_state.confirmed_report = False
            st.caption("Reports stay in this session. They are never added to the reference index.")
            return None, None, False, True

        fingerprint = hashlib.sha256(report_bytes + str(report_name).encode()).hexdigest()
        if st.session_state.report_fingerprint != fingerprint:
            clear_result()
            st.session_state.report_fingerprint = fingerprint
            st.session_state.confirmed_report = False
        try:
            preview = parse_report(report_bytes, str(report_name))
        except ReportError as exc:
            st.error(str(exc))
            st.button("Remove report", on_click=remove_report, width="stretch")
            return report_bytes, report_name, False, False
        facts = preview.get("pathology_facts", {})
        missing = preview.get("missing_facts", [])
        st.markdown("**Extracted facts**")
        if facts:
            st.dataframe(
                [{"Field": field.replace("_", " ").title(), "Reported value": str(value)}
                 for field, value in facts.items()],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No recognized facts were found.")
        if missing:
            st.warning("Unknown: " + ", ".join(str(field).replace("_", " ") for field in missing))
        st.caption("Missing information stays unknown. No diagnosis or stage is inferred.")
        confirmed = st.checkbox(
            "I reviewed these facts and confirm this report is synthetic.",
            key="confirmed_report",
            on_change=confirmation_changed,
        )
        st.button("Remove report", on_click=remove_report, width="stretch")
        return report_bytes, report_name, confirmed, True


def render_answer(result: dict) -> None:
    decision = result.get("decision", "")
    if result.get("error"):
        st.error(result["error"])
        st.code("uv run python ingest.py", language="powershell")
        return
    if result.get("awaiting_confirmation"):
        st.info("Review the report facts and explicitly confirm them before retrieval.")
        return
    if decision == "block":
        st.error("Blocked · This request exceeds the educational scope.")
    elif decision == "escalate":
        st.warning("Clinician review required")
    elif decision == "rewrite":
        st.info("Rewritten and revalidated for evidence support")
    else:
        st.success("Assurance complete · Educational output")
    answer = result.get("final_answer", "")
    if answer:
        # The exact, revalidated synthesis template escapes report values.
        # Streamlit's default renderer disables raw HTML.
        st.markdown(answer)
    else:
        st.info("No answer was released. Try a question about oncology or cardiology.")
    if result.get("missing_facts"):
        st.caption("Unknown report fields: " + ", ".join(result["missing_facts"]))


def render_evidence(result: dict) -> None:
    evidence = result.get("evidence", [])
    st.caption("Exact excerpts from the synthetic reference library. Similarity ranks retrieval; it is not clinical confidence.")
    if not evidence:
        st.info("No reference excerpts were returned for this run.")
    for index, item in enumerate(evidence, 1):
        title = f"{index:02d} · {item.get('domain', '').title()} · {item.get('section', 'Reference excerpt')}"
        with st.expander(title, expanded=index == 1):
            st.text(item.get("text", ""))
            st.caption(f"Source: {item.get('source', '')}  ·  Version: {item.get('version', '')}")
            st.code(item.get("chunk_id", ""), language=None)
            score = item.get("score")
            if isinstance(score, (int, float)):
                st.caption(f"Cosine similarity: {score:.3f} · retrieval ranking only")


def render_trace(result: dict) -> None:
    st.caption("NODE = DO · EDGE = GO · STATE = KNOW. This trace shows metadata; raw report text is omitted.")
    st.caption("Cloud tracing: " + result.get("cloud_trace_status", "disabled"))
    nodes = result.get("executed_nodes", [])
    if nodes:
        # All HTML labels come from this fixed allowlist, never from state text.
        labels = [NODE_LABELS.get(node, "Workflow step") for node in nodes]
        st.markdown(
            '<div class="cg-flow">' + "".join(f"<span>{label}</span>" for label in labels) + "</div>",
            unsafe_allow_html=True,
        )
    trace = result.get("trace", [])
    if trace:
        st.dataframe(
            [{
                "Step": i,
                "Node": event.get("node", ""),
                "Changed fields": ", ".join(event.get("changed_keys", [])),
            } for i, event in enumerate(trace, 1)],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Before / after state summaries"):
            st.json(trace)
    with st.expander("Final state overview"):
        st.json({
            "route": result.get("route", "fallback"),
            "domains": result.get("domains", []),
            "report_confirmed": bool(result.get("report_confirmed", False)),
            "fact_fields": list(result.get("pathology_facts", {}).keys()),
            "missing_facts": result.get("missing_facts", []),
            "evidence_count": len(result.get("evidence", [])),
            "source_ids": [item.get("chunk_id") for item in result.get("evidence", [])],
            "decision": result.get("decision"),
            "executed_nodes": nodes,
        })


init_session()

with st.sidebar:
    st.markdown(
        '<div class="cg-brand"><div class="cg-mark">✳</div><div>'
        '<div class="cg-brand-name">CareGraph AI</div>'
        '<div class="cg-brand-sub">EVIDENCE WORKSPACE</div></div></div>'
        '<div class="cg-local">● &nbsp; LOCAL APPLICATION</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("Explore the evidence.")
    st.caption("Follow a question from routing to cited explanation, one graph step at a time.")
    st.divider()
    st.markdown("**Reference library**")
    status_label, status_detail = index_summary()
    st.write(f"◉ {status_label}")
    st.caption(status_detail)
    st.caption("2 synthetic domains · Oncology + Cardiology")
    if status_label != "Library ready":
        st.code("uv run python ingest.py", language="powershell")
    st.divider()
    st.markdown("**How this workspace works**")
    st.caption("01  Ask an educational question")
    st.caption("02  Retrieve source excerpts")
    st.caption("03  Inspect the assurance decision")
    st.divider()
    st.button("Clear session", icon=":material/restart_alt:", on_click=clear_workspace, width="stretch")
    st.caption("Clears this workspace's question, report, and result from session state.")
    st.caption("Local embeddings · Extractive synthesis · No paid API")

st.markdown('<div class="cg-eyebrow">SYNTHETIC CLINICAL EDUCATION</div>', unsafe_allow_html=True)
heading, badge = st.columns([5, 1.4], vertical_alignment="center")
with heading:
    st.title("From question to evidence.")
    st.markdown(
        '<div class="cg-subtitle">Explore oncology and cardiology through a transparent, local workflow. '
        'See the sources, follow the graph, and understand every step.</div>',
        unsafe_allow_html=True,
    )
with badge:
    st.caption("POWERED BY")
    st.markdown("**LangGraph + FAISS**")
    st.caption("Deterministic · Source grounded")
st.markdown(
    '<div class="cg-scope">Educational demonstration using synthetic references only. '
    'Do not enter real patient data. Outputs are for clinician review; they do not diagnose, stage, prescribe, or select treatment.</div>',
    unsafe_allow_html=True,
)

question_column, report_column = st.columns([1.8, 1], gap="large")
with report_column:
    report_bytes, report_name, report_confirmed, report_valid = render_report()

with question_column:
    with st.container(border=True):
        st.markdown('<div class="cg-step">01 / YOUR QUESTION</div>', unsafe_allow_html=True)
        st.subheader("What would you like to explore?")
        st.text_area(
            "Educational question",
            key="question",
            placeholder="For example, what considerations apply to cancer plus heart disease?",
            height=135,
            max_chars=4000,
            label_visibility="collapsed",
        )
        st.caption("START WITH AN EXAMPLE")
        example_columns = st.columns(3)
        for column, (label, example) in zip(example_columns, EXAMPLES.items()):
            column.button(
                label, on_click=choose_example, args=(example,), width="stretch",
                key="example_" + label,
            )
        st.write("")
        needs_confirmation = report_bytes is not None and not report_confirmed
        run_clicked = st.button(
            "Explore evidence",
            icon=":material/arrow_forward:",
            type="primary",
            width="stretch",
            disabled=not report_valid or needs_confirmation,
            help="Confirm extracted report facts first." if needs_confirmation else "Run the local graph.",
        )
        if needs_confirmation:
            st.caption("Confirm the report facts to enable retrieval.")
        else:
            st.caption("Searches the existing reference index. No report data is indexed.")

    if run_clicked:
        with st.spinner("Routing your question and retrieving local evidence…"):
            try:
                st.session_state.result = run_workflow(
                    question=st.session_state.question,
                    report_bytes=report_bytes,
                    report_name=report_name,
                    report_confirmed=report_confirmed,
                    index_dir=DEFAULT_INDEX_DIR,
                    embedder=get_embedder(),
                )
                st.session_state.result_question = st.session_state.question
            except Exception as exc:
                # Keep library/setup errors readable without dumping tracebacks
                # or report bytes into the browser.
                st.session_state.result = {
                    "error": f"The workflow could not complete: {exc}",
                    "evidence": [],
                    "executed_nodes": [],
                }
                st.session_state.result_question = st.session_state.question

result = st.session_state.result
if result is None:
    with st.container(border=True):
        st.markdown(
            '<div class="cg-empty"><div class="cg-empty-icon">✳</div>'
            '<div class="cg-empty-title">Make the evidence visible.</div>'
            '<div class="cg-empty-copy">Your cited explanation, source excerpts, and graph trace<br>'
            'will appear here after you explore a question.</div></div>',
            unsafe_allow_html=True,
        )
else:
    st.write("")
    st.subheader("Your evidence workspace")
    st.caption("RESULT FOR")
    st.text(st.session_state.result_question.strip() or "(empty question)")
    metrics = st.columns(3)
    metrics[0].metric("Selected route", str(result.get("route", "—")).title())
    metrics[1].metric("Reference excerpts", len(result.get("evidence", [])))
    metrics[2].metric("Graph steps", len(result.get("executed_nodes", [])))
    answer_tab, evidence_tab, trace_tab = st.tabs(["Explanation", "Evidence", "Execution trace"])
    with answer_tab:
        render_answer(result)
    with evidence_tab:
        render_evidence(result)
    with trace_tab:
        render_trace(result)

st.markdown(
    '<div class="cg-footer">CareGraph AI · Local learning project · Synthetic content, '
    'not NCCN guidance · No clinical validation</div>',
    unsafe_allow_html=True,
)
