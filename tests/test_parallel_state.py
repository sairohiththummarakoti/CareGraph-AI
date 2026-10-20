from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from caregraph.graph import run_workflow
from caregraph.retrieval import ReferenceIndex


def test_true_parallel_retrieval_and_single_join(reference_index, embedder):
    barrier = Barrier(2, timeout=10)
    lock = Lock()
    completed = []
    reference = ReferenceIndex(reference_index, embedder)

    class WaitingRetriever:
        def search(self, query, domain, top_k):
            # This fails if graph branches execute sequentially.
            barrier.wait()
            results = reference.search(query, domain, top_k)
            with lock:
                completed.append(domain)
            return results

    state = run_workflow("What considerations apply to cancer plus heart disease?", retriever=WaitingRetriever())
    nodes = state["executed_nodes"]
    assert set(completed) == {"oncology", "cardiology"}
    assert {c["domain"] for c in state["evidence"]} == {"oncology", "cardiology"}
    for name in ("join_evidence", "synthesis", "assurance"):
        assert nodes.count(name) == 1
    assert nodes.index("join_evidence") > max(nodes.index("retrieve_oncology"), nodes.index("retrieve_cardiology"))
    join_trace = next(event for event in state["trace"] if event["node"] == "join_evidence")
    assert set(join_trace["before"]["completed_domains"]) == {"oncology", "cardiology"}
    assert join_trace["before"]["evidence_count"] == len(state["evidence"])
    assert state["decision"] == "release"


@pytest.mark.parametrize("question, domain", [
    ("What is oncology care?", "oncology"), ("What is a heart attack?", "cardiology"),
])
def test_single_domain_does_not_wait_for_other_branch(reference_index, embedder, question, domain):
    state = run_workflow(question, index_dir=reference_index, embedder=embedder)
    assert state["domains"] == [domain]
    assert state["executed_nodes"].count("join_evidence") == 1
    assert state["executed_nodes"].count("synthesis") == 1
    assert state["evidence"]
    for chunk in state["evidence"]:
        assert chunk["domain"] == domain
        assert chunk["text"] in state["final_answer"]
        assert chunk["chunk_id"] in state["final_answer"]
        assert chunk["source"] in state["final_answer"]
        assert chunk["section"] in state["final_answer"]


def test_sessions_do_not_share_state(reference_index, embedder):
    with ThreadPoolExecutor(2) as executor:
        states = list(executor.map(lambda q: run_workflow(q, index_dir=reference_index, embedder=embedder), [
            "What is oncology care?", "What is a heart attack?",
        ]))
    assert states[0]["completed_domains"] == ["oncology"]
    assert states[1]["completed_domains"] == ["cardiology"]
