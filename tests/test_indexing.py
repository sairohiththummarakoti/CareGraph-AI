from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from caregraph.retrieval import (
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_DIR,
    REFERENCE_SOURCES,
    IndexSetupError,
    ReferenceIndex,
    SentenceTransformerEmbedder,
    build_index,
)
from tests.helpers import DeterministicEmbedder


@pytest.fixture
def reference_index(tmp_path):
    embedder = DeterministicEmbedder()
    directory = tmp_path / "index"
    build_index(directory, embedder)
    return ReferenceIndex(directory, embedder)


def snapshot(directory: Path):
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in directory.iterdir()
    }


def test_persisted_faiss_semantic_alias_and_citation_provenance(reference_index):
    evidence = reference_index.search("What is a heart attack?", "cardiology")
    assert evidence
    assert any("myocardial infarction" in item["text"].lower() for item in evidence)
    assert all(item["domain"] == "cardiology" for item in evidence)
    assert all(item["source"] == "cardiology_reference.txt" for item in evidence)
    assert all(item["section"] and item["version"] and item["chunk_id"] for item in evidence)
    assert (reference_index.index_dir / "references.faiss").read_bytes()[:4] == b"IxFI"


def test_allowlist_excludes_reports_and_extra_files(tmp_path):
    references = tmp_path / "references"
    shutil.copytree(REFERENCE_DIR, references)
    (references / "unapproved.txt").write_text("SYNTHETIC secret report information", encoding="utf-8")
    build_index(tmp_path / "index", DeterministicEmbedder(), reference_dir=references)
    metadata = json.loads((tmp_path / "index" / "metadata.json").read_text())
    assert set(metadata["sources"]) == set(REFERENCE_SOURCES)
    assert all("report.txt" not in chunk["source"] for chunk in metadata["chunks"])
    assert not any("secret report" in chunk["text"] for chunk in metadata["chunks"])


def test_search_and_unchanged_ingestion_never_modify_index(reference_index):
    before = snapshot(reference_index.index_dir)
    for _ in range(2):
        reference_index.search("What is oncology care?", "oncology")
        reference_index.search("Cancer plus heart disease", "cardiology")
    status = build_index(reference_index.index_dir, DeterministicEmbedder())
    assert status["rebuilt"] is False
    assert snapshot(reference_index.index_dir) == before


def test_irrelevant_and_empty_queries_return_no_evidence(reference_index):
    assert reference_index.search("banana telescopes holiday astronomy", "oncology") == []
    assert reference_index.search("", "oncology") == []


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(IndexSetupError, match="ingest.py"):
        ReferenceIndex(tmp_path, DeterministicEmbedder())


@pytest.mark.parametrize("damage", ["binary", "json", "model", "dimension", "provenance", "header", "timestamp"])
def test_corruption_and_incompatibility_fail_before_search(reference_index, damage):
    directory = reference_index.index_dir
    metadata_path = directory / "metadata.json"
    binary_path = directory / "references.faiss"
    metadata = json.loads(metadata_path.read_text())
    if damage == "binary":
        binary_path.write_bytes(b"not an index")
    elif damage == "json":
        metadata_path.write_text("{broken", encoding="utf-8")
    elif damage == "model":
        metadata["model"]["revision"] = "different"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "dimension":
        metadata["dimension"] = 900000000
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "provenance":
        metadata["chunks"][0]["text"] = "injected unsupported claim"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "header":
        raw = b"IxF2" + binary_path.read_bytes()[4:]
        binary_path.write_bytes(raw)
        metadata["index_sha256"] = hashlib.sha256(raw).hexdigest()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif damage == "timestamp":
        del metadata["created_at"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IndexSetupError, match="ingest.py"):
        ReferenceIndex(directory, DeterministicEmbedder())


def test_changed_source_requires_explicit_rebuild(tmp_path):
    references = tmp_path / "references"
    shutil.copytree(REFERENCE_DIR, references)
    directory = tmp_path / "index"
    build_index(directory, DeterministicEmbedder(), reference_dir=references)
    source = references / "oncology_reference.txt"
    source.write_text(source.read_text(encoding="utf-8") + "\nEdited synthetic reference.\n", encoding="utf-8")
    with pytest.raises(IndexSetupError, match="reference source content changed"):
        ReferenceIndex(directory, DeterministicEmbedder(), reference_dir=references)


def test_query_dimension_mismatch_fails(reference_index):
    class WrongDimension(DeterministicEmbedder):
        def encode(self, texts):
            return np.ones((len(texts), 7), dtype=np.float32)

    index = ReferenceIndex(reference_index.index_dir, WrongDimension())
    with pytest.raises(IndexSetupError, match="dimension"):
        index.search("heart attack", "cardiology")


def test_local_encoder_pins_revision_and_forbids_remote_code(monkeypatch):
    import sentence_transformers

    calls = {}

    class FakeModel:
        def __init__(self, model_id, **kwargs):
            calls.update(model_id=model_id, **kwargs)

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 8))

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    SentenceTransformerEmbedder().encode(["heart attack"])
    assert calls["model_id"] == MODEL_ID
    assert calls["revision"] == MODEL_REVISION
    assert calls["local_files_only"] is True
    assert calls["trust_remote_code"] is False
    assert calls["token"] is False
    assert calls["model_kwargs"]["use_safetensors"] is True


def test_runtime_never_calls_ingestion_chunker(reference_index, monkeypatch):
    import caregraph.retrieval as retrieval

    def forbidden(*args, **kwargs):
        pytest.fail("Runtime must not ingest or rechunk the reference corpus.")

    monkeypatch.setattr(retrieval, "_read_references", forbidden)
    monkeypatch.setattr(retrieval, "_chunks", forbidden)
    index = ReferenceIndex(reference_index.index_dir, DeterministicEmbedder())
    assert index.search("heart attack", "cardiology")


def test_explicit_ingestion_restores_model_cache_without_rewriting_index(reference_index):
    class CacheRestoringEmbedder(SentenceTransformerEmbedder):
        model_id = DeterministicEmbedder.model_id
        revision = DeterministicEmbedder.revision
        restored = False

        def _load(self):
            self.restored = True

        def encode(self, texts):
            pytest.fail("An unchanged corpus must not be reembedded.")

    embedder = CacheRestoringEmbedder(offline=False)
    before = snapshot(reference_index.index_dir)
    result = build_index(reference_index.index_dir, embedder)
    assert embedder.restored
    assert result["rebuilt"] is False
    assert snapshot(reference_index.index_dir) == before


def _refresh_chunk_ids(chunks):
    """Simulate an attacker who can recompute local metadata checksums."""
    ordinals = {}
    for chunk in chunks:
        key = (chunk["source"], chunk["section"])
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        digest = hashlib.sha256(
            f"{chunk['source']}|{chunk['version']}|{chunk['section']}|{ordinal}|{chunk['text']}".encode()
        ).hexdigest()[:16]
        chunk["chunk_id"] = f"{chunk['domain']}-{digest}"


@pytest.mark.parametrize("forgery", ["invented-text", "wrong-section", "wrong-version"])
def test_recomputed_metadata_ids_cannot_forge_provenance(reference_index, forgery):
    path = reference_index.index_dir / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    chunk = metadata["chunks"][0]
    if forgery == "invented-text":
        chunk["text"] = "This invented educational claim does not occur in the approved reference source."
    elif forgery == "wrong-section":
        chunk["section"] = metadata["chunks"][1]["section"]
    else:
        chunk["version"] = "invented-version"
    _refresh_chunk_ids(metadata["chunks"])
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IndexSetupError, match="does not match the approved source"):
        ReferenceIndex(reference_index.index_dir, DeterministicEmbedder())


@pytest.mark.parametrize("damage", ["long-heading", "duplicate-heading"])
def test_invalid_reference_cannot_replace_existing_index(tmp_path, damage):
    references = tmp_path / "references"
    shutil.copytree(REFERENCE_DIR, references)
    directory = tmp_path / "index"
    build_index(directory, DeterministicEmbedder(), reference_dir=references)
    before = snapshot(directory)
    path = references / "oncology_reference.txt"
    text = path.read_text(encoding="utf-8")
    if damage == "long-heading":
        text = text.replace("## Oncology care and shared review", "## " + "x" * 201)
    else:
        text = text.replace("## Pathology report facts and unknowns", "## Oncology care and shared review")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(IndexSetupError, match="section heading"):
        build_index(directory, DeterministicEmbedder(), reference_dir=references, rebuild=True)
    assert snapshot(directory) == before


def test_full_validation_precedes_index_publication(reference_index, monkeypatch):
    import caregraph.retrieval as retrieval

    before = snapshot(reference_index.index_dir)

    def rejected(*args, **kwargs):
        raise ValueError("simulated invalid generated metadata")

    monkeypatch.setattr(retrieval, "_validate_chunk_metadata", rejected)
    with pytest.raises(IndexSetupError, match="simulated invalid"):
        build_index(reference_index.index_dir, DeterministicEmbedder(), rebuild=True)
    assert snapshot(reference_index.index_dir) == before


def test_query_windows_include_late_facts_and_reuse_cached_model(monkeypatch):
    import sentence_transformers

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def encode(self, text, **kwargs):
            return text.split()

        def decode(self, tokens, **kwargs):
            return " ".join(tokens)

    loaded = []
    encoded = []

    class FakeModel:
        tokenizer = Tokenizer()
        max_seq_length = 10

        def __init__(self, *args, **kwargs):
            loaded.append(kwargs)

        def encode(self, texts, **kwargs):
            encoded.extend(texts)
            # Like the real model, silently truncates inputs at its budget.
            return np.asarray([
                [1, " ".join(text.split()[:8]).count("HER2"), " ".join(text.split()[:8]).count("heart")]
                for text in texts
            ], dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    embedder = SentenceTransformerEmbedder()
    prefix = "background " * 40
    query = prefix + "Confirmed synthetic report fields: biomarkers HER2 positive"
    without_facts = embedder.encode_queries([prefix])[0]
    encoded.clear()
    with_facts = embedder.encode_queries([query])[0]
    assert with_facts[1] > without_facts[1]
    assert " ".join(encoded).split() == query.split()
    assert all(len(window.split()) <= 8 for window in encoded)
    assert len(loaded) == 1
    assert loaded[0]["local_files_only"] is True
    assert np.array_equal(embedder.encode_queries(["heart overview"]), embedder.encode(["heart overview"]))


def test_search_uses_query_encoder_without_changing_index(reference_index):
    class QueryEmbedder(DeterministicEmbedder):
        def __init__(self):
            self.queries = []

        def encode_queries(self, texts):
            self.queries.extend(texts)
            return self.encode(texts)

    embedder = QueryEmbedder()
    before = snapshot(reference_index.index_dir)
    index = ReferenceIndex(reference_index.index_dir, embedder)
    assert index.search("heart attack", "cardiology")
    assert embedder.queries == ["heart attack"]
    assert snapshot(reference_index.index_dir) == before


@pytest.mark.skipif(
    os.environ.get("CAREGRAPH_MODEL_SMOKE") != "1",
    reason="Set CAREGRAPH_MODEL_SMOKE=1 after caching the pinned model to test real embeddings.",
)
def test_actual_pinned_model_semantics_offline(tmp_path):
    embedder = SentenceTransformerEmbedder(offline=True)
    directory = tmp_path / "actual-model"
    build_index(directory, embedder)
    index = ReferenceIndex(directory, embedder)
    before = snapshot(directory)
    results = index.search("What is a heart attack?", "cardiology")
    assert results
    assert "myocardial infarction" in results[0]["text"].lower()
    assert index.status()["dimension"] == 384
    prefix = "Explain oncology care in this synthetic educational example. " * 50
    with_facts = prefix + "\nConfirmed synthetic report fields: biomarkers: HER2 positive"
    # The original encoder truncates this suffix. Query pooling must retain it.
    assert np.allclose(embedder.encode([prefix]), embedder.encode([with_facts]))
    assert not np.allclose(embedder.encode_queries([prefix]), embedder.encode_queries([with_facts]))
    assert snapshot(directory) == before
