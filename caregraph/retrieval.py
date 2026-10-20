"""Explicit reference ingestion and read-only semantic search.

Only the two bundled synthetic reference files can enter this index. Reports
and question text are never written here. The persisted FAISS format is a
constrained IndexFlatIP, with JSON provenance rather than Python pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = ROOT / "indexes"
REFERENCE_DIR = ROOT / "data" / "synthetic"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
REFERENCE_SOURCES = {
    "oncology_reference.txt": "oncology",
    "cardiology_reference.txt": "cardiology",
}
SCHEMA_VERSION = 1
MIN_SIMILARITY = 0.25
CHUNK_WORDS = 110
CHUNK_OVERLAP_WORDS = 20
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_QUERY_TOKENS = 8192
_FLAT_HEADER = struct.Struct("<4siqqqBiQ")
REBUILD_HINT = "Run `uv run python ingest.py --rebuild` to build the approved references."


class IndexSetupError(RuntimeError):
    """An absent, stale, corrupt, or incompatible reference index."""


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Pinned local embedding model, lazily loaded without runtime downloads."""

    model_id = MODEL_ID
    revision = MODEL_REVISION

    def __init__(self, offline: bool = True) -> None:
        self.offline = offline
        self._model: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(
                            self.model_id,
                            revision=self.revision,
                            device="cpu",
                            local_files_only=self.offline,
                            trust_remote_code=False,
                            token=False,
                            model_kwargs={"use_safetensors": True},
                        )
                    except Exception as exc:
                        action = (
                            "The pinned embedding model is not available in the local cache. "
                            "Run `uv run python ingest.py` once with internet access."
                            if self.offline
                            else "The pinned embedding model could not be downloaded or loaded. "
                            "Check internet access and retry `uv run python ingest.py`."
                        )
                        raise IndexSetupError(action) from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        try:
            return np.asarray(
                self._load().encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
        except IndexSetupError:
            raise
        except Exception as exc:
            raise IndexSetupError("Local embedding failed. Check the model cache and restart.") from exc


    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Pool bounded token windows so late report fields cannot be truncated.

        Corpus embeddings and short queries keep their original encoding. Long
        queries use an equal mean of window embeddings, normalized by search.
        No query windows or report fields are persisted.
        """
        model = self._load()
        try:
            tokenizer = model.tokenizer
            budget = int(model.max_seq_length) - tokenizer.num_special_tokens_to_add(pair=False)
            if budget < 8:
                raise ValueError("invalid model token budget")
            results = []
            for text in texts:
                tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
                if len(tokens) > _MAX_QUERY_TOKENS:
                    raise IndexSetupError(
                        "The question and confirmed report exceed the 8,192-token query limit. "
                        "Shorten the question or report fields."
                    )
                if len(tokens) <= budget:
                    results.append(self.encode([text])[0])
                    continue
                windows = []
                start = 0
                while start < len(tokens):
                    end = min(start + budget, len(tokens))
                    # A window can begin inside a WordPiece. Decode it as text,
                    # then check the actual re-tokenized length before encoding.
                    while end > start:
                        window = tokenizer.decode(
                            tokens[start:end],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        if len(tokenizer.encode(window, add_special_tokens=False, truncation=False, verbose=False)) <= budget:
                            break
                        end -= 1
                    if end == start:
                        raise ValueError("query token cannot fit the model budget")
                    windows.append(window)
                    start = end
                results.append(self.encode(windows).mean(axis=0))
            return np.asarray(results, dtype=np.float32)
        except IndexSetupError:
            raise
        except Exception as exc:
            raise IndexSetupError("The complete query could not be embedded safely. Shorten the question and retry.") from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(embedder: Embedder) -> dict[str, str]:
    identity = {
        "id": getattr(embedder, "model_id", ""),
        "revision": getattr(embedder, "revision", ""),
    }
    if not all(isinstance(value, str) and value for value in identity.values()):
        raise IndexSetupError("The embedder must declare model_id and a pinned revision.")
    return identity


def _vectors(embedder: Embedder, texts: list[str], *, query: bool = False) -> np.ndarray:
    encode = getattr(embedder, "encode_queries", embedder.encode) if query else embedder.encode
    vectors = np.asarray(encode(texts), dtype=np.float32)
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(texts)
        or not 1 <= vectors.shape[1] <= 4096
        or not np.isfinite(vectors).all()
    ):
        raise IndexSetupError("The embedding model returned invalid vectors.")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise IndexSetupError("The embedding model returned an empty vector.")
    vectors = np.ascontiguousarray(vectors)
    faiss.normalize_L2(vectors)
    return vectors


def _chunks(text: str) -> list[str]:
    """Word windows preserve a section and overlap by twenty words."""
    words = text.split()
    windows = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_WORDS, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS
    return windows


def _source_catalog(reference_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Read approved source sections for provenance checks, without chunking."""
    catalog = {}
    source_hashes = {}
    for filename, domain in REFERENCE_SOURCES.items():
        try:
            source = reference_dir / filename
            if source.stat().st_size > 1024 * 1024:
                raise IndexSetupError(f"Approved reference {filename} exceeds the size limit.")
            raw = source.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise IndexSetupError(f"Approved reference {filename} is missing or unreadable. {REBUILD_HINT}") from exc
        preamble = text.split("\n## ", 1)[0]
        version_match = re.search(r"^Version:\s*([A-Za-z0-9._-]+)\s*$", preamble, re.MULTILINE)
        if (
            "SYNTHETIC EDUCATIONAL REFERENCE" not in preamble
            or f"Domain: {domain}" not in preamble
            or not version_match
        ):
            raise IndexSetupError(f"{filename} lacks its synthetic label, domain, or version.")
        version = version_match.group(1)
        sections = re.split(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if len(sections) < 3:
            raise IndexSetupError(f"{filename} has no reference sections.")
        source_hashes[filename] = _sha(raw)
        section_bodies = {}
        for section, body in zip(sections[1::2], sections[2::2]):
            if len(section) > 200 or "\n" in section:
                raise IndexSetupError(f"{filename} has an invalid section heading (maximum 200 characters).")
            if section in section_bodies:
                raise IndexSetupError(f"{filename} contains a duplicate section heading: {section}.")
            section_bodies[section] = " ".join(body.split())
        catalog[filename] = {"version": version, "sections": section_bodies}
    return catalog, source_hashes


def _read_references(reference_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    chunks: list[dict[str, Any]] = []
    catalog, source_hashes = _source_catalog(reference_dir)
    for filename, source in catalog.items():
        domain = REFERENCE_SOURCES[filename]
        version = source["version"]
        for section, body in source["sections"].items():
            for ordinal, chunk in enumerate(_chunks(body)):
                if len(chunk) < 40:
                    raise IndexSetupError(f"{filename}, {section} has an unusably short section.")
                digest = _sha(f"{filename}|{version}|{section}|{ordinal}|{chunk}".encode())[:16]
                chunks.append({
                    "domain": domain,
                    "text": chunk,
                    "source": filename,
                    "section": section,
                    "version": version,
                    "chunk_id": f"{domain}-{digest}",
                })
    if not chunks or len(chunks) > 10000:
        raise IndexSetupError("The approved reference library has an invalid chunk count.")
    return chunks, source_hashes


def _validate_chunk_metadata(chunks: Any, catalog: dict[str, dict]) -> None:
    """Validate stored provenance; the ingestion chunker is never called."""
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= 10000:
        raise ValueError("invalid chunk metadata")
    ordinals: dict[tuple[str, str], int] = {}
    required = {"domain", "text", "source", "section", "version", "chunk_id"}
    seen_ids = set()
    for chunk in chunks:
        if not isinstance(chunk, dict) or set(chunk) != required:
            raise ValueError("invalid chunk metadata fields")
        if not all(isinstance(value, str) and value.strip() for value in chunk.values()):
            raise ValueError("invalid chunk metadata values")
        source, section, version = chunk["source"], chunk["section"], chunk["version"]
        if REFERENCE_SOURCES.get(source) != chunk["domain"]:
            raise ValueError("unapproved source provenance")
        if len(section) > 200 or "\n" in section or not re.fullmatch(r"[A-Za-z0-9._-]+", version):
            raise ValueError("invalid section or version provenance")
        if len(chunk["text"]) < 40 or len(chunk["text"].split()) > CHUNK_WORDS:
            raise ValueError("invalid stored chunk length")
        approved_source = catalog[source]
        approved_section = approved_source["sections"].get(section)
        if (
            version != approved_source["version"]
            or approved_section is None
            or f" {chunk['text']} " not in f" {approved_section} "
        ):
            raise ValueError("stored excerpt, section, or version does not match the approved source")
        key = (source, section)
        ordinal = ordinals.get(key, 0)
        ordinals[key] = ordinal + 1
        digest = _sha(f"{source}|{version}|{section}|{ordinal}|{chunk['text']}".encode())[:16]
        expected_id = f"{chunk['domain']}-{digest}"
        if chunk["chunk_id"] != expected_id or expected_id in seen_ids:
            raise ValueError("chunk provenance identifier mismatch")
        seen_ids.add(expected_id)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def build_index(
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    embedder: Embedder | None = None,
    *,
    reference_dir: Path | str = REFERENCE_DIR,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Explicit offline ingestion; never called by the application workflow."""
    index_dir = Path(index_dir)
    reference_dir = Path(reference_dir)
    embedder = embedder or SentenceTransformerEmbedder(offline=False)
    chunks, source_hashes = _read_references(reference_dir)
    identity = _identity(embedder)
    if not rebuild and (index_dir / "metadata.json").exists():
        try:
            current = ReferenceIndex(index_dir, embedder, reference_dir=reference_dir)
            if isinstance(embedder, SentenceTransformerEmbedder):
                # Explicit ingestion also restores a removed model cache even
                # when the persisted vectors are still current.
                embedder._load()
            return {**current.status(), "rebuilt": False}
        except IndexSetupError:
            pass
    vectors = _vectors(embedder, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    serialized = faiss.serialize_index(index).tobytes()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "index_type": "IndexFlatIP",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": identity,
        "dimension": int(vectors.shape[1]),
        "count": len(chunks),
        "normalized": True,
        "metric": "inner_product",
        "chunking": {"words": CHUNK_WORDS, "overlap_words": CHUNK_OVERLAP_WORDS},
        "sources": source_hashes,
        "index_sha256": _sha(serialized),
        "chunks": chunks,
    }
    # Run the same complete validation used by runtime BEFORE replacing either
    # existing artifact. A malformed source cannot destroy a working index.
    ReferenceIndex._validate_data(metadata, serialized, embedder, reference_dir)
    metadata_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
    index_dir.mkdir(parents=True, exist_ok=True)
    # Publish metadata last. Interrupted writes are detected by checksum on load.
    _atomic_write(index_dir / "references.faiss", serialized)
    _atomic_write(
        index_dir / "metadata.json",
        metadata_bytes,
    )
    return {**ReferenceIndex(index_dir, embedder, reference_dir=reference_dir).status(), "rebuilt": True}


class ReferenceIndex:
    """Read-only, integrity-checked FAISS search with domain provenance."""

    def __init__(
        self,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        embedder: Embedder | None = None,
        *,
        reference_dir: Path | str = REFERENCE_DIR,
        min_similarity: float = MIN_SIMILARITY,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedder = embedder or SentenceTransformerEmbedder(offline=True)
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between zero and one.")
        self.min_similarity = min_similarity
        metadata_path = self.index_dir / "metadata.json"
        index_path = self.index_dir / "references.faiss"
        try:
            if metadata_path.stat().st_size > _MAX_METADATA_BYTES:
                raise ValueError("metadata size limit")
            if index_path.stat().st_size > _MAX_INDEX_BYTES:
                raise ValueError("index size limit")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw = index_path.read_bytes()
        except FileNotFoundError as exc:
            raise IndexSetupError(f"Reference index is missing. {REBUILD_HINT}") from exc
        except (OSError, ValueError, UnicodeError) as exc:
            raise IndexSetupError(f"Reference index is unreadable or corrupt. {REBUILD_HINT}") from exc
        self._index = self._validate_data(metadata, raw, self.embedder, Path(reference_dir))
        self._metadata = metadata

    @staticmethod
    def _validate_data(metadata: Any, raw: bytes, embedder: Embedder, reference_dir: Path) -> Any:
        try:
            if len(raw) > _MAX_INDEX_BYTES:
                raise ValueError("index size limit")
            if len(json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")) > _MAX_METADATA_BYTES:
                raise ValueError("metadata size limit")
            catalog, source_hashes = _source_catalog(reference_dir)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            if metadata.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("unsupported schema")
            if metadata.get("model") != _identity(embedder):
                raise ValueError("embedding model or pinned revision mismatch")
            if metadata.get("sources") != source_hashes:
                raise ValueError("reference source content changed")
            _validate_chunk_metadata(metadata.get("chunks"), catalog)
            created_at = metadata["created_at"]
            if not isinstance(created_at, str) or datetime.fromisoformat(created_at).tzinfo is None:
                raise ValueError("invalid creation timestamp")
            dimension, count = metadata["dimension"], metadata["count"]
            if type(dimension) is not int or type(count) is not int:
                raise ValueError("invalid dimensions")
            if not 1 <= dimension <= 4096 or count != len(metadata["chunks"]):
                raise ValueError("invalid vector dimensions or count")
            if (
                metadata.get("index_type") != "IndexFlatIP"
                or metadata.get("metric") != "inner_product"
                or metadata.get("normalized") is not True
                or metadata.get("chunking") != {
                    "words": CHUNK_WORDS, "overlap_words": CHUNK_OVERLAP_WORDS
                }
            ):
                raise ValueError("incompatible index configuration")
            if _sha(raw) != metadata.get("index_sha256"):
                raise ValueError("index checksum mismatch")
            return ReferenceIndex._safe_flat_index(raw, dimension, count)
        except (KeyError, TypeError, ValueError, struct.error) as exc:
            raise IndexSetupError(f"Reference index is stale, corrupt, or incompatible ({exc}). {REBUILD_HINT}") from exc

    @staticmethod
    def _safe_flat_index(raw: bytes, dimension: int, count: int) -> Any:
        """Read only bounded dense-vector data, never a general FAISS deserializer.

        FAISS binary loaders accept arbitrary index structures. We accept only
        the small IndexFlatIP layout this application writes and reconstruct it
        from finite normalized floats. No pickle or untrusted object loader runs.
        """
        if len(raw) != _FLAT_HEADER.size + count * dimension * 4:
            raise ValueError("invalid FAISS byte length")
        tag, stored_dim, stored_count, _, _, trained, metric, vector_size = _FLAT_HEADER.unpack_from(raw)
        if (
            tag != b"IxFI"
            or stored_dim != dimension
            or stored_count != count
            or trained != 1
            or metric != faiss.METRIC_INNER_PRODUCT
            or vector_size != count * dimension
        ):
            raise ValueError("incompatible FAISS flat-index header")
        vectors = np.frombuffer(raw, dtype="<f4", offset=_FLAT_HEADER.size).reshape(count, dimension)
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError("invalid stored embedding vectors")
        index = faiss.IndexFlatIP(dimension)
        index.add(np.ascontiguousarray(vectors))
        return index

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "index_dir": str(self.index_dir),
            "count": self._metadata["count"],
            "dimension": self._metadata["dimension"],
            "model": dict(self._metadata["model"]),
            "created_at": self._metadata["created_at"],
            "sources": list(self._metadata["sources"]),
            "synthetic_only": True,
        }

    def search(self, query: str, domain: str, top_k: int = 3) -> list[dict[str, Any]]:
        if domain not in REFERENCE_SOURCES.values():
            raise ValueError("Search domain must be oncology or cardiology.")
        if not isinstance(query, str) or not query.strip():
            return []
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20.")
        vectors = _vectors(self.embedder, [query], query=True)
        if vectors.shape[1] != self._index.d:
            raise IndexSetupError(f"Query embedding dimension does not match the index. {REBUILD_HINT}")
        # Searching this small reference library in full preserves domain top-k
        # without allowing another domain to consume the candidate budget.
        scores, ids = self._index.search(vectors, self._index.ntotal)
        results = []
        for score, row in zip(scores[0], ids[0]):
            chunk = self._metadata["chunks"][int(row)]
            if chunk["domain"] == domain and float(score) >= self.min_similarity:
                results.append({**chunk, "score": round(float(score), 6)})
                if len(results) >= top_k:
                    break
        return results
