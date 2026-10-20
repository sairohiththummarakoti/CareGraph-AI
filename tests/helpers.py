"""Deterministic injectable embeddings for fast real-FAISS integration tests."""

from __future__ import annotations

import hashlib
import re

import numpy as np


class DeterministicEmbedder:
    """A tiny test encoder; production always uses the pinned neural model."""

    model_id = "tests/deterministic-embedder"
    revision = "v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            value = text.lower()
            row = np.zeros(128, dtype=np.float32)
            # Domain and alias features make tests meaningful without downloading
            # weights. The separate model smoke check tests actual semantics.
            groups = (
                ("heart", "cardio", "myocardial", "coronary"),
                ("cancer", "oncology", "tumor", "pathology", "biomarker"),
                ("heart attack", "myocardial infarction"),
                ("stage", "grade", "histology", "specimen", "report"),
                ("biomarker", "her2", "estrogen", "progesterone"),
                ("coordinat", "both", "together"),
            )
            for index, words in enumerate(groups):
                row[index] = 4 if any(word in value for word in words) else 0
            for word in re.findall(r"[a-z]+", value):
                if len(word) > 3:
                    bucket = 8 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:2], "little") % 120
                    row[bucket] += 0.2
            # Keep every input nonzero so irrelevant input reaches low-score logic.
            row[127] += 0.01
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)
