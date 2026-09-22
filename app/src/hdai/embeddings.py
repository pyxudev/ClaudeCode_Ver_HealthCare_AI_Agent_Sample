"""Deterministic local text embeddings for the pgvector leg of hybrid search.

Why not a transformer model: the POC must come up with one `docker-compose up
-d` on any machine. Downloading a sentence-transformer at build or boot time
adds ~500MB, a network dependency, and a class of startup failures that is
hard to diagnose. This is a signed feature-hashing embedder (the classic
hashing-trick vectoriser) over word tokens and character trigrams:

  * no model download, no GPU, sub-millisecond
  * language agnostic, so Japanese input works without a tokenizer
  * deterministic across processes - blake2b, NOT Python's hash(), which is
    randomised per process by PYTHONHASHSEED and would silently produce a
    different vector in the ingest worker than in the query path

It is a lexical signal, not a semantic one. Section "Not covered" of the test
report calls this out: swapping in a real embedding model is a drop-in change
to embed() plus a re-ingest, and is the right move before production.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Iterable, Sequence

DEFAULT_DIM = 256

_TOKEN_RE = re.compile(r"[0-9a-z]+|[぀-ヿ]|[一-鿿]", re.U)


def _normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower().strip()


def _features(text: str) -> Iterable[str]:
    """Word tokens plus character trigrams.

    Trigrams give partial credit for morphology ("cardiology"/"cardiologist")
    and make the representation work for Japanese, which has no spaces.
    """
    normalised = _normalise(text)
    tokens = _TOKEN_RE.findall(normalised)
    for token in tokens:
        yield f"w:{token}"
    compact = "".join(tokens)
    for i in range(len(compact) - 2):
        yield f"c:{compact[i:i + 3]}"


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    # Lowest bit picks the sign; signed hashing keeps collisions unbiased.
    sign = 1.0 if value & 1 else -1.0
    return (value >> 1) % dim, sign


def embed(text: str, dim: int = DEFAULT_DIM) -> list[float]:
    """L2-normalised vector. Empty/blank input yields the zero vector."""
    vector = [0.0] * dim
    for feature in _features(text):
        index, sign = _bucket(feature, dim)
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return sum(x * y for x, y in zip(a, b))


def to_pgvector(vector: Sequence[float]) -> str:
    """pgvector's text input format, used with an explicit ::vector cast.

    Avoids taking a dependency on the pgvector python package just to register
    a type adapter.
    """
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


def doctor_profile_text(
    *,
    name: str,
    expertise: str,
    hospital: str,
    region: str,
    languages: Sequence[str],
    personality: str,
    keywords: Sequence[str] = (),
) -> str:
    """The document that gets embedded for a doctor.

    Expertise is repeated because specialty is the dominant matching dimension
    (35% of the ranking weight) and repetition raises its share of the vector.
    """
    parts = [
        name,
        expertise, expertise, expertise,
        hospital,
        region,
        " ".join(languages),
        personality,
        " ".join(keywords),
    ]
    return " ".join(p for p in parts if p).strip()
