"""Embeddings: determinism is the property that matters most.

If the ingest worker and the query path disagree on a vector, hybrid search
silently returns nonsense with no error anywhere.
"""

from __future__ import annotations

import math
import subprocess
import sys

import pytest

from hdai.embeddings import cosine, doctor_profile_text, embed, to_pgvector


class TestShape:
    def test_dimension_and_normalisation(self):
        vector = embed("cardiology tokyo", 256)
        assert len(vector) == 256
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_custom_dimension(self):
        assert len(embed("x", 64)) == 64

    def test_empty_input_is_the_zero_vector_not_a_crash(self):
        vector = embed("", 32)
        assert vector == [0.0] * 32

    def test_whitespace_and_punctuation_only(self):
        assert embed("  !!! ", 32) == [0.0] * 32


class TestDeterminism:
    def test_same_input_same_vector(self):
        assert embed("Pediatrics Tokyo") == embed("Pediatrics Tokyo")

    def test_stable_across_processes_with_different_hash_seeds(self):
        """Regression guard: Python's hash() is per-process randomised.

        Using it here would give the ingest worker and the API different
        vectors for identical text.
        """
        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from hdai.embeddings import embed;"
            "print(sum(embed('Otolaryngology Yokohama English')))"
        )
        outputs = set()
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin", "PYTHONPATH": "src"},
            )
            outputs.add(result.stdout.strip())
        assert len(outputs) == 1, f"embedding changed with PYTHONHASHSEED: {outputs}"


class TestSemantics:
    def test_identical_text_has_similarity_one(self):
        vector = embed("Cardiology Osaka")
        assert cosine(vector, vector) == pytest.approx(1.0, abs=1e-9)

    def test_related_text_beats_unrelated(self):
        query = embed("pediatrics tokyo japanese english kind")
        near = embed("Dr. Aiko Tanaka Pediatrics Tokyo Japanese English kind")
        far = embed("Dr. Kenta Aoki Ophthalmology Sapporo Japanese humor")
        assert cosine(query, near) > cosine(query, far)

    def test_morphological_overlap_is_captured_by_trigrams(self):
        assert cosine(embed("cardiology"), embed("cardiologist")) > 0.3

    def test_works_on_japanese_without_a_tokenizer(self):
        similar = cosine(embed("小児科 東京"), embed("小児科 東京 日本語"))
        different = cosine(embed("小児科 東京"), embed("眼科 札幌"))
        assert similar > different

    def test_dimension_mismatch_is_an_error_not_a_silent_wrong_answer(self):
        with pytest.raises(ValueError):
            cosine(embed("a", 32), embed("a", 64))


class TestSerialisation:
    def test_pgvector_literal_format(self):
        literal = to_pgvector([0.5, -0.25, 0.0])
        assert literal.startswith("[") and literal.endswith("]")
        assert literal == "[0.500000,-0.250000,0.000000]"

    def test_profile_text_weights_expertise(self):
        text = doctor_profile_text(
            name="Dr. A", expertise="Cardiology", hospital="H", region="Tokyo",
            languages=["Japanese"], personality="kind", keywords=["heart"],
        )
        assert text.count("Cardiology") == 3
        assert "heart" in text
