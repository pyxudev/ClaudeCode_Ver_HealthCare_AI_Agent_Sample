"""Closed-vocabulary coercion and rule-based symptom classification."""

from __future__ import annotations

import pytest

from hdai.taxonomy import (
    SPECIALTIES,
    classify_symptom_rules,
    coerce_gender,
    coerce_language,
    coerce_personality,
    coerce_region,
    coerce_specialty,
    parse_languages,
)


class TestSpecialtyCoercion:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("ENT", "Otolaryngology"),
            ("ent", "Otolaryngology"),
            ("耳鼻科", "Otolaryngology"),
            ("Pedia", "Pediatrics"),
            ("paediatrics", "Pediatrics"),
            ("heart doctor", "Cardiology"),
            ("Cardiology", "Cardiology"),
            ("OB/GYN", "Obstetrics and Gynecology"),
            ("内科", "General Practice"),
        ],
    )
    def test_aliases_from_the_dataset_and_the_call_floor(self, value, expected):
        assert coerce_specialty(value) == expected

    @pytest.mark.parametrize("value", ["Hogwarts Medicine", "", None, "xyzzy", "!!!"])
    def test_unknown_values_return_none_rather_than_a_guess(self, value):
        assert coerce_specialty(value) is None

    def test_a_db_alias_overrides_the_static_map(self):
        assert coerce_specialty("ENT", {"ENT": "Pediatrics"}) == "Pediatrics"

    def test_every_canonical_value_round_trips(self):
        for specialty in SPECIALTIES:
            assert coerce_specialty(specialty) == specialty


class TestOtherVocabularies:
    @pytest.mark.parametrize(
        "value,expected",
        [("tokyo", "Tokyo"), ("東京", "Tokyo"), ("Kanagawa", "Yokohama"),
         ("HOKKAIDO", "Sapporo"), ("Atlantis", None)],
    )
    def test_region(self, value, expected):
        assert coerce_region(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [("english", "English"), ("英語", "English"), ("Mandarin", "Chinese"), ("Klingon", None)],
    )
    def test_language(self, value, expected):
        assert coerce_language(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [("nice", "kind"), ("優しい", "kind"), ("funny", "humor"),
         ("listens", "empathetic"), ("grumpy", None)],
    )
    def test_personality(self, value, expected):
        assert coerce_personality(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [("woman", "female"), ("女性", "female"), ("man", "male"), ("robot", None)],
    )
    def test_gender(self, value, expected):
        assert coerce_gender(value) == expected


class TestLanguageParsing:
    def test_handles_the_dataset_s_comma_separated_string(self):
        assert parse_languages("Japanese, English, Chinese") == ["Japanese", "English", "Chinese"]

    def test_handles_a_list(self):
        assert parse_languages(["Japanese", "English"]) == ["Japanese", "English"]

    def test_handles_japanese_separators(self):
        assert parse_languages("日本語、英語") == ["Japanese", "English"]

    def test_empty_is_empty(self):
        assert parse_languages(None) == [] and parse_languages("") == []

    def test_keeps_an_unknown_language_rather_than_dropping_data(self):
        assert "Swahili" in parse_languages("Japanese, Swahili")

    def test_deduplicates(self):
        assert parse_languages("English, english") == ["English"]


class TestSymptomRules:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("my son has an earache", "Otolaryngology"),
            ("I have a bad rash on my arm", "Dermatology"),
            ("heart palpitations at night", "Cardiology"),
            ("my knee pain won't go away", "Orthopedics"),
            ("terrible migraine for two days", "Neurology"),
            ("feeling anxious and can't sleep", "Psychiatry"),
            ("acid reflux after meals", "Gastroenterology"),
            ("blurry vision in my right eye", "Ophthalmology"),
            ("I think I'm pregnant", "Obstetrics and Gynecology"),
            ("耳が痛いです", "Otolaryngology"),
            ("湿疹がひどい", "Dermatology"),
            ("腰痛がつらい", "Orthopedics"),
        ],
    )
    def test_common_presentations(self, text, expected):
        specialty, confidence, _ = classify_symptom_rules(text)
        assert specialty == expected
        assert confidence > 0.3

    def test_vague_input_falls_back_to_general_practice_with_low_confidence(self):
        specialty, confidence, _ = classify_symptom_rules("I just don't feel right")
        assert specialty == "General Practice"
        assert confidence <= 0.35

    def test_explicit_department_beats_symptom_inference(self):
        specialty, confidence, _ = classify_symptom_rules("I need an ENT, my knee also hurts")
        assert specialty == "Otolaryngology"
        assert confidence >= 0.9

    def test_ambiguous_input_is_penalised(self):
        _, ambiguous, _ = classify_symptom_rules("I have a rash and my knee hurts")
        _, clear, _ = classify_symptom_rules("I have a rash")
        assert ambiguous < clear

    def test_evidence_is_returned_for_audit(self):
        _, _, evidence = classify_symptom_rules("terrible migraine")
        assert "migraine" in evidence

    def test_is_deterministic(self):
        text = "my child has an earache and a fever"
        assert classify_symptom_rules(text) == classify_symptom_rules(text)
