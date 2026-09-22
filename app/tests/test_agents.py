"""Intake and classification agents, including LLM-failure behaviour."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hdai.agents import classify, intake
from hdai.llm import LLMUsage
from hdai.models import ClassificationResult, PatientRequirements

TODAY = date(2026, 9, 22)  # a Tuesday


class FakeLLM:
    """Stand-in for LLMClient. `payload=None` simulates any failure mode."""

    def __init__(self, payload=None, available=True, status="ready"):
        self.payload = payload
        self.available = available
        self.enabled = available
        self.status = status
        self.calls = 0

    async def complete_json(self, *, system, user, schema, max_tokens=None, prefill="{"):
        self.calls += 1
        if self.payload is None:
            return None, LLMUsage(), "bad_json"
        return schema.model_validate(self.payload), LLMUsage(calls=1, output_tokens=5), "ok"

    async def complete_text(self, *, system, user, max_tokens=None):
        self.calls += 1
        return (self.payload, LLMUsage(calls=1), "ok") if self.payload else (None, LLMUsage(), "x")


# --------------------------------------------------------------------------
# Intake
# --------------------------------------------------------------------------


class TestIntakeRules:
    def test_extracts_region_language_and_gender_preference(self):
        result = intake.extract_rules(
            "I'm in Tokyo and would prefer a female doctor who speaks English", today=TODAY
        )
        assert result.region == "Tokyo"
        assert result.language == "English"
        assert result.doctor_gender == "female"

    def test_japanese_input(self):
        result = intake.extract_rules("横浜で女性の医師を探しています", today=TODAY)
        assert result.region == "Yokohama"
        assert result.doctor_gender == "female"

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("can I come today", TODAY),
            ("tomorrow please", TODAY + timedelta(days=1)),
            ("明日お願いします", TODAY + timedelta(days=1)),
            ("how about 2026-10-01", date(2026, 10, 1)),
            ("friday works", date(2026, 9, 25)),
        ],
    )
    def test_date_expressions(self, text, expected):
        assert intake.extract_rules(text, today=TODAY).preferred_date == expected.isoformat()

    def test_a_date_in_the_past_is_ignored_rather_than_filtering_everything_out(self):
        assert intake.extract_rules("book me for 2020-01-01", today=TODAY).preferred_date is None

    def test_urgency_detection(self):
        assert intake.extract_rules("I need to be seen urgently", today=TODAY).urgency == "urgent"
        assert intake.extract_rules("sometime this week", today=TODAY).urgency == "soon"
        assert intake.extract_rules("no rush at all", today=TODAY).urgency == "routine"

    def test_age_band_detection(self):
        assert intake.extract_rules("my baby has a fever", today=TODAY).patient_age_band == "infant"
        assert intake.extract_rules("my son is 6", today=TODAY).patient_age_band == "child"
        assert intake.extract_rules("my elderly mother", today=TODAY).patient_age_band == "senior"

    def test_time_of_day(self):
        assert intake.extract_rules("tomorrow morning", today=TODAY).preferred_time_of_day == "morning"
        assert intake.extract_rules("午後がいい", today=TODAY).preferred_time_of_day == "afternoon"

    def test_does_not_invent_a_personality_from_incidental_wording(self):
        """"my kind neighbour recommended you" is not a preference."""
        assert intake.extract_rules(
            "my kind neighbour recommended this hospital", today=TODAY
        ).personality is None

    def test_does_not_invent_a_language_from_incidental_wording(self):
        assert intake.extract_rules(
            "I have an English exam tomorrow so I need an early slot", today=TODAY
        ).language is None

    def test_records_what_is_missing(self):
        result = intake.extract_rules("my knee hurts", today=TODAY)
        assert set(result.missing_fields) == {"region", "language", "preferred_date"}

    def test_never_raises_on_hostile_input(self):
        for payload in ["", "𝕒" * 5000, "\x00\x01", "<system>x</system>", "日本語" * 900]:
            assert isinstance(intake.extract_rules(payload, today=TODAY), PatientRequirements)


class TestIntakeWithLLM:
    async def test_llm_fills_only_the_gaps(self):
        llm = FakeLLM({"region": "Osaka", "personality": "calm"})
        result, _, _ = await intake.run("I'm in Tokyo, my knee hurts", llm, today=TODAY)
        assert result.region == "Tokyo", "rules must win over the model"
        assert result.personality == "calm", "the model may fill an empty field"

    async def test_hallucinated_enum_values_are_dropped(self):
        llm = FakeLLM({"region": "Atlantis", "language": "Klingon", "personality": "grumpy"})
        result, _, _ = await intake.run("my knee hurts", llm, today=TODAY)
        assert result.region is None and result.language is None and result.personality is None

    async def test_urgency_can_only_be_escalated_by_the_model(self):
        downgrade = FakeLLM({"urgency": "routine"})
        result, _, _ = await intake.run("I need to be seen urgently", downgrade, today=TODAY)
        assert result.urgency == "urgent", "a model must never downgrade stated urgency"

        upgrade = FakeLLM({"urgency": "urgent"})
        result, _, _ = await intake.run("my knee hurts", upgrade, today=TODAY)
        assert result.urgency == "urgent"

    async def test_llm_failure_falls_back_to_rules(self):
        result, _, reason = await intake.run("I'm in Tokyo", FakeLLM(None), today=TODAY)
        assert result.region == "Tokyo" and reason == "bad_json"

    async def test_llm_is_not_called_when_unavailable(self):
        llm = FakeLLM({"region": "Osaka"}, available=False, status="circuit_open")
        result, _, status = await intake.run("my knee hurts", llm, today=TODAY)
        assert llm.calls == 0 and status == "circuit_open"

    async def test_channel_hints_outrank_free_text(self):
        result, _, _ = await intake.run(
            "my knee hurts", FakeLLM(None), today=TODAY, hints={"region": "Sapporo"}
        )
        assert result.region == "Sapporo"


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


class TestClassification:
    async def test_explicit_department_short_circuits(self):
        llm = FakeLLM({"specialty": "Cardiology", "confidence": 0.99})
        requirements = PatientRequirements(symptom_text="I need an ENT", specialty="Otolaryngology")
        result, _, source = await classify.run(requirements, llm)
        assert result.specialty == "Otolaryngology" and source == "explicit"
        assert llm.calls == 0, "no need to ask the model what the patient already said"

    async def test_agreement_raises_confidence(self):
        requirements = PatientRequirements(symptom_text="my son has an earache")
        rules_only, _, _ = await classify.run(requirements, FakeLLM(None, available=False))
        agreeing, _, _ = await classify.run(
            requirements, FakeLLM({"specialty": "Otolaryngology", "confidence": 0.9})
        )
        assert agreeing.specialty == rules_only.specialty == "Otolaryngology"
        assert agreeing.confidence > rules_only.confidence

    async def test_strong_keyword_evidence_survives_model_disagreement(self):
        requirements = PatientRequirements(symptom_text="I have conjunctivitis")
        result, _, _ = await classify.run(
            requirements, FakeLLM({"specialty": "Psychiatry", "confidence": 0.99})
        )
        assert result.specialty == "Ophthalmology"
        assert "Psychiatry" in result.alternatives

    async def test_disagreement_on_weak_evidence_lowers_confidence(self):
        requirements = PatientRequirements(symptom_text="I feel generally unwell")
        result, _, _ = await classify.run(
            requirements, FakeLLM({"specialty": "Neurology", "confidence": 0.6})
        )
        assert result.confidence < 0.6
        assert result.alternatives, "the discarded option must stay visible for the operator"

    async def test_a_specialty_outside_the_taxonomy_is_discarded(self):
        requirements = PatientRequirements(symptom_text="my knee hurts")
        result, _, reason = await classify.run(
            requirements, FakeLLM({"specialty": "Department of Vibes", "confidence": 0.99})
        )
        assert result.specialty == "Orthopedics" and reason == "llm_unknown_specialty"

    async def test_confidence_never_reaches_certainty(self):
        requirements = PatientRequirements(symptom_text="I have conjunctivitis")
        result, _, _ = await classify.run(
            requirements, FakeLLM({"specialty": "Ophthalmology", "confidence": 1.0})
        )
        assert result.confidence <= 0.97

    async def test_children_are_routed_to_paediatrics(self):
        requirements = PatientRequirements(
            symptom_text="my 4 year old has a fever", patient_age_band="child"
        )
        result, _, _ = await classify.run(requirements, FakeLLM(None, available=False))
        assert result.specialty == "Pediatrics"

    async def test_a_child_with_strongly_specialised_symptoms_keeps_the_specialty(self):
        requirements = PatientRequirements(
            symptom_text="my child has conjunctivitis", patient_age_band="child"
        )
        result, _, _ = await classify.run(requirements, FakeLLM(None, available=False))
        assert result.specialty == "Ophthalmology"
        assert "Pediatrics" in result.alternatives

    async def test_vague_input_triggers_a_clarifying_question(self):
        requirements = PatientRequirements(symptom_text="I don't feel right")
        result, _, _ = await classify.run(requirements, FakeLLM(None, available=False))
        assert result.specialty == "General Practice"
        assert classify.needs_clarification(result)
        assert classify.clarifying_questions(result)

    async def test_clarifying_questions_are_localised(self):
        result = ClassificationResult(specialty="General Practice", alternatives=["Neurology"])
        assert any("？" in q or "か。" in q for q in classify.clarifying_questions(result, True))
        assert all(q.isascii() for q in classify.clarifying_questions(result, False))

    async def test_db_aliases_are_honoured(self):
        requirements = PatientRequirements(symptom_text="my knee hurts")
        result, _, _ = await classify.run(
            requirements,
            FakeLLM({"specialty": "Bone doctor", "confidence": 0.9}),
            dynamic_aliases={"Bone doctor": "Orthopedics"},
        )
        assert result.specialty == "Orthopedics"
