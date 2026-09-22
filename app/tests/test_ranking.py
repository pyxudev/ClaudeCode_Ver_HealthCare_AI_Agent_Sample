"""Ranking arithmetic - the contract with the business (design doc section 7)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from hdai.agents.ranking import (
    WEIGHTS,
    Candidate,
    overall_confidence,
    rank,
    score_availability,
    score_language,
    score_personality,
    score_rating,
    score_region,
    score_specialty,
)
from hdai.models import DoctorOut, PatientRequirements, SlotOut

NOW = datetime(2026, 9, 22, 9, 0, tzinfo=timezone.utc)


def make_candidate(
    doctor_id: int = 1,
    *,
    expertise: str = "Pediatrics",
    region: str = "Tokyo",
    languages: tuple[str, ...] = ("Japanese", "English"),
    personality: str = "kind",
    rating: float = 4.8,
    hospital_rating: float = 4.6,
    score: int = 90,
    gender: str = "female",
    slot_offsets_hours: tuple[float, ...] = (26,),
) -> Candidate:
    return Candidate(
        doctor=DoctorOut(
            doctor_id=doctor_id,
            internal_number=f"D-{1000 + doctor_id}",
            name=f"Dr. Test {doctor_id}",
            gender=gender,
            age=40,
            expertise=expertise,
            region=region,
            languages=list(languages),
            personality=personality,
            rating=rating,
            score=score,
            hospital_name="Test Hospital",
            hospital_rating=hospital_rating,
        ),
        slots=[
            SlotOut(
                schedule_id=doctor_id * 100 + i,
                available_time=NOW + timedelta(hours=offset),
                status="available",
            )
            for i, offset in enumerate(slot_offsets_hours)
        ],
    )


class TestWeights:
    def test_weights_match_the_design_document(self):
        assert WEIGHTS == {
            "specialty": 0.35, "availability": 0.20, "region": 0.15, "language": 0.10,
            "doctor_rating": 0.10, "hospital_rating": 0.05, "personality": 0.05,
        }

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_a_perfect_candidate_scores_one(self):
        candidate = make_candidate(
            rating=5.0, hospital_rating=5.0, slot_offsets_hours=(2,)
        )
        requirements = PatientRequirements(
            specialty="Pediatrics", region="Tokyo", language="English", personality="kind"
        )
        result = rank([candidate], requirements, now=NOW, top_n=1)
        assert result[0].score == pytest.approx(1.0, abs=1e-6)

    def test_breakdown_sums_to_the_total(self):
        candidate = make_candidate()
        requirements = PatientRequirements(specialty="Pediatrics", region="Tokyo")
        [result] = rank([candidate], requirements, now=NOW, top_n=1)
        assert result.breakdown.total() == pytest.approx(result.score, abs=1e-4)


class TestDimensions:
    def test_specialty_exact_related_and_miss(self):
        assert score_specialty(make_candidate(expertise="Pediatrics"), "Pediatrics")[0] == 1.0
        assert 0 < score_specialty(make_candidate(expertise="General Practice"), "Cardiology")[0] < 1
        assert score_specialty(make_candidate(expertise="Ophthalmology"), "Cardiology")[0] == 0.0

    def test_no_preference_is_neutral_not_zero(self):
        """Absent preference must not silently shrink the score range."""
        candidate = make_candidate()
        assert score_specialty(candidate, None)[0] == 1.0
        assert score_region(candidate, None)[0] == 1.0
        assert score_language(candidate, None)[0] == 1.0
        assert score_personality(candidate, None)[0] == 1.0

    def test_availability_prefers_sooner(self):
        today = score_availability(make_candidate(slot_offsets_hours=(4,)), NOW)[0]
        tomorrow = score_availability(make_candidate(slot_offsets_hours=(26,)), NOW)[0]
        next_month = score_availability(make_candidate(slot_offsets_hours=(24 * 40,)), NOW)[0]
        assert today > tomorrow > next_month > 0

    def test_availability_is_zero_without_a_future_slot(self):
        past_only = make_candidate(slot_offsets_hours=(-48,))
        assert score_availability(past_only, NOW)[0] == 0.0

    def test_preferred_date_hit_beats_a_sooner_miss(self):
        wanted = (NOW + timedelta(days=5)).date().isoformat()
        exact = make_candidate(slot_offsets_hours=(24 * 5 + 1,))
        sooner = make_candidate(slot_offsets_hours=(26,))
        assert score_availability(exact, NOW, wanted)[0] > score_availability(sooner, NOW, wanted)[0]

    def test_urgent_penalises_a_distant_slot(self):
        candidate = make_candidate(slot_offsets_hours=(24 * 6,))
        routine = score_availability(candidate, NOW, None, "routine")[0]
        urgent = score_availability(candidate, NOW, None, "urgent")[0]
        assert urgent < routine

    def test_adjacent_region_gets_partial_credit(self):
        assert score_region(make_candidate(region="Yokohama"), "Tokyo")[0] == 0.5
        assert score_region(make_candidate(region="Osaka"), "Tokyo")[0] == 0.0

    def test_language_is_all_or_nothing(self):
        assert score_language(make_candidate(languages=("Japanese", "English")), "English")[0] == 1.0
        assert score_language(make_candidate(languages=("Japanese",)), "English")[0] == 0.0

    def test_personality_cluster_gets_partial_credit(self):
        assert score_personality(make_candidate(personality="kind"), "kind")[0] == 1.0
        assert score_personality(make_candidate(personality="gentle"), "kind")[0] == 0.6
        assert score_personality(make_candidate(personality="professional"), "humor")[0] == 0.0

    def test_rating_scale_spreads_the_realistic_band(self):
        assert score_rating(5.0) == 1.0
        assert score_rating(3.5) == 0.0
        assert score_rating(0) == 0.0
        assert score_rating(4.8) > score_rating(4.4) > score_rating(4.0)


class TestOrdering:
    def test_specialty_dominates_a_high_rating(self):
        right = make_candidate(1, expertise="Cardiology", rating=4.0, hospital_rating=4.0)
        wrong = make_candidate(2, expertise="Ophthalmology", rating=5.0, hospital_rating=5.0)
        result = rank([wrong, right], PatientRequirements(specialty="Cardiology"), now=NOW)
        assert result[0].doctor.doctor_id == 1

    def test_candidates_without_availability_are_dropped(self):
        result = rank(
            [make_candidate(1, slot_offsets_hours=(-10,))],
            PatientRequirements(specialty="Pediatrics"),
            now=NOW,
        )
        assert result == []

    def test_respects_top_n(self):
        candidates = [make_candidate(i) for i in range(1, 11)]
        assert len(rank(candidates, PatientRequirements(), now=NOW, top_n=3)) == 3

    def test_is_deterministic_regardless_of_input_order(self):
        candidates = [make_candidate(i) for i in range(1, 6)]
        forward = [r.doctor.doctor_id for r in rank(candidates, PatientRequirements(), now=NOW, top_n=5)]
        backward = [
            r.doctor.doctor_id
            for r in rank(list(reversed(candidates)), PatientRequirements(), now=NOW, top_n=5)
        ]
        assert forward == backward

    def test_min_score_filters_weak_matches(self):
        weak = make_candidate(1, expertise="Ophthalmology", region="Osaka",
                              languages=("Japanese",), personality="humor",
                              rating=3.5, hospital_rating=3.5,
                              slot_offsets_hours=(24 * 60,))
        requirements = PatientRequirements(
            specialty="Cardiology", region="Tokyo", language="English", personality="kind"
        )
        assert rank([weak], requirements, now=NOW, min_score=0.25) == []


class TestConfidence:
    def test_no_results_means_no_confidence(self):
        assert overall_confidence([], 0.9) == 0.0

    def test_uncertain_classification_caps_confidence(self):
        recs = rank([make_candidate()], PatientRequirements(specialty="Pediatrics"), now=NOW)
        assert overall_confidence(recs, 0.3) < overall_confidence(recs, 0.95)

    def test_a_close_runner_up_lowers_confidence(self):
        twins = [make_candidate(1), make_candidate(2)]
        clear = [make_candidate(1, slot_offsets_hours=(2,)),
                 make_candidate(2, expertise="Ophthalmology", rating=3.6,
                                slot_offsets_hours=(24 * 30,))]
        requirements = PatientRequirements(specialty="Pediatrics")
        close_conf = overall_confidence(rank(twins, requirements, now=NOW), 0.9)
        clear_conf = overall_confidence(rank(clear, requirements, now=NOW), 0.9)
        assert close_conf < clear_conf

    def test_degraded_mode_is_reported_as_less_confident(self):
        recs = rank([make_candidate()], PatientRequirements(specialty="Pediatrics"), now=NOW)
        assert overall_confidence(recs, 0.9, degraded=True) < overall_confidence(recs, 0.9)

    def test_confidence_stays_in_range(self):
        recs = rank([make_candidate(rating=5.0, hospital_rating=5.0, slot_offsets_hours=(1,))],
                    PatientRequirements(), now=NOW)
        assert 0.0 <= overall_confidence(recs, 1.0) <= 1.0
