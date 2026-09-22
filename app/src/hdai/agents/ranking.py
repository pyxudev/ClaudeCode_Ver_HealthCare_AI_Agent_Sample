"""Agent 4 - Ranking. Pure, deterministic, no LLM.

The weights are the product's contract with the business (design doc section
7), so the arithmetic lives here in Python and is unit-tested. An LLM is never
asked to compute or adjust a score: language models are unreliable at
arithmetic, cannot be audited, and would make the same input rank differently
on different days - which is exactly the "inconsistent recommendations"
problem this platform exists to fix.

Neutral-dimension rule: when the patient expressed no preference on a
dimension, every candidate scores 1.0 there. Scoring 0 instead would quietly
shrink the usable range of the total and make scores incomparable between
requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

from ..models import (
    DoctorOut,
    PatientRequirements,
    Recommendation,
    ScoreBreakdown,
    SlotOut,
)
from ..taxonomy import ADJACENT_REGIONS, PERSONALITY_CLUSTERS, RELATED_SPECIALTIES
from ..timeutil import days_between, local_date

WEIGHTS: dict[str, float] = {
    "specialty": 0.35,
    "availability": 0.20,
    "region": 0.15,
    "language": 0.10,
    "doctor_rating": 0.10,
    "hospital_rating": 0.05,
    "personality": 0.05,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "ranking weights must sum to 1.0"


@dataclass
class Candidate:
    """A doctor row joined with its hospital, open slots and vector distance."""

    doctor: DoctorOut
    slots: list[SlotOut] = field(default_factory=list)
    vector_similarity: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Dimension scores - each returns a value in [0, 1] plus a human reason
# --------------------------------------------------------------------------


def score_specialty(candidate: Candidate, wanted: str | None) -> tuple[float, str | None]:
    if not wanted:
        return 1.0, None
    actual = candidate.doctor.expertise
    if actual == wanted:
        return 1.0, f"specialises in {actual}"
    if actual in RELATED_SPECIALTIES.get(wanted, frozenset()):
        if actual == "General Practice":
            return 0.45, "general practitioner who can assess and refer"
        return 0.60, f"{actual} is a related specialty"
    return 0.0, None


def score_availability(
    candidate: Candidate,
    now: datetime,
    preferred_date: str | None = None,
    urgency: str = "routine",
) -> tuple[float, str | None]:
    """Sooner is better; an exact preferred-date hit is worth full marks."""
    future = [s for s in candidate.slots if s.available_time > now]
    if not future:
        return 0.0, None
    earliest = min(s.available_time for s in future)

    # Dates are compared in clinic local time, never UTC: an 08:30 JST slot is
    # 23:30 UTC the previous day and would otherwise be counted a day early.
    if preferred_date:
        matching = [s for s in future if local_date(s.available_time).isoformat() == preferred_date]
        if matching:
            return 1.0, f"has a slot on the requested date ({preferred_date})"

    days = days_between(now, earliest)
    if days <= 0:
        base, reason = 1.0, "can be seen today"
    elif days == 1:
        base, reason = 0.90, "available tomorrow"
    elif days <= 3:
        base, reason = 0.75, f"available in {days} days"
    elif days <= 7:
        base, reason = 0.55, "available this week"
    elif days <= 14:
        base, reason = 0.35, "available within two weeks"
    else:
        base, reason = 0.15, f"next opening in {days} days"

    if urgency == "urgent" and days > 1:
        base *= 0.6  # a wait hurts more when the patient said it is urgent
        reason = f"{reason} (later than requested)"
    if preferred_date:
        base *= 0.85  # wanted a specific date and did not get it
    return round(base, 4), reason


def score_region(candidate: Candidate, wanted: str | None) -> tuple[float, str | None]:
    if not wanted:
        return 1.0, None
    actual = candidate.doctor.region
    if actual == wanted:
        return 1.0, f"located in {actual}"
    if actual in ADJACENT_REGIONS.get(wanted, frozenset()):
        return 0.50, f"in {actual}, commutable from {wanted}"
    return 0.0, None


def score_language(candidate: Candidate, wanted: str | None) -> tuple[float, str | None]:
    if not wanted:
        return 1.0, None
    languages = candidate.doctor.languages or []
    if wanted in languages:
        return 1.0, f"speaks {wanted}"
    if any(wanted.lower() == lang.lower() for lang in languages):
        return 1.0, f"speaks {wanted}"
    return 0.0, None


def score_personality(candidate: Candidate, wanted: str | None) -> tuple[float, str | None]:
    if not wanted:
        return 1.0, None
    actual = (candidate.doctor.personality or "").lower()
    if not actual:
        return 0.3, None
    if actual == wanted.lower():
        return 1.0, f"described as {actual}"
    if actual in PERSONALITY_CLUSTERS.get(wanted.lower(), frozenset()):
        return 0.60, f"described as {actual}"
    return 0.0, None


def score_rating(value: float) -> float:
    """Ratings live on 0-5 but real ones cluster in 4.0-5.0.

    Rescaling from 3.5 spreads that cluster across the usable range; a flat
    /5 would compress every candidate into 0.86-0.98 and make the dimension
    contribute almost nothing.
    """
    if value <= 0:
        return 0.0
    normalised = (min(value, 5.0) - 3.5) / 1.5
    return round(min(max(normalised, 0.0), 1.0), 4)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def rank(
    candidates: Sequence[Candidate],
    requirements: PatientRequirements,
    *,
    now: datetime,
    top_n: int = 3,
    min_score: float = 0.0,
    require_availability: bool = True,
) -> list[Recommendation]:
    scored: list[tuple[float, ScoreBreakdown, Candidate, list[str]]] = []

    for candidate in candidates:
        reasons: list[str] = []
        parts: dict[str, float] = {}

        for key, (value, reason) in {
            "specialty": score_specialty(candidate, requirements.specialty),
            "availability": score_availability(
                candidate, now, requirements.preferred_date, requirements.urgency
            ),
            "region": score_region(candidate, requirements.region),
            "language": score_language(candidate, requirements.language),
            "personality": score_personality(candidate, requirements.personality),
        }.items():
            parts[key] = value
            if reason:
                reasons.append(reason)

        parts["doctor_rating"] = score_rating(candidate.doctor.rating)
        parts["hospital_rating"] = score_rating(candidate.doctor.hospital_rating)
        if candidate.doctor.rating >= 4.7:
            reasons.append(f"rated {candidate.doctor.rating:.1f}/5 by patients")

        # A doctor with nothing bookable is not a recommendation, it is a
        # dead end for the caller - drop rather than rank low.
        if require_availability and parts["availability"] <= 0.0:
            continue

        breakdown = ScoreBreakdown(**{k: round(v * WEIGHTS[k], 6) for k, v in parts.items()})
        total = breakdown.total()
        if total < min_score:
            continue
        scored.append((total, breakdown, candidate, reasons))

    # Deterministic tie-break: score, then vector similarity, then internal
    # score, then id. Never rely on the DB's row order.
    scored.sort(
        key=lambda item: (
            -item[0],
            -item[2].vector_similarity,
            -item[2].doctor.score,
            item[2].doctor.doctor_id,
        )
    )

    out: list[Recommendation] = []
    for index, (total, breakdown, candidate, reasons) in enumerate(scored[:top_n], start=1):
        future_slots = sorted(
            (s for s in candidate.slots if s.available_time > now),
            key=lambda s: s.available_time,
        )[:3]
        out.append(
            Recommendation(
                rank=index,
                doctor=candidate.doctor,
                score=round(total, 4),
                breakdown=breakdown,
                vector_similarity=round(candidate.vector_similarity, 4),
                next_slots=future_slots,
                reasons=reasons[:4],
            )
        )
    return out


def overall_confidence(
    recommendations: Sequence[Recommendation],
    classification_confidence: float,
    *,
    degraded: bool = False,
) -> float:
    """How much the platform trusts its own answer.

    Three independent things have to go right: the specialty had to be
    classified correctly, the top candidate has to score well, and it has to
    be clearly better than the runner-up. A confident-looking score on a
    guessed specialty is the failure mode this is designed to expose.
    """
    if not recommendations:
        return 0.0
    top = recommendations[0].score
    margin = top - recommendations[1].score if len(recommendations) > 1 else 0.10
    margin_factor = min(margin / 0.15, 1.0) * 0.2 + 0.8
    confidence = top * classification_confidence * margin_factor
    if degraded:
        confidence *= 0.85
    return round(min(max(confidence, 0.0), 1.0), 3)


def explain(recommendation: Recommendation) -> dict[str, float]:
    """Per-dimension contribution, for the operator screen and audit log."""
    return recommendation.breakdown.model_dump()


def next_slot_within(candidate: Candidate, now: datetime, window: timedelta) -> bool:
    return any(now < s.available_time <= now + window for s in candidate.slots)
