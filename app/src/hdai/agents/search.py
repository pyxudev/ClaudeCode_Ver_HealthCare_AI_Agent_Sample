"""Agent 3 - Search. Builds the structured query and runs hybrid retrieval.

Hybrid, per design doc section 8:
  structured SQL filter  -> correctness (only doctors who really match)
  pgvector similarity    -> ordering within the filtered set
  top-20 retrieved, top-3 returned after business re-ranking

The LLM does not write SQL. It never sees the database. Query construction is
a pure function of the validated PatientRequirements object, so there is no
path from patient text to a query predicate - only to a parameter value.

Progressive relaxation matters as much as the query itself: with 15 doctors,
a fully strict filter frequently returns nothing, and "no doctors found" for a
patient who would happily see someone in the next prefecture is a worse
failure than a slightly imperfect match. Every relaxation applied is recorded
and surfaced to the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..db import Database
from ..embeddings import embed, to_pgvector
from ..models import DoctorOut, PatientRequirements, SlotOut
from ..taxonomy import ADJACENT_REGIONS, RELATED_SPECIALTIES
from .ranking import Candidate

log = logging.getLogger("hdai.agents.search")


@dataclass
class SearchQuery:
    """The structured query. Every field is already coerced and safe."""

    specialty: str | None = None
    include_related_specialties: bool = False
    region: str | None = None
    include_adjacent_regions: bool = False
    language: str | None = None
    gender: str | None = None
    require_availability: bool = True
    semantic_text: str = ""
    limit: int = 20

    def describe(self) -> dict[str, Any]:
        return {
            "specialty": self.specialty,
            "related_specialties": self.include_related_specialties,
            "region": self.region,
            "adjacent_regions": self.include_adjacent_regions,
            "language": self.language,
            "gender": self.gender,
            "require_availability": self.require_availability,
            "limit": self.limit,
        }


@dataclass
class SearchResult:
    candidates: list[Candidate]
    query: SearchQuery
    relaxations: list[str]
    total_considered: int


def build_query(
    requirements: PatientRequirements, specialty: str, *, limit: int = 20
) -> SearchQuery:
    semantic_parts = [
        requirements.symptom_text,
        specialty,
        requirements.region or "",
        requirements.personality or "",
        requirements.language or "",
    ]
    return SearchQuery(
        specialty=specialty,
        region=requirements.region,
        language=requirements.language,
        gender=requirements.doctor_gender,
        semantic_text=" ".join(p for p in semantic_parts if p),
        limit=limit,
    )


# Order matters: drop the cheapest preference first, keep clinical fit
# (specialty) and the patient's language until last.
_RELAXATION_STEPS: tuple[tuple[str, str], ...] = (
    ("include_adjacent_regions", "widened to neighbouring areas"),
    ("gender", "dropped the doctor-gender preference"),
    ("include_related_specialties", "included related departments"),
    ("region", "searched all regions"),
    ("language", "dropped the language preference"),
)
# Note: availability is deliberately NOT relaxable. Ranking drops candidates
# with no future slot anyway, so relaxing it does no work and puts a
# misleading "I widened the search" note in front of the patient.


async def run(
    db: Database,
    requirements: PatientRequirements,
    specialty: str,
    *,
    now: datetime,
    limit: int = 20,
    embedding_dim: int = 256,
    min_candidates: int = 3,
) -> SearchResult:
    query = build_query(requirements, specialty, limit=limit)
    relaxations: list[str] = []

    rows = await _execute(db, query, now=now, embedding_dim=embedding_dim)

    for attribute, description in _RELAXATION_STEPS:
        if len(rows) >= min_candidates:
            break
        # Only relax a constraint that is actually constraining anything.
        if attribute == "include_adjacent_regions":
            if not query.region or not ADJACENT_REGIONS.get(query.region):
                continue
            query.include_adjacent_regions = True
        elif attribute == "include_related_specialties":
            if not query.specialty or not RELATED_SPECIALTIES.get(query.specialty):
                continue
            query.include_related_specialties = True
        else:
            if getattr(query, attribute) is None:
                continue
            setattr(query, attribute, None)

        relaxations.append(description)
        rows = await _execute(db, query, now=now, embedding_dim=embedding_dim)

    if relaxations:
        log.info(
            "search relaxed",
            extra={"relaxations": relaxations, "found": len(rows), "specialty": specialty},
        )

    candidates = await _attach_slots(db, rows, now=now)
    return SearchResult(
        candidates=candidates,
        query=query,
        relaxations=relaxations,
        total_considered=len(rows),
    )


async def _execute(
    db: Database, query: SearchQuery, *, now: datetime, embedding_dim: int
) -> list[dict[str, Any]]:
    vector = to_pgvector(embed(query.semantic_text, embedding_dim))

    specialties: list[str] | None = None
    if query.specialty:
        specialties = [query.specialty]
        if query.include_related_specialties:
            specialties += sorted(RELATED_SPECIALTIES.get(query.specialty, frozenset()))

    regions: list[str] | None = None
    if query.region:
        regions = [query.region]
        if query.include_adjacent_regions:
            regions += sorted(ADJACENT_REGIONS.get(query.region, frozenset()))

    sql = """
        WITH open_slots AS (
            SELECT doctor_id, COUNT(*) AS open_count, MIN(available_time) AS earliest
            FROM doctor_schedule
            WHERE status = 'available' AND available_time > %(now)s
            GROUP BY doctor_id
        )
        SELECT d.doctor_id, d.internal_number, d.name, d.gender, d.age, d.expertise,
               d.region, d.languages, d.rating, d.score, d.personality,
               h.name AS hospital_name, h.rating AS hospital_rating, h.address AS hospital_address,
               COALESCE(os.open_count, 0) AS open_count,
               os.earliest AS earliest_slot,
               CASE WHEN d.embedding IS NULL THEN 0.0
                    ELSE 1 - (d.embedding <=> %(vector)s::vector) END AS similarity
        FROM doctor d
        JOIN hospital h ON h.hospital_id = d.hospital_id
        LEFT JOIN open_slots os ON os.doctor_id = d.doctor_id
        WHERE d.leave_date IS NULL
          AND (%(specialties)s::text[] IS NULL OR d.expertise = ANY(%(specialties)s::text[]))
          AND (%(regions)s::text[] IS NULL OR d.region = ANY(%(regions)s::text[]))
          AND (%(language)s::text IS NULL OR %(language)s = ANY(d.languages))
          AND (%(gender)s::text IS NULL OR d.gender = %(gender)s)
          AND (%(require_availability)s = false OR COALESCE(os.open_count, 0) > 0)
        ORDER BY
            CASE WHEN %(exact_specialty)s::text IS NOT NULL
                      AND d.expertise = %(exact_specialty)s THEN 0 ELSE 1 END,
            similarity DESC,
            d.score DESC,
            d.doctor_id ASC
        LIMIT %(limit)s
    """
    params = {
        "now": now,
        "vector": vector,
        "specialties": specialties,
        "regions": regions,
        "language": query.language,
        "gender": query.gender,
        "require_availability": query.require_availability,
        "exact_specialty": query.specialty,
        "limit": max(1, min(query.limit, 100)),
    }
    try:
        return await db.fetch_all(sql, params)
    except Exception as exc:  # noqa: BLE001
        log.error("search query failed", extra={"error": str(exc), "query": query.describe()})
        raise


async def _attach_slots(
    db: Database, rows: list[dict[str, Any]], *, now: datetime
) -> list[Candidate]:
    if not rows:
        return []
    doctor_ids = [int(r["doctor_id"]) for r in rows]
    slot_rows = await db.fetch_all(
        """
        SELECT schedule_id, doctor_id, available_time, status
        FROM doctor_schedule
        WHERE doctor_id = ANY(%s) AND status = 'available' AND available_time > %s
        ORDER BY doctor_id, available_time
        """,
        (doctor_ids, now),
    )
    by_doctor: dict[int, list[SlotOut]] = {}
    for row in slot_rows:
        by_doctor.setdefault(int(row["doctor_id"]), []).append(
            SlotOut(
                schedule_id=int(row["schedule_id"]),
                available_time=row["available_time"],
                status=str(row["status"]),
            )
        )

    candidates: list[Candidate] = []
    for row in rows:
        doctor_id = int(row["doctor_id"])
        candidates.append(
            Candidate(
                doctor=DoctorOut(
                    doctor_id=doctor_id,
                    internal_number=str(row["internal_number"]),
                    name=str(row["name"]),
                    gender=row.get("gender"),
                    age=row.get("age"),
                    expertise=str(row["expertise"]),
                    region=str(row["region"]),
                    languages=list(row.get("languages") or []),
                    personality=row.get("personality"),
                    rating=float(row.get("rating") or 0),
                    score=int(row.get("score") or 0),
                    hospital_name=str(row["hospital_name"]),
                    hospital_rating=float(row.get("hospital_rating") or 0),
                    hospital_address=row.get("hospital_address"),
                ),
                slots=by_doctor.get(doctor_id, []),
                vector_similarity=float(row.get("similarity") or 0.0),
                raw=row,
            )
        )
    return candidates


async def load_aliases(db: Database) -> dict[str, str]:
    """alias -> canonical specialty, from the DB (authoritative over the static map)."""
    try:
        rows = await db.fetch_all("SELECT alias, keyword FROM keyword_alias")
    except Exception as exc:  # noqa: BLE001
        log.warning("alias load failed, using static map", extra={"error": str(exc)})
        return {}
    return {str(r["alias"]): str(r["keyword"]) for r in rows}
