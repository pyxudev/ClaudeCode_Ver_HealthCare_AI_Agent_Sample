"""Data ingestion: JSON -> validate -> normalise -> dedupe -> upsert -> embed.

Mirrors design doc section 13. Properties that matter:

  * Idempotent - safe to run on every boot; re-running does not duplicate rows
    and never destroys a booked slot.
  * Total - a single malformed record is recorded in ingest_log and skipped,
    it does not abort the run. A dataset that half-loads silently is how a
    doctor disappears from search with nobody noticing.
  * Explicit about the sample data's dates - every slot in sample_doctors.json
    is 2026-08-xx. Loaded literally, the POC has zero bookable availability.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from psycopg.types.json import Jsonb

from .config import Settings
from .db import Database
from .embeddings import doctor_profile_text, embed, to_pgvector
from .timeutil import LOCAL_TZ as _LOCAL_TZ
from .taxonomy import (
    PERSONALITIES,
    SPECIALTIES,
    coerce_gender,
    coerce_personality,
    coerce_region,
    coerce_specialty,
    parse_languages,
)

log = logging.getLogger("hdai.ingest")

# Slot strings in the dataset carry no offset; they are clinic local time.
LOCAL_TZ = _LOCAL_TZ
_SLOT_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


@dataclass
class IngestReport:
    hospitals: int = 0
    doctors: int = 0
    slots: int = 0
    aliases: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    shifted_days: int = 0

    def reject(self, kind: str, identifier: str, reason: str) -> None:
        self.rejected.append({"kind": kind, "id": identifier, "reason": reason})
        log.warning("record rejected", extra={"kind": kind, "record": identifier, "reason": reason})

    def as_dict(self) -> dict[str, Any]:
        return {
            "hospitals": self.hospitals,
            "doctors": self.doctors,
            "slots": self.slots,
            "aliases": self.aliases,
            "rejected": self.rejected,
            "notes": self.notes,
            "shifted_days": self.shifted_days,
        }


# --------------------------------------------------------------------------
# Normalisation helpers - each returns a value plus never raises
# --------------------------------------------------------------------------


def clean_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def clamp_rating(value: Any, report: IngestReport, who: str) -> float:
    try:
        rating = float(value)
    except (TypeError, ValueError):
        report.notes.append(f"{who}: unreadable rating {value!r} -> 0.0")
        return 0.0
    if rating < 0 or rating > 5:
        clamped = min(max(rating, 0.0), 5.0)
        report.notes.append(f"{who}: rating {rating} out of range -> {clamped}")
        return clamped
    return round(rating, 1)


def clamp_score(value: Any, report: IngestReport, who: str) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        report.notes.append(f"{who}: unreadable score {value!r} -> 0")
        return 0
    if score < 0 or score > 100:
        clamped = min(max(score, 0), 100)
        report.notes.append(f"{who}: score {score} out of range -> {clamped}")
        return clamped
    return score


def parse_slot(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=LOCAL_TZ)
    text = clean_text(raw, 40)
    if not text:
        return None
    for fmt in _SLOT_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=LOCAL_TZ)
        except ValueError:
            continue
    try:  # last resort: full ISO 8601 with offset
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=LOCAL_TZ)
    except ValueError:
        return None


def parse_date(raw: Any) -> date | None:
    text = clean_text(raw, 32)
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def synthetic_internal_number(prefix: str, *parts: str) -> str:
    """Stable surrogate key for a record whose internal_number is missing."""
    import hashlib

    digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=5).hexdigest()
    return f"{prefix}-GEN-{digest.upper()}"


def compute_slot_shift(slots: list[datetime], now: datetime, enabled: bool) -> timedelta:
    """Whole-day offset that moves a stale dataset into the near future.

    Preserves relative day gaps and time-of-day so the availability weighting
    still exercises 'today vs next week' logic.
    """
    if not enabled or not slots:
        return timedelta(0)
    latest = max(slots)
    if latest >= now:
        return timedelta(0)
    earliest = min(slots)
    target = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return timedelta(days=(target.date() - earliest.date()).days)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


async def ingest_file(db: Database, settings: Settings, *, now: datetime | None = None) -> IngestReport:
    report = IngestReport()
    path = pathlib.Path(settings.data_file)
    now = now or datetime.now(tz=LOCAL_TZ)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report.reject("file", str(path), "dataset file not found")
        log.error("dataset missing; starting with whatever is already in the DB",
                  extra={"path": str(path)})
        return report
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        report.reject("file", str(path), f"unreadable dataset: {exc}")
        log.error("dataset unreadable", extra={"path": str(path), "error": str(exc)})
        return report

    if not isinstance(raw, dict):
        report.reject("file", str(path), "top level JSON is not an object")
        return report

    hospitals = raw.get("hospitals") or []
    doctors = raw.get("doctors") or []
    aliases = raw.get("aliases") or []
    if not isinstance(hospitals, list) or not isinstance(doctors, list) or not isinstance(aliases, list):
        report.reject("file", str(path), "hospitals/doctors/aliases must be arrays")
        return report

    async with db.connection() as conn:
        # Serialise concurrent ingests from multiple replicas.
        await conn.execute("SELECT pg_advisory_lock(%s)", (824_199_013,))
        try:
            hospital_ids = await _upsert_hospitals(conn, hospitals, report)
            await _upsert_aliases(conn, aliases, report)
            await _upsert_doctors(conn, doctors, hospital_ids, report, settings, now)
            await conn.execute(
                """
                INSERT INTO ingest_log (source, finished_at, hospitals_upserted, doctors_upserted,
                                        slots_upserted, aliases_upserted, rejected, notes)
                VALUES (%s, now(), %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(path),
                    report.hospitals,
                    report.doctors,
                    report.slots,
                    report.aliases,
                    Jsonb(report.rejected),
                    "; ".join(report.notes[:50]) or None,
                ),
            )
        finally:
            await conn.execute("SELECT pg_advisory_unlock(%s)", (824_199_013,))

    log.info(
        "ingest complete",
        extra={
            "hospitals": report.hospitals,
            "doctors": report.doctors,
            "slots": report.slots,
            "aliases": report.aliases,
            "rejected": len(report.rejected),
            "shifted_days": report.shifted_days,
        },
    )
    return report


async def _upsert_hospitals(conn, hospitals: list[Any], report: IngestReport) -> dict[str, int]:
    """Returns {lowercased hospital name: hospital_id}."""
    mapping: dict[str, int] = {}
    seen_numbers: set[str] = set()

    for item in hospitals:
        if not isinstance(item, dict):
            report.reject("hospital", str(item)[:60], "not an object")
            continue
        name = clean_text(item.get("name"))
        if not name:
            report.reject("hospital", str(item)[:60], "missing name")
            continue
        region = coerce_region(item.get("region")) or clean_text(item.get("region")) or "Unknown"
        internal = clean_text(item.get("internal_number"), 40) or synthetic_internal_number("H", name)
        if internal in seen_numbers:
            report.notes.append(f"hospital {name}: duplicate internal_number {internal}, last wins")
        seen_numbers.add(internal)
        rating = clamp_rating(item.get("rating", 0), report, f"hospital {name}")
        address = clean_text(item.get("address"), 300) or None

        try:
            cur = await conn.execute(
                """
                INSERT INTO hospital (internal_number, name, region, rating, address)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (internal_number) DO UPDATE
                    SET name = EXCLUDED.name,
                        region = EXCLUDED.region,
                        rating = EXCLUDED.rating,
                        address = EXCLUDED.address,
                        updated_at = now()
                RETURNING hospital_id
                """,
                (internal, name, region, rating, address),
            )
            row = await cur.fetchone()
            mapping[name.lower()] = int(row["hospital_id"])
            report.hospitals += 1
        except Exception as exc:  # noqa: BLE001
            report.reject("hospital", name, f"upsert failed: {exc}")

    # Hospitals already in the DB from an earlier run are still valid targets.
    cur = await conn.execute("SELECT hospital_id, name FROM hospital")
    for row in await cur.fetchall():
        mapping.setdefault(str(row["name"]).lower(), int(row["hospital_id"]))
    return mapping


async def _upsert_aliases(conn, aliases: list[Any], report: IngestReport) -> None:
    for item in aliases:
        if not isinstance(item, dict):
            report.reject("alias", str(item)[:60], "not an object")
            continue
        keyword = clean_text(item.get("keyword"), 80)
        alias = clean_text(item.get("alias"), 80)
        if not keyword or not alias:
            report.reject("alias", f"{keyword}/{alias}", "missing keyword or alias")
            continue
        canonical = coerce_specialty(keyword)
        if canonical is None:
            # Keep it - the dataset may legitimately extend the taxonomy - but
            # flag it, because ranking can only score known specialties.
            report.notes.append(f"alias {alias!r} maps to unknown specialty {keyword!r}")
            canonical = keyword
        try:
            await conn.execute(
                """
                INSERT INTO keyword_alias (keyword, alias) VALUES (%s, %s)
                ON CONFLICT (lower(alias)) DO UPDATE SET keyword = EXCLUDED.keyword
                """,
                (canonical, alias),
            )
            report.aliases += 1
        except Exception as exc:  # noqa: BLE001
            report.reject("alias", alias, f"upsert failed: {exc}")


async def _upsert_doctors(
    conn,
    doctors: list[Any],
    hospital_ids: dict[str, int],
    report: IngestReport,
    settings: Settings,
    now: datetime,
) -> None:
    # Pass 1: parse every slot so the demo shift can be computed globally.
    parsed: list[dict[str, Any]] = []
    all_slots: list[datetime] = []
    seen_numbers: set[str] = set()

    for item in doctors:
        if not isinstance(item, dict):
            report.reject("doctor", str(item)[:60], "not an object")
            continue
        name = clean_text(item.get("name"))
        if not name:
            report.reject("doctor", str(item)[:60], "missing name")
            continue

        hospital_name = clean_text(item.get("hospital"))
        hospital_id = hospital_ids.get(hospital_name.lower())
        if hospital_id is None:
            # Deliberately not auto-creating a stub hospital: a patient sent to
            # an address we invented is worse than a doctor missing from search.
            report.reject("doctor", name, f"unknown hospital {hospital_name!r}")
            continue

        expertise = coerce_specialty(item.get("expertise"))
        if expertise is None:
            raw_expertise = clean_text(item.get("expertise"), 80)
            if not raw_expertise:
                report.reject("doctor", name, "missing expertise")
                continue
            expertise = raw_expertise
            report.notes.append(f"doctor {name}: expertise {raw_expertise!r} outside known taxonomy")

        internal = clean_text(item.get("internal_number"), 40) or synthetic_internal_number(
            "D", name, hospital_name
        )
        if internal in seen_numbers:
            report.notes.append(f"doctor {name}: duplicate internal_number {internal}, last wins")
        seen_numbers.add(internal)

        age_raw = item.get("age")
        try:
            age = int(age_raw) if age_raw is not None else None
            if age is not None and not (18 <= age <= 100):
                report.notes.append(f"doctor {name}: implausible age {age} -> null")
                age = None
        except (TypeError, ValueError):
            age = None

        personality = coerce_personality(item.get("personality"))
        if personality is None and clean_text(item.get("personality")):
            personality = clean_text(item.get("personality"), 40)
            report.notes.append(f"doctor {name}: unmapped personality {personality!r}")

        keywords = item.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        keywords = [clean_text(k, 40) for k in keywords if clean_text(k, 40)]

        slots: list[datetime] = []
        raw_slots = item.get("available_slots") or []
        if not isinstance(raw_slots, list):
            report.notes.append(f"doctor {name}: available_slots is not a list, ignored")
            raw_slots = []
        for raw_slot in raw_slots:
            slot = parse_slot(raw_slot)
            if slot is None:
                report.notes.append(f"doctor {name}: unparseable slot {raw_slot!r}")
                continue
            slots.append(slot)
        # De-duplicate inside one record; the DB unique constraint would
        # otherwise turn a duplicated slot into a failed insert.
        slots = sorted(set(slots))
        all_slots.extend(slots)

        register_date = parse_date(item.get("register_date"))
        leave_date = parse_date(item.get("leave_date"))
        if register_date and leave_date and leave_date < register_date:
            report.notes.append(f"doctor {name}: leave_date before register_date -> leave_date dropped")
            leave_date = None

        parsed.append(
            {
                "internal": internal,
                "name": name,
                "gender": coerce_gender(item.get("gender")),
                "age": age,
                "expertise": expertise,
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "region": coerce_region(item.get("region")) or clean_text(item.get("region")) or "Unknown",
                "languages": parse_languages(item.get("language")),
                "rating": clamp_rating(item.get("rating", 0), report, f"doctor {name}"),
                "score": clamp_score(item.get("score", 0), report, f"doctor {name}"),
                "personality": personality,
                "keywords": keywords,
                "register_date": register_date,
                "leave_date": leave_date,
                "slots": slots,
            }
        )

    shift = compute_slot_shift(all_slots, now, settings.demo_shift_past_slots)
    if shift:
        report.shifted_days = shift.days
        report.notes.append(
            f"every slot in the dataset was in the past; shifted forward by {shift.days} days "
            "(HDAI_DEMO_SHIFT_PAST_SLOTS=true)"
        )
        log.warning("demo slot shift applied", extra={"days": shift.days})

    # Pass 2: write.
    for record in parsed:
        try:
            profile = doctor_profile_text(
                name=record["name"],
                expertise=record["expertise"],
                hospital=record["hospital_name"],
                region=record["region"],
                languages=record["languages"],
                personality=record["personality"] or "",
                keywords=record["keywords"],
            )
            vector = to_pgvector(embed(profile, settings.embedding_dim))
            cur = await conn.execute(
                """
                INSERT INTO doctor (internal_number, name, gender, age, expertise, hospital_id,
                                    region, languages, rating, score, personality, keywords,
                                    register_date, leave_date, profile_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (internal_number) DO UPDATE SET
                    name = EXCLUDED.name, gender = EXCLUDED.gender, age = EXCLUDED.age,
                    expertise = EXCLUDED.expertise, hospital_id = EXCLUDED.hospital_id,
                    region = EXCLUDED.region, languages = EXCLUDED.languages,
                    rating = EXCLUDED.rating, score = EXCLUDED.score,
                    personality = EXCLUDED.personality, keywords = EXCLUDED.keywords,
                    register_date = EXCLUDED.register_date, leave_date = EXCLUDED.leave_date,
                    profile_text = EXCLUDED.profile_text, embedding = EXCLUDED.embedding,
                    updated_at = now()
                RETURNING doctor_id
                """,
                (
                    record["internal"], record["name"], record["gender"], record["age"],
                    record["expertise"], record["hospital_id"], record["region"],
                    record["languages"], record["rating"], record["score"],
                    record["personality"], record["keywords"], record["register_date"],
                    record["leave_date"], profile, vector,
                ),
            )
            row = await cur.fetchone()
            doctor_id = int(row["doctor_id"])
            report.doctors += 1
            report.slots += await _sync_slots(conn, doctor_id, record["slots"], shift)
        except Exception as exc:  # noqa: BLE001
            report.reject("doctor", record["name"], f"upsert failed: {exc}")


async def _sync_slots(conn, doctor_id: int, slots: list[datetime], shift: timedelta) -> int:
    """Replace this doctor's open slots, preserving anything already booked."""
    shifted = [slot + shift for slot in slots]
    await conn.execute(
        """
        DELETE FROM doctor_schedule
        WHERE doctor_id = %s
          AND status = 'available'
          AND schedule_id NOT IN (SELECT schedule_id FROM reservation)
        """,
        (doctor_id,),
    )
    written = 0
    for slot in shifted:
        cur = await conn.execute(
            """
            INSERT INTO doctor_schedule (doctor_id, available_time, status)
            VALUES (%s, %s, 'available')
            ON CONFLICT (doctor_id, available_time) DO NOTHING
            """,
            (doctor_id, slot),
        )
        written += cur.rowcount or 0
    return written


def known_specialties() -> tuple[str, ...]:
    return SPECIALTIES


def known_personalities() -> tuple[str, ...]:
    return PERSONALITIES
