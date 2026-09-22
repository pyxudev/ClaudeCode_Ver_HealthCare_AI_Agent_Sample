"""Appointment booking.

Two callers racing for the last slot is the concurrency bug that matters here
(two patients told they have the same appointment), so booking is a
compare-and-set inside one transaction:

    UPDATE doctor_schedule SET status='booked'
    WHERE schedule_id = %s AND status = 'available'

Exactly one transaction sees rowcount 1; the loser gets 409. A UNIQUE
constraint on reservation.schedule_id backs this up at the storage layer so
the invariant holds even if some future code path forgets the guard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg

from .db import Database, write_audit
from .safety import hash_identifier

log = logging.getLogger("hdai.reservations")


class BookingError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass
class Booking:
    reservation_id: int
    status: str
    doctor_id: int
    doctor_name: str
    hospital_name: str
    slot_time: datetime
    idempotent_replay: bool = False


async def book(
    db: Database,
    *,
    schedule_id: int,
    caller_id: str | None,
    session_id: str | None,
    idempotency_key: str | None,
    request_id: str | None = None,
) -> Booking:
    patient_hash = hash_identifier(caller_id)

    if idempotency_key:
        existing = await _find_by_idempotency_key(db, idempotency_key)
        if existing is not None:
            existing.idempotent_replay = True
            return existing

    async with db.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                """
                SELECT s.schedule_id, s.doctor_id, s.available_time, s.status,
                       d.name AS doctor_name, h.name AS hospital_name
                FROM doctor_schedule s
                JOIN doctor d ON d.doctor_id = s.doctor_id
                JOIN hospital h ON h.hospital_id = d.hospital_id
                WHERE s.schedule_id = %s
                FOR UPDATE OF s
                """,
                (schedule_id,),
            )
            slot = await cur.fetchone()
            if slot is None:
                raise BookingError("slot_not_found", "That appointment slot does not exist.", 404)
            if slot["status"] != "available":
                raise BookingError(
                    "slot_taken",
                    "That slot has just been taken. Please choose another time.",
                    409,
                )
            if slot["available_time"] <= datetime.now(tz=timezone.utc):
                raise BookingError(
                    "slot_in_past", "That appointment time has already passed.", 409
                )

            updated = await conn.execute(
                """
                UPDATE doctor_schedule SET status = 'booked', updated_at = now()
                WHERE schedule_id = %s AND status = 'available'
                """,
                (schedule_id,),
            )
            if updated.rowcount != 1:
                raise BookingError(
                    "slot_taken",
                    "That slot has just been taken. Please choose another time.",
                    409,
                )

            try:
                cur = await conn.execute(
                    """
                    INSERT INTO reservation (patient_id_hash, doctor_id, schedule_id, slot_time,
                                             status, idempotency_key, session_id)
                    VALUES (%s, %s, %s, %s, 'confirmed', %s, %s)
                    RETURNING reservation_id, status
                    """,
                    (
                        patient_hash, int(slot["doctor_id"]), schedule_id,
                        slot["available_time"], idempotency_key, session_id,
                    ),
                )
            except psycopg.errors.UniqueViolation as exc:
                # The storage-level guard fired. Report it as the business
                # outcome it is, not as a 500.
                raise BookingError(
                    "slot_taken",
                    "That slot has just been taken. Please choose another time.",
                    409,
                ) from exc
            created = await cur.fetchone()

            await write_audit(
                conn,
                action="INSERT",
                table_name="reservation",
                record_id=str(created["reservation_id"]),
                before={"schedule_id": schedule_id, "status": "available"},
                after={
                    "schedule_id": schedule_id,
                    "status": "booked",
                    "doctor_id": int(slot["doctor_id"]),
                    "patient_id_hash": patient_hash,
                },
                request_id=request_id,
            )

    log.info(
        "reservation created",
        extra={"reservation_id": int(created["reservation_id"]), "schedule_id": schedule_id,
               "doctor_id": int(slot["doctor_id"])},
    )
    return Booking(
        reservation_id=int(created["reservation_id"]),
        status=str(created["status"]),
        doctor_id=int(slot["doctor_id"]),
        doctor_name=str(slot["doctor_name"]),
        hospital_name=str(slot["hospital_name"]),
        slot_time=slot["available_time"],
    )


async def _find_by_idempotency_key(db: Database, key: str) -> Booking | None:
    row = await db.fetch_one(
        """
        SELECT r.reservation_id, r.status, r.doctor_id, r.slot_time,
               d.name AS doctor_name, h.name AS hospital_name
        FROM reservation r
        JOIN doctor d ON d.doctor_id = r.doctor_id
        JOIN hospital h ON h.hospital_id = d.hospital_id
        WHERE r.idempotency_key = %s
        """,
        (key,),
    )
    if row is None:
        return None
    return Booking(
        reservation_id=int(row["reservation_id"]),
        status=str(row["status"]),
        doctor_id=int(row["doctor_id"]),
        doctor_name=str(row["doctor_name"]),
        hospital_name=str(row["hospital_name"]),
        slot_time=row["slot_time"],
    )


async def cancel(db: Database, reservation_id: int, *, request_id: str | None = None) -> bool:
    """Release the slot back to the pool. Idempotent."""
    async with db.connection() as conn:
        async with conn.transaction():
            cur = await conn.execute(
                "SELECT schedule_id, status FROM reservation WHERE reservation_id = %s FOR UPDATE",
                (reservation_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return False
            if row["status"] == "cancelled":
                return True
            await conn.execute(
                "UPDATE reservation SET status = 'cancelled' WHERE reservation_id = %s",
                (reservation_id,),
            )
            await conn.execute(
                "UPDATE doctor_schedule SET status = 'available', updated_at = now() "
                "WHERE schedule_id = %s",
                (row["schedule_id"],),
            )
            await write_audit(
                conn,
                action="UPDATE",
                table_name="reservation",
                record_id=str(reservation_id),
                before={"status": row["status"]},
                after={"status": "cancelled"},
                request_id=request_id,
            )
    return True
