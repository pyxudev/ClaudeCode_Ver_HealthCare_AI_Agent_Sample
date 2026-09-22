"""FastAPI application: lifespan, middleware, routes.

Startup is deliberately forgiving. A container that crash-loops on boot
because Postgres needed three more seconds, or because the dataset had one bad
row, is a worse operational outcome than one that comes up and reports itself
as not-ready. So startup records failures on app.state and /readyz tells the
truth about them, rather than the process dying.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError

from . import reservations
from .agents.orchestrator import Orchestrator
from .cache import Cache
from .config import Settings, get_settings
from .db import Database
from .ingest import ingest_file
from .llm import LLMClient
from .logging_conf import configure_logging
from .metrics import METRICS
from .models import (
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    ReservationRequest,
    ReservationResponse,
)
from .safety import looks_japanese, sanitise

log = logging.getLogger("hdai.api")

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info(
        "starting",
        extra={"llm": settings.llm_active, "model": settings.llm_model,
               "data_file": settings.data_file},
    )

    app.state.settings = settings
    app.state.startup_errors = []
    app.state.ingest_report = None
    app.state.started_at = datetime.now(tz=timezone.utc)

    db = Database(settings)
    app.state.db = db
    cache = Cache(settings.redis_url)
    app.state.cache = cache
    llm = LLMClient(settings)
    app.state.llm = llm

    try:
        await db.connect()
        await db.migrate()
    except Exception as exc:  # noqa: BLE001
        log.exception("database initialisation failed")
        app.state.startup_errors.append(f"database: {exc}")

    await cache.connect()

    if settings.ingest_on_startup and db.is_open:
        try:
            report = await ingest_file(db, settings)
            app.state.ingest_report = report.as_dict()
            if report.rejected:
                app.state.startup_errors.append(
                    f"ingest rejected {len(report.rejected)} record(s)"
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("ingestion failed")
            app.state.startup_errors.append(f"ingest: {exc}")

    app.state.orchestrator = Orchestrator(db, llm, settings)
    log.info("ready", extra={"startup_errors": app.state.startup_errors})

    try:
        yield
    finally:
        await llm.aclose()
        await cache.close()
        await db.close()
        log.info("stopped")


app = FastAPI(
    title="Healthcare Doctor Recommendation AI Platform",
    version="1.0.0",
    description=(
        "Voice / SMS / web-chat doctor recommendation. Five-agent pipeline over "
        "PostgreSQL + pgvector hybrid search with deterministic business ranking."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Middleware
# --------------------------------------------------------------------------


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    route = request.url.path
    try:
        response = await call_next(request)
    except Exception:
        METRICS.inc("hdai_http_requests_total", {"route": route, "status": "500"})
        log.exception("unhandled request error", extra={"request_id": request_id, "route": route})
        return JSONResponse(
            status_code=500,
            content={"detail": "internal error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
    elapsed = time.perf_counter() - started
    METRICS.inc("hdai_http_requests_total", {"route": route, "status": str(response.status_code)})
    METRICS.observe("hdai_http_request_seconds", elapsed, {"route": route})
    response.headers["x-request-id"] = request_id
    return response


async def rate_limit(request: Request) -> None:
    settings: Settings = request.app.state.settings
    cache: Cache = request.app.state.cache
    client = request.client.host if request.client else "unknown"
    if not await cache.allow(f"{client}:{request.url.path}", settings.rate_limit_per_minute):
        METRICS.inc("hdai_rate_limited_total", {"route": request.url.path})
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="service still starting")
    return orchestrator


def get_db(request: Request) -> Database:
    db: Database = request.app.state.db
    if not db.is_open:
        raise HTTPException(status_code=503, detail="database unavailable")
    return db


# --------------------------------------------------------------------------
# Health / ops
# --------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz() -> HealthResponse:
    """Liveness: is the process able to serve at all."""
    return HealthResponse(status="alive")


@app.get("/readyz", tags=["ops"])
async def readyz(request: Request) -> JSONResponse:
    """Readiness: can this instance actually answer a patient."""
    db: Database = request.app.state.db
    cache: Cache = request.app.state.cache
    llm: LLMClient = request.app.state.llm

    db_ok = await db.ping() if db.is_open else False
    doctor_count = 0
    if db_ok:
        try:
            row = await db.fetch_one("SELECT COUNT(*) AS n FROM doctor WHERE leave_date IS NULL")
            doctor_count = int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            db_ok = False

    redis_ok = await cache.ping()
    ready = db_ok and doctor_count > 0

    checks: dict[str, Any] = {
        "database": "ok" if db_ok else "down",
        "doctors_loaded": doctor_count,
        # Redis is explicitly non-fatal: rate limiting fails open.
        "redis": "ok" if redis_ok else "degraded",
        "llm": llm.status,
        "startup_errors": request.app.state.startup_errors,
        "ingest": request.app.state.ingest_report,
    }
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks,
                 "version": app.version},
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics(request: Request) -> PlainTextResponse:
    llm: LLMClient = request.app.state.llm
    METRICS.set_gauge("hdai_llm_tokens_input_total", llm.total_usage.input_tokens)
    METRICS.set_gauge("hdai_llm_tokens_output_total", llm.total_usage.output_tokens)
    METRICS.set_gauge("hdai_llm_cost_usd_total", llm.total_usage.cost_usd)
    METRICS.set_gauge("hdai_llm_failures_total", llm.total_usage.failures)
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")


# --------------------------------------------------------------------------
# Core API
# --------------------------------------------------------------------------


@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    tags=["recommendation"],
    dependencies=[Depends(rate_limit)],
)
async def recommend(
    payload: RecommendRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> RecommendResponse:
    started = time.perf_counter()
    response = await orchestrator.recommend(payload)
    METRICS.observe("hdai_recommendation_seconds", time.perf_counter() - started)
    METRICS.inc("hdai_recommendations_total", {"status": response.status})
    if response.status == "emergency":
        METRICS.inc("hdai_emergency_escalations_total")
    if response.degraded_components:
        METRICS.inc("hdai_degraded_total")
    return response


@app.post(
    "/api/v1/reservations",
    response_model=ReservationResponse,
    tags=["reservation"],
    dependencies=[Depends(rate_limit)],
)
async def create_reservation(
    payload: ReservationRequest,
    request: Request,
    db: Database = Depends(get_db),
) -> ReservationResponse:
    try:
        booking = await reservations.book(
            db,
            schedule_id=payload.schedule_id,
            caller_id=payload.caller_id,
            session_id=payload.session_id,
            idempotency_key=payload.idempotency_key,
            request_id=getattr(request.state, "request_id", None),
        )
    except reservations.BookingError as exc:
        METRICS.inc("hdai_reservations_total", {"result": exc.code})
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    METRICS.inc(
        "hdai_reservations_total",
        {"result": "replay" if booking.idempotent_replay else "created"},
    )
    return ReservationResponse(**booking.__dict__)


@app.delete("/api/v1/reservations/{reservation_id}", tags=["reservation"])
async def delete_reservation(
    reservation_id: int, request: Request, db: Database = Depends(get_db)
) -> dict[str, Any]:
    ok = await reservations.cancel(
        db, reservation_id, request_id=getattr(request.state, "request_id", None)
    )
    if not ok:
        raise HTTPException(status_code=404, detail="reservation not found")
    METRICS.inc("hdai_reservations_total", {"result": "cancelled"})
    return {"reservation_id": reservation_id, "status": "cancelled"}


@app.get("/api/v1/doctors", tags=["catalogue"])
async def list_doctors(
    db: Database = Depends(get_db),
    specialty: str | None = None,
    region: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT d.doctor_id, d.internal_number, d.name, d.expertise, d.region, d.languages,
               d.rating, d.score, d.personality, d.gender, d.age,
               h.name AS hospital_name, h.rating AS hospital_rating,
               (SELECT COUNT(*) FROM doctor_schedule s
                 WHERE s.doctor_id = d.doctor_id AND s.status = 'available'
                   AND s.available_time > now()) AS open_slots
        FROM doctor d
        JOIN hospital h ON h.hospital_id = d.hospital_id
        WHERE d.leave_date IS NULL
          AND (%s::text IS NULL OR d.expertise = %s)
          AND (%s::text IS NULL OR d.region = %s)
        ORDER BY d.score DESC, d.doctor_id
        LIMIT %s
        """,
        (specialty, specialty, region, region, max(1, min(limit, 200))),
    )
    return {"count": len(rows), "doctors": rows}


@app.get("/api/v1/doctors/{doctor_id}/slots", tags=["catalogue"])
async def doctor_slots(doctor_id: int, db: Database = Depends(get_db)) -> dict[str, Any]:
    rows = await db.fetch_all(
        """
        SELECT schedule_id, available_time, status
        FROM doctor_schedule
        WHERE doctor_id = %s AND available_time > now()
        ORDER BY available_time
        """,
        (doctor_id,),
    )
    return {"doctor_id": doctor_id, "slots": rows}


@app.get("/api/v1/stats", tags=["ops"])
async def stats(db: Database = Depends(get_db)) -> dict[str, Any]:
    """Operational summary for the supervisor screen (design doc section 15)."""
    row = await db.fetch_one(
        """
        SELECT
          (SELECT COUNT(*) FROM doctor WHERE leave_date IS NULL) AS doctors,
          (SELECT COUNT(*) FROM hospital) AS hospitals,
          (SELECT COUNT(*) FROM doctor_schedule WHERE status = 'available'
             AND available_time > now()) AS open_slots,
          (SELECT COUNT(*) FROM reservation WHERE status = 'confirmed') AS reservations,
          (SELECT COUNT(*) FROM contact_log) AS contacts,
          (SELECT COUNT(*) FROM contact_log WHERE emergency_flag) AS emergencies,
          (SELECT COUNT(*) FROM contact_log WHERE injection_flag) AS injection_attempts,
          (SELECT ROUND(AVG(latency_ms)) FROM contact_log) AS avg_latency_ms,
          (SELECT COALESCE(SUM(token_cost), 0) FROM contact_log) AS token_cost_usd,
          (SELECT COUNT(*) FROM audit_log) AS audit_rows
        """
    )
    return row or {}


# --------------------------------------------------------------------------
# Channel webhooks (design doc section 5). Twilio-shaped, no SDK required.
# --------------------------------------------------------------------------


@app.post("/webhooks/sms", tags=["channels"], dependencies=[Depends(rate_limit)])
async def sms_webhook(
    request: Request,
    Body: str = Form(default=""),
    From: str = Form(default=""),
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> Response:
    """Inbound SMS. Replies with TwiML; the raw number is never persisted."""
    response = await _handle_channel(orchestrator, Body, From, channel="sms")
    body = _xml_escape(_shorten(response.message, 600))
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{body}</Message></Response>',
        media_type="application/xml",
    )


@app.post("/webhooks/voice", tags=["channels"], dependencies=[Depends(rate_limit)])
async def voice_webhook(
    request: Request,
    SpeechResult: str = Form(default=""),
    From: str = Form(default=""),
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> Response:
    """Inbound voice turn. SpeechResult is the STT transcript from the gateway."""
    response = await _handle_channel(orchestrator, SpeechResult, From, channel="voice")
    japanese = looks_japanese(response.message)
    say = _xml_escape(_shorten(response.message, 800))
    voice_attrs = 'language="ja-JP"' if japanese else 'language="en-US"'
    return Response(
        content=(
            '<?xml version="1.0" encoding="UTF-8"?><Response>'
            f"<Say {voice_attrs}>{say}</Say>"
            "</Response>"
        ),
        media_type="application/xml",
    )


async def _handle_channel(
    orchestrator: Orchestrator, text: str, caller: str, *, channel: str
) -> RecommendResponse:
    cleaned = sanitise(text, 2000)
    if not cleaned:
        return RecommendResponse(
            request_id=uuid.uuid4().hex,
            status="clarification_needed",
            message="Could you tell me what symptoms you're having, and which area you're in?",
        )
    try:
        payload = RecommendRequest(
            message=cleaned, channel=channel, caller_id=caller or None  # type: ignore[arg-type]
        )
    except ValidationError:
        return RecommendResponse(
            request_id=uuid.uuid4().hex,
            status="clarification_needed",
            message="Sorry, I didn't catch that. Could you say it again?",
        )
    return await orchestrator.recommend(payload)


def _shorten(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&apos;")
    )


# --------------------------------------------------------------------------
# Demo web chat
# --------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def index() -> Response:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return PlainTextResponse("UI not bundled. API docs are at /docs")
    return FileResponse(page)
