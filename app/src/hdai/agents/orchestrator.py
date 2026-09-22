"""Agent orchestrator: safety gates, the five agents, persistence, fallbacks.

The contract this file enforces is that /recommend ALWAYS returns a usable
reply. A voice gateway cannot render an HTTP 500 - the caller just hears dead
air - so every failure is converted into a spoken answer plus a degraded
status, and the exception detail goes to the logs and the contact log.

Degradation ladder, worst case last:
  full pipeline -> LLM unavailable (rules only) -> search failed (operator
  transfer) -> unhandled exception (operator transfer)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from ..config import Settings
from ..db import Database, write_audit
from ..llm import LLMClient, LLMUsage
from ..models import (
    ClassificationResult,
    PatientRequirements,
    RecommendRequest,
    RecommendResponse,
)
from ..safety import (
    detect_injection,
    hash_identifier,
    looks_japanese,
    sanitise,
    scrub_pii,
    triage,
)
from . import classify, intake, ranking, respond, search

log = logging.getLogger("hdai.orchestrator")

OPERATOR_TRANSFER_EN = (
    "I'm sorry, I'm having trouble looking that up right now. "
    "Let me put you through to one of our operators who can help."
)
OPERATOR_TRANSFER_JA = (
    "申し訳ありません。ただいま検索に問題が発生しています。"
    "担当オペレーターにおつなぎいたします。"
)


class Orchestrator:
    def __init__(self, db: Database, llm: LLMClient, settings: Settings) -> None:
        self._db = db
        self._llm = llm
        self._settings = settings
        self._alias_cache: dict[str, str] = {}
        self._alias_loaded_at: float = 0.0

    async def aliases(self) -> dict[str, str]:
        """Cached for 5 minutes: the table is tiny but this is on the hot path."""
        if time.monotonic() - self._alias_loaded_at > 300 or not self._alias_cache:
            self._alias_cache = await search.load_aliases(self._db)
            self._alias_loaded_at = time.monotonic()
        return self._alias_cache

    # ------------------------------------------------------------------ main
    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._settings.request_timeout_s):
                return await self._recommend(request, request_id, started)
        except asyncio.TimeoutError:
            log.error(
                "request exceeded end-to-end budget",
                extra={"request_id": request_id, "budget_s": self._settings.request_timeout_s},
            )
            return await self._bail(
                request, request_id, started, "timeout", ["end_to_end_timeout"]
            )
        except Exception as exc:  # noqa: BLE001 - never surface a 500 to a phone line
            log.exception("unhandled orchestrator error", extra={"request_id": request_id})
            return await self._bail(
                request, request_id, started, f"error:{type(exc).__name__}", ["unhandled_error"]
            )

    async def _recommend(
        self, request: RecommendRequest, request_id: str, started: float
    ) -> RecommendResponse:
        now = datetime.now(tz=timezone.utc)
        usage = LLMUsage()
        degraded: list[str] = []

        message = sanitise(request.message, self._settings.max_input_chars)
        japanese = request.locale == "ja" or (request.locale is None and looks_japanese(message))

        if not message:
            return await self._finish(
                request, request_id, started, usage,
                RecommendResponse(
                    request_id=request_id,
                    status="clarification_needed",
                    message=("ご症状を教えていただけますか。" if japanese
                             else "Could you tell me what symptoms you're having?"),
                    follow_up_questions=classify.clarifying_questions(
                        ClassificationResult(specialty="General Practice"), japanese
                    ),
                    disclaimers=respond.disclaimers(japanese),
                ),
                degraded=degraded,
            )

        # --- Gate 1: prompt injection ------------------------------------
        injection = detect_injection(message)
        if injection.detected:
            log.warning(
                "prompt injection detected",
                extra={"request_id": request_id, "categories": list(injection.categories),
                       "severity": injection.severity},
            )
            if injection.severity == "high":
                # Do not hand a hostile payload to the model at all; the
                # deterministic path cannot be talked out of its behaviour.
                degraded.append("llm_bypassed_injection")

        use_llm = self._llm.available and "llm_bypassed_injection" not in degraded
        llm = self._llm if use_llm else _NullLLM()
        if not self._llm.available and self._llm.status != "disabled":
            degraded.append(f"llm_{self._llm.status}")

        # --- Gate 2: medical red flags -----------------------------------
        verdict = triage(message)
        if verdict.emergency:
            log.warning(
                "emergency red flag",
                extra={"request_id": request_id, "categories": list(verdict.categories)},
            )
            return await self._finish(
                request, request_id, started, usage,
                RecommendResponse(
                    request_id=request_id,
                    status="emergency",
                    message=verdict.message(japanese),
                    confidence=1.0,
                    disclaimers=respond.disclaimers(japanese),
                ),
                degraded=degraded,
                emergency=True,
                injection=injection.detected,
            )

        # --- Agent 1: intake ---------------------------------------------
        requirements, intake_usage, _ = await intake.run(
            message, llm, today=intake.local_today(),
            hints={
                "region": request.region,
                "language": request.language,
                "preferred_date": request.preferred_date,
            },
        )
        usage.merge(intake_usage)

        # --- Agent 2: classification -------------------------------------
        classification, classify_usage, _ = await classify.run(
            requirements, llm, dynamic_aliases=await self.aliases()
        )
        usage.merge(classify_usage)
        requirements.specialty = classification.specialty

        # --- Agent 3: search ----------------------------------------------
        try:
            found = await search.run(
                self._db, requirements, classification.specialty,
                now=now,
                limit=self._settings.retrieve_top_k,
                embedding_dim=self._settings.embedding_dim,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("search failed", extra={"request_id": request_id, "error": str(exc)})
            return await self._finish(
                request, request_id, started, usage,
                RecommendResponse(
                    request_id=request_id,
                    status="degraded",
                    message=OPERATOR_TRANSFER_JA if japanese else OPERATOR_TRANSFER_EN,
                    understood=requirements,
                    classification=classification,
                    degraded_components=degraded + ["search_unavailable"],
                    disclaimers=respond.disclaimers(japanese),
                ),
                degraded=degraded + ["search_unavailable"],
                injection=injection.detected,
            )

        # --- Agent 4: ranking (deterministic) -----------------------------
        recommendations = ranking.rank(
            found.candidates, requirements,
            now=now,
            top_n=self._settings.return_top_n,
            min_score=self._settings.min_recommendation_score,
        )
        confidence = ranking.overall_confidence(
            recommendations, classification.confidence, degraded=bool(degraded)
        )

        # --- Agent 5: response --------------------------------------------
        reply, respond_usage, _ = await respond.run(
            recommendations, requirements, classification.specialty, llm,
            japanese=japanese, relaxations=found.relaxations,
        )
        usage.merge(respond_usage)

        status = "ok"
        questions: list[str] = []
        if not recommendations:
            status = "no_match"
        elif classify.needs_clarification(classification):
            status = "clarification_needed"
            questions = classify.clarifying_questions(classification, japanese)
        elif degraded:
            status = "degraded"

        response = RecommendResponse(
            request_id=request_id,
            status=status,
            message=reply,
            recommendations=recommendations,
            confidence=confidence,
            understood=requirements,
            classification=classification,
            follow_up_questions=questions,
            disclaimers=respond.disclaimers(japanese),
            degraded_components=degraded + [f"search:{r}" for r in found.relaxations],
            llm_used=usage.calls > 0,
        )
        return await self._finish(
            request, request_id, started, usage, response,
            degraded=degraded, injection=injection.detected,
        )

    # -------------------------------------------------------------- fallback
    async def _bail(
        self,
        request: RecommendRequest,
        request_id: str,
        started: float,
        outcome: str,
        degraded: list[str],
    ) -> RecommendResponse:
        japanese = request.locale == "ja" or looks_japanese(request.message or "")
        response = RecommendResponse(
            request_id=request_id,
            status="degraded",
            message=OPERATOR_TRANSFER_JA if japanese else OPERATOR_TRANSFER_EN,
            degraded_components=degraded,
            disclaimers=respond.disclaimers(japanese),
        )
        try:
            return await self._finish(
                request, request_id, started, LLMUsage(), response,
                degraded=degraded, outcome=outcome,
            )
        except Exception:  # noqa: BLE001 - logging must not mask the fallback
            log.exception("failed to persist fallback contact log")
            response.latency_ms = int((time.perf_counter() - started) * 1000)
            return response

    # ------------------------------------------------------------ persistence
    async def _finish(
        self,
        request: RecommendRequest,
        request_id: str,
        started: float,
        usage: LLMUsage,
        response: RecommendResponse,
        *,
        degraded: list[str],
        emergency: bool = False,
        injection: bool = False,
        outcome: str | None = None,
    ) -> RecommendResponse:
        response.latency_ms = int((time.perf_counter() - started) * 1000)
        response.llm_used = usage.calls > 0

        keywords: dict[str, Any] = {}
        if response.understood is not None:
            keywords = response.understood.model_dump(
                include={"specialty", "region", "language", "personality",
                         "doctor_gender", "preferred_date", "urgency"}
            )
        keywords["symptom_excerpt"] = scrub_pii(sanitise(request.message, 200))

        top_doctor = response.recommendations[0].doctor.doctor_id if response.recommendations else None

        try:
            async with self._db.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO contact_log (request_id, session_id, channel, end_time, phone_hash,
                                             keywords, doctor_recommended, token_cost, latency_ms,
                                             outcome, llm_used, injection_flag, emergency_flag, degraded)
                    VALUES (%s, %s, %s, now(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    (
                        request_id,
                        request.session_id,
                        request.channel,
                        hash_identifier(request.caller_id),
                        Jsonb(keywords),
                        top_doctor,
                        usage.cost_usd,
                        response.latency_ms,
                        outcome or response.status,
                        response.llm_used,
                        injection,
                        emergency,
                        degraded,
                    ),
                )
                await write_audit(
                    conn,
                    action="READ",
                    table_name="doctor",
                    record_id=str(top_doctor) if top_doctor else None,
                    after={
                        "status": response.status,
                        "confidence": response.confidence,
                        "specialty": response.classification.specialty if response.classification else None,
                        "ranked": [r.doctor.internal_number for r in response.recommendations],
                        "latency_ms": response.latency_ms,
                    },
                    request_id=request_id,
                )
        except Exception as exc:  # noqa: BLE001
            # Losing a log line must not lose the patient's answer.
            log.error("contact log write failed",
                      extra={"request_id": request_id, "error": str(exc)})

        log.info(
            "recommendation served",
            extra={
                "request_id": request_id,
                "status": response.status,
                "specialty": response.classification.specialty if response.classification else None,
                "confidence": response.confidence,
                "latency_ms": response.latency_ms,
                "llm_used": response.llm_used,
                "token_cost_usd": usage.cost_usd,
                "results": len(response.recommendations),
                "degraded": degraded,
            },
        )
        return response


class _NullLLM:
    """Stands in for the client when the LLM must be skipped for one request."""

    available = False
    enabled = False
    status = "bypassed"

    async def complete_json(self, **_: Any):  # pragma: no cover - never called
        return None, LLMUsage(), "bypassed"

    async def complete_text(self, **_: Any):  # pragma: no cover - never called
        return None, LLMUsage(), "bypassed"
