"""Claude client with the guardrails an LLM needs before it faces patients.

Calls the Messages API over httpx directly rather than pulling in a framework:
one less dependency to break the build, and full control over timeouts,
retries and the circuit breaker - which is the part that actually determines
whether the service stays inside its 10s end-to-end budget when the provider
is slow.

Failure policy, in order:
  timeout / 429 / 5xx  -> bounded retry with jittered backoff
  repeated failures    -> circuit opens, every caller instantly gets None
  malformed JSON       -> one structural repair attempt, then None
  schema violation     -> None
None always means "use the deterministic rule-based path".
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings

log = logging.getLogger("hdai.llm")

T = TypeVar("T", bound=BaseModel)

# Rough Haiku 4.5 pricing in USD per million tokens, for the token_cost metric.
# Wrong pricing only makes a cost dashboard wrong, never a recommendation.
_PRICE_IN_PER_MTOK = 1.00
_PRICE_OUT_PER_MTOK = 5.00


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failures: int = 0

    @property
    def cost_usd(self) -> float:
        return round(
            self.input_tokens / 1e6 * _PRICE_IN_PER_MTOK
            + self.output_tokens / 1e6 * _PRICE_OUT_PER_MTOK,
            6,
        )

    def merge(self, other: "LLMUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.failures += other.failures


@dataclass
class _Breaker:
    threshold: int
    cooldown_s: float
    consecutive_failures: int = 0
    opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self.consecutive_failures < self.threshold:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_s:
            # Half-open: let one probe through.
            self.consecutive_failures = self.threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.opened_at = time.monotonic()


class LLMClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._client = client
        self._breaker = _Breaker(settings.llm_breaker_threshold, settings.llm_breaker_cooldown_s)
        self.total_usage = LLMUsage()

    # ---------------------------------------------------------------- state
    @property
    def enabled(self) -> bool:
        return self._settings.llm_active and bool(self._settings.anthropic_api_key.strip())

    @property
    def available(self) -> bool:
        return self.enabled and not self._breaker.is_open

    @property
    def status(self) -> str:
        # Order matters: "turned off" and "nobody gave us a key" are different
        # operational problems and the readiness probe reports them separately.
        if self._settings.llm_enabled == "off":
            return "disabled"
        if not self._settings.anthropic_api_key.strip():
            return "no_api_key"
        return "circuit_open" if self._breaker.is_open else "ready"

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.llm_base_url,
                timeout=httpx.Timeout(self._settings.llm_timeout_s, connect=3.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    # ----------------------------------------------------------------- call
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: Type[T],
        max_tokens: int | None = None,
        prefill: str = "{",
    ) -> tuple[T | None, LLMUsage, str]:
        """Returns (parsed | None, usage, reason).

        `reason` is one of ok / disabled / circuit_open / http_error /
        timeout / bad_json / schema_error - it lands in the contact log so the
        fallback rate per cause is measurable.
        """
        usage = LLMUsage()
        if not self.enabled:
            return None, usage, self.status
        if self._breaker.is_open:
            log.warning("llm circuit open, skipping call")
            return None, usage, "circuit_open"

        text, usage, reason = await self._request(system, user, max_tokens, prefill)
        self.total_usage.merge(usage)
        if text is None:
            return None, usage, reason

        # With an assistant prefill the API omits it from the completion, so it
        # has to be glued back on - unless the model echoed it anyway.
        stitched = text if not prefill or text.lstrip().startswith("{") else prefill + text
        parsed_dict = _extract_json(stitched)
        if parsed_dict is None:
            log.warning("llm returned unparseable json", extra={"snippet": (text or "")[:200]})
            return None, usage, "bad_json"

        try:
            return schema.model_validate(parsed_dict), usage, "ok"
        except ValidationError as exc:
            log.warning(
                "llm json failed schema validation",
                extra={"errors": exc.errors()[:3], "schema": schema.__name__},
            )
            return None, usage, "schema_error"

    async def complete_text(
        self, *, system: str, user: str, max_tokens: int | None = None
    ) -> tuple[str | None, LLMUsage, str]:
        usage = LLMUsage()
        if not self.enabled:
            return None, usage, self.status
        if self._breaker.is_open:
            return None, usage, "circuit_open"
        text, usage, reason = await self._request(system, user, max_tokens, prefill="")
        self.total_usage.merge(usage)
        return text, usage, reason

    async def _request(
        self, system: str, user: str, max_tokens: int | None, prefill: str
    ) -> tuple[str | None, LLMUsage, str]:
        usage = LLMUsage()
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        if prefill:
            # Assistant prefill forces the reply to start as JSON, which
            # removes the "Sure! Here is the JSON:" preamble failure mode.
            messages.append({"role": "assistant", "content": prefill})

        payload = {
            "model": self._settings.llm_model,
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "temperature": 0,  # recommendations must be reproducible
            "system": system,
            "messages": messages,
        }
        headers = {
            "x-api-key": self._settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_reason = "http_error"
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = await self._http().post("/v1/messages", json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_reason = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport_error"
                log.warning("llm request failed", extra={"attempt": attempt, "error": str(exc)})
                if attempt >= self._settings.llm_max_retries:
                    break
                await self._backoff(attempt)
                continue

            if response.status_code == 200:
                try:
                    body = response.json()
                    usage.calls = 1
                    usage.input_tokens = int(body.get("usage", {}).get("input_tokens", 0))
                    usage.output_tokens = int(body.get("usage", {}).get("output_tokens", 0))
                    blocks = body.get("content") or []
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                except (ValueError, AttributeError, TypeError) as exc:
                    self._breaker.record_failure()
                    usage.failures = 1
                    log.error("llm response shape unexpected", extra={"error": str(exc)})
                    return None, usage, "bad_response"
                self._breaker.record_success()
                return text, usage, "ok"

            # 429 / 5xx / 529 overloaded are retryable; 4xx is not.
            retryable = response.status_code == 429 or response.status_code >= 500
            last_reason = f"http_{response.status_code}"
            log.warning(
                "llm http error",
                extra={"status": response.status_code, "attempt": attempt,
                       "body": response.text[:200], "retryable": retryable},
            )
            if not retryable or attempt >= self._settings.llm_max_retries:
                break
            await self._backoff(attempt, response.headers.get("retry-after"))

        self._breaker.record_failure()
        usage.failures = 1
        return None, usage, last_reason

    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 3.0))
                return
            except ValueError:
                pass
        # Jitter so N concurrent calls do not retry in lockstep.
        delay = min(0.25 * (2 ** attempt), 2.0) * (0.5 + random.random())
        await asyncio.sleep(delay)


# --------------------------------------------------------------------------
# JSON extraction - models wrap JSON in prose or fences more often than not
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates: list[str] = []

    fenced = _FENCE_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for attempt in (candidate, _repair(candidate)):
            try:
                loaded = json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(loaded, dict):
                return loaded
    return None


def _first_balanced_object(text: str) -> str | None:
    """Scan for the first {...} whose braces balance outside of strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _repair(text: str) -> str:
    """Cheap structural fixes for the ways models truncate or over-format JSON."""
    repaired = text.strip()
    repaired = re.sub(r"^```(?:json)?|```$", "", repaired).strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)  # trailing commas
    if repaired.rstrip().endswith(","):
        repaired = repaired.rstrip().rstrip(",")

    # Truncated output: close what is still open, innermost first. Counting
    # braces and brackets separately would emit them in the wrong order and
    # turn `{"a": ["b"` into `{"a": ["b"}]`.
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in repaired:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if in_string:
        repaired += '"'
    return repaired + "".join(reversed(stack))
