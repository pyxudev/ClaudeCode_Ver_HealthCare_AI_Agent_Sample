"""LLM client guardrails: JSON extraction, retries, circuit breaker, schemas.

Every test here is a model-failure mode we expect in production, asserted to
degrade to None (= "use the rules") rather than to propagate garbage.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel

from hdai.config import Settings
from hdai.llm import LLMClient, _extract_json, _repair


class Draft(BaseModel):
    specialty: str
    confidence: float = 0.0


def settings(**overrides) -> Settings:
    base = {
        "anthropic_api_key": "test-key",
        "llm_enabled": "on",
        "llm_max_retries": 1,
        "llm_timeout_s": 1.0,
        "llm_breaker_threshold": 2,
        "llm_breaker_cooldown_s": 60.0,
    }
    base.update(overrides)
    return Settings(**base)


def client_with(handler, **overrides) -> LLMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="https://api.test")
    return LLMClient(settings(**overrides), client=http)


def reply(text: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 11, "output_tokens": 7},
        },
    )


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


class TestJsonExtraction:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"specialty": "Cardiology"}',
            'Sure! Here is the JSON:\n{"specialty": "Cardiology"}\nHope that helps.',
            '```json\n{"specialty": "Cardiology"}\n```',
            '```\n{"specialty": "Cardiology"}\n```',
            '{"specialty": "Cardiology",}',                 # trailing comma
            '{"specialty": "Cardiology"',                   # truncated
            '{"specialty": "Cardiology", "notes": ["a",',   # truncated mid-array
            '{"note": "he said \\"}\\" once", "specialty": "Cardiology"}',  # brace in a string
        ],
    )
    def test_survives_the_usual_model_formatting(self, raw):
        parsed = _extract_json(raw)
        assert parsed is not None and parsed.get("specialty") == "Cardiology"

    @pytest.mark.parametrize("raw", ["", "no json at all", "[1,2,3]", None])
    def test_returns_none_when_there_is_nothing_usable(self, raw):
        assert _extract_json(raw) is None

    def test_repair_is_idempotent_on_valid_json(self):
        assert _repair('{"a": 1}') == '{"a": 1}'


# --------------------------------------------------------------------------
# Call behaviour
# --------------------------------------------------------------------------


class TestCallPath:
    async def test_happy_path_parses_and_records_usage(self):
        llm = client_with(lambda r: reply('"specialty": "Cardiology", "confidence": 0.8}'))
        parsed, usage, reason = await llm.complete_json(
            system="s", user="u", schema=Draft
        )
        assert reason == "ok"
        assert parsed.specialty == "Cardiology" and parsed.confidence == 0.8
        assert usage.input_tokens == 11 and usage.output_tokens == 7
        assert llm.total_usage.cost_usd > 0
        await llm.aclose()

    async def test_disabled_without_an_api_key(self):
        llm = LLMClient(Settings(anthropic_api_key="", llm_enabled="auto"))
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "no_api_key"
        assert not llm.available

    async def test_schema_violation_is_rejected(self):
        llm = client_with(lambda r: reply('"wrong_field": 1}'))
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "schema_error"
        await llm.aclose()

    async def test_unparseable_output_is_rejected(self):
        llm = client_with(lambda r: reply("I'm sorry, I can't help with that."))
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "bad_json"
        await llm.aclose()

    async def test_retries_a_500_then_succeeds(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, json={"error": "server"})
            return reply('"specialty": "Neurology"}')

        llm = client_with(handler)
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert calls["n"] == 2 and reason == "ok" and parsed.specialty == "Neurology"
        await llm.aclose()

    async def test_does_not_retry_a_400(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(400, json={"error": "bad request"})

        llm = client_with(handler)
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert calls["n"] == 1 and parsed is None and reason == "http_400"
        await llm.aclose()

    async def test_timeout_is_handled_not_raised(self):
        def handler(request):
            raise httpx.ReadTimeout("too slow", request=request)

        llm = client_with(handler)
        parsed, usage, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "timeout" and usage.failures == 1
        await llm.aclose()

    async def test_malformed_http_body_is_handled(self):
        llm = client_with(lambda r: httpx.Response(200, content=b"not json"))
        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "bad_response"
        await llm.aclose()

    async def test_temperature_is_zero_for_reproducibility(self):
        seen = {}

        def handler(request):
            import json as _json

            seen.update(_json.loads(request.content))
            return reply('"specialty": "Cardiology"}')

        llm = client_with(handler)
        await llm.complete_json(system="s", user="u", schema=Draft)
        assert seen["temperature"] == 0
        assert seen["messages"][-1]["role"] == "assistant"  # JSON prefill
        await llm.aclose()


class TestCircuitBreaker:
    async def test_opens_after_repeated_failures_and_stops_calling(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(503, json={"error": "overloaded"})

        llm = client_with(handler, llm_max_retries=0, llm_breaker_threshold=2)
        for _ in range(2):
            await llm.complete_json(system="s", user="u", schema=Draft)
        assert calls["n"] == 2
        assert llm.status == "circuit_open" and not llm.available

        parsed, _, reason = await llm.complete_json(system="s", user="u", schema=Draft)
        assert parsed is None and reason == "circuit_open"
        assert calls["n"] == 2, "an open circuit must not issue a request"
        await llm.aclose()

    async def test_a_success_resets_the_failure_count(self):
        state = {"fail": True}

        def handler(request):
            if state["fail"]:
                return httpx.Response(500)
            return reply('"specialty": "Cardiology"}')

        llm = client_with(handler, llm_max_retries=0, llm_breaker_threshold=3)
        await llm.complete_json(system="s", user="u", schema=Draft)
        state["fail"] = False
        await llm.complete_json(system="s", user="u", schema=Draft)
        state["fail"] = True
        await llm.complete_json(system="s", user="u", schema=Draft)
        assert llm.available, "breaker should not be open after an interleaved success"
        await llm.aclose()

    async def test_cooldown_allows_a_probe(self):
        llm = client_with(lambda r: httpx.Response(500), llm_max_retries=0,
                          llm_breaker_threshold=1, llm_breaker_cooldown_s=0.0)
        await llm.complete_json(system="s", user="u", schema=Draft)
        assert llm.available, "a zero cooldown should half-open immediately"
        await llm.aclose()
