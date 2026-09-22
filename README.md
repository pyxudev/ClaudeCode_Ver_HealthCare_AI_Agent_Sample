# Healthcare Doctor Recommendation AI Platform

Implementation of `Healthcare_Doctor_AI_System_Design.md`: an AI contact-centre
platform that takes a patient enquiry by voice, SMS or web chat, works out which
department they need, and recommends bookable doctors with appointment slots.

## Run it

```bash
cp .env.example .env    # sets POSTGRES_PASSWORD - compose refuses to start without it
docker-compose up -d
```

No API key required. Then:

| | |
|---|---|
| Web chat demo | http://localhost:8080/ |
| API docs | http://localhost:8080/docs |
| Readiness + ingest report | http://localhost:8080/readyz |
| Prometheus metrics | http://localhost:8080/metrics |

```bash
curl -s localhost:8080/api/v1/recommend -H 'content-type: application/json' -d '{
  "message": "My 5-year-old son has had an earache since yesterday. We are in Tokyo and would prefer an English-speaking doctor."
}' | python3 -m json.tool
```

Published ports are `8080`, `55432` (Postgres) and `56379` (Redis) — deliberately
not the defaults, because 5432/6379 are usually already taken. Override with
`API_HOST_PORT` / `POSTGRES_HOST_PORT` / `REDIS_HOST_PORT`.

Tests: `docker-compose exec api pytest` (282 tests, ~3s).
Teardown: `docker-compose down -v`.

## Optional: enable Claude

The platform runs fully without an LLM. To turn it on:

```bash
# in .env, set: ANTHROPIC_API_KEY=sk-ant-...
docker-compose up -d
```

`HDAI_LLM_ENABLED` is `auto` (use it if a key exists), `on`, or `off`.

## Architecture

```
voice / SMS / web ─► FastAPI ─► Intake ─► Classification ─► Search ─► Ranking ─► Response
                        │                                      │         │
                        │                       PostgreSQL 16 + pgvector │
                        │                       (SQL filter + vector)    │
                        └─► Redis (rate limit)   deterministic weights ───┘
                            contact_log + audit_log on every request
```

| Agent | Job | If the LLM fails |
|---|---|---|
| 1 Intake | patient's words → structured requirements | regex/keyword extraction |
| 2 Classification | symptoms → one of 11 departments | keyword rule table |
| 3 Search | structured SQL + pgvector, top-20 | unaffected — never uses the LLM |
| 4 Ranking | weighted score, top-3 | unaffected — never uses the LLM |
| 5 Response | natural-language reply | deterministic template |

Ranking weights, exactly as specified: specialty 35%, availability 20%, region
15%, language 10%, doctor rating 10%, hospital rating 5%, personality 5%.

### Deliberate deviations from the design doc

* **No LangChain.** The 5-agent orchestration is ~200 lines of explicit async
  Python calling the Messages API over `httpx`. Fewer dependencies in the boot
  path, and full control of timeouts, retries and the circuit breaker — which is
  what actually keeps the service inside its 10s budget.
* **Hash-based embeddings, not a transformer.** A real model would add ~500MB
  and a network dependency to `docker-compose up`. `embeddings.py` is a
  drop-in seam; see "Not covered" in the test report.
* The design doc says Claude Haiku 4.5, the PNG says OpenAI. This implements
  Claude (`claude-haiku-4-5-20251001`, configurable).

## Things that will bite you if you don't know them

* **The sample data is stale.** Every slot in `sample_doctors.json` is dated
  2026-08, which is in the past. Loaded literally the POC has zero bookable
  availability, so ingest shifts the whole schedule forward, preserving relative
  gaps and time-of-day. It logs loudly and reports the offset at `/readyz`.
  Set `HDAI_DEMO_SHIFT_PAST_SLOTS=false` for real data.
* **Times are stored UTC, displayed Asia/Tokyo.** An 08:30 JST slot is 23:30 UTC
  the *previous* day; all day arithmetic goes through `timeutil.py`.
* **Doctors reference hospitals by name string.** A doctor whose hospital name
  does not match is rejected and recorded in `ingest_log`, not silently dropped
  and not attached to an invented hospital.
* Ingest is idempotent and runs on every boot. It never deletes a booked slot.

## Failure behaviour

`/api/v1/recommend` always returns 200 with a usable reply — a voice gateway
cannot render a 500, the caller just hears silence. The degradation ladder:

| Condition | Behaviour |
|---|---|
| No API key / LLM down / circuit open | rules-only, `degraded_components` set |
| High-severity prompt injection | LLM bypassed entirely for that request |
| Emergency red flag | escalate to 119, no booking, no doctor |
| Generated reply fails grounding | silently replaced by the template |
| Search error / timeout / crash | operator-transfer message |
| Redis down | rate limiting fails open, service continues |

See `TEST_REPORT.md` for what is verified and what is not.
