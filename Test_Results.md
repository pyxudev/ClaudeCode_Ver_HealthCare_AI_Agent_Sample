# Test Results

> Built and running. docker-compose down -v && docker-compose up -d came up healthy in 10s with zero errors, and 286 tests pass.

## What's there

FastAPI + Python 3.13, PostgreSQL 16 + pgvector, Redis. The five agents from §7 (Intake → Classification → Search → Ranking → Response), hybrid retrieval (top-20 → top-3), all eight tables from §9, Twilio-shaped SMS/voice webhooks, Prometheus metrics, and a web chat UI at localhost:8080.

Two deliberate deviations, both documented in the README: no LangChain (explicit async orchestration over httpx — fewer ways for the build to fail, full control of timeouts and the circuit breaker) and hash-based embeddings instead of a transformer, so up -d needs no 500MB download. The LLM is optional throughout — with no key the whole pipeline runs deterministically.

## On risks

The design principle I worked to: the LLM never touches anything a mistake would make unsafe. It doesn't compute scores, write SQL, or see the database. It proposes; rules dispose. 30 risks are tabulated in TEST_REPORT.md §1 with the mitigation and the test that proves it.

Testing found five real bugs — four only visible when running it:

1. Cancel-then-rebook returned HTTP 500 (a UNIQUE constraint blocked reusing a freed slot).
2. All times displayed in UTC — a 15:00 JST appointment was read back as "6:00 AM".
3. The model invented a gender, calling a female doctor "he".
4. The model invented kanji for a name: "Dr. Yuki Nakamura" → 中村由紀.
5. "ent" matched inside "Kenta" — a patient named Kenta got routed to Otolaryngology.

Also worth flagging: every slot in sample_doctors.json is dated 2026-08, i.e. in the past. Loaded literally the POC has zero bookable availability, so ingest shifts the schedule forward (preserving gaps and time-of-day), logs it, and reports the offset at /readyz. One env var turns it off for real data.

## On what isn't covered

TEST_REPORT.md §5 is the part I'd want read most carefully. The honest headlines:

- Recommendation accuracy is not tested. Everything asserts plumbing and policy; nothing asserts a clinician would agree with the department chosen. The >95% target needs labelled real enquiries.
- Emergency triage is not clinically reviewed and its false-negative rate is unknown. That needs sign-off before it goes near a patient.
- Twilio signature verification is implemented and unit-tested but not wired into the routes — the webhooks accept unsigned requests. Must be enforced before any public exposure.
- No auth on any endpoint, no STT testing, no load testing, no HA/backup, and none of §10–11's WAF/Azure AD/Vault/K8s.

Verified live, not just asserted: a 10-way concurrent booking race gave exactly 1×200 and 9×409; Redis stopped mid-flight kept serving (fails open); and boots against a missing and a deliberately corrupt dataset stayed up with restarts=0, rejecting 4 bad records with reasons while loading the valid ones.
