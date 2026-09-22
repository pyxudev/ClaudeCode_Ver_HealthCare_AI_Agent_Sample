# Test Report

**286 automated tests, all passing** (`docker-compose exec api pytest`, ~3s),
plus the manual verifications in section 3. Run against the real stack on a
clean volume: `docker-compose down -v && docker-compose up -d` came up healthy
in 10s with no errors.

---

## 1. Risks identified up front, and what was done about each

The design doc specifies *what* the system does. These are the failure modes I
worked backwards from — most of them are places where a language model's
mistake becomes a patient's problem.

| # | Risk | Mitigation | Verified by |
|---|---|---|---|
| R1 | Model invents a doctor / slot that doesn't exist | Reply is checked against the chosen candidates; any unknown name, time, or diagnosis-like phrasing falls back to the template | `test_safety.py::TestGrounding` (8) |
| R2 | Model invents a **gender** for a doctor | Facts carry no gender → any `he/she` is ungrounded and rejected | `test_rejects_an_invented_pronoun`; **found live**, §4 |
| R3 | Model **transliterates** a romaji name into invented kanji | Top pick's name must appear verbatim | `test_rejects_a_transliterated_name`; **found live**, §4 |
| R4 | Model hallucinates a specialty / region / language | Everything is coerced onto a closed vocabulary; unmappable → dropped, not guessed | `test_taxonomy.py` (28), `test_hallucinated_enum_values_are_dropped` |
| R5 | Model does the ranking arithmetic | It never sees it. Ranking is pure Python, weights asserted against the doc | `test_ranking.py` (26) |
| R6 | Model **downgrades** stated urgency | Merge takes the max; escalation only | `test_urgency_can_only_be_escalated_by_the_model` |
| R7 | Model over-reads ("she said Tokyo so she wants a woman") | Rules run first and win; the LLM may only fill gaps | `test_llm_fills_only_the_gaps` |
| R8 | Prompt injection redirects the recommendation | Detected pre-pipeline; high-severity bypasses the LLM entirely; ranking is deterministic so it cannot be talked into anything | `test_safety.py::TestInjection` (13), `test_prompt_injection_cannot_choose_the_doctor` |
| R9 | Injection hidden with full-width/zero-width characters | NFKC + control-character stripping before matching | `test_normalises_fullwidth_so_filters_cannot_be_bypassed` |
| R10 | **An emergency gets booked as a routine appointment** | 8 red-flag categories, EN+JA, short-circuit the whole pipeline to a 119 escalation | `TestTriage` (12), `test_emergency_is_escalated_and_never_booked` |
| R11 | System implies a diagnosis | Prompts forbid it, grounding rejects it, every reply carries a disclaimer | `test_rejects_anything_that_reads_as_medical_advice`, `test_every_reply_carries_a_disclaimer` |
| R12 | Malformed JSON from the model | Fence-stripping, balanced-brace scan, trailing-comma and truncation repair, then schema validation | `TestJsonExtraction` (12) |
| R13 | LLM slow/down → blows the 10s budget | 8s timeout, bounded retry, circuit breaker, whole request capped by `asyncio.timeout` | `TestCircuitBreaker` (3), `TestCallPath` (9) |
| R14 | No API key at all | Full rules-only pipeline; `llm_enabled=auto` | 286 tests run this way; §3-F |
| R15 | **Two patients given the same slot** | Compare-and-set in one transaction + partial unique index | `TestBooking` (6); 10-way live race, §3-A |
| R16 | Retried booking creates two appointments | Idempotency key, replayed not re-booked | `test_idempotency_key_replays_instead_of_double_booking` |
| R17 | Raw phone numbers stored or logged | HMAC-SHA256 at every boundary; symptom excerpts scrubbed | `TestPII` (4), `test_patient_identifier_is_stored_hashed`, `TestContactLog` |
| R18 | Bad data row silently drops a doctor from search | Per-record rejection recorded in `ingest_log` and surfaced at `/readyz` | `test_ingest.py` (19); §3-E |
| R19 | **Every slot in the sample data is in the past** → zero availability | Whole-schedule shift preserving gaps and time-of-day, logged loudly | `TestSlotShift` (5), `test_the_dataset_really_is_stale` |
| R20 | UTC/JST off-by-one (08:30 JST = 23:30 UTC *previous day*) | All day arithmetic and display via `timeutil.py` | `test_ranking.py` availability tests; **found live**, §4 |
| R21 | Ingest on every boot duplicates rows / destroys bookings | Upsert by `internal_number`; only unbooked slots are replaced | `TestIdempotentIngest`, §3-C |
| R22 | Strict filters return nothing for a willing patient | Progressive relaxation in a defined order, each step disclosed | `test_impossible_constraints_relax_rather_than_return_nothing` |
| R23 | Embeddings differ between ingest and query | blake2b, never `hash()` (which is per-process randomised) | `test_stable_across_processes_with_different_hash_seeds` |
| R24 | SQL injection via patient text | Parameters only; the model never writes SQL | `test_sql_injection_in_free_text_is_inert` |
| R25 | Redis down takes the service down | Fails open with a logged warning | §3-B |
| R26 | An exception returns 500 → dead air on a phone call | Every path returns 200 + an operator-transfer message | `Orchestrator._bail`, `TestValidation` |
| R27 | Confident-looking answer built on a guessed specialty | Confidence multiplies classification × score × runner-up margin; low → ask a question | `TestConfidence` (5), `test_vague_symptoms_ask_a_question` |
| R28 | A child routed to an adult department | Paediatric override unless specialty evidence is strong | `test_children_are_routed_to_paediatrics` + the converse |
| R29 | Crash-loop on boot (bad data, slow DB) | Connect retries; startup errors recorded, not fatal | §3-D, §3-E |
| R30 | Non-deterministic recommendations | `temperature=0`, explicit tie-breakers, no `hash()`/`random` in scoring | `test_is_deterministic_regardless_of_input_order`, `test_is_reproducible` |

---

## 2. Automated test coverage (286 tests)

| File | Tests | Covers |
|---|---:|---|
| `test_safety.py` | 44 | sanitisation, 8 injection patterns + 4 false-positive guards, 8 emergency categories EN/JA, PII hashing/scrubbing, grounding (invented doctor / slot / diagnosis / pronoun / transliteration), Twilio signature |
| `test_ranking.py` | 26 | weights vs the doc, perfect-score case, breakdown sums to total, each of 7 dimensions, neutral-preference rule, availability decay, urgency penalty, ordering, top-N, determinism, min-score, confidence |
| `test_taxonomy.py` | 40 | alias coercion (dataset's ENT/Pedia/Heart-doctor + JA), unknown → `None`, DB aliases override, language parsing, 12 symptom→specialty cases, ambiguity penalty |
| `test_llm.py` | 27 | 8 malformed-JSON shapes, schema violation, HTTP 400/500/timeout/bad-body, retry policy, circuit breaker open/reset/half-open, `temperature=0`, JSON prefill |
| `test_agents.py` | 33 | intake extraction (region/language/gender/date/urgency/age/time-of-day), false-positive guards, hostile input, LLM gap-filling, hallucination dropping, urgency escalation-only; classification reconciliation, unknown-specialty discard, paediatric override, clarification |
| `test_embeddings.py` | 16 | dimension, L2 norm, empty input, cross-process determinism, similarity ordering, Japanese, pgvector literal |
| `test_ingest.py` | 19 | cleaning, rating/score clamping, 4 slot formats + garbage, JST handling, slot shift (4 cases), synthetic keys, rejection recording, **+5 assertions against the real `sample_doctors.json`** |
| `test_api_integration.py` | 81 | boot/readiness/metrics/OpenAPI/UI, 12 recommendation scenarios, 7 safety scenarios, 8 validation cases, 6 booking cases incl. double-booking and idempotency, 5 channel webhooks, contact-log/audit persistence, re-ingest idempotency |

Integration tests pin `HDAI_LLM_ENABLED=off` so they are deterministic and free;
the LLM path is covered by `test_llm.py` with a mock transport, and manually
against the live API (§3-G).

---

## 3. Manual verification (executed, with results)

| | Check | Result |
|---|---|---|
| A | 10 parallel bookings on one slot | **1 × HTTP 200, 9 × HTTP 409**, exactly one `confirmed` row |
| B | `docker-compose stop redis`, then recommend | HTTP 200; `/readyz` = ready, `redis: degraded` — fails open as designed |
| C | Second `up -d`, then `restart api` | 0 errors; 15 doctors / 4 hospitals / 43 slots unchanged |
| D | Boot with a **missing** dataset | `running, restarts=0`; `/readyz` reports `dataset file not found` — no crash-loop |
| E | Boot with a **corrupt** dataset (null name, non-object row, unknown hospital, `rating:"n/a"`, `score:500`, `age:3`, `leave_date < register_date`, garbage + duplicate slots, alias with no keyword) | `running, restarts=0`; 4 records rejected with reasons, 8 normalisation notes, valid records still loaded |
| F | `HDAI_LLM_ENABLED=off` | Correct answer (Otolaryngology, Tokyo, English) in **12 ms** via the template |
| G | Live LLM, 4 scenarios | Paediatric ENT 2.9s/conf 0.85; emergency escalated with 0 recommendations; injection → Dermatology (not the demanded doctor); Japanese in → Japanese out |
| H | Clean slate `down -v` → `up -d` | Healthy in 10s, 0 errors, 4 hospitals / 15 doctors / 43 slots / 3 aliases / **0 rejected** |

---

## 4. Defects found and fixed during testing

Five real bugs, four of which only appeared when running the thing:

1. **Cancel-then-rebook returned HTTP 500.** `UNIQUE(schedule_id)` on
   `reservation` blocked rebooking a slot whose reservation was cancelled. Fixed
   with a partial unique index over active statuses only, plus a
   `UniqueViolation` → 409 guard so a race is never a 500.
2. **All times shown in UTC.** A 15:00 JST appointment was read back to the
   patient as "6:00 AM". Fixed by centralising display and day arithmetic in
   `timeutil.py` — this also removed an off-by-one for 08:30 JST slots, which
   fall on the previous UTC day.
3. **The model invented a gender**, describing a female doctor as "he". The
   facts contain no gender, so gendered pronouns are now a grounding failure.
4. **The model invented kanji for a name**, rendering "Dr. Yuki Nakamura" as
   中村由紀 in the Japanese reply. Names must now appear verbatim.
5. **`"ent"` matched inside `"Kenta"` and `"Department"`**, routing a patient
   named Kenta to Otolaryngology. Alias matching is now whole-word for Latin
   text, plain substring for CJK (which has no word boundaries).

Also fixed pre-deployment: missing `python-multipart` (container crash-looped),
JSON repair closing braces before brackets, and a relaxation step that
advertised "I widened the search" for candidates ranking then discarded.

---

## 5. NOT covered by these tests

Being explicit about this, because the gaps matter more than the green ticks.

### Not tested at all

- **Real Twilio.** Webhooks are Twilio-*shaped* and signature verification is
  implemented and unit-tested, but it is **not wired into the routes** — there is
  no `TWILIO_AUTH_TOKEN` in the config and the endpoints accept unsigned
  requests. Fine behind a private ingress; **must be enforced before any public
  deployment**. No real call, STT, or carrier round trip was exercised.
- **Speech-to-text.** `SpeechResult` is assumed to be a clean transcript.
  Mis-hears ("earache"→"ear rake"), partial utterances, code-switching mid-call,
  barge-in, and multi-turn state are untested. Real STT error rates are the
  single biggest untested threat to the 95% accuracy target.
- **Recommendation *accuracy*.** Every test asserts plumbing and policy. Nothing
  asserts that a clinician would agree with the department chosen. Hitting >95%
  needs a labelled set of real enquiries with expert adjudication; the 12
  symptom cases here are my own assumptions, not ground truth.
- **Load, concurrency at scale, soak.** The 3s/10s targets are verified on one
  request at a time on an idle laptop (12ms rules-only, ~2.9s with the LLM).
  No k6/Locust run, no connection-pool exhaustion test, no memory-leak soak. The
  10-way booking race is the only concurrency test.
- **Failover and backup/restore.** Single Postgres node. The doc's 3-node HA,
  WAL archiving, hourly/daily backups and 90-day retention are neither built nor
  tested. `docker-compose down -v` loses everything.
- **Kubernetes, WAF, network zones, Azure AD / OIDC / MFA, Vault, TLS 1.3.**
  None implemented. Compose gives an `internal: true` data network as a nod to
  the zone split; everything else in sections 10-11 is production work.
- **Authentication and authorisation.** Every endpoint is open. The `operator`
  table exists with a role column; nothing reads it. Anyone who can reach the
  port can book, cancel, and read the doctor catalogue.
- **Prometheus/Grafana/Loki.** `/metrics` is Prometheus-formatted and logs are
  JSON, but nothing scrapes or ships them; no dashboards or alerts.
- **Multi-turn conversation.** Each request is independent. `session_id` is
  recorded but no state is carried between turns, so "actually, make it Tuesday"
  is not understood.
- **Cancellation/reschedule by a patient**, waitlists, reminders, calendar sync.

### Tested only shallowly

- **The LLM path.** Mocked exhaustively; run live only for the handful of
  scenarios in §3-G. There is no regression suite over real model outputs, so a
  model or prompt change could degrade quality silently. Prompt-output
  regression tests are the highest-value next investment.
- **Prompt injection.** 8 known patterns. A pattern list is a floor, not a
  ceiling — novel phrasings will get through the *detector*. The real defence is
  structural (deterministic ranking, closed vocabularies, output grounding), and
  that is what §3-G actually demonstrates.
- **Emergency triage.** 8 categories from common presentations. Not
  clinically reviewed, and **the false-negative rate is unknown** — an emergency
  phrased unusually gets treated as routine. This needs sign-off from a
  clinician before it goes anywhere near a patient, and should fail safe by
  escalating more, not less.
- **Japanese.** Tested throughout, but by a non-native reading of the dataset's
  vocabulary. Keigo, dialect, and colloquial symptom descriptions are unverified.
- **Only 15 doctors, 4 hospitals, 11 specialties.** Relaxation logic that looks
  sensible here will behave differently at 10,000 doctors, where the top-20
  retrieval actually binds and an ANN index becomes necessary.

### Known weaknesses of the implementation itself

- **Embeddings are lexical, not semantic.** The hash-based embedder matches
  character overlap, so "my tummy is upset" is not close to "Gastroenterology"
  in vector space — the keyword rules are doing the real work. This is a
  deliberate trade for a dependency-free `docker-compose up`. `embeddings.py` is
  the seam: swap `embed()` for a real model and re-ingest.
- **No ANN index** on the vector column. Exact search is correct and fast at 15
  rows; add IVFFlat/HNSW before ~10k.
- **Rate limiting is per-instance fixed-window** and fails open. A burst at a
  window boundary can pass 2× the limit.
- **The PII salt defaults to a hard-coded value.** Fine for a POC, and it must
  come from Vault before production or the hashes are guessable.
- **`token_cost` uses hard-coded prices.** Wrong prices make a dashboard wrong,
  never a recommendation.
