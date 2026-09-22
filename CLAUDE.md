# CLAUDE.md

## What this repository is

A **design-stage** repository for a Healthcare Doctor Recommendation AI Platform (an AI call-center agent that takes patient inquiries by voice/SMS/web chat and recommends doctors and appointment slots). There is **no source code, build system, package manager, test suite, or git repo here yet** — only specification artifacts. Do not invent build/test commands; there are none to run.

Three files, all at the repo root:

- `Healthcare_Doctor_AI_System_Design.md` — the authoritative written spec (16 sections: business problem, success criteria, agent architecture, DB schema, network/security, ops).
- `Sample_Desgin.png` — a large Japanese-language design poster covering the same system in more visual detail (ER diagram, network zones, workflow, tech stack table). Note the filename typo ("Desgin"); keep it unless asked to rename.
- `sample_doctors.json` — the seed dataset: 4 hospitals, 15 doctors, 3 specialty aliases.

## Domain model (from `sample_doctors.json`)

Top-level keys: `hospitals`, `aliases`, `doctors`.

- **hospital**: `name`, `region`, `rating`, `address`, `internal_number` (e.g. `H-TOKYO-001`).
- **doctor**: `name`, `gender`, `age`, `expertise`, `hospital` (joined by *name string*, not id), `region`, `language` (comma-separated string, not an array), `internal_number` (e.g. `D-1001`), `rating` (0–5), `score` (0–100), `personality`, `available_slots` (array of `"YYYY-MM-DD HH:MM"` strings), `register_date`, optional `keywords`.
- **aliases**: `{keyword, alias}` pairs mapping canonical specialty → colloquial term (`Otolaryngology → ENT`, `Pediatrics → Pedia`, `Cardiology → Heart doctor`). Alias resolution is a required step in the search path — patients say "ENT", the DB stores "Otolaryngology".

Shape gotchas if you write an ingestion step: doctors reference hospitals by name only; `language` needs splitting on `", "`; `available_slots` is denormalized and the spec expects it exploded into a `doctor_schedule` table; `keywords` is present on only one doctor; no `leave_date` appears in the sample though the schema defines it.

## Architecture the spec commits to

Five sequential agents: **Intake → Medical Classification → Search → Ranking → Response**.

Retrieval is hybrid: structured SQL + pgvector similarity, then business re-ranking. **Retrieve top-20, return top-3.**

Ranking weights (must sum to 100%) — specialty 35, availability 20, region 15, language 10, doctor rating 10, hospital rating 5, personality 5.

Core tables: `hospital`, `doctor`, `doctor_schedule`, `keyword_alias`, `reservation`, `contact_log`, `audit_log`, `operator`. Patient identity is always hashed (`patient_id_hash`, `phone_hash`) — never store raw patient identifiers.

Performance targets that constrain design choices: search < 3s, end-to-end < 10s, 99.9% availability.

## Known spec conflicts

The markdown and the PNG disagree on the stack. Surface the conflict rather than silently picking one:

| | `Healthcare_Doctor_AI_System_Design.md` | `Sample_Desgin.png` |
|---|---|---|
| LLM | Claude Haiku 4.5 | OpenAI / Azure OpenAI |
| Framework | LangChain | Python (FastAPI) + gRPC |
| STT | unspecified | Whisper / Azure Speech |

Agreed on both: Python, PostgreSQL + pgvector, Redis, Docker (POC) → Kubernetes (production), Prometheus/Grafana/Loki, Twilio for voice/SMS, on-premise deployment target.

## Working here

- Both spec documents describe the *same* system. When changing one, check whether the other needs the matching edit — especially schema fields and ranking weights.
- The PNG is the only source for several details (operator table columns, `contact_log` fields like `accuracy_flag`/`csat`/`token_cost`, the 4-phase roadmap). Read it when the markdown seems thin.
- If asked to build the system, expect to create the project scaffold from scratch; confirm the LLM/framework choice against the conflict table above before starting.
