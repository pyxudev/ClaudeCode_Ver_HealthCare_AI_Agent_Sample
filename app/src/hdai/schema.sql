-- Idempotent schema (design doc section 9).
-- Applied by the API at startup under a pg advisory lock, so N replicas can
-- boot concurrently without racing. {EMBEDDING_DIM} is substituted in db.py.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- hospital
CREATE TABLE IF NOT EXISTS hospital (
    hospital_id      SERIAL PRIMARY KEY,
    internal_number  TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    region           TEXT NOT NULL,
    rating           NUMERIC(2,1) NOT NULL DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    address          TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS hospital_name_uidx ON hospital (lower(name));
CREATE INDEX IF NOT EXISTS hospital_region_idx ON hospital (region);

-- ------------------------------------------------------------------ doctor
CREATE TABLE IF NOT EXISTS doctor (
    doctor_id        SERIAL PRIMARY KEY,
    internal_number  TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    gender           TEXT CHECK (gender IN ('male', 'female', 'other')),
    age              INTEGER CHECK (age BETWEEN 18 AND 100),
    expertise        TEXT NOT NULL,
    hospital_id      INTEGER NOT NULL REFERENCES hospital (hospital_id) ON DELETE RESTRICT,
    region           TEXT NOT NULL,
    languages        TEXT[] NOT NULL DEFAULT '{}',
    rating           NUMERIC(2,1) NOT NULL DEFAULT 0 CHECK (rating >= 0 AND rating <= 5),
    score            INTEGER NOT NULL DEFAULT 0 CHECK (score BETWEEN 0 AND 100),
    personality      TEXT,
    keywords         TEXT[] NOT NULL DEFAULT '{}',
    register_date    DATE,
    leave_date       DATE,
    profile_text     TEXT,
    embedding        vector({EMBEDDING_DIM}),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT doctor_dates_ordered CHECK (leave_date IS NULL OR register_date IS NULL OR leave_date >= register_date)
);
CREATE INDEX IF NOT EXISTS doctor_expertise_idx ON doctor (expertise);
CREATE INDEX IF NOT EXISTS doctor_region_idx ON doctor (region);
CREATE INDEX IF NOT EXISTS doctor_languages_idx ON doctor USING GIN (languages);
-- Partial index: every search path filters out doctors who have left.
CREATE INDEX IF NOT EXISTS doctor_active_idx ON doctor (expertise, region) WHERE leave_date IS NULL;

-- --------------------------------------------------------- doctor_schedule
CREATE TABLE IF NOT EXISTS doctor_schedule (
    schedule_id      SERIAL PRIMARY KEY,
    doctor_id        INTEGER NOT NULL REFERENCES doctor (doctor_id) ON DELETE CASCADE,
    available_time   TIMESTAMPTZ NOT NULL,
    status           TEXT NOT NULL DEFAULT 'available'
                     CHECK (status IN ('available', 'held', 'booked', 'cancelled')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT doctor_schedule_slot_uniq UNIQUE (doctor_id, available_time)
);
CREATE INDEX IF NOT EXISTS schedule_open_idx
    ON doctor_schedule (doctor_id, available_time) WHERE status = 'available';
CREATE INDEX IF NOT EXISTS schedule_time_idx ON doctor_schedule (available_time);

-- ----------------------------------------------------------- keyword_alias
CREATE TABLE IF NOT EXISTS keyword_alias (
    id               SERIAL PRIMARY KEY,
    keyword          TEXT NOT NULL,   -- canonical, e.g. Otolaryngology
    alias            TEXT NOT NULL,   -- what patients say, e.g. ENT
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS keyword_alias_uidx ON keyword_alias (lower(alias));

-- ---------------------------------------------------------------- operator
CREATE TABLE IF NOT EXISTS operator (
    operator_id      SERIAL PRIMARY KEY,
    employee_id      TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    gender           TEXT,
    role             TEXT NOT NULL DEFAULT 'OPERATOR'
                     CHECK (role IN ('ADMIN', 'MANAGER', 'OPERATOR')),
    phone_number     TEXT,
    join_date        DATE,
    leave_date       DATE
);

-- ------------------------------------------------------------- reservation
CREATE TABLE IF NOT EXISTS reservation (
    reservation_id   SERIAL PRIMARY KEY,
    patient_id_hash  TEXT,
    doctor_id        INTEGER NOT NULL REFERENCES doctor (doctor_id) ON DELETE RESTRICT,
    schedule_id      INTEGER NOT NULL REFERENCES doctor_schedule (schedule_id) ON DELETE RESTRICT,
    slot_time        TIMESTAMPTZ NOT NULL,
    status           TEXT NOT NULL DEFAULT 'confirmed'
                     CHECK (status IN ('confirmed', 'cancelled', 'completed', 'no_show')),
    idempotency_key  TEXT UNIQUE,
    session_id       TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS reservation_patient_idx ON reservation (patient_id_hash);

-- Last line of defence against a double booking, even if the application-level
-- compare-and-set is ever bypassed. PARTIAL, not a plain UNIQUE column: a
-- cancelled reservation must not stop the slot being booked by someone else.
ALTER TABLE reservation DROP CONSTRAINT IF EXISTS reservation_schedule_id_key;
CREATE UNIQUE INDEX IF NOT EXISTS reservation_active_slot_uidx
    ON reservation (schedule_id) WHERE status IN ('confirmed', 'completed');

-- ------------------------------------------------------------- contact_log
CREATE TABLE IF NOT EXISTS contact_log (
    contact_id       BIGSERIAL PRIMARY KEY,
    request_id       TEXT NOT NULL UNIQUE,
    session_id       TEXT,
    channel          TEXT NOT NULL DEFAULT 'web',
    start_time       TIMESTAMPTZ NOT NULL DEFAULT now(),
    end_time         TIMESTAMPTZ,
    phone_hash       TEXT,                 -- never the raw number
    keywords         JSONB NOT NULL DEFAULT '{}'::jsonb,
    doctor_recommended INTEGER REFERENCES doctor (doctor_id) ON DELETE SET NULL,
    accuracy_flag    BOOLEAN,
    csat             NUMERIC(2,1) CHECK (csat IS NULL OR (csat >= 1 AND csat <= 5)),
    token_cost       NUMERIC(12,6) NOT NULL DEFAULT 0,
    latency_ms       INTEGER,
    outcome          TEXT,
    llm_used         BOOLEAN NOT NULL DEFAULT false,
    injection_flag   BOOLEAN NOT NULL DEFAULT false,
    emergency_flag   BOOLEAN NOT NULL DEFAULT false,
    degraded         TEXT[] NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS contact_log_start_idx ON contact_log (start_time DESC);

-- --------------------------------------------------------------- audit_log
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id         BIGSERIAL PRIMARY KEY,
    actor            TEXT NOT NULL DEFAULT 'system',
    action           TEXT NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE', 'READ', 'REJECT')),
    table_name       TEXT NOT NULL,
    record_id        TEXT,
    value_before     JSONB,
    value_after      JSONB,
    request_id       TEXT,
    action_time      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_log_time_idx ON audit_log (action_time DESC);
CREATE INDEX IF NOT EXISTS audit_log_table_idx ON audit_log (table_name, record_id);

-- -------------------------------------------------------------- ingest_log
-- Rejected source records are kept, not silently dropped: a doctor missing
-- from search because of a bad row is invisible otherwise.
CREATE TABLE IF NOT EXISTS ingest_log (
    ingest_id        BIGSERIAL PRIMARY KEY,
    source           TEXT NOT NULL,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    hospitals_upserted INTEGER NOT NULL DEFAULT 0,
    doctors_upserted   INTEGER NOT NULL DEFAULT 0,
    slots_upserted     INTEGER NOT NULL DEFAULT 0,
    aliases_upserted   INTEGER NOT NULL DEFAULT 0,
    rejected         JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes            TEXT
);
