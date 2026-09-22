"""PostgreSQL access: pool lifecycle, migrations, audit helper.

Design decisions worth knowing:
  * Migrations run at API startup behind a pg advisory lock, so the two API
    replicas the design doc calls for can boot simultaneously.
  * A statement_timeout is set on every connection. A runaway query would
    otherwise blow the <3s search / <10s end-to-end targets and pin a worker.
  * The app never interpolates user input into SQL - psycopg parameters only.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .config import Settings

log = logging.getLogger("hdai.db")

_SCHEMA_PATH = pathlib.Path(__file__).with_name("schema.sql")
# Arbitrary but fixed: all replicas must agree on the same lock key.
_MIGRATION_LOCK_KEY = 8_241_990_11


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: AsyncConnectionPool | None = None

    # -------------------------------------------------------------- lifecycle
    async def connect(self, retries: int = 30, delay: float = 1.0) -> None:
        """Open the pool, retrying while Postgres finishes starting.

        compose already gates on the postgres healthcheck, but a healthy
        postgres can still refuse connections for a beat during init, and an
        API that dies on boot never comes back on its own.
        """
        pool = AsyncConnectionPool(
            conninfo=self._settings.database_url,
            min_size=self._settings.db_pool_min,
            max_size=self._settings.db_pool_max,
            timeout=self._settings.db_connect_timeout_s,
            max_idle=300.0,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
            configure=self._configure_connection,
        )
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                await pool.open(wait=True, timeout=self._settings.db_connect_timeout_s)
                self._pool = pool
                log.info("database pool open", extra={"attempt": attempt})
                return
            except Exception as exc:  # noqa: BLE001 - retry on anything
                last_error = exc
                log.warning("database not ready", extra={"attempt": attempt, "error": str(exc)})
                await asyncio.sleep(min(delay * attempt, 5.0))
        raise RuntimeError(f"could not connect to postgres after {retries} attempts: {last_error}")

    async def _configure_connection(self, conn: psycopg.AsyncConnection) -> None:
        await conn.execute(
            f"SET statement_timeout = {int(self._settings.db_statement_timeout_ms)}"
        )
        await conn.execute("SET idle_in_transaction_session_timeout = 30000")
        # UTC everywhere; slot times are compared across process boundaries.
        await conn.execute("SET TIME ZONE 'UTC'")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def is_open(self) -> bool:
        return self._pool is not None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection]:
        if self._pool is None:
            raise RuntimeError("database pool is not open")
        async with self._pool.connection() as conn:
            yield conn

    # ------------------------------------------------------------- migrations
    async def migrate(self) -> None:
        ddl = _SCHEMA_PATH.read_text(encoding="utf-8").replace(
            "{EMBEDDING_DIM}", str(self._settings.embedding_dim)
        )
        async with self.connection() as conn:
            await conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
            try:
                # Long DDL can exceed the per-statement timeout on a cold volume.
                await conn.execute("SET statement_timeout = 60000")
                await conn.execute(ddl)
                await self._verify_embedding_dim(conn)
                log.info("schema applied")
            finally:
                await conn.execute(
                    f"SET statement_timeout = {int(self._settings.db_statement_timeout_ms)}"
                )
                await conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))

    async def _verify_embedding_dim(self, conn: psycopg.AsyncConnection) -> None:
        """Catch a dimension change against an existing volume.

        CREATE TABLE IF NOT EXISTS silently keeps the old vector(N); every
        insert would then fail at runtime with a confusing error.
        """
        cur = await conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'doctor'::regclass AND attname = 'embedding'"
        )
        row = await cur.fetchone()
        if not row or row["atttypmod"] in (None, -1):
            return
        actual = int(row["atttypmod"])
        if actual != self._settings.embedding_dim:
            raise RuntimeError(
                f"existing doctor.embedding is vector({actual}) but HDAI_EMBEDDING_DIM="
                f"{self._settings.embedding_dim}. Drop the pgdata volume or restore the old value."
            )

    # ------------------------------------------------------------- shortcuts
    async def fetch_all(self, sql: str, params: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        async with self.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchall()

    async def fetch_one(self, sql: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        async with self.connection() as conn:
            cur = await conn.execute(sql, params)
            return await cur.fetchone()

    async def execute(self, sql: str, params: Sequence[Any] | None = None) -> int:
        async with self.connection() as conn:
            cur = await conn.execute(sql, params)
            return cur.rowcount

    async def ping(self) -> bool:
        try:
            async with self.connection() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("db ping failed", extra={"error": str(exc)})
            return False


async def write_audit(
    conn: psycopg.AsyncConnection,
    *,
    action: str,
    table_name: str,
    record_id: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    actor: str = "ai-agent",
    request_id: str | None = None,
) -> None:
    """Append-only audit trail (design doc section 11).

    Audit failure must never fail the business transaction it describes, so
    this swallows its own errors after logging them loudly.
    """
    try:
        await conn.execute(
            """
            INSERT INTO audit_log (actor, action, table_name, record_id,
                                   value_before, value_after, request_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                actor,
                action,
                table_name,
                record_id,
                Jsonb(before) if before is not None else None,
                Jsonb(after) if after is not None else None,
                request_id,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        log.error("audit write failed", extra={"error": str(exc), "table": table_name})
