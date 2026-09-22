"""Runtime configuration.

Every setting has a default that works in the compose stack, so a missing .env
can never be the reason the container fails to boot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HDAI_", extra="ignore")

    # --- Infrastructure ---
    database_url: str = "postgresql://hdai:hdai_local_dev_pw@postgres:5432/hdai"
    redis_url: str = "redis://redis:6379/0"
    data_file: str = "/data/sample_doctors.json"
    log_level: str = "INFO"

    db_pool_min: int = 1
    db_pool_max: int = 8
    db_connect_timeout_s: float = 10.0
    db_statement_timeout_ms: int = 5_000  # success criterion: search < 3s

    # --- LLM ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_enabled: Literal["auto", "on", "off"] = "auto"
    llm_base_url: str = "https://api.anthropic.com"
    llm_timeout_s: float = 8.0
    llm_max_retries: int = 2
    llm_max_output_tokens: int = 1024
    # Circuit breaker: after N consecutive failures stop calling for M seconds.
    llm_breaker_threshold: int = 3
    llm_breaker_cooldown_s: float = 30.0

    # --- Retrieval / ranking (design doc section 8) ---
    retrieve_top_k: int = 20
    return_top_n: int = 3
    embedding_dim: int = 256
    # Minimum final score for a candidate to be offered at all.
    min_recommendation_score: float = 0.25

    # --- Safety / limits ---
    max_input_chars: int = 2_000
    rate_limit_per_minute: int = 60
    request_timeout_s: float = 10.0  # success criterion: end-to-end < 10s

    # --- Data ingestion ---
    demo_shift_past_slots: bool = True
    ingest_on_startup: bool = True

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.upper()
        return v if v in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"

    @property
    def llm_active(self) -> bool:
        """Whether the LLM path should be attempted at all."""
        if self.llm_enabled == "off":
            return False
        if self.llm_enabled == "on":
            return True
        return bool(self.anthropic_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
