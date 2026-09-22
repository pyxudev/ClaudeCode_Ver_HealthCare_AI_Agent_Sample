"""Request/response contracts and the internal objects passed between agents."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000, description="Patient's own words")
    channel: Literal["voice", "sms", "web"] = "web"
    session_id: Optional[str] = Field(default=None, max_length=64)
    caller_id: Optional[str] = Field(default=None, max_length=64, description="Phone/patient id; hashed before storage")
    locale: Optional[Literal["en", "ja"]] = None
    # Structured hints from an IVR or web form; merged with what intake extracts.
    region: Optional[str] = Field(default=None, max_length=64)
    language: Optional[str] = Field(default=None, max_length=64)
    preferred_date: Optional[str] = Field(default=None, max_length=32)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class SlotOut(BaseModel):
    schedule_id: int
    available_time: datetime
    status: str


class DoctorOut(BaseModel):
    doctor_id: int
    internal_number: str
    name: str
    gender: Optional[str] = None
    age: Optional[int] = None
    expertise: str
    region: str
    languages: list[str] = []
    personality: Optional[str] = None
    rating: float = 0.0
    score: int = 0
    hospital_name: str
    hospital_rating: float = 0.0
    hospital_address: Optional[str] = None


class ScoreBreakdown(BaseModel):
    """Per-dimension contribution, so every recommendation is explainable."""

    specialty: float = 0.0
    availability: float = 0.0
    region: float = 0.0
    language: float = 0.0
    doctor_rating: float = 0.0
    hospital_rating: float = 0.0
    personality: float = 0.0

    def total(self) -> float:
        return round(sum(self.model_dump().values()), 6)


class Recommendation(BaseModel):
    rank: int
    doctor: DoctorOut
    score: float
    breakdown: ScoreBreakdown
    vector_similarity: float = 0.0
    next_slots: list[SlotOut] = []
    reasons: list[str] = []


class PatientRequirements(BaseModel):
    """Output of the Intake agent; the only thing downstream agents trust."""

    model_config = ConfigDict(extra="ignore")

    symptom_text: str = ""
    specialty: Optional[str] = None
    region: Optional[str] = None
    language: Optional[str] = None
    personality: Optional[str] = None
    doctor_gender: Optional[str] = None
    preferred_date: Optional[str] = None  # ISO date
    preferred_time_of_day: Optional[Literal["morning", "afternoon", "evening"]] = None
    urgency: Literal["routine", "soon", "urgent"] = "routine"
    patient_age_band: Optional[Literal["infant", "child", "adult", "senior"]] = None
    missing_fields: list[str] = []
    notes: list[str] = []


class ClassificationResult(BaseModel):
    specialty: str
    confidence: float = 0.0
    source: Literal["llm", "rules", "explicit", "fallback"] = "rules"
    evidence: list[str] = []
    alternatives: list[str] = []


class RecommendResponse(BaseModel):
    request_id: str
    status: Literal["ok", "emergency", "no_match", "clarification_needed", "degraded"]
    message: str
    recommendations: list[Recommendation] = []
    confidence: float = 0.0
    understood: PatientRequirements | None = None
    classification: ClassificationResult | None = None
    follow_up_questions: list[str] = []
    disclaimers: list[str] = []
    degraded_components: list[str] = []
    latency_ms: int = 0
    llm_used: bool = False


class ReservationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schedule_id: int = Field(gt=0)
    caller_id: Optional[str] = Field(default=None, max_length=64)
    session_id: Optional[str] = Field(default=None, max_length=64)
    idempotency_key: Optional[str] = Field(default=None, max_length=80)


class ReservationResponse(BaseModel):
    reservation_id: int
    status: str
    doctor_id: int
    doctor_name: str
    hospital_name: str
    slot_time: datetime
    idempotent_replay: bool = False


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, Any] = {}
    version: str = "1.0.0"
