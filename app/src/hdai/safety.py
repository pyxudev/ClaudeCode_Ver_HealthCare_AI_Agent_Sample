"""Safety layer: the things that go wrong when an LLM talks to patients.

Covers, in the order a request hits them:
  1. sanitise()          - control chars, length bombs, unicode tricks
  2. detect_injection()  - prompt-injection attempts in patient free text
  3. triage()            - medical red flags that must NOT become a booking
  4. hash_identifier()   - patient id / phone never stored in the clear
  5. ground_response()   - the generated reply may only name real candidates
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# 1. Input sanitisation
# --------------------------------------------------------------------------

# Everything in Unicode category Cc/Cf except tab/newline. Cf catches the
# bidi-override and zero-width characters used to hide injected instructions.
_CONTROL_RE = re.compile(
    "["
    "\u0000-\u0008\u000b-\u001f\u007f-\u009f"  # C0/C1 controls (keeps \t and \n)
    "\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff"  # zero-width + bidi overrides
    "]"
)
_WHITESPACE_RE = re.compile(r"[ \t]{3,}")
_NEWLINES_RE = re.compile(r"\n{4,}")


def sanitise(text: str | None, max_chars: int = 2000) -> str:
    """Normalise untrusted text. Never raises; worst case returns ""."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    # NFKC folds full-width latin (Ｉｇｎｏｒｅ) onto ASCII so the injection
    # patterns below cannot be bypassed with full-width characters.
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub("  ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " …[truncated]"
    return text


# --------------------------------------------------------------------------
# 2. Prompt injection
# --------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("override", re.compile(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,30}\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}\b(instruction|prompt|rule|direction)", re.I)),
    ("override_ja", re.compile(r"(これまでの|以前の|上記の)?(指示|命令|ルール|プロンプト)(を|は)?(無視|忘れ|破棄)")),
    ("system_prompt", re.compile(r"\b(system\s*prompt|developer\s*message|initial\s*instructions?|your\s+instructions)\b", re.I)),
    ("role_play", re.compile(r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be|from\s+now\s+on\s+you)\b", re.I)),
    ("role_marker", re.compile(r"(^|\n)\s*(system|assistant|human|user)\s*:", re.I)),
    ("tag_injection", re.compile(r"</?\s*(system|instructions?|patient_message|admin)\s*>", re.I)),
    ("exfiltration", re.compile(r"\b(reveal|print|repeat|output|show)\b[^.\n]{0,25}\b(prompt|instructions?|api[\s_-]?key|secret|token)\b", re.I)),
    # No [^.] here: "recommend Dr. Sato first" contains a full stop.
    ("ranking_tamper", re.compile(r"\b(rank|recommend|score|return)\b[^\n]{0,40}\b(first|top|only|always|highest)\b", re.I)),
    ("sql", re.compile(r"(\bunion\s+select\b|\bdrop\s+table\b|\bor\s+1\s*=\s*1\b|;--)", re.I)),
)


@dataclass(frozen=True)
class InjectionVerdict:
    detected: bool
    categories: tuple[str, ...] = ()

    @property
    def severity(self) -> str:
        if not self.detected:
            return "none"
        hard = {"exfiltration", "sql", "system_prompt", "tag_injection"}
        return "high" if hard & set(self.categories) else "medium"


def detect_injection(text: str) -> InjectionVerdict:
    hits = tuple(name for name, pattern in _INJECTION_PATTERNS if pattern.search(text or ""))
    return InjectionVerdict(detected=bool(hits), categories=hits)


def wrap_untrusted(text: str) -> str:
    """Fence patient text so the model can tell data from instructions.

    Any literal fence in the input is neutralised first, otherwise the patient
    could close the fence and continue as if they were the system.
    """
    cleaned = re.sub(r"</?patient_message>", "", text or "", flags=re.I)
    return f"<patient_message>\n{cleaned}\n</patient_message>"


# --------------------------------------------------------------------------
# 3. Medical triage red flags
#
# This system recommends a doctor; it must never triage-down an emergency or
# imply a diagnosis. Anything matching here short-circuits the whole pipeline.
# --------------------------------------------------------------------------

_EMERGENCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cardiac", re.compile(r"(chest\s+(pain|pressure|tightness)|crushing\s+chest|heart\s+attack|胸(の)?(痛|が痛)|心筋梗塞)", re.I)),
    ("stroke", re.compile(r"(stroke|(face|facial)[^\n]{0,15}droop|droop[^\n]{0,15}face|slurred\s+speech|speech[^\n]{0,12}slurred|sudden\s+(numbness|weakness)\s+on\s+one\s+side|脳卒中|ろれつ|呂律|半身(の)?(麻痺|しびれ))", re.I)),
    ("breathing", re.compile(r"(can'?t\s+breathe|cannot\s+breathe|difficulty\s+breathing|struggling\s+to\s+breathe|choking|anaphyla|息が(でき|出来)ない|呼吸困難|窒息)", re.I)),
    ("consciousness", re.compile(r"(unconscious|unresponsive|passed\s+out|fainted\s+and|seizure|convuls|意識(が)?(ない|無い|不明|ありません|あり ません)|反応がな|けいれん|痙攣)", re.I)),
    ("haemorrhage", re.compile(r"(severe\s+bleeding|bleeding\s+heavily|won'?t\s+stop\s+bleeding|coughing\s+up\s+blood|vomiting\s+blood|大量(の)?出血|吐血|血が止まらない)", re.I)),
    ("self_harm", re.compile(r"(suicid|kill\s+myself|end\s+my\s+life|self\s*-?\s*harm|自殺|死にたい)", re.I)),
    ("obstetric", re.compile(r"(in\s+labou?r\s+now|water\s+broke|heavy\s+bleeding\s+pregnan|破水|陣痛)", re.I)),
    ("trauma", re.compile(r"(hit\s+by\s+a\s+car|serious\s+accident|deep\s+wound|broken\s+bone\s+through|大事故|交通事故)", re.I)),
)

EMERGENCY_MESSAGE_EN = (
    "This may be a medical emergency. Please stop using this service and call "
    "emergency services now (119 in Japan) or go to the nearest emergency room. "
    "I am not able to book an appointment for symptoms like these."
)
EMERGENCY_MESSAGE_JA = (
    "緊急性の高い症状の可能性があります。本サービスのご利用を中止し、"
    "直ちに119番通報するか、最寄りの救急外来を受診してください。"
    "これらの症状について予約をお取りすることはできません。"
)


@dataclass(frozen=True)
class TriageVerdict:
    emergency: bool
    categories: tuple[str, ...] = ()

    def message(self, japanese: bool = False) -> str:
        return EMERGENCY_MESSAGE_JA if japanese else EMERGENCY_MESSAGE_EN


def triage(text: str) -> TriageVerdict:
    hits = tuple(name for name, pattern in _EMERGENCY_PATTERNS if pattern.search(text or ""))
    return TriageVerdict(emergency=bool(hits), categories=hits)


_JA_RE = re.compile(r"[぀-ヿ一-鿿]")


def looks_japanese(text: str) -> bool:
    if not text:
        return False
    return len(_JA_RE.findall(text)) >= max(2, len(text) // 20)


# --------------------------------------------------------------------------
# 4. PII
# --------------------------------------------------------------------------

_SALT_ENV = "HDAI_PII_SALT"
# A process-stable salt. In production this comes from Vault / Key Vault
# (design doc section 11); a random per-process salt would break the ability to
# correlate a returning caller within a deployment, so it is explicit config.
_DEFAULT_SALT = "hdai-local-poc-salt"


def _salt() -> bytes:
    return os.environ.get(_SALT_ENV, _DEFAULT_SALT).encode("utf-8")


def hash_identifier(value: str | None) -> str | None:
    """Keyed hash of a phone number / patient id. One-way, correlatable."""
    if value is None:
        return None
    normalised = re.sub(r"[\s\-()]", "", str(value)).strip().lower()
    if not normalised:
        return None
    return hmac.new(_salt(), normalised.encode("utf-8"), hashlib.sha256).hexdigest()


_PII_SCRUBBERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?\d[\d\-\s()]{7,}\d)(?!\d)"), "[phone]"),
    (re.compile(r"\b\d{12}\b"), "[id]"),  # My Number style
)


def scrub_pii(text: str) -> str:
    """Remove obvious direct identifiers before text is logged or embedded."""
    out = text or ""
    for pattern, repl in _PII_SCRUBBERS:
        out = pattern.sub(repl, out)
    return out


# --------------------------------------------------------------------------
# 5. Response grounding
# --------------------------------------------------------------------------


@dataclass
class GroundingResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


_DOCTOR_MENTION_RE = re.compile(r"\bDr\.?\s+([A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?)", re.U)
_SLOT_MENTION_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})[ T](\d{2}:\d{2})")
_PRONOUN_RE = re.compile(r"\b(he|him|his|she|her|hers)\b", re.I)
_DIAGNOSIS_RE = re.compile(
    r"\b(you\s+(have|are\s+suffering\s+from)|this\s+is\s+(definitely|certainly)|"
    r"diagnos(is|ed)\s+(is|as)|you\s+(definitely\s+)?need\s+surgery|"
    r"take\s+\d+\s*mg|prescrib)", re.I,
)


def ground_response(
    text: str,
    allowed_doctor_names: Iterable[str],
    allowed_slots: Iterable[str],
    *,
    forbid_gendered_pronouns: bool = True,
    require_verbatim: Iterable[str] = (),
) -> GroundingResult:
    """Reject a generated reply that invents doctors, slots, or a diagnosis.

    An LLM that hallucinates a doctor here sends a patient to a clinic that
    does not exist, so the check fails closed: anything unrecognised is a fail
    and the caller falls back to the deterministic template.
    """
    reasons: list[str] = []
    allowed_norm = {_norm_name(n) for n in allowed_doctor_names}
    allowed_slot_set = {s.strip()[:16].replace("T", " ") for s in allowed_slots}

    for match in _DOCTOR_MENTION_RE.finditer(text or ""):
        mentioned = _norm_name(match.group(1))
        if not mentioned:
            continue
        if not any(mentioned in a or a in mentioned for a in allowed_norm):
            reasons.append(f"unknown doctor mentioned: {match.group(0)!r}")

    for match in _SLOT_MENTION_RE.finditer(text or ""):
        slot = f"{match.group(1)} {match.group(2)}"
        if slot not in allowed_slot_set:
            reasons.append(f"unknown slot mentioned: {slot!r}")

    if _DIAGNOSIS_RE.search(text or ""):
        reasons.append("reply reads as a diagnosis or prescription")

    # The facts handed to the model carry no gender, so any gendered pronoun
    # is invented. Observed in practice: a female doctor read back as "he".
    if forbid_gendered_pronouns and _PRONOUN_RE.search(text or ""):
        reasons.append("reply asserts a gender that was never provided")

    # Catches transliteration: a Japanese reply that renders "Dr. Yuki
    # Nakamura" as 中村由紀 has invented kanji the hospital may not use, and
    # the regex above cannot see it because the name is no longer latin.
    lowered = (text or "").lower()
    for name in require_verbatim:
        if name and name.lower() not in lowered:
            reasons.append(f"doctor name not written verbatim: {name!r}")

    return GroundingResult(ok=not reasons, reasons=reasons)


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z ]", "", (name or "").lower().replace("dr.", "").replace("dr ", "")).strip()


def verify_twilio_signature(auth_token: str, url: str, params: dict[str, str], signature: str) -> bool:
    """Twilio webhook authenticity (design doc section 11).

    Returns False when no auth token is configured - an unconfigured webhook is
    an open webhook, so the route must refuse to trust it.
    """
    if not auth_token or not signature:
        return False
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode("utf-8"), hashlib.sha1).digest()
    import base64

    expected = base64.b64encode(digest).decode()
    return hmac.compare_digest(expected, signature)


def any_of(text: str, needles: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(n.lower() in lowered for n in needles)
