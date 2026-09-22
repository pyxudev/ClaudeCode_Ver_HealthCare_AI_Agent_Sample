"""Medical / geographic taxonomy used by classification and ranking.

Deliberately data-driven and deterministic. This is the safety net under the
LLM: whatever the model returns is coerced into these vocabularies, and if it
cannot be coerced the rule-based path here produces the answer instead.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# Canonical specialties. Anything outside this set is rejected as a
# hallucination (see coerce_specialty).
SPECIALTIES: tuple[str, ...] = (
    "Cardiology",
    "Dermatology",
    "Gastroenterology",
    "General Practice",
    "Neurology",
    "Obstetrics and Gynecology",
    "Ophthalmology",
    "Orthopedics",
    "Otolaryngology",
    "Pediatrics",
    "Psychiatry",
)

FALLBACK_SPECIALTY = "General Practice"

# Colloquial -> canonical. The three from sample_doctors.json plus the ones a
# call centre actually hears. The DB keyword_alias table is loaded from the
# dataset at ingest time and takes precedence over this static map.
STATIC_ALIASES: dict[str, str] = {
    "ent": "Otolaryngology",
    "ear nose and throat": "Otolaryngology",
    "ear nose throat": "Otolaryngology",
    "耳鼻科": "Otolaryngology",
    "耳鼻咽喉科": "Otolaryngology",
    "pedia": "Pediatrics",
    "peds": "Pediatrics",
    "paediatrics": "Pediatrics",
    "children's doctor": "Pediatrics",
    "小児科": "Pediatrics",
    "heart doctor": "Cardiology",
    "cardiologist": "Cardiology",
    "循環器科": "Cardiology",
    "心臓": "Cardiology",
    "skin doctor": "Dermatology",
    "dermatologist": "Dermatology",
    "皮膚科": "Dermatology",
    "eye doctor": "Ophthalmology",
    "optometrist": "Ophthalmology",
    "眼科": "Ophthalmology",
    "bone doctor": "Orthopedics",
    "orthopaedics": "Orthopedics",
    "整形外科": "Orthopedics",
    "stomach doctor": "Gastroenterology",
    "gi": "Gastroenterology",
    "消化器科": "Gastroenterology",
    "brain doctor": "Neurology",
    "神経内科": "Neurology",
    "脳神経内科": "Neurology",
    "mental health": "Psychiatry",
    "psychiatrist": "Psychiatry",
    "therapist": "Psychiatry",
    "精神科": "Psychiatry",
    "心療内科": "Psychiatry",
    "obgyn": "Obstetrics and Gynecology",
    "ob/gyn": "Obstetrics and Gynecology",
    "gynecology": "Obstetrics and Gynecology",
    "gynaecology": "Obstetrics and Gynecology",
    "産婦人科": "Obstetrics and Gynecology",
    "family doctor": "General Practice",
    "gp": "General Practice",
    "internal medicine": "General Practice",
    "内科": "General Practice",
    "総合診療": "General Practice",
}

# Symptom keyword -> specialty. Ordered by weight: a match on a more specific
# phrase beats a generic one. Kept explicit rather than clever because a wrong
# specialty is the single most damaging error this system can make.
SYMPTOM_RULES: tuple[tuple[str, str, float], ...] = (
    # Otolaryngology
    ("ear ache", "Otolaryngology", 0.9), ("earache", "Otolaryngology", 0.9),
    ("ear pain", "Otolaryngology", 0.9), ("ear infection", "Otolaryngology", 0.9),
    ("sore throat", "Otolaryngology", 0.8), ("tonsil", "Otolaryngology", 0.85),
    ("sinus", "Otolaryngology", 0.85), ("runny nose", "Otolaryngology", 0.6),
    ("nosebleed", "Otolaryngology", 0.8), ("hoarse", "Otolaryngology", 0.75),
    ("hearing loss", "Otolaryngology", 0.9), ("tinnitus", "Otolaryngology", 0.9),
    ("耳が痛", "Otolaryngology", 0.9), ("のどが痛", "Otolaryngology", 0.8),
    ("喉が痛", "Otolaryngology", 0.8), ("鼻血", "Otolaryngology", 0.8),
    ("難聴", "Otolaryngology", 0.9), ("耳鳴り", "Otolaryngology", 0.9),
    # Pediatrics (age cue, not symptom - handled additionally in classify)
    ("my child", "Pediatrics", 0.8), ("my son", "Pediatrics", 0.8),
    ("my daughter", "Pediatrics", 0.8), ("my baby", "Pediatrics", 0.85),
    ("my toddler", "Pediatrics", 0.85), ("infant", "Pediatrics", 0.8),
    ("子供", "Pediatrics", 0.8), ("子ども", "Pediatrics", 0.8),
    ("赤ちゃん", "Pediatrics", 0.85), ("息子", "Pediatrics", 0.8), ("娘", "Pediatrics", 0.8),
    # Dermatology
    ("rash", "Dermatology", 0.85), ("eczema", "Dermatology", 0.9),
    ("acne", "Dermatology", 0.9), ("itchy skin", "Dermatology", 0.85),
    ("mole", "Dermatology", 0.8), ("hives", "Dermatology", 0.85),
    ("psoriasis", "Dermatology", 0.9), ("skin", "Dermatology", 0.6),
    ("発疹", "Dermatology", 0.85), ("湿疹", "Dermatology", 0.9),
    ("かゆみ", "Dermatology", 0.7), ("にきび", "Dermatology", 0.85),
    # Cardiology (non-emergency presentations only; red flags are triaged out)
    ("palpitation", "Cardiology", 0.9), ("irregular heartbeat", "Cardiology", 0.9),
    ("high blood pressure", "Cardiology", 0.85), ("hypertension", "Cardiology", 0.85),
    ("cholesterol", "Cardiology", 0.7), ("heart murmur", "Cardiology", 0.9),
    ("動悸", "Cardiology", 0.9), ("高血圧", "Cardiology", 0.85), ("不整脈", "Cardiology", 0.9),
    # Orthopedics
    ("back pain", "Orthopedics", 0.8), ("knee pain", "Orthopedics", 0.9),
    ("shoulder pain", "Orthopedics", 0.85), ("sprain", "Orthopedics", 0.85),
    ("fracture", "Orthopedics", 0.85), ("joint pain", "Orthopedics", 0.85),
    ("arthritis", "Orthopedics", 0.8), ("neck pain", "Orthopedics", 0.8),
    ("腰痛", "Orthopedics", 0.85), ("膝が痛", "Orthopedics", 0.9),
    ("肩が痛", "Orthopedics", 0.85), ("骨折", "Orthopedics", 0.85),
    ("関節", "Orthopedics", 0.8),
    # Neurology
    ("migraine", "Neurology", 0.9), ("headache", "Neurology", 0.7),
    ("dizziness", "Neurology", 0.7), ("vertigo", "Neurology", 0.75),
    ("tremor", "Neurology", 0.85), ("memory loss", "Neurology", 0.85),
    ("numbness", "Neurology", 0.7),
    ("片頭痛", "Neurology", 0.9), ("頭痛", "Neurology", 0.7), ("めまい", "Neurology", 0.7),
    # Psychiatry
    ("anxiety", "Psychiatry", 0.9), ("depress", "Psychiatry", 0.9),
    ("panic attack", "Psychiatry", 0.9), ("insomnia", "Psychiatry", 0.8),
    ("can't sleep", "Psychiatry", 0.7), ("stress", "Psychiatry", 0.6),
    ("不安", "Psychiatry", 0.8), ("うつ", "Psychiatry", 0.9),
    ("不眠", "Psychiatry", 0.85), ("眠れない", "Psychiatry", 0.7),
    # Gastroenterology
    ("stomach pain", "Gastroenterology", 0.85), ("stomach ache", "Gastroenterology", 0.85),
    ("abdominal pain", "Gastroenterology", 0.85), ("diarrhea", "Gastroenterology", 0.85),
    ("diarrhoea", "Gastroenterology", 0.85), ("constipation", "Gastroenterology", 0.85),
    ("heartburn", "Gastroenterology", 0.85), ("acid reflux", "Gastroenterology", 0.9),
    ("nausea", "Gastroenterology", 0.6), ("bloating", "Gastroenterology", 0.8),
    ("腹痛", "Gastroenterology", 0.85), ("胃が痛", "Gastroenterology", 0.85),
    ("下痢", "Gastroenterology", 0.85), ("便秘", "Gastroenterology", 0.85),
    ("胸やけ", "Gastroenterology", 0.85),
    # Ophthalmology
    ("blurry vision", "Ophthalmology", 0.9), ("eye pain", "Ophthalmology", 0.9),
    ("red eye", "Ophthalmology", 0.85), ("dry eyes", "Ophthalmology", 0.85),
    ("conjunctivitis", "Ophthalmology", 0.95), ("cataract", "Ophthalmology", 0.9),
    ("目が痛", "Ophthalmology", 0.9), ("視力", "Ophthalmology", 0.85),
    ("目のかすみ", "Ophthalmology", 0.9), ("充血", "Ophthalmology", 0.8),
    # Obstetrics and Gynecology
    ("pregnan", "Obstetrics and Gynecology", 0.9),
    ("menstrual", "Obstetrics and Gynecology", 0.9),
    ("period pain", "Obstetrics and Gynecology", 0.85),
    ("menopause", "Obstetrics and Gynecology", 0.9),
    ("prenatal", "Obstetrics and Gynecology", 0.95),
    ("妊娠", "Obstetrics and Gynecology", 0.9), ("生理", "Obstetrics and Gynecology", 0.85),
    ("更年期", "Obstetrics and Gynecology", 0.9),
    # General Practice
    # "X hurts" is how patients actually phrase it on the phone; without these
    # the rules fall through to General Practice for very specific complaints.
    ("ear hurts", "Otolaryngology", 0.85), ("throat hurts", "Otolaryngology", 0.8),
    ("knee hurts", "Orthopedics", 0.85), ("back hurts", "Orthopedics", 0.8),
    ("shoulder hurts", "Orthopedics", 0.8), ("hip hurts", "Orthopedics", 0.8),
    ("stomach hurts", "Gastroenterology", 0.85), ("belly hurts", "Gastroenterology", 0.8),
    ("eye hurts", "Ophthalmology", 0.85), ("eyes hurt", "Ophthalmology", 0.85),
    ("skin is itchy", "Dermatology", 0.8), ("head hurts", "Neurology", 0.7),
    ("fever", "General Practice", 0.55), ("cough", "General Practice", 0.55),
    ("cold", "General Practice", 0.5), ("flu", "General Practice", 0.6),
    ("checkup", "General Practice", 0.8), ("check-up", "General Practice", 0.8),
    ("vaccination", "General Practice", 0.8), ("tired", "General Practice", 0.5),
    ("熱がある", "General Practice", 0.6), ("発熱", "General Practice", 0.6),
    ("咳", "General Practice", 0.55), ("風邪", "General Practice", 0.6),
    ("健康診断", "General Practice", 0.85),
)

# Specialties that are plausible neighbours. Used for partial credit so a near
# miss scores above an unrelated specialty instead of dropping to zero.
RELATED_SPECIALTIES: dict[str, frozenset[str]] = {
    "Cardiology": frozenset({"General Practice"}),
    "Dermatology": frozenset({"General Practice"}),
    "Gastroenterology": frozenset({"General Practice"}),
    "General Practice": frozenset(SPECIALTIES) - {"General Practice"},
    "Neurology": frozenset({"General Practice", "Psychiatry", "Otolaryngology"}),
    "Obstetrics and Gynecology": frozenset({"General Practice"}),
    "Ophthalmology": frozenset({"General Practice"}),
    "Orthopedics": frozenset({"General Practice"}),
    "Otolaryngology": frozenset({"General Practice", "Neurology"}),
    "Pediatrics": frozenset({"General Practice"}),
    "Psychiatry": frozenset({"General Practice", "Neurology"}),
}

REGIONS: tuple[str, ...] = ("Tokyo", "Osaka", "Yokohama", "Sapporo")

REGION_ALIASES: dict[str, str] = {
    "tokyo": "Tokyo", "東京": "Tokyo", "とうきょう": "Tokyo", "chiyoda": "Tokyo",
    "shinjuku": "Tokyo", "shibuya": "Tokyo",
    "osaka": "Osaka", "大阪": "Osaka", "おおさか": "Osaka", "kita-ku": "Osaka",
    "yokohama": "Yokohama", "横浜": "Yokohama", "よこはま": "Yokohama",
    "kanagawa": "Yokohama", "神奈川": "Yokohama",
    "sapporo": "Sapporo", "札幌": "Sapporo", "さっぽろ": "Sapporo",
    "hokkaido": "Sapporo", "北海道": "Sapporo",
}

# Commutable pairs get partial region credit rather than zero.
ADJACENT_REGIONS: dict[str, frozenset[str]] = {
    "Tokyo": frozenset({"Yokohama"}),
    "Yokohama": frozenset({"Tokyo"}),
    "Osaka": frozenset(),
    "Sapporo": frozenset(),
}

LANGUAGES: tuple[str, ...] = ("Japanese", "English", "Chinese", "Korean")

LANGUAGE_ALIASES: dict[str, str] = {
    "japanese": "Japanese", "jp": "Japanese", "nihongo": "Japanese", "日本語": "Japanese",
    "english": "English", "en": "English", "英語": "English", "eigo": "English",
    "chinese": "Chinese", "mandarin": "Chinese", "中国語": "Chinese", "中文": "Chinese",
    "korean": "Korean", "韓国語": "Korean",
}

PERSONALITIES: tuple[str, ...] = (
    "kind", "gentle", "professional", "empathetic", "calm", "humor", "friendly",
)

PERSONALITY_ALIASES: dict[str, str] = {
    "kind": "kind", "nice": "kind", "warm": "kind", "優しい": "kind", "親切": "kind",
    "gentle": "gentle", "soft": "gentle", "穏やか": "gentle",
    "professional": "professional", "serious": "professional", "experienced": "professional",
    "thorough": "professional", "専門的": "professional", "きちんと": "professional",
    "empathetic": "empathetic", "understanding": "empathetic", "listens": "empathetic",
    "共感": "empathetic", "話をよく聞": "empathetic",
    "calm": "calm", "patient": "calm", "落ち着": "calm",
    "humor": "humor", "humour": "humor", "funny": "humor", "cheerful": "humor",
    "面白": "humor", "明るい": "humor",
    "friendly": "friendly", "approachable": "friendly", "フレンドリー": "friendly",
    "話しやすい": "friendly",
}

# Personalities that a patient asking for X would also accept.
PERSONALITY_CLUSTERS: dict[str, frozenset[str]] = {
    "kind": frozenset({"gentle", "empathetic", "friendly"}),
    "gentle": frozenset({"kind", "calm", "empathetic"}),
    "empathetic": frozenset({"kind", "gentle", "calm"}),
    "calm": frozenset({"gentle", "professional", "empathetic"}),
    "professional": frozenset({"calm"}),
    "humor": frozenset({"friendly"}),
    "friendly": frozenset({"kind", "humor"}),
}

GENDERS: tuple[str, ...] = ("male", "female")

GENDER_ALIASES: dict[str, str] = {
    "male": "male", "man": "male", "men": "male", "he": "male", "男性": "male", "男": "male",
    "female": "female", "woman": "female", "women": "female", "she": "female",
    "lady": "female", "女性": "female", "女": "female",
}


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "")).strip().lower()


def _coerce(value: str | None, vocabulary: Iterable[str], aliases: dict[str, str]) -> str | None:
    """Map free text onto a closed vocabulary, or None.

    Returning None is the important part: an unmappable value is dropped rather
    than guessed, so a hallucinated enum never reaches the database.
    """
    if not value:
        return None
    folded = _fold(value)
    if not folded:
        return None
    for canonical in vocabulary:
        if folded == canonical.lower():
            return canonical
    if folded in aliases:
        return aliases[folded]
    # Longest alias first so "ear nose throat" beats "ent".
    for alias in sorted(aliases, key=len, reverse=True):
        if _contains(folded, alias):
            return aliases[alias]
    for canonical in vocabulary:
        if _contains(folded, canonical.lower()):
            return canonical
    return None


def _contains(haystack: str, needle: str) -> bool:
    """Whole-word match for latin text, plain substring for CJK.

    A bare substring test is wrong for short latin aliases: "ent" is inside
    "Kenta" and "Department", which once made a patient named Kenta get routed
    to Otolaryngology. Japanese has no word boundaries, and 東京 / 内科 / 眼科
    are two characters each, so \\b cannot be used there.
    """
    if not needle:
        return False
    if needle.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None
    return needle in haystack


def coerce_specialty(value: str | None, dynamic_aliases: dict[str, str] | None = None) -> str | None:
    if not value:
        return None
    folded = _fold(value)
    if dynamic_aliases:
        for alias, canonical in dynamic_aliases.items():
            if _fold(alias) == folded and canonical in SPECIALTIES:
                return canonical
    return _coerce(value, SPECIALTIES, STATIC_ALIASES)


def coerce_region(value: str | None) -> str | None:
    return _coerce(value, REGIONS, REGION_ALIASES)


def coerce_language(value: str | None) -> str | None:
    return _coerce(value, LANGUAGES, LANGUAGE_ALIASES)


def coerce_personality(value: str | None) -> str | None:
    return _coerce(value, PERSONALITIES, PERSONALITY_ALIASES)


def coerce_gender(value: str | None) -> str | None:
    return _coerce(value, GENDERS, GENDER_ALIASES)


def parse_languages(raw: str | list[str] | None) -> list[str]:
    """`"Japanese, English"` and `["Japanese","English"]` both accepted."""
    if not raw:
        return []
    items = raw if isinstance(raw, list) else re.split(r"[,/、･・]| and ", str(raw))
    out: list[str] = []
    for item in items:
        canonical = coerce_language(item)
        if canonical and canonical not in out:
            out.append(canonical)
        elif not canonical and item.strip():
            cleaned = item.strip()
            if cleaned not in out:
                out.append(cleaned)  # keep unknown languages rather than lose data
    return out


def classify_symptom_rules(text: str) -> tuple[str, float, list[str]]:
    """Deterministic symptom -> specialty. Returns (specialty, confidence, evidence)."""
    folded = _fold(text)
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for keyword, specialty, weight in SYMPTOM_RULES:
        if keyword in folded:
            # Longer keyword matches are more specific, so nudge them up.
            adjusted = weight + min(len(keyword), 20) / 200.0
            if adjusted > scores.get(specialty, 0.0):
                scores[specialty] = adjusted
            evidence.setdefault(specialty, []).append(keyword)

    # An explicit specialty / alias mention beats any symptom inference.
    explicit = coerce_specialty(text)
    if explicit and any(
        _fold(a) in folded for a in list(STATIC_ALIASES) + [s.lower() for s in SPECIALTIES]
    ):
        scores[explicit] = max(scores.get(explicit, 0.0), 0.95)
        evidence.setdefault(explicit, []).append("explicit specialty mention")

    if not scores:
        return FALLBACK_SPECIALTY, 0.30, []

    best = max(scores, key=lambda s: (scores[s], s))
    confidence = min(scores[best], 0.95)
    # Ambiguity penalty: two specialties scoring close together is a weak signal.
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) > 1 and ranked[0] - ranked[1] < 0.1:
        confidence *= 0.8
    return best, round(confidence, 3), evidence.get(best, [])
