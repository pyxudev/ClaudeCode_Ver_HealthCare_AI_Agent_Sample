"""Agent 5 - Response. Turns the ranked list into something a patient hears.

The deterministic template is the product; the LLM is a rewrite pass on top of
it. Generation is constrained three ways:

  1. the model is given ONLY the already-chosen facts, never the raw dataset,
     so there is nothing extra to leak
  2. the output is checked against those facts (safety.ground_response) - an
     invented doctor, an invented slot, or anything that reads like a
     diagnosis fails the check
  3. any failure silently falls back to the template, which is always correct

The template is not a degraded mode users should dread: it is fully formed
prose, and in an A/B it is the safe arm.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

from ..llm import LLMClient, LLMUsage
from ..models import PatientRequirements, Recommendation
from ..safety import ground_response
from ..timeutil import to_local

log = logging.getLogger("hdai.agents.respond")

DISCLAIMER_EN = (
    "This is an appointment suggestion, not medical advice or a diagnosis. "
    "If your symptoms get worse or feel severe, please seek urgent care."
)
DISCLAIMER_JA = (
    "これは受診先のご提案であり、診断や医学的助言ではありません。"
    "症状が悪化した場合や重い場合は、速やかに医療機関を受診してください。"
)

_SYSTEM_PROMPT = """You write the spoken reply for a hospital call centre.

You are given a JSON object of doctors that have ALREADY been selected. Your
only job is to read them back naturally and warmly.

Absolute rules:
- Mention ONLY the doctors, hospitals, times and languages present in the JSON.
  Never add, merge, or adjust a detail. Never invent a doctor or a time.
- Never give a diagnosis, a cause, a treatment, or a medication.
- Never write he, him, his, she, her or hers about a doctor. The JSON does not
  say what anyone's pronouns are, so use the doctor's name or they/them.
- Write every doctor's name exactly as it appears in the JSON, in Latin script,
  even when the rest of your reply is in Japanese. Do not translate, transliterate
  or guess kanji for a name.
- Do not re-order or re-rank: the list is already in the correct order.
- Keep it under 90 words. Plain sentences, no bullet points, no markdown.
- End by offering to book the first suggestion.
- Reply in the requested language.
"""


def _format_slot(when: datetime, japanese: bool) -> str:
    """Always clinic local time - a patient hears "15:00", not the UTC 06:00."""
    local = to_local(when)
    if japanese:
        return local.strftime("%-m月%-d日 %H:%M")
    return local.strftime("%a %d %b at %H:%M")


def _slot_key(when: datetime) -> str:
    """Canonical local-time string shared by the prompt and the grounding check."""
    return to_local(when).strftime("%Y-%m-%d %H:%M")


def render_template(
    recommendations: Sequence[Recommendation],
    requirements: PatientRequirements,
    specialty: str,
    *,
    japanese: bool = False,
    relaxations: Sequence[str] = (),
) -> str:
    """Deterministic, always-correct reply. Never raises."""
    if not recommendations:
        return _no_match_text(specialty, japanese)

    lines: list[str] = []
    if japanese:
        lines.append(f"{specialty} の担当医を{len(recommendations)}名ご案内します。")
    else:
        article = "an" if specialty[0].upper() in "AEIOU" else "a"
        lines.append(
            f"Based on what you described, {article} {specialty} doctor looks right. "
            f"Here {'is' if len(recommendations) == 1 else 'are'} "
            f"{len(recommendations)} {'option' if len(recommendations) == 1 else 'options'}:"
        )

    for rec in recommendations:
        doctor = rec.doctor
        slot = rec.next_slots[0].available_time if rec.next_slots else None
        when = _format_slot(slot, japanese) if slot else ("空き枠なし" if japanese else "no open slot")
        if japanese:
            languages = "・".join(doctor.languages) or "日本語"
            lines.append(
                f"{rec.rank}. {doctor.name}（{doctor.expertise}、{doctor.hospital_name}／"
                f"{doctor.region}）評価 {doctor.rating:.1f}、対応言語 {languages}。"
                f"直近の空き枠は {when} です。"
            )
        else:
            languages = ", ".join(doctor.languages) or "Japanese"
            lines.append(
                f"{rec.rank}. {doctor.name} - {doctor.expertise} at {doctor.hospital_name}, "
                f"{doctor.region}. Rated {doctor.rating:.1f}/5, speaks {languages}. "
                f"Next opening: {when}."
            )

    if relaxations:
        note = "; ".join(relaxations)
        lines.append(
            f"（条件を一部広げて検索しました: {note}）" if japanese
            else f"(To find these I widened the search: {note}.)"
        )

    lines.append(
        "1番目の枠でご予約をお取りしましょうか。" if japanese
        else "Would you like me to book the first one for you?"
    )
    return "\n".join(lines)


def _no_match_text(specialty: str, japanese: bool) -> str:
    if japanese:
        return (
            f"申し訳ありません。ご希望の条件に合う {specialty} の医師で、"
            "現在空きのある方が見つかりませんでした。"
            "地域や日時のご希望を変更いただくか、オペレーターにおつなぎします。"
        )
    return (
        f"I'm sorry - I couldn't find a {specialty} doctor with an opening that matches "
        "what you asked for. If you can be flexible on the area or the date I can look "
        "again, or I can put you through to an operator."
    )


def _facts(recommendations: Sequence[Recommendation], japanese: bool) -> dict:
    return {
        "language": "Japanese" if japanese else "English",
        "timezone": "clinic local time - read the times back exactly as written",
        "doctors": [
            {
                "rank": rec.rank,
                "name": rec.doctor.name,
                "specialty": rec.doctor.expertise,
                "hospital": rec.doctor.hospital_name,
                "region": rec.doctor.region,
                "rating": rec.doctor.rating,
                "languages": rec.doctor.languages,
                "personality": rec.doctor.personality,
                "next_slots": [_slot_key(s.available_time) for s in rec.next_slots[:2]],
                "why": rec.reasons,
            }
            for rec in recommendations
        ],
    }


async def run(
    recommendations: Sequence[Recommendation],
    requirements: PatientRequirements,
    specialty: str,
    llm: LLMClient,
    *,
    japanese: bool = False,
    relaxations: Sequence[str] = (),
) -> tuple[str, LLMUsage, str]:
    template = render_template(
        recommendations, requirements, specialty, japanese=japanese, relaxations=relaxations
    )
    if not recommendations or not llm.available:
        return template, LLMUsage(), llm.status if not llm.available else "no_candidates"

    import json

    text, usage, reason = await llm.complete_text(
        system=_SYSTEM_PROMPT,
        user=json.dumps(_facts(recommendations, japanese), ensure_ascii=False),
        max_tokens=400,
    )
    if not text or not text.strip():
        return template, usage, reason or "empty"

    allowed_names = [rec.doctor.name for rec in recommendations]
    allowed_slots = [
        _slot_key(s.available_time) for rec in recommendations for s in rec.next_slots
    ]
    verdict = ground_response(
        text, allowed_names, allowed_slots,
        # The top pick must be named exactly as the hospital records it.
        require_verbatim=allowed_names[:1],
    )
    if not verdict.ok:
        log.warning(
            "generated reply failed grounding, using template",
            extra={"reasons": verdict.reasons[:3]},
        )
        return template, usage, "ungrounded"

    return text.strip(), usage, "ok"


def disclaimers(japanese: bool) -> list[str]:
    return [DISCLAIMER_JA if japanese else DISCLAIMER_EN]
