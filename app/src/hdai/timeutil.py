"""One place that decides what "today" and "10:00" mean.

Slots are stored as timestamptz (UTC) but every human-facing decision -
"tomorrow", "this week", "09:30" - is in clinic local time. Mixing the two is
a silent off-by-one: a 08:30 JST slot is 23:30 UTC the *previous* day, so a
UTC-based day count tells the patient the wrong day.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _resolve() -> ZoneInfo:
    name = os.environ.get("HDAI_DISPLAY_TIMEZONE", "Asia/Tokyo")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


LOCAL_TZ = _resolve()


def to_local(when: datetime) -> datetime:
    """Convert to clinic local time. Naive input is assumed to be local."""
    if when.tzinfo is None:
        return when.replace(tzinfo=LOCAL_TZ)
    return when.astimezone(LOCAL_TZ)


def local_date(when: datetime) -> date:
    return to_local(when).date()


def days_between(start: datetime, end: datetime) -> int:
    """Whole calendar days apart in clinic local time."""
    return (local_date(end) - local_date(start)).days
