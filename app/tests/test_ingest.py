"""Ingestion normalisation: the pure helpers, tested against dirty input."""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timedelta

import pytest

from hdai.ingest import (
    LOCAL_TZ,
    IngestReport,
    clamp_rating,
    clamp_score,
    clean_text,
    compute_slot_shift,
    parse_date,
    parse_slot,
    synthetic_internal_number,
)

DATASET = pathlib.Path("/data/sample_doctors.json")


class TestCleaning:
    def test_collapses_whitespace_and_trims(self):
        assert clean_text("  Dr.   Aiko\n Tanaka  ") == "Dr. Aiko Tanaka"

    def test_none_becomes_empty(self):
        assert clean_text(None) == ""

    def test_enforces_a_length_limit(self):
        assert len(clean_text("x" * 5000, limit=50)) == 50


class TestNumericClamping:
    @pytest.mark.parametrize("value,expected", [(4.8, 4.8), (9.9, 5.0), (-3, 0.0), ("4.5", 4.5)])
    def test_rating(self, value, expected):
        assert clamp_rating(value, IngestReport(), "x") == expected

    def test_unreadable_rating_becomes_zero_and_is_noted(self):
        report = IngestReport()
        assert clamp_rating("very good", report, "Dr. X") == 0.0
        assert report.notes

    @pytest.mark.parametrize("value,expected", [(92, 92), (500, 100), (-1, 0), ("88", 88)])
    def test_score(self, value, expected):
        assert clamp_score(value, IngestReport(), "x") == expected


class TestSlotParsing:
    @pytest.mark.parametrize(
        "raw",
        ["2026-08-04 10:00", "2026-08-04T10:00", "2026-08-04 10:00:00", "2026-08-04T10:00:00"],
    )
    def test_accepted_formats(self, raw):
        parsed = parse_slot(raw)
        assert parsed is not None and parsed.hour == 10 and parsed.tzinfo is not None

    def test_dataset_slots_are_treated_as_local_time(self):
        assert parse_slot("2026-08-04 10:00").utcoffset() == timedelta(hours=9)

    @pytest.mark.parametrize("raw", ["", None, "not a date", "2026-13-45 99:99", "tomorrow"])
    def test_garbage_returns_none_instead_of_raising(self, raw):
        assert parse_slot(raw) is None

    def test_date_parsing(self):
        assert parse_date("2018-04-01").year == 2018
        assert parse_date("2018/04/01").month == 4
        assert parse_date("garbage") is None
        assert parse_date(None) is None


class TestSlotShift:
    NOW = datetime(2026, 9, 22, 12, 0, tzinfo=LOCAL_TZ)

    def test_a_fully_stale_dataset_is_moved_to_tomorrow(self):
        """Every slot in sample_doctors.json is 2026-08-xx: without this the
        POC would come up with zero bookable availability."""
        slots = [
            datetime(2026, 8, 4, 10, 0, tzinfo=LOCAL_TZ),
            datetime(2026, 8, 8, 16, 0, tzinfo=LOCAL_TZ),
        ]
        shift = compute_slot_shift(slots, self.NOW, True)
        assert (min(slots) + shift).date() == (self.NOW + timedelta(days=1)).date()

    def test_relative_gaps_and_time_of_day_are_preserved(self):
        slots = [
            datetime(2026, 8, 4, 10, 0, tzinfo=LOCAL_TZ),
            datetime(2026, 8, 8, 16, 30, tzinfo=LOCAL_TZ),
        ]
        shift = compute_slot_shift(slots, self.NOW, True)
        shifted = [s + shift for s in slots]
        assert (shifted[1] - shifted[0]) == (slots[1] - slots[0])
        assert shifted[0].hour == 10 and shifted[1].minute == 30

    def test_current_data_is_left_alone(self):
        slots = [self.NOW + timedelta(days=3)]
        assert compute_slot_shift(slots, self.NOW, True) == timedelta(0)

    def test_disabled_by_configuration(self):
        slots = [datetime(2020, 1, 1, 9, 0, tzinfo=LOCAL_TZ)]
        assert compute_slot_shift(slots, self.NOW, False) == timedelta(0)

    def test_empty_input(self):
        assert compute_slot_shift([], self.NOW, True) == timedelta(0)


class TestSyntheticKeys:
    def test_stable_for_the_same_inputs(self):
        assert synthetic_internal_number("D", "a", "b") == synthetic_internal_number("D", "a", "b")

    def test_distinct_for_different_inputs(self):
        assert synthetic_internal_number("D", "a") != synthetic_internal_number("D", "b")

    def test_prefixed_for_traceability(self):
        assert synthetic_internal_number("D", "a").startswith("D-GEN-")


class TestReport:
    def test_rejections_are_recorded_not_swallowed(self):
        report = IngestReport()
        report.reject("doctor", "Dr. X", "unknown hospital")
        assert report.as_dict()["rejected"] == [
            {"kind": "doctor", "id": "Dr. X", "reason": "unknown hospital"}
        ]


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not mounted")
class TestAgainstTheRealDataset:
    @pytest.fixture(scope="class")
    def data(self):
        return json.loads(DATASET.read_text(encoding="utf-8"))

    def test_shape_is_what_the_loader_expects(self, data):
        assert set(data) >= {"hospitals", "doctors", "aliases"}
        assert len(data["hospitals"]) == 4 and len(data["doctors"]) == 15

    def test_every_doctor_references_a_known_hospital(self, data):
        names = {h["name"] for h in data["hospitals"]}
        missing = [d["name"] for d in data["doctors"] if d.get("hospital") not in names]
        assert not missing, f"these doctors would be rejected at ingest: {missing}"

    def test_every_slot_parses(self, data):
        bad = [
            (d["name"], s)
            for d in data["doctors"]
            for s in d.get("available_slots", [])
            if parse_slot(s) is None
        ]
        assert not bad

    def test_internal_numbers_are_unique(self, data):
        numbers = [d["internal_number"] for d in data["doctors"]]
        assert len(numbers) == len(set(numbers))

    def test_the_dataset_really_is_stale(self, data):
        """Documents the assumption behind HDAI_DEMO_SHIFT_PAST_SLOTS."""
        latest = max(
            parse_slot(s)
            for d in data["doctors"]
            for s in d.get("available_slots", [])
        )
        assert latest < datetime.now(tz=LOCAL_TZ), (
            "sample data is no longer stale - the demo shift can be turned off"
        )
