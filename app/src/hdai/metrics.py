"""Minimal Prometheus text-format metrics (design doc section 15).

Hand-rolled rather than prometheus_client: ~60 lines against one more
dependency in the boot path, and the metric set here is fixed.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._hist: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(
            lambda: [0.0] * (len(_LATENCY_BUCKETS) + 2)  # buckets + count + sum
        )

    @staticmethod
    def _key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((labels or {}).items()))

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[(name, self._key(labels))] += value

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._gauges[(name, self._key(labels))] = value

    def observe(self, name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            slot = self._hist[(name, self._key(labels))]
            for i, bound in enumerate(_LATENCY_BUCKETS):
                if seconds <= bound:
                    slot[i] += 1
            slot[-2] += 1
            slot[-1] += seconds

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{_fmt(labels)} {value:g}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"{name}{_fmt(labels)} {value:g}")
            for (name, labels), slot in sorted(self._hist.items()):
                cumulative = 0.0
                for i, bound in enumerate(_LATENCY_BUCKETS):
                    cumulative = slot[i]
                    lines.append(f"{name}_bucket{_fmt(labels, le=str(bound))} {cumulative:g}")
                lines.append(f"{name}_bucket{_fmt(labels, le='+Inf')} {slot[-2]:g}")
                lines.append(f"{name}_count{_fmt(labels)} {slot[-2]:g}")
                lines.append(f"{name}_sum{_fmt(labels)} {slot[-1]:g}")
        return "\n".join(lines) + "\n"


def _fmt(labels: Iterable[tuple[str, str]], le: str | None = None) -> str:
    pairs = [f'{k}="{_escape(v)}"' for k, v in labels]
    if le is not None:
        pairs.append(f'le="{le}"')
    return "{" + ",".join(pairs) + "}" if pairs else ""


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


METRICS = Metrics()
