"""Low-overhead runtime latency aggregation for UniVTAC rollouts."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Iterator


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass
class LatencySeries:
    """Bounded samples plus an exact count/total/max accumulator."""

    max_samples: int = 10_000
    count: int = 0
    total_ns: int = 0
    maximum_ns: int = 0
    samples_ns: list[int] = field(default_factory=list)

    def add(self, duration_ns: int) -> None:
        duration_ns = max(0, int(duration_ns))
        self.count += 1
        self.total_ns += duration_ns
        self.maximum_ns = max(self.maximum_ns, duration_ns)
        if len(self.samples_ns) < self.max_samples:
            self.samples_ns.append(duration_ns)
        else:
            # Deterministic reservoir-like replacement keeps memory bounded and
            # avoids random-number overhead in the action loop.
            self.samples_ns[self.count % self.max_samples] = duration_ns

    def summary(self) -> dict[str, int | float]:
        values_ms = [value / 1_000_000 for value in self.samples_ns]
        if not values_ms:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "max_ms": 0.0,
                "total_ms": 0.0,
            }
        return {
            "count": self.count,
            "mean_ms": self.total_ns / self.count / 1_000_000,
            "p50_ms": _percentile(values_ms, 0.50),
            "p95_ms": _percentile(values_ms, 0.95),
            "max_ms": self.maximum_ns / 1_000_000,
            "total_ms": self.total_ns / 1_000_000,
        }


class RuntimeProfiler:
    """Thread-safe named timing recorder based on ``perf_counter_ns``."""

    def __init__(self, *, enabled: bool = True, max_samples: int = 10_000):
        self.enabled = enabled
        self.max_samples = max_samples
        self._series: dict[str, LatencySeries] = {}
        self._lock = threading.Lock()

    def record_ns(self, stage: str, duration_ns: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            series = self._series.setdefault(stage, LatencySeries(self.max_samples))
            series.add(duration_ns)

    def record_ms(self, stage: str, duration_ms: float) -> None:
        self.record_ns(stage, round(duration_ms * 1_000_000))

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_ns(stage, time.perf_counter_ns() - started)

    def summary(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            return {name: series.summary() for name, series in sorted(self._series.items())}

    def merge_summary(self, summary: dict[str, Any], *, prefix: str = "") -> None:
        """Merge another process's summarized data without fabricating samples."""
        for name, row in summary.items():
            if not isinstance(row, dict):
                continue
            count = row.get("count")
            total_ms = row.get("total_ms")
            max_ms = row.get("max_ms")
            if not isinstance(count, int) or count <= 0 or not isinstance(total_ms, (int, float)):
                continue
            stage = f"{prefix}{name}"
            with self._lock:
                series = self._series.setdefault(stage, LatencySeries(self.max_samples))
                series.count += count
                series.total_ns += round(float(total_ms) * 1_000_000)
                if isinstance(max_ms, (int, float)):
                    series.maximum_ns = max(series.maximum_ns, round(float(max_ms) * 1_000_000))

    def write_json(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clock": "time.perf_counter_ns",
            "gpu_synchronized": False,
            "stages": self.summary(),
            "metadata": metadata or {},
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)


GLOBAL_PROFILER = RuntimeProfiler()
