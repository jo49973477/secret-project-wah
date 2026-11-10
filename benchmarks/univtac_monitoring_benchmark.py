#!/usr/bin/env python3
"""CPU-only fake-worker benchmark for UniVTAC rollout monitoring.

This benchmark measures monitoring mechanics only. It does not estimate Isaac
Sim or GPU inference speedups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "UniVTAC"))

from univtac_rollout.telemetry import EventReader, EventWriter  # noqa: E402


LEGACY_PREFIX = "@@WAH_ROLLOUT@@"


def _latencies(values_ns: list[int]) -> dict[str, float | int | None]:
    if not values_ns:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    values = sorted(value / 1_000_000 for value in values_ns)

    def percentile(fraction: float) -> float:
        return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "max_ms": max(values),
    }


def _fake_compute(index: int) -> int:
    value = index
    for _ in range(20):
        value = ((value * 1_103_515_245) + 12_345) & 0x7FFFFFFF
    return value


def run_case(mode: str, *, actions: int, action_seconds: float) -> dict[str, Any]:
    if mode not in {"disabled", "legacy", "optimized"}:
        raise ValueError(mode)
    with tempfile.TemporaryDirectory(prefix="univtac-monitor-bench-") as directory:
        root = Path(directory)
        log_path = root / ("rollout.log" if mode == "legacy" else "events.jsonl")
        stop = threading.Event()
        display_latencies: list[int] = []
        processed_events = 0

        def monitor_legacy() -> None:
            nonlocal processed_events
            offset = 0
            pending = ""
            while not stop.wait(0.5):
                if not log_path.exists():
                    continue
                with log_path.open(encoding="utf-8") as stream:
                    stream.seek(offset)
                    pending += stream.read()
                    offset = stream.tell()
                lines = pending.split("\n")
                pending = lines.pop()
                for line in lines:
                    if not line.startswith(LEGACY_PREFIX):
                        continue
                    payload = json.loads(line[len(LEGACY_PREFIX) :])
                    created = payload.get("created_monotonic_ns")
                    if isinstance(created, int):
                        display_latencies.append(time.monotonic_ns() - created)
                    processed_events += 1

        def monitor_optimized() -> None:
            nonlocal processed_events
            reader = EventReader(log_path)
            last_render = 0.0
            pending_created: list[int] = []
            while not stop.wait(0.05):
                for event in reader.read():
                    created = event.get("created_monotonic_ns")
                    if isinstance(created, int):
                        pending_created.append(created)
                    processed_events += 1
                now = time.monotonic()
                if pending_created and now - last_render >= 1.0:
                    displayed = time.monotonic_ns()
                    display_latencies.extend(displayed - created for created in pending_created)
                    pending_created.clear()
                    last_render = now
            for event in reader.read():
                created = event.get("created_monotonic_ns")
                if isinstance(created, int):
                    pending_created.append(created)
                processed_events += 1
            displayed = time.monotonic_ns()
            display_latencies.extend(displayed - created for created in pending_created)

        monitor = None
        if mode == "legacy":
            monitor = threading.Thread(target=monitor_legacy, daemon=True)
        elif mode == "optimized":
            monitor = threading.Thread(target=monitor_optimized, daemon=True)
        if monitor is not None:
            monitor.start()

        tracemalloc.start()
        cpu_started = time.process_time_ns()
        wall_started = time.perf_counter_ns()
        checksum = 0
        writer = EventWriter(log_path, source="benchmark") if mode == "optimized" else None
        legacy_log = log_path.open("w", encoding="utf-8", buffering=1) if mode == "legacy" else None
        telemetry_every = max(1, round(1.0 / max(action_seconds, 1e-9)))
        try:
            if writer is not None:
                writer.emit(
                    "episode_started",
                    task="fake_task",
                    episode_index=0,
                    max_actions=actions,
                )
            for index in range(actions):
                checksum ^= _fake_compute(index)
                if action_seconds:
                    time.sleep(action_seconds)
                if legacy_log is not None:
                    for phase in (
                        "inference_started",
                        "inference_completed",
                        "execute_action",
                        "observation_started",
                        "observation_completed",
                    ):
                        legacy_log.write(f"[rollout-worker] phase={phase} action={index}\n")
                        legacy_log.flush()
                if writer is not None and (
                    (index + 1) % telemetry_every == 0 or index + 1 == actions
                ):
                    writer.emit(
                        "episode_progress",
                        task="fake_task",
                        episode_index=0,
                        action_count=index + 1,
                        max_actions=actions,
                    )
            if legacy_log is not None:
                payload = {
                    "kind": "episode",
                    "status": "success",
                    "created_monotonic_ns": time.monotonic_ns(),
                }
                legacy_log.write(LEGACY_PREFIX + json.dumps(payload) + "\n")
                legacy_log.flush()
            if writer is not None:
                writer.emit(
                    "episode_completed",
                    task="fake_task",
                    episode_index=0,
                    status="success",
                    termination_reason="success",
                    elapsed_seconds=actions * action_seconds,
                )
        finally:
            if writer is not None:
                writer.close()
            if legacy_log is not None:
                legacy_log.close()
        # Allow one real polling interval, then force a final drain.
        time.sleep(0.55 if mode == "legacy" else 0.10 if mode == "optimized" else 0.0)
        stop.set()
        if monitor is not None:
            monitor.join(timeout=2.0)
        wall_ns = time.perf_counter_ns() - wall_started
        cpu_ns = time.process_time_ns() - cpu_started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "mode": mode,
            "actions": actions,
            "wall_seconds": wall_ns / 1_000_000_000,
            "cpu_seconds": cpu_ns / 1_000_000_000,
            "events_processed": processed_events,
            "event_processing_throughput_per_second": processed_events / (wall_ns / 1e9),
            "event_to_display": _latencies(display_latencies),
            "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
            "peak_python_memory_bytes": peak_bytes,
            "checksum": checksum,
        }


def benchmark(*, actions: int, action_seconds: float) -> dict[str, Any]:
    cases = [
        run_case(mode, actions=actions, action_seconds=action_seconds)
        for mode in ("disabled", "legacy", "optimized")
    ]
    baseline = cases[0]["wall_seconds"]
    for row in cases:
        row["wall_overhead_percent_vs_disabled"] = (
            (row["wall_seconds"] / baseline - 1) * 100 if baseline else None
        )
    return {
        "scope": "CPU-only fake worker; not a GPU/Isaac speedup measurement",
        "action_seconds": action_seconds,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=int, default=300)
    parser.add_argument("--action-seconds", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(actions=args.actions, action_seconds=args.action_seconds)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
