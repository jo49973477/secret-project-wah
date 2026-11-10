"""Structured, crash-tolerant telemetry for UniVTAC closed-loop rollouts."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable
from uuid import uuid4

from .core import CANONICAL_TASKS, TASK_DISPLAY_NAMES, classify_result
from .profiling import GLOBAL_PROFILER, RuntimeProfiler


EVENT_SCHEMA_VERSION = 1
DIAGNOSTIC_LEVELS = ("off", "standard", "full")
LOG_LEVELS = ("quiet", "normal", "debug")
TASK_STATUSES = ("Pending", "Initializing", "Running", "Completed", "Error")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


class EventWriter:
    """Append one complete JSON record with one OS write.

    Readers only advance through newline-terminated records, so a killed writer can
    leave at most one ignored partial record. Event IDs make replay idempotent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source: str,
        profiler: RuntimeProfiler | None = None,
    ):
        self.path = Path(path)
        self.source = source
        self.sequence = 0
        self.profiler = profiler or GLOBAL_PROFILER
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)

    def emit(
        self,
        event: str,
        *,
        event_id: str | None = None,
        flush: bool = False,
        **fields: Any,
    ) -> dict[str, Any]:
        generation_started = time.perf_counter_ns()
        with self._lock:
            self.sequence += 1
            created_ns = time.monotonic_ns()
            payload = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "event": event,
                "event_id": event_id
                or f"{self.source}:{os.getpid()}:{self.sequence}:{uuid4().hex}",
                "timestamp": utc_now(),
                "monotonic_seconds": created_ns / 1_000_000_000,
                "created_monotonic_ns": created_ns,
                "source": self.source,
                "pid": os.getpid(),
                **fields,
            }
            serialization_started = time.perf_counter_ns()
            encoded = (
                json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
            ).encode()
            self.profiler.record_ns(
                "monitoring.event_serialization",
                time.perf_counter_ns() - serialization_started,
            )
            write_started = time.perf_counter_ns()
            written = os.write(self._descriptor, encoded)
            if written != len(encoded):  # pragma: no cover - defensive for exotic filesystems
                raise OSError(f"short telemetry write: {written}/{len(encoded)} bytes")
            if flush:
                os.fsync(self._descriptor)
            self.profiler.record_ns(
                "monitoring.event_write", time.perf_counter_ns() - write_started
            )
        self.profiler.record_ns(
            "monitoring.event_generation", time.perf_counter_ns() - generation_started
        )
        return payload

    def close(self) -> None:
        with self._lock:
            descriptor = self._descriptor
            self._descriptor = -1
            if descriptor >= 0:
                os.close(descriptor)

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


class EventReader:
    """Incrementally read complete JSONL records without consuming partial writes."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_record_bytes: int = 1_000_000,
        max_read_bytes: int = 4_000_000,
        profiler: RuntimeProfiler | None = None,
    ):
        self.path = Path(path)
        self.offset = 0
        self.malformed_records = 0
        self.max_record_bytes = max_record_bytes
        self.max_read_bytes = max_read_bytes
        self._pending = b""
        self.profiler = profiler or GLOBAL_PROFILER

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        size = self.path.stat().st_size
        if size < self.offset:  # A replaced/truncated diagnostic stream starts a new generation.
            self.offset = 0
            self._pending = b""
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read(self.max_read_bytes)
        self.offset += len(data)
        if not data:
            return []
        data = self._pending + data
        records = data.split(b"\n")
        self._pending = records.pop()
        if len(self._pending) > self.max_record_bytes:
            self._pending = b""
            self.malformed_records += 1
        events: list[dict[str, Any]] = []
        parse_started = time.perf_counter_ns()
        received_ns = time.monotonic_ns()
        for raw in records:
            if not raw.strip():
                continue
            if len(raw) > self.max_record_bytes:
                self.malformed_records += 1
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.malformed_records += 1
                continue
            if not isinstance(value, dict) or not isinstance(value.get("event"), str):
                self.malformed_records += 1
                continue
            value["_monitor_received_monotonic_ns"] = received_ns
            created_ns = value.get("created_monotonic_ns")
            if isinstance(created_ns, int) and received_ns >= created_ns:
                self.profiler.record_ns("monitoring.event_delivery", received_ns - created_ns)
            events.append(value)
        self.profiler.record_ns("monitoring.event_parsing", time.perf_counter_ns() - parse_started)
        return events


class EpisodeProgressEmitter:
    """Publish the latest episode phase at most once per telemetry interval.

    A small heartbeat thread only writes telemetry. It never executes actions or
    accesses simulator state, so scientific evaluation order remains unchanged.
    """

    def __init__(
        self,
        writer: EventWriter,
        *,
        interval_seconds: float = 1.0,
        **common_fields: Any,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("telemetry interval must be finite and positive")
        self.writer = writer
        self.interval_seconds = interval_seconds
        self.common_fields = common_fields
        self._latest: tuple[str, dict[str, Any]] | None = None
        self._last_emit = 0.0
        self._episode_started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="episode-telemetry", daemon=True)
        self._thread.start()

    def update(self, event: str, *, force: bool = False, **fields: Any) -> bool:
        with self._lock:
            self._latest = (event, fields)
        return self._emit_if_due(force=force)

    def _emit_if_due(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        with self._lock:
            if self._latest is None or (
                not force and now - self._last_emit < self.interval_seconds
            ):
                return False
            event, fields = self._latest
            payload = {
                **self.common_fields,
                **fields,
                "elapsed_seconds": max(0.0, now - self._episode_started),
            }
            self._last_emit = now
        self.writer.emit(event, **payload)
        return True

    def _run(self) -> None:
        tick = min(0.2, self.interval_seconds / 2)
        while not self._stop.wait(tick):
            self._emit_if_due()

    def close(self, *, emit_latest: bool = False) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        if emit_latest:
            self._emit_if_due(force=True)

    def __enter__(self) -> "EpisodeProgressEmitter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass
class EpisodeTelemetry:
    task: str = ""
    episode_index: int | None = None
    seed: int | None = None
    action_count: int = 0
    max_actions: int | None = None
    execution_horizon: int = 1
    started_monotonic: float | None = None
    elapsed_seconds: float = 0.0
    inference_latency_ms: float | None = None
    simulation_fps: float | None = None
    phase: str = "Waiting"
    last_action_monotonic: float | None = None
    last_event_monotonic: float | None = None

    @property
    def live_elapsed_seconds(self) -> float:
        if self.started_monotonic is None or self.phase == "Episode complete":
            return self.elapsed_seconds
        return max(self.elapsed_seconds, time.monotonic() - self.started_monotonic)


@dataclass
class TaskTelemetry:
    task: str
    planned: int
    status: str = "Pending"
    successes: int = 0
    failures: int = 0
    errors: int = 0
    completed: int = 0
    unexecuted: int = 0
    total_steps: int = 0
    elapsed_seconds: float = 0.0
    started_monotonic: float | None = None
    completed_keys: set[tuple[str, int]] = field(default_factory=set, repr=False)
    episode_durations: deque[float] = field(default_factory=lambda: deque(maxlen=20), repr=False)

    @property
    def evaluated(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float | None:
        return self.successes / self.evaluated if self.evaluated else None

    @property
    def mean_episode_length(self) -> float | None:
        return self.total_steps / self.evaluated if self.evaluated else None


class RolloutAggregator:
    """Single logical source of rollout statistics for UI, files, and trainctl."""

    def __init__(
        self,
        tasks: Iterable[str] = (),
        episodes_per_task: int = 0,
        *,
        eta_history_size: int = 50,
        stall_warning_seconds: float = 30.0,
    ):
        self.tasks: dict[str, TaskTelemetry] = {
            task: TaskTelemetry(task, episodes_per_task) for task in tasks
        }
        self.episodes_per_task = episodes_per_task
        self.seen_event_ids: set[str] = set()
        self._seen_event_order: deque[str] = deque(maxlen=100_000)
        self.completed_episode_keys: set[tuple[str, int]] = set()
        self.current = EpisodeTelemetry()
        self.started_monotonic: float | None = None
        self.finished_monotonic: float | None = None
        self.phase = "Waiting"
        self.experiment_name = "UniVTAC Rollout"
        self.checkpoint = "N/A"
        self.checkpoint_identity = "N/A"
        self.policy_type = "N/A"
        self.sidecar_type = "N/A"
        self.execution_horizon = 1
        self.status = "Pending"
        self.diagnostics: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.inference_latencies_ms: list[float] = []
        self.simulation_fps_samples: list[float] = []
        self.error_categories: defaultdict[str, int] = defaultdict(int)
        self.last_event: dict[str, Any] | None = None
        self.episode_durations: deque[float] = deque(maxlen=eta_history_size)
        self.setup_seconds: float | None = None
        self.stall_warning_seconds = stall_warning_seconds
        self.last_meaningful_progress_monotonic: float | None = None

    def _ensure_task(self, task: str) -> TaskTelemetry:
        if task not in self.tasks:
            self.tasks[task] = TaskTelemetry(task, self.episodes_per_task)
        return self.tasks[task]

    def apply(self, event: dict[str, Any]) -> bool:
        event_id = str(event.get("event_id", ""))
        if event_id and event_id in self.seen_event_ids:
            return False
        if event_id:
            if len(self._seen_event_order) == self._seen_event_order.maxlen:
                expired = self._seen_event_order.popleft()
                self.seen_event_ids.discard(expired)
            self._seen_event_order.append(event_id)
            self.seen_event_ids.add(event_id)
        kind = str(event.get("event", ""))
        self.last_event = event
        phase = event.get("phase")
        if isinstance(phase, str) and phase:
            self.phase = phase

        if kind == "rollout_started":
            self.status = "Running"
            self.phase = str(event.get("phase", "Starting"))
            self.started_monotonic = (
                finite_number(event.get("monotonic_seconds")) or time.monotonic()
            )
            self.experiment_name = str(event.get("experiment_name") or self.experiment_name)
            self.checkpoint = str(event.get("checkpoint") or self.checkpoint)
            self.checkpoint_identity = str(
                event.get("checkpoint_identity") or self.checkpoint_identity
            )
            self.policy_type = str(event.get("policy_type") or self.policy_type).upper()
            self.sidecar_type = str(event.get("sidecar_type") or self.sidecar_type)
            self.execution_horizon = int(event.get("execution_horizon") or 1)
            planned = int(event.get("episodes_per_task") or self.episodes_per_task or 0)
            self.episodes_per_task = planned
            for task in event.get("tasks", ()):
                self._ensure_task(str(task)).planned = planned
        elif kind in {"server_starting", "server_ready"}:
            self.phase = "Server starting" if kind == "server_starting" else "Server ready"
            self.diagnostics.update(
                {key: value for key, value in event.items() if key not in _EVENT_BASE_FIELDS}
            )
        elif kind == "task_started":
            task = self._ensure_task(str(event["task"]))
            if task.status == "Pending":
                task.status = "Initializing"
            task.started_monotonic = (
                finite_number(event.get("monotonic_seconds")) or time.monotonic()
            )
            task.planned = int(event.get("planned_episodes") or task.planned)
            self.phase = f"Initializing {task.task}"
        elif kind == "episode_started":
            task_name = str(event["task"])
            episode_index = int(event.get("episode_index", 0))
            if (task_name, episode_index) in self.completed_episode_keys:
                return False
            task = self._ensure_task(task_name)
            task.status = "Running"
            self.current = EpisodeTelemetry(
                task=task.task,
                episode_index=episode_index,
                seed=int(event.get("seed", 0)),
                max_actions=_optional_int(event.get("max_actions")),
                execution_horizon=int(event.get("execution_horizon") or self.execution_horizon),
                started_monotonic=finite_number(event.get("monotonic_seconds")),
                phase=str(event.get("phase", "Episode reset")),
                last_event_monotonic=finite_number(event.get("monotonic_seconds")),
            )
            self.last_meaningful_progress_monotonic = self.current.started_monotonic
            self.phase = self.current.phase
        elif kind in {"episode_progress", "inference_started", "inference_completed"}:
            task_name = str(event.get("task", self.current.task))
            episode_index = int(event.get("episode_index", self.current.episode_index or 0))
            if (task_name, episode_index) in self.completed_episode_keys:
                return False
            previous_count = self.current.action_count
            self.current.action_count = int(event.get("action_count") or previous_count)
            self.current.max_actions = (
                _optional_int(event.get("max_actions")) or self.current.max_actions
            )
            self.current.elapsed_seconds = (
                finite_number(event.get("elapsed_seconds")) or self.current.elapsed_seconds
            )
            latency = finite_number(event.get("inference_latency_ms"))
            if latency is not None:
                self.current.inference_latency_ms = latency
            fps = finite_number(event.get("simulation_fps"))
            if fps is not None:
                self.current.simulation_fps = fps
            self.current.phase = str(event.get("phase", self.current.phase))
            event_monotonic = finite_number(event.get("monotonic_seconds")) or time.monotonic()
            self.current.last_event_monotonic = event_monotonic
            if self.current.action_count > previous_count:
                self.current.last_action_monotonic = event_monotonic
                self.last_meaningful_progress_monotonic = event_monotonic
            self.phase = self.current.phase
            self._sample_performance(event)
            diagnostics = event.get("diagnostics")
            if isinstance(diagnostics, dict):
                self.diagnostics.update(diagnostics)
        elif kind == "episode_completed":
            task_name = str(event["task"])
            index = int(event["episode_index"])
            key = (task_name, index)
            if key in self.completed_episode_keys:
                return False
            self.completed_episode_keys.add(key)
            task = self._ensure_task(task_name)
            if task.status not in {"Completed", "Error"}:
                task.status = "Running"
            classification = event.get("status") or classify_result(
                event.get("success"), str(event["termination_reason"])
            )
            if classification == "success":
                task.successes += 1
                task.total_steps += int(event.get("num_steps") or 0)
            elif classification == "failure":
                task.failures += 1
                task.total_steps += int(event.get("num_steps") or 0)
            else:
                task.errors += 1
                category = str(
                    event.get("error_category") or event.get("termination_reason") or "error"
                )
                self.error_categories[category] += 1
            task.completed += 1
            task.elapsed_seconds += finite_number(event.get("elapsed_seconds")) or 0.0
            duration = finite_number(event.get("elapsed_seconds"))
            if duration is not None and duration > 0:
                task.episode_durations.append(duration)
                self.episode_durations.append(duration)
            self.current.elapsed_seconds = finite_number(event.get("elapsed_seconds")) or 0.0
            self.current.phase = "Episode complete"
            self.phase = "Episode complete"
            self.last_meaningful_progress_monotonic = (
                finite_number(event.get("monotonic_seconds")) or time.monotonic()
            )
            self._sample_performance(event)
        elif kind == "task_completed":
            task = self._ensure_task(str(event["task"]))
            task.status = "Completed"
            task.elapsed_seconds = (
                finite_number(event.get("elapsed_seconds")) or task.elapsed_seconds
            )
            self.phase = f"Completed {task.task}"
        elif kind in {"task_failed", "rollout_failed"}:
            task_name = event.get("task")
            if task_name:
                task = self._ensure_task(str(task_name))
                task.status = "Error"
                task.unexecuted = int(event.get("unexecuted_episodes") or 0)
            self.status = "Failed" if kind == "rollout_failed" else self.status
            self.phase = str(event.get("phase") or "Failed")
        elif kind == "rollout_completed":
            self.status = "Completed"
            self.phase = "Completed"
            self.finished_monotonic = (
                finite_number(event.get("monotonic_seconds")) or time.monotonic()
            )
        elif kind == "system_warning":
            message = str(event.get("message", "warning"))
            if message not in self.warnings:
                self.warnings.append(message)
        elif kind == "system_diagnostics":
            metrics = event.get("diagnostics")
            if isinstance(metrics, dict):
                self.diagnostics.update(metrics)
        if kind == "server_ready" and self.started_monotonic is not None:
            ready = finite_number(event.get("monotonic_seconds"))
            if ready is not None:
                self.setup_seconds = max(0.0, ready - self.started_monotonic)
        return True

    def _sample_performance(self, event: dict[str, Any]) -> None:
        latency = finite_number(event.get("inference_latency_ms"))
        fps = finite_number(event.get("simulation_fps"))
        if latency is not None:
            self.inference_latencies_ms.append(latency)
        if fps is not None:
            self.simulation_fps_samples.append(fps)

    @property
    def total_planned(self) -> int:
        return sum(task.planned for task in self.tasks.values())

    @property
    def total_completed(self) -> int:
        return sum(task.completed for task in self.tasks.values())

    @property
    def total_successes(self) -> int:
        return sum(task.successes for task in self.tasks.values())

    @property
    def total_failures(self) -> int:
        return sum(task.failures for task in self.tasks.values())

    @property
    def total_errors(self) -> int:
        return sum(task.errors for task in self.tasks.values())

    @property
    def evaluated(self) -> int:
        return self.total_successes + self.total_failures

    @property
    def micro_success_rate(self) -> float | None:
        return self.total_successes / self.evaluated if self.evaluated else None

    @property
    def macro_success_rate(self) -> float | None:
        rates = [task.success_rate for task in self.tasks.values()]
        return (
            sum(rates) / len(rates) if rates and all(rate is not None for rate in rates) else None
        )

    @property
    def error_rate(self) -> float | None:
        return self.total_errors / self.total_completed if self.total_completed else None

    @property
    def elapsed_seconds(self) -> float:
        if self.started_monotonic is None:
            return 0.0
        end = self.finished_monotonic or time.monotonic()
        return max(0.0, end - self.started_monotonic)

    @property
    def eta_seconds(self) -> float | None:
        if self.total_planned <= self.total_completed:
            return (
                0.0 if self.total_planned and self.total_completed >= self.total_planned else None
            )
        global_mean = (
            _mean(list(self.episode_durations)) if len(self.episode_durations) >= 3 else None
        )
        estimate = 0.0
        for task in self.tasks.values():
            remaining = max(0, task.planned - task.completed)
            if not remaining:
                continue
            task_mean = (
                _mean(list(task.episode_durations)) if len(task.episode_durations) >= 2 else None
            )
            duration = task_mean if task_mean is not None else global_mean
            if duration is None or not math.isfinite(duration):
                return None
            estimate += remaining * duration
        return estimate if math.isfinite(estimate) else None

    @property
    def stalled_seconds(self) -> float | None:
        if self.current.episode_index is None or self.current.phase == "Episode complete":
            return None
        baseline = self.current.last_action_monotonic or self.current.started_monotonic
        if baseline is None:
            return None
        return max(0.0, time.monotonic() - baseline)

    @property
    def stall_warning(self) -> str | None:
        stalled = self.stalled_seconds
        if stalled is None or stalled < self.stall_warning_seconds:
            return None
        return (
            f"No completed action for {stalled:.0f} seconds; phase={self.current.phase}; "
            f"last_action={self.current.action_count}; "
            f"server_status={self.diagnostics.get('server_status', 'N/A')}"
        )

    def percentile_latency(self, percentile: float) -> float | None:
        if not self.inference_latencies_ms:
            return None
        values = sorted(self.inference_latencies_ms)
        index = min(len(values) - 1, max(0, math.ceil(percentile * len(values)) - 1))
        return values[index]

    def trainctl_metrics(self) -> dict[str, int | float]:
        elapsed = self.elapsed_seconds
        values: dict[str, Any] = {
            "rollout/completed_episodes": self.total_completed,
            "rollout/total_episodes": self.total_planned,
            "rollout/successes": self.total_successes,
            "rollout/failures": self.total_failures,
            "rollout/errors": self.total_errors,
            "rollout/success_rate": self.micro_success_rate,
            "rollout/error_rate": self.error_rate,
            "rollout/episodes_per_second": self.total_completed / elapsed if elapsed > 0 else None,
            "rollout/inference_latency_ms": _mean(self.inference_latencies_ms),
            "rollout/simulation_fps": _mean(self.simulation_fps_samples),
            "rollout/gpu_memory_used_mib": self.diagnostics.get("gpu_memory_used_mib"),
        }
        return {
            key: int(value) if isinstance(value, int) else float(value)
            for key, value in values.items()
            if finite_number(value) is not None
        }


_EVENT_BASE_FIELDS = {
    "schema_version",
    "event",
    "event_id",
    "timestamp",
    "monotonic_seconds",
    "created_monotonic_ns",
    "_monitor_received_monotonic_ns",
    "source",
    "pid",
}


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def task_display_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())


def default_task_order(tasks: Iterable[str]) -> list[str]:
    order = list(tasks)
    return sorted(
        order, key=lambda task: CANONICAL_TASKS.index(task) if task in CANONICAL_TASKS else 99
    )
