"""Independent background samplers used by the rollout dashboard."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .profiling import GLOBAL_PROFILER, RuntimeProfiler


class AsyncSampler:
    """Run one non-overlapping sample at a configurable interval.

    Sampling failures are retained as status only; consumers keep the last valid
    value. A single worker thread guarantees subprocesses never overlap.
    """

    def __init__(
        self,
        sample: Callable[[], Any],
        *,
        interval_seconds: float,
        stage: str,
        profiler: RuntimeProfiler | None = None,
    ):
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.sample = sample
        self.interval_seconds = interval_seconds
        self.stage = stage
        self.profiler = profiler or GLOBAL_PROFILER
        self.last_valid: Any = None
        self.last_error: str | None = None
        self.last_sample_monotonic: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"{self.stage}-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter_ns()
            try:
                value = self.sample()
                if value is not None:
                    with self._lock:
                        self.last_valid = value
                        self.last_error = None
                        self.last_sample_monotonic = time.monotonic()
            except Exception as exc:  # telemetry must never stop evaluation
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self.profiler.record_ns(self.stage, time.perf_counter_ns() - started)
            if self._stop.wait(self.interval_seconds):
                break

    def snapshot(self) -> tuple[Any, str | None, float | None]:
        with self._lock:
            return self.last_valid, self.last_error, self.last_sample_monotonic

    def close(self, timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self) -> "AsyncSampler":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
