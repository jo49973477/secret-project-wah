"""Human-readable, dependency-free progress display for UniVTAC rollouts.

The Isaac worker writes machine-readable events to its normal log. The outer
deployment process reads those events from ``rollout.log`` and renders progress
on the user's terminal, without flooding it with Isaac Sim diagnostics.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import TextIO


EVENT_PREFIX = "@@WAH_ROLLOUT@@"


def emit_event(kind: str, **values: object) -> None:
    """Emit a line-buffer-friendly event that survives nested subprocess pipes."""
    print(EVENT_PREFIX + json.dumps({"kind": kind, **values}, ensure_ascii=False), flush=True)


class RolloutProgress:
    """Display episode progress; keep non-TTY logs clean and deterministic."""

    def __init__(self, stream: TextIO | None = None, *, tty: bool | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self.color = (
            self.tty and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
        )
        self.total = 0
        self.completed = 0
        self.success = 0
        self.failure = 0
        self.error = 0
        self.task = "waiting for simulator"
        self.started = time.monotonic()
        self._last_render = 0.0
        self._line_visible = False

    def _color(self, value: str, code: str) -> str:
        return f"\x1b[{code}m{value}\x1b[0m" if self.color else value

    def consume(self, line: str) -> bool:
        """Return True only for valid progress events (not ordinary log lines)."""
        if not line.startswith(EVENT_PREFIX):
            return False
        try:
            event = json.loads(line[len(EVENT_PREFIX) :])
        except (ValueError, TypeError):
            return False
        if not isinstance(event, dict):
            return False
        kind = event.get("kind")
        if kind == "start":
            total = event.get("total")
            if type(total) is not int or total < 1:
                return False
            self.total = total
            self.task = "initializing task"
            self._announce(f"WAH rollout started · {total} episodes")
        elif kind == "task":
            task = event.get("task")
            if not isinstance(task, str) or not task:
                return False
            self.task = task
            self._announce(f"Task {event.get('index', '?')}/{event.get('task_count', '?')}: {task}")
        elif kind == "episode":
            status = event.get("status")
            if status not in {"success", "failure", "error"} or self.completed >= self.total:
                return False
            self.completed += 1
            setattr(self, status, getattr(self, status) + 1)
            if not self.tty:
                self._announce(self._status_text())
        else:
            return False
        self.render(force=True)
        return True

    def _announce(self, message: str) -> None:
        if self._line_visible:
            self.stream.write("\r\x1b[2K\n")
            self._line_visible = False
        self.stream.write(message + "\n")
        self.stream.flush()

    def _status_text(self) -> str:
        percent = self.completed / self.total * 100 if self.total else 0
        elapsed = max(0, int(time.monotonic() - self.started))
        elapsed_str = f"{elapsed // 60:02}:{elapsed % 60:02}"
        if self.completed and self.total > self.completed:
            remaining = int(elapsed / self.completed * (self.total - self.completed))
            eta = f" ETA {remaining // 60:02}:{remaining % 60:02}"
        else:
            eta = ""
        return (
            f"{self.completed}/{self.total or '?'} {percent:5.1f}% "
            f"| success {self.success} failure {self.failure} error {self.error} "
            f"| {self.task} | {elapsed_str}{eta}"
        )

    def render(self, *, force: bool = False) -> None:
        if not self.tty or not self.total:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 1:
            return
        self._last_render = now
        width = shutil.get_terminal_size(fallback=(100, 24)).columns
        bar_width = max(8, min(24, width - 70))
        filled = int(bar_width * self.completed / self.total)
        bar = self._color("━" * filled, "36") + self._color("╺" * (bar_width - filled), "90")
        counts = (
            f"{self._color('✓', '32')}{self.success} "
            f"{self._color('✗', '33')}{self.failure} "
            f"{self._color('!', '31')}{self.error}"
        )
        progress = f"{self.completed}/{self.total} {self.completed / self.total:5.1%}"
        elapsed = int(now - self.started)
        eta = (
            f" ETA {int(elapsed / self.completed * (self.total - self.completed))}s"
            if self.completed and self.completed < self.total
            else ""
        )
        suffix = f" | {elapsed}s{eta}"
        prefix_visible = (
            f"WAH [{'━' * filled}{'╺' * (bar_width - filled)}] "
            f"{progress} | ✓{self.success} ✗{self.failure} !{self.error} | "
        )
        task_width = max(0, width - 1 - len(prefix_visible) - len(suffix))
        task = self.task[:task_width]
        line = f"WAH [{bar}] {progress} | {counts} | {task}{suffix}"
        self.stream.write("\r\x1b[2K" + line)
        self.stream.flush()
        self._line_visible = True

    def finish(self) -> None:
        if self._line_visible:
            self.stream.write("\n")
            self.stream.flush()
            self._line_visible = False
