"""Rich terminal dashboard and clean redirected-output fallback for rollouts."""

from __future__ import annotations

import atexit
import math
import sys
import time
from typing import Any, TextIO

from .profiling import GLOBAL_PROFILER, RuntimeProfiler
from .telemetry import RolloutAggregator, TaskTelemetry, task_display_name


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "N/A"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _percent(value: float | None) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{100 * value:.1f}%"


def _number(value: Any, suffix: str = "") -> str:
    return (
        "N/A"
        if not isinstance(value, (int, float)) or not math.isfinite(value)
        else f"{value:.2f}{suffix}"
    )


class RolloutDashboard:
    """Render a control-side dashboard; workers never need terminal ownership."""

    def __init__(
        self,
        state: RolloutAggregator,
        *,
        enabled: bool | None = None,
        refresh_seconds: float = 1.0,
        stream: TextIO | None = None,
        screen: bool = True,
        profiler: RuntimeProfiler | None = None,
    ):
        if not math.isfinite(refresh_seconds) or refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be finite and positive")
        self.state = state
        self.stream = stream or sys.stdout
        self.requested = enabled
        self.disabled = enabled is False
        self.interactive = bool(self.stream.isatty()) and enabled is not False
        if enabled is True and not self.stream.isatty():
            self.interactive = False
        self.refresh_seconds = refresh_seconds
        self.screen = screen
        self.profiler = profiler or GLOBAL_PROFILER
        self._live = None
        self._console = None
        self._last_refresh = 0.0
        self._last_text_completed = -1
        self._last_text_phase = ""
        self._last_text_action = -1
        self._last_text_refresh = 0.0
        self._pending_display_created_ns: list[int] = []

    def start(self) -> None:
        if not self.interactive or self._live is not None:
            return
        try:
            from rich.console import Console
            from rich.live import Live
        except ImportError:
            if self.requested is True:
                raise RuntimeError(
                    "Rich dashboard was explicitly requested but Rich is unavailable in this Python."
                ) from None
            self.interactive = False
            return

        self._console = Console(file=self.stream, force_terminal=True)
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=max(1.0, 1.0 / self.refresh_seconds),
            screen=self.screen,
            transient=not self.screen,
            vertical_overflow="crop",
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()
        atexit.register(self.close)

    def close(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            finally:
                self._live = None
                try:
                    atexit.unregister(self.close)
                except Exception:  # pragma: no cover - interpreter shutdown
                    pass

    def update(self, events: list[dict[str, Any]], *, force: bool = False) -> None:
        accepted = [event for event in events if self.state.apply(event)]
        if self.disabled:
            return
        self._pending_display_created_ns.extend(
            created
            for event in accepted
            if isinstance((created := event.get("created_monotonic_ns")), int)
        )
        now = time.monotonic()
        if self.interactive:
            if self._live is None:
                self.start()
            if self._live is not None and (
                force or now - self._last_refresh >= self.refresh_seconds
            ):
                render_started = time.perf_counter_ns()
                self._live.update(self, refresh=True)
                self.profiler.record_ns(
                    "monitoring.terminal_render", time.perf_counter_ns() - render_started
                )
                self._last_refresh = now
                self._record_display_delay()
            return
        render_started = time.perf_counter_ns()
        rendered = self._text_update(accepted, force=force)
        if rendered:
            self.profiler.record_ns(
                "monitoring.terminal_render", time.perf_counter_ns() - render_started
            )
            self._record_display_delay()

    def _record_display_delay(self) -> None:
        displayed_ns = time.monotonic_ns()
        for created_ns in self._pending_display_created_ns:
            if displayed_ns >= created_ns:
                self.profiler.record_ns("monitoring.event_to_display", displayed_ns - created_ns)
        self._pending_display_created_ns.clear()

    def _text_update(self, events: list[dict[str, Any]], *, force: bool) -> bool:
        important = {
            "rollout_started",
            "server_starting",
            "server_ready",
            "task_started",
            "task_completed",
            "task_failed",
            "rollout_completed",
            "rollout_failed",
            "system_warning",
        }
        has_important = any(event.get("event") in important for event in events)
        now = time.monotonic()
        if (
            not force
            and not has_important
            and self.state.total_completed == self._last_text_completed
            and self.state.current.action_count == self._last_text_action
            and now - self._last_text_refresh < self.refresh_seconds
        ):
            return False
        total = self.state.total_planned
        completed = self.state.total_completed
        progress = completed / total if total else 0.0
        current = self.state.current
        current_task = self.state.tasks.get(current.task)
        task_progress = (
            f"{current_task.completed}/{current_task.planned}"
            if current_task is not None
            else "N/A"
        )
        action_progress = f"{current.action_count}/{current.max_actions or '?'}"
        eta = (
            "Estimating..." if self.state.eta_seconds is None else _duration(self.state.eta_seconds)
        )
        line = (
            f"[rollout] phase={self.state.phase} overall={completed}/{total or '?'} ({progress:.1%}) "
            f"task={current.task or 'N/A'} {task_progress} actions={action_progress} "
            f"success={self.state.total_successes} failure={self.state.total_failures} "
            f"error={self.state.total_errors} success_rate={_percent(self.state.micro_success_rate)} "
            f"episode_elapsed={_duration(current.live_elapsed_seconds)} "
            f"elapsed={_duration(self.state.elapsed_seconds)} eta={eta}"
        )
        print(line, file=self.stream, flush=True)
        for event in events:
            if event.get("event") == "system_warning":
                print(
                    f"[rollout warning] {event.get('message', 'unknown warning')}", file=self.stream
                )
        if self.state.stall_warning:
            print(f"[rollout warning] {self.state.stall_warning}", file=self.stream, flush=True)
        self._last_text_completed = completed
        self._last_text_phase = self.state.phase
        self._last_text_action = current.action_count
        self._last_text_refresh = now
        return True

    def __rich_console__(self, console, options):
        yield self._render(console.size.width, console.size.height)

    def _render(self, width: int, height: int):
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table
        from rich.text import Text

        total = max(self.state.total_planned, 1)
        progress = Progress(
            TextColumn("[bold]WAH Rollout"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            expand=True,
        )
        progress.add_task("Rollout", total=total, completed=self.state.total_completed)
        title = Text(
            f"Florence | {self.state.experiment_name}",
            style="bold cyan",
            no_wrap=True,
            overflow="ellipsis",
        )
        identity = Table.grid(expand=True, padding=(0, 1))
        identity.add_column(style="cyan", no_wrap=True)
        identity.add_column(overflow="ellipsis")
        identity.add_column(style="cyan", no_wrap=True)
        identity.add_column(justify="right", overflow="ellipsis")
        identity.add_row(
            "Checkpoint",
            self.state.checkpoint,
            "Identity",
            self.state.checkpoint_identity,
        )
        identity.add_row(
            "Policy",
            self.state.policy_type,
            "Sidecar",
            self.state.sidecar_type,
        )
        identity.add_row(
            "Phase",
            self.state.phase,
            "Horizon",
            str(self.state.execution_horizon),
        )
        identity.add_row(
            "Tasks / Episodes",
            f"{len(self.state.tasks)} / {self.state.total_completed}/{self.state.total_planned}",
            "Success / Failure / Error",
            f"{self.state.total_successes} / {self.state.total_failures} / {self.state.total_errors}",
        )
        identity.add_row(
            "Elapsed / ETA",
            f"{_duration(self.state.elapsed_seconds)} / "
            f"{_duration(self.state.eta_seconds) if self.state.eta_seconds is not None else 'Estimating...'}",
            "Macro / Micro",
            f"{_percent(self.state.macro_success_rate)} / {_percent(self.state.micro_success_rate)}",
        )

        parts: list[Any] = [title, progress, Panel(identity, title="Rollout")]
        active_task = self.state.tasks.get(self.state.current.task)
        if active_task is not None:
            task_progress = Progress(
                TextColumn(f"[bold]Current Task  {task_display_name(active_task.task)}"),
                BarColumn(),
                TextColumn("{task.completed:.0f}/{task.total:.0f}"),
                expand=True,
            )
            task_progress.add_task(
                active_task.task, total=max(active_task.planned, 1), completed=active_task.completed
            )
            parts.append(task_progress)
        if height >= 14:
            parts.append(self._task_table(compact=width < 105))
        if height >= 24:
            parts.append(self._episode_panel())
        if height >= 31 and width >= 70:
            parts.append(self._diagnostics_panel(compact=height < 39))
        return Group(*parts)

    def _task_table(self, *, compact: bool):
        from rich.table import Table

        table = Table(title="Tasks", expand=True, pad_edge=False)
        table.add_column("Task", overflow="ellipsis", ratio=2)
        table.add_column("Status", no_wrap=True)
        table.add_column("Episodes", justify="right")
        table.add_column("S/F/E", justify="right")
        table.add_column("Success", justify="right")
        if not compact:
            table.add_column("Mean Len", justify="right")
            table.add_column("Elapsed", justify="right")
        for task in self.state.tasks.values():
            cells = [
                task_display_name(task.task),
                task.status,
                f"{task.completed}/{task.planned}"
                + (f" ({task.unexecuted} not run)" if task.unexecuted else ""),
                f"{task.successes}/{task.failures}/{task.errors}",
                _percent(task.success_rate),
            ]
            if not compact:
                cells.extend((_number(task.mean_episode_length), _duration(task.elapsed_seconds)))
            table.add_row(*cells, style=_task_style(task))
        return table

    def _episode_panel(self):
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn
        from rich.table import Table

        current = self.state.current
        maximum = current.max_actions or max(current.action_count, 1)
        progress = Progress(
            TextColumn("Actions"),
            BarColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            expand=True,
        )
        progress.add_task("episode", total=maximum, completed=min(current.action_count, maximum))
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column()
        table.add_column(style="cyan")
        table.add_column(justify="right")
        index = "N/A" if current.episode_index is None else str(current.episode_index + 1)
        table.add_row(
            "Task / Episode",
            f"{task_display_name(current.task)} / {index}",
            "Seed",
            str(current.seed or "N/A"),
        )
        table.add_row("Phase", current.phase, "Horizon", str(current.execution_horizon))
        table.add_row(
            "Episode elapsed",
            _duration(current.live_elapsed_seconds),
            "Inference / FPS",
            f"{_number(current.inference_latency_ms, ' ms')} / {_number(current.simulation_fps)}",
        )
        warning = self.state.stall_warning
        content: list[Any] = [table, progress]
        if warning:
            from rich.text import Text

            content.append(Text(f"WARNING: {warning}", style="bold yellow"))
        return Panel(Group(*content), title="Current Episode")

    def _diagnostics_panel(self, *, compact: bool):
        from rich.panel import Panel
        from rich.table import Table

        values = self.state.diagnostics
        rows = [
            (
                "Server / Simulator",
                f"{values.get('server_status', 'N/A')} / {values.get('simulator_status', 'N/A')}",
            ),
            (
                "Server PID / exit",
                f"{values.get('server_pid', 'N/A')} / {values.get('server_exit_code', 'N/A')}",
            ),
            ("GPU used/free/total", _gpu_text(values)),
            ("GPU peak", _number(values.get("gpu_memory_peak_mib"), " MiB")),
            (
                "Inference mean / p95",
                f"{_number(_mean(self.state.inference_latencies_ms), ' ms')} / "
                f"{_number(self.state.percentile_latency(0.95), ' ms')}",
            ),
            (
                "Server initialization",
                _number(values.get("server_initialization_seconds"), " s"),
            ),
            ("Simulation FPS", _number(_mean(self.state.simulation_fps_samples))),
            (
                "GPU / CUDA visibility",
                f"{values.get('gpu_index', 'N/A')} {values.get('gpu_model', '')} / "
                f"{values.get('cuda_visible_devices', 'N/A')}",
            ),
            ("Python", str(values.get("python_executable", "N/A"))),
            (
                "Isaac Sim / Lab",
                f"{values.get('isaac_sim_version', 'N/A')} / {values.get('isaac_lab_version', 'N/A')}",
            ),
            ("Disk output / temp / root / shm", _disk_text(values.get("filesystems"))),
        ]
        if compact:
            rows = rows[:6]
        table = Table.grid(expand=True, padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="ellipsis")
        for label, value in rows:
            table.add_row(label, str(value))
        return Panel(table, title="System & Diagnostics")


def _task_style(task: TaskTelemetry) -> str:
    if task.status == "Error":
        return "red"
    if task.status == "Completed":
        return "green"
    if task.status in {"Initializing", "Running"}:
        return "cyan"
    return "dim"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _gpu_text(values: dict[str, Any]) -> str:
    used = values.get("gpu_memory_used_mib")
    free = values.get("gpu_memory_free_mib")
    total = values.get("gpu_memory_total_mib")
    return " / ".join(_number(item, " MiB") for item in (used, free, total))


def _disk_text(filesystems: Any) -> str:
    if not isinstance(filesystems, dict):
        return "N/A"
    parts = []
    for name in ("output", "temp", "root", "dev_shm"):
        row = filesystems.get(name)
        if isinstance(row, dict):
            inodes = row.get("free_inodes")
            inode_text = f", {int(inodes):,} inodes" if isinstance(inodes, int) else ""
            parts.append(f"{name}={_number(row.get('free_gib'), ' GiB')}{inode_text}")
    return " | ".join(parts) or "N/A"
