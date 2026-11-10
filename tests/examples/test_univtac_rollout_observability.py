from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


UNIVTAC_ROOT = Path(__file__).parents[2] / "examples" / "UniVTAC"
sys.path.insert(0, str(UNIVTAC_ROOT))
sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from run_closed_loop_eval import _stream_process  # noqa: E402
from univtac_deployment import gpu_memory  # noqa: E402
from univtac_rollout.dashboard import RolloutDashboard  # noqa: E402
from univtac_rollout.diagnostics import collect_filesystems, dependency_preflight  # noqa: E402
from univtac_rollout.monitoring import AsyncSampler  # noqa: E402
from univtac_rollout.notifications import ROLLOUT_MESSAGES, RolloutTrainctlReporter, rollout_message  # noqa: E402
from univtac_rollout.profiling import RuntimeProfiler  # noqa: E402
from univtac_rollout.telemetry import (  # noqa: E402
    EpisodeProgressEmitter,
    EventReader,
    EventWriter,
    RolloutAggregator,
)


def event(kind: str, event_id: str, **fields):
    return {"event": kind, "event_id": event_id, **fields}


def started(**fields):
    return event(
        "rollout_started",
        "start",
        tasks=["lift_can", "lift_bottle"],
        episodes_per_task=2,
        execution_horizon=4,
        checkpoint="checkpoint-10",
        **fields,
    )


def completed(task: str, index: int, status: str, reason: str, **fields):
    success = True if status == "success" else False if status == "failure" else None
    return event(
        "episode_completed",
        f"complete:{task}:{index}",
        task=task,
        episode_index=index,
        seed=100 + index,
        status=status,
        success=success,
        termination_reason=reason,
        num_steps=10,
        elapsed_seconds=2.0,
        **fields,
    )


def test_overall_and_task_counts_update_after_each_episode() -> None:
    state = RolloutAggregator()
    state.apply(started())
    state.apply(completed("lift_can", 0, "success", "success"))
    task = state.tasks["lift_can"]
    assert state.total_planned == 4
    assert state.total_completed == 1
    assert (task.successes, task.failures, task.errors) == (1, 0, 0)
    assert task.success_rate == 1.0


def test_error_excluded_from_success_rate_and_zero_valid_is_na() -> None:
    state = RolloutAggregator(["lift_can"], 2)
    state.apply(completed("lift_can", 0, "error", "invalid_action"))
    assert state.micro_success_rate is None
    assert state.tasks["lift_can"].success_rate is None
    state.apply(completed("lift_can", 1, "failure", "step_limit"))
    assert state.micro_success_rate == 0.0
    assert state.error_rate == 0.5


def test_duplicate_event_and_duplicate_episode_are_not_counted() -> None:
    state = RolloutAggregator(["lift_can"], 1)
    row = completed("lift_can", 0, "success", "success")
    assert state.apply(row)
    assert not state.apply(row)
    replay = dict(row, event_id="different-id")
    assert not state.apply(replay)
    assert state.total_successes == 1


def test_out_of_order_events_remain_consistent() -> None:
    state = RolloutAggregator()
    state.apply(completed("lift_can", 0, "failure", "early_stop"))
    state.apply(started())
    state.apply(event("episode_started", "late-start", task="lift_can", episode_index=0, seed=1))
    assert state.total_completed == 1
    assert state.total_failures == 1
    assert state.tasks["lift_can"].status == "Running"
    assert state.current.episode_index is None


def test_reader_ignores_partial_and_malformed_records(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, source="test")
    writer.emit("rollout_started", event_id="one")
    with path.open("ab") as stream:
        stream.write(b"{malformed}\n")
        stream.write(b'{"event":"episode_started"')
    reader = EventReader(path)
    assert [row["event_id"] for row in reader.read()] == ["one"]
    assert reader.malformed_records == 1
    assert reader.read() == []


def test_non_tty_dashboard_has_no_ansi_sequences() -> None:
    state = RolloutAggregator()
    stream = StringIO()
    dashboard = RolloutDashboard(state, stream=stream)
    dashboard.update([started()], force=True)
    dashboard.update([completed("lift_can", 0, "success", "success")], force=True)
    dashboard.close()
    assert "\x1b[" not in stream.getvalue()
    assert "success=1" in stream.getvalue()


def test_episode_action_progress_does_not_advance_overall_count() -> None:
    state = RolloutAggregator(["lift_can"], 2)
    state.apply(
        event("episode_started", "episode:0", task="lift_can", episode_index=0, max_actions=100)
    )
    state.apply(
        event(
            "episode_progress",
            "progress:0:26",
            task="lift_can",
            episode_index=0,
            action_count=26,
            max_actions=100,
            phase="Executing action",
        )
    )
    assert state.current.action_count == 26
    assert state.tasks["lift_can"].completed == 0
    assert state.total_completed == 0


def test_inference_pending_keeps_last_action_and_live_elapsed(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("univtac_rollout.telemetry.time.monotonic", lambda: now[0])
    state = RolloutAggregator(["lift_can"], 1)
    state.apply(
        event(
            "episode_started",
            "episode:0",
            task="lift_can",
            episode_index=0,
            max_actions=100,
            monotonic_seconds=90.0,
        )
    )
    state.apply(
        event(
            "inference_started",
            "inference:0",
            task="lift_can",
            episode_index=0,
            action_count=12,
            phase="GR00T inference",
            monotonic_seconds=100.0,
        )
    )
    now[0] = 107.0
    assert state.current.action_count == 12
    assert state.current.live_elapsed_seconds == 17.0
    assert state.current.phase == "GR00T inference"


def test_episode_progress_emitter_throttles_and_heartbeats_during_inference(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, source="worker")
    emitter = EpisodeProgressEmitter(
        writer,
        interval_seconds=0.05,
        task="lift_can",
        episode_index=0,
        max_actions=100,
    )
    try:
        assert emitter.update("inference_started", action_count=4, phase="GR00T inference")
        assert not emitter.update("episode_progress", action_count=5, phase="Executing action")
        time.sleep(0.08)
    finally:
        emitter.close()
        writer.close()
    rows = EventReader(path).read()
    assert len(rows) == 2
    assert rows[0]["event"] == "inference_started"
    assert rows[1]["action_count"] == 5


def test_delayed_progress_after_completion_is_ignored() -> None:
    state = RolloutAggregator(["lift_can"], 1)
    state.apply(completed("lift_can", 0, "success", "success"))
    accepted = state.apply(
        event(
            "episode_progress",
            "late-progress",
            task="lift_can",
            episode_index=0,
            action_count=999,
        )
    )
    assert not accepted
    assert state.total_completed == 1
    assert state.current.action_count == 0


def test_slow_gpu_sampler_does_not_block_progress_processing() -> None:
    release = threading.Event()

    def slow_sample():
        release.wait(0.2)
        return [{"used_mib": 10}]

    sampler = AsyncSampler(
        slow_sample, interval_seconds=1.0, stage="monitoring.gpu", profiler=RuntimeProfiler()
    )
    sampler.start()
    started_at = time.perf_counter()
    state = RolloutAggregator(["lift_can"], 1)
    state.apply(event("episode_started", "start:0", task="lift_can", episode_index=0))
    state.apply(
        event(
            "episode_progress",
            "progress:0",
            task="lift_can",
            episode_index=0,
            action_count=1,
        )
    )
    assert time.perf_counter() - started_at < 0.05
    release.set()
    sampler.close(timeout=1.0)
    assert sampler.snapshot()[0] == [{"used_mib": 10}]


def test_gpu_sampler_timeout_or_failure_keeps_last_valid_sample() -> None:
    call_count = 0

    def sample():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [{"used_mib": 10}]
        raise RuntimeError("nvidia-smi timeout")

    sampler = AsyncSampler(
        sample, interval_seconds=0.02, stage="monitoring.gpu", profiler=RuntimeProfiler()
    )
    sampler.start()
    time.sleep(0.06)
    sampler.close()
    value, error, _ = sampler.snapshot()
    assert value == [{"used_mib": 10}]
    assert error is not None and "timeout" in error


def test_nvidia_smi_timeout_is_nonfatal(monkeypatch) -> None:
    monkeypatch.setattr(
        "univtac_deployment.subprocess.run",
        Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 0.01)),
    )
    assert gpu_memory(timeout_seconds=0.01) == []


def test_stalled_progress_warning_identifies_phase(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr("univtac_rollout.telemetry.time.monotonic", lambda: now[0])
    state = RolloutAggregator(["lift_can"], 1, stall_warning_seconds=30.0)
    state.apply(
        event(
            "episode_started",
            "start:0",
            task="lift_can",
            episode_index=0,
            monotonic_seconds=60.0,
        )
    )
    state.apply(
        event(
            "inference_started",
            "inference:0",
            task="lift_can",
            episode_index=0,
            action_count=7,
            phase="GR00T inference",
            monotonic_seconds=65.0,
        )
    )
    warning = state.stall_warning
    assert warning is not None
    assert "GR00T inference" in warning
    assert "last_action=7" in warning


def test_eta_uses_episode_durations_and_needs_enough_samples() -> None:
    state = RolloutAggregator(["lift_can", "lift_bottle"], 3)
    assert state.eta_seconds is None
    for index, duration in enumerate((10.0, 12.0, 11.0)):
        state.apply(
            event(
                "episode_completed",
                f"complete:lift_can:{index}",
                task="lift_can",
                episode_index=index,
                status="success",
                success=True,
                termination_reason="success",
                num_steps=10,
                elapsed_seconds=duration,
            )
        )
    assert state.eta_seconds == pytest.approx(33.0)


def test_event_writer_cleanup_and_bounded_partial_record(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, source="test")
    writer.emit("episode_started", event_id="one")
    writer.close()
    writer.close()
    with path.open("ab") as stream:
        stream.write(b"x" * 1_000)
    reader = EventReader(path, max_record_bytes=512)
    assert [row["event_id"] for row in reader.read()] == ["one"]
    assert reader.malformed_records == 1


def test_runtime_profiler_reports_required_summary_statistics() -> None:
    profiler = RuntimeProfiler()
    for value in (1_000_000, 2_000_000, 10_000_000):
        profiler.record_ns("simulation.action_execution", value)
    row = profiler.summary()["simulation.action_execution"]
    assert row["count"] == 3
    assert row["mean_ms"] == pytest.approx(13 / 3)
    assert row["p50_ms"] == 2.0
    assert row["p95_ms"] == 10.0
    assert row["max_ms"] == 10.0
    assert row["total_ms"] == 13.0


def test_worker_monitor_ticks_while_child_has_no_stdout(tmp_path: Path) -> None:
    ticks: list[float] = []
    outcome = _stream_process(
        [sys.executable, "-c", "import time; time.sleep(0.35)"],
        cwd=tmp_path,
        log_path=tmp_path / "worker.log",
        on_output=lambda: ticks.append(time.monotonic()),
        echo=False,
        poll_seconds=0.05,
    )
    assert outcome.return_code == 0
    assert len(ticks) >= 4
    assert "worker process finished" in (tmp_path / "worker.log").read_text(encoding="utf-8")


class TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_dashboard_starts_and_restores_live_display() -> None:
    stream = TTYBuffer()
    dashboard = RolloutDashboard(RolloutAggregator(), stream=stream, screen=False)
    dashboard.start()
    dashboard.update([started()], force=True)
    dashboard.close()
    assert dashboard._live is None


def test_task_initialization_failure_tracks_unexecuted_not_robot_failure() -> None:
    state = RolloutAggregator(["lift_can"], 100)
    state.apply(
        event(
            "task_failed",
            "task-failed",
            task="lift_can",
            error_category="worker_failure",
            unexecuted_episodes=100,
        )
    )
    assert state.tasks["lift_can"].status == "Error"
    assert state.tasks["lift_can"].unexecuted == 100
    assert state.total_failures == 0
    assert state.total_completed == 0


def test_trainctl_lifecycle_and_task_notifications_are_exactly_once() -> None:
    tracker = SimpleNamespace(
        init=Mock(), define_metrics=Mock(), log=Mock(), notify=Mock(), finish=Mock(), fail=Mock()
    )
    state = RolloutAggregator()
    start = started()
    state.apply(start)
    reporter = RolloutTrainctlReporter(tracker, state, name="rollout", message_mode="koyori")
    reporter.initialize()
    reporter.process(start)
    reporter.process(start)
    for index, status in enumerate(("success", "failure")):
        row = completed(
            "lift_can", index, status, "success" if status == "success" else "step_limit"
        )
        state.apply(row)
        reporter.process(row)
    task_done = event("task_completed", "task-done", task="lift_can")
    state.apply(task_done)
    reporter.process(task_done)
    reporter.process(task_done)
    done = event("rollout_completed", "done")
    state.apply(done)
    reporter.process(done)
    reporter.process(done)
    titles = [call.kwargs["message"].splitlines()[0] for call in tracker.notify.call_args_list]
    assert titles.count("🤖 Rollout Started") == 1
    assert titles.count("📈 Rollout Halfway") == 1
    assert titles.count("✅ Task Completed: lift_can") == 1
    assert titles.count("✅ Rollout Completed") == 1
    tracker.finish.assert_called_once()


def test_repeated_errors_are_throttled_and_critical_error_is_immediate(monkeypatch) -> None:
    tracker = SimpleNamespace(
        init=Mock(), define_metrics=Mock(), log=Mock(), notify=Mock(), finish=Mock(), fail=Mock()
    )
    state = RolloutAggregator(["lift_can"], 10)
    reporter = RolloutTrainctlReporter(
        tracker, state, name="rollout", notify_halfway=False, error_throttle_seconds=100
    )
    for index in range(3):
        row = completed("lift_can", index, "error", "invalid_action")
        state.apply(row)
        reporter.process(row)
    assert (
        sum("Rollout Error" in call.kwargs["message"] for call in tracker.notify.call_args_list)
        == 1
    )
    failure = event("rollout_failed", "failed", message="server crashed")
    state.apply(failure)
    reporter.process(failure)
    assert tracker.notify.call_args_list[-1].kwargs["message"].startswith("🚨 Rollout Aborted")
    tracker.fail.assert_called_once()


@pytest.mark.parametrize("mode", ["koyori", "roboco", "neuro", "sopia"])
@pytest.mark.parametrize("style", [1, 2, 3])
def test_all_rollout_persona_styles_are_available(mode: str, style: int) -> None:
    for kind in ("start", "halfway", "completed", "error"):
        assert rollout_message(mode, style, kind) == ROLLOUT_MESSAGES[mode][style][kind]


def test_notification_transport_failure_does_not_interrupt_rollout() -> None:
    failing = Mock(side_effect=RuntimeError("secret transport details"))
    tracker = SimpleNamespace(
        init=Mock(), define_metrics=Mock(), log=Mock(), notify=failing, finish=Mock(), fail=Mock()
    )
    state = RolloutAggregator()
    row = started()
    state.apply(row)
    reporter = RolloutTrainctlReporter(tracker, state, name="rollout")
    reporter.process(row)


def test_filesystem_diagnostics_distinguish_devices(tmp_path: Path) -> None:
    diagnostics = collect_filesystems(tmp_path)
    assert {"output", "temp", "root"}.issubset(diagnostics)
    assert all("device" in row and "free_gib" in row for row in diagnostics.values())


def test_missing_ffmpeg_is_actionable(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("univtac_rollout.diagnostics.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="FFmpeg executable unavailable"):
        dependency_preflight(
            output_dir=tmp_path / "output",
            checkpoint=checkpoint,
            server_host="127.0.0.1",
            server_port=0,
            require_server_port=False,
            sensor_contract=None,
            python_packages=(),
        )


def test_disk_full_preflight_reports_temporary_filesystem(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    real = collect_filesystems(tmp_path)
    real["temp"]["free_gib"] = 0.01
    monkeypatch.setattr("univtac_rollout.diagnostics.collect_filesystems", lambda output: real)
    monkeypatch.setattr("univtac_rollout.diagnostics.shutil.which", lambda name: "/usr/bin/ffmpeg")
    with pytest.raises(OSError, match="temporary filesystem"):
        dependency_preflight(
            output_dir=tmp_path / "output",
            checkpoint=checkpoint,
            server_host="127.0.0.1",
            server_port=0,
            require_server_port=False,
            sensor_contract=None,
            python_packages=(),
        )
