"""CPU-only regression tests for live UniVTAC rollout progress events."""

import io
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

from univtac_progress import EVENT_PREFIX, RolloutProgress  # noqa: E402


def _event(kind: str, **kwargs: object) -> str:
    return EVENT_PREFIX + json.dumps({"kind": kind, **kwargs})


def test_plain_progress_counts_success_failure_and_error() -> None:
    output = io.StringIO()
    progress = RolloutProgress(output, tty=False)
    assert not progress.consume("[rollout-worker] phase=gr00t_inference: starting")
    assert progress.consume(_event("start", total=3))
    assert progress.consume(_event("task", task="lift_can", index=1, task_count=1))
    for status in ("success", "failure", "error"):
        assert progress.consume(_event("episode", task="lift_can", status=status))
    progress.finish()

    assert (progress.completed, progress.total) == (3, 3)
    assert (progress.success, progress.failure, progress.error) == (1, 1, 1)
    text = output.getvalue()
    assert "WAH rollout started" in text
    assert "3/3 100.0%" in text
    assert "success 1 failure 1 error 1" in text
    assert "\x1b[" not in text


def test_invalid_or_unrecognized_events_do_not_corrupt_progress() -> None:
    progress = RolloutProgress(io.StringIO(), tty=False)
    for payload in (
        "not json",
        "[]",
        '{"kind":"start","total":0}',
        '{"kind":"episode","status":"success"}',
    ):
        assert not progress.consume(EVENT_PREFIX + payload)
    assert progress.completed == 0
    assert progress.total == 0
    assert progress.consume(_event("start", total=1))
    assert not progress.consume(_event("episode", status="unknown"))
    assert progress.consume(_event("episode", status="success"))
    assert progress.completed == 1


def test_interactive_progress_renders_single_line_and_finishes() -> None:
    output = io.StringIO()
    progress = RolloutProgress(output, tty=True)
    progress.color = False
    progress.consume(_event("start", total=2))
    progress.consume(_event("episode", status="failure"))
    progress.finish()
    assert "\r\x1b[2K" in output.getvalue()
    assert "1/2 50.0%" in output.getvalue()
    assert output.getvalue().endswith("\n")
