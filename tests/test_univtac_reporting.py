from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


UNIVTAC_ROOT = Path(__file__).parents[1] / "examples" / "UniVTAC"
sys.path.insert(0, str(UNIVTAC_ROOT))

from univtac_rollout.core import EpisodeResult, aggregate_results  # noqa: E402
from univtac_rollout.reporting.analytics import (  # noqa: E402
    comparison,
    error_analysis,
    protocol_warnings,
    run_summary,
)
from univtac_rollout.reporting.generator import generate_report  # noqa: E402
from univtac_rollout.reporting.parser import load_run  # noqa: E402


def episode(
    task: str,
    index: int,
    success: bool | None,
    reason: str,
    *,
    error: str = "",
    video: str = "",
) -> dict:
    return {
        "task": task,
        "seed": 100 + index,
        "episode_index": index,
        "success": success,
        "num_steps": 10 + index,
        "sim_steps": 20 + index,
        "execution_horizon": 2,
        "checkpoint": "/models/checkpoint-42",
        "instruction": "perform task",
        "elapsed_seconds": 1.5 + index,
        "termination_reason": reason,
        "video_path": video,
        "error": error,
    }


def write_run(
    root: Path,
    rows: list[dict],
    *,
    tasks: list[str] | None = None,
    episodes_per_task: int = 3,
    metadata: dict | None = None,
) -> Path:
    root.mkdir(parents=True)
    tasks = tasks or sorted({row["task"] for row in rows}) or ["lift_can"]
    payload = {
        "checkpoint": "/models/checkpoint-42",
        "execution_horizon": 2,
        "episodes_per_task": episodes_per_task,
        "task_order": tasks,
        "summary": {},
        "episodes": rows,
        "metadata": {"seed_start": 100, "evaluation_status": "completed", **(metadata or {})},
    }
    (root / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("rows", "success_rate", "error_rate", "overall"),
    [
        ([episode("lift_can", 0, True, "success")], 1.0, 0.0, 1.0),
        ([episode("lift_can", 0, False, "step_limit")], 0.0, 0.0, 0.0),
        ([episode("lift_can", 0, None, "invalid_action")], None, 1.0, 0.0),
        (
            [
                episode("lift_can", 0, True, "success"),
                episode("lift_can", 1, False, "early_stop"),
                episode("lift_can", 2, None, "exception"),
            ],
            0.5,
            1 / 3,
            1 / 3,
        ),
    ],
)
def test_outcome_metrics_follow_core_semantics(
    tmp_path: Path,
    rows: list[dict],
    success_rate: float | None,
    error_rate: float,
    overall: float,
) -> None:
    run = load_run(write_run(tmp_path / "run", rows, episodes_per_task=len(rows)))
    summary = run_summary(run)
    assert summary["success_rate"] == success_rate
    assert summary["error_rate"] == pytest.approx(error_rate)
    assert summary["overall_success_fraction"] == pytest.approx(overall)
    core_rows = [EpisodeResult(**row) for row in rows]
    core = aggregate_results(core_rows, ["lift_can"])
    assert summary["success_rate"] == core["micro_average"]
    assert summary["error_rate"] == core["error_rate"]


def test_missing_metrics_are_none_for_empty_rollout(tmp_path: Path) -> None:
    run = load_run(write_run(tmp_path / "empty", [], episodes_per_task=2))
    summary = run_summary(run)
    assert summary["success_rate"] is None
    assert summary["error_rate"] is None
    assert summary["macro_success_rate"] is None
    assert summary["coverage"] == 0.0
    assert summary["unexecuted"] == 2


def test_summary_is_authoritative_and_episode_rows_are_not_duplicated(tmp_path: Path) -> None:
    root = write_run(tmp_path / "run", [episode("lift_can", 0, True, "success")])
    with (root / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(episode("lift_can", 1, False, "step_limit"))
        )
        writer.writeheader()
        writer.writerow(episode("lift_can", 1, False, "step_limit"))
    workers = root / "logs" / "worker_results"
    workers.mkdir(parents=True)
    (workers / "lift_can.json").write_text(
        json.dumps({"episodes": [episode("lift_can", 2, None, "worker_failure")]}),
        encoding="utf-8",
    )
    run = load_run(root)
    assert [(row.episode_index, row.outcome) for row in run.episodes] == [(0, "success")]
    assert run.source == "summary.json"


def test_incomplete_worker_fallback_is_deduplicated(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    workers = root / "logs" / "worker_results"
    workers.mkdir(parents=True)
    row = episode("lift_can", 0, None, "worker_failure", error="worker crashed")
    (workers / "a.json").write_text(json.dumps({"episodes": [row, row]}), encoding="utf-8")
    run = load_run(root)
    assert len(run.episodes) == 1
    assert run.episodes[0].outcome == "error"
    assert "fallback" in run.source


def test_video_links_resolve_and_paths_outside_root_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    rows = [
        episode("lift_can", 0, True, "success", video="videos/a.mp4"),
        episode("lift_can", 1, False, "step_limit", video="../secret.mp4"),
    ]
    write_run(root, rows, episodes_per_task=2)
    (root / "videos").mkdir()
    (root / "videos" / "a.mp4").write_bytes(b"mp4")
    (tmp_path / "secret.mp4").write_bytes(b"secret")
    output = tmp_path / "reports" / "index.html"
    run = load_run(root, output_dir=output.parent)
    assert run.episodes[0].video_available is True
    assert run.episodes[0].video_url == "../run/videos/a.mp4"
    assert run.episodes[1].video_available is False
    assert run.episodes[1].video_url == ""


def test_malformed_jsonl_is_ignored_and_latency_can_be_missing(tmp_path: Path) -> None:
    root = write_run(tmp_path / "run", [episode("lift_can", 0, True, "success")])
    logs = root / "logs"
    logs.mkdir()
    (logs / "rollout_events.jsonl").write_text(
        '{"event":"rollout_started","monotonic_seconds":1}\nnot-json\n'
        '{"event":"rollout_completed","monotonic_seconds":4}\n',
        encoding="utf-8",
    )
    run = load_run(root)
    assert run.malformed_telemetry_records == 1
    assert run.wall_clock_seconds == 3.0
    assert run.latency == []


def test_error_diagnostics_extract_recorded_joint_values_only(tmp_path: Path) -> None:
    text = (
        "GR00T qpos action exceeds UniVTAC articulation limits: dof 7: value=0.051, range=[0, 0.04]"
    )
    root = write_run(
        tmp_path / "run",
        [episode("lift_can", 0, None, "invalid_action", error=text)],
        episodes_per_task=1,
    )
    run = load_run(root)
    violation = run.episodes[0].diagnostics["joint_limit_violations"][0]
    assert violation == {
        "dof": 7,
        "predicted_qpos": 0.051,
        "lower": 0.0,
        "upper": 0.04,
        "violation_magnitude": pytest.approx(0.011),
    }
    analysis = error_analysis(run)
    assert analysis["by_category"] == {"joint_limit_violation": 1}
    assert analysis["by_domain"] == {"invalid model output": 1}


def test_comparison_matches_task_seed_and_warns_on_protocol_mismatch(tmp_path: Path) -> None:
    first = load_run(
        write_run(
            tmp_path / "first",
            [episode("lift_can", 0, True, "success")],
            episodes_per_task=1,
            metadata={"instruction_type": "seen"},
        ),
        label="Baseline",
    )
    second_row = episode("lift_can", 0, None, "invalid_action", error="NaN in action")
    second = load_run(
        write_run(
            tmp_path / "second",
            [second_row],
            episodes_per_task=2,
            metadata={"instruction_type": "unseen"},
        ),
        label="WAH",
    )
    result = comparison([first, second])
    assert result["paired"] is True
    assert result["matched"][0]["transition"] == "success → error"
    warnings = protocol_warnings([first, second])
    assert any("episode counts" in warning for warning in warnings)
    assert any("instruction types" in warning for warning in warnings)


def test_html_is_offline_escaped_and_handles_dangerous_error(tmp_path: Path) -> None:
    dangerous = '</script><script>alert("xss")</script><img src=x onerror=alert(1)>'
    root = write_run(
        tmp_path / "run",
        [episode("lift_can", 0, None, "exception", error=dangerous)],
        episodes_per_task=1,
    )
    destination = generate_report([root], tmp_path / "report" / "index.html")
    html = destination.read_text(encoding="utf-8")
    assert "WAH Rollout Analyzer" in html
    assert "https://" not in html
    assert "<script>alert" not in html
    assert "\\u003c/script>" in html
    assert "Detailed action diagnostics were not recorded" in html


def test_multi_run_report_generation(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", [episode("lift_can", 0, True, "success")], episodes_per_task=1)
    b = write_run(
        tmp_path / "b",
        [episode("lift_can", 0, False, "step_limit")],
        episodes_per_task=1,
    )
    output = generate_report([a, b], tmp_path / "compare", labels=["GR00T", "WAH"])
    assert output.name == "index.html"
    text = output.read_text(encoding="utf-8")
    assert '"paired":true' in text
    assert "GR00T" in text and "WAH" in text


def test_missing_optional_metadata_and_video_do_not_block_html(tmp_path: Path) -> None:
    root = write_run(
        tmp_path / "minimal",
        [episode("lift_can", 0, False, "step_limit", video="videos/missing.mp4")],
        episodes_per_task=1,
        metadata={},
    )
    run = load_run(root)
    assert run.checkpoint_identity is None
    assert run.policy_type is None
    assert run.episodes[0].video_available is False
    assert generate_report([root], tmp_path / "minimal.html").is_file()


def test_cli_defaults_to_report_inside_single_run(tmp_path: Path) -> None:
    root = write_run(
        tmp_path / "cli-run",
        [episode("lift_can", 0, True, "success")],
        episodes_per_task=1,
    )
    script = Path(__file__).parents[1] / "scripts" / "report_univtac_rollout.py"
    result = subprocess.run(
        [sys.executable, str(script), "--run", str(root)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "report" / "index.html").is_file()
    assert "http://localhost:8000/report/index.html" in result.stdout
