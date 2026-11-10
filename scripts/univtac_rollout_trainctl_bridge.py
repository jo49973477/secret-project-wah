#!/usr/bin/env python3
"""Control-side trainctl SDK bridge for simulator-isolated rollout workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "UniVTAC"))

from univtac_rollout.notifications import RolloutTrainctlReporter  # noqa: E402
from univtac_rollout.telemetry import EventReader, EventWriter, RolloutAggregator  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--message-mode",
        choices=("standard", "koyori", "roboco", "neuro", "sopia", "sophia"),
        default="standard",
    )
    parser.add_argument("--message-style", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--notify-task-complete", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--notify-halfway", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-errors", action=argparse.BooleanOptionalAction, default=True)
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a rollout command is required after --")
    return args, command


def run(args: argparse.Namespace, command: list[str]) -> int:
    if not os.environ.get("TRAINCTL_RUN_ID") or not os.environ.get("TRAINCTL_SOCKET"):
        raise RuntimeError(
            "trainctl was requested, but its SDK endpoint is unavailable. "
            "Inspect the trainctl supervisor startup log and Unix-socket permissions."
        )
    from trainctl import tracker

    reader = EventReader(args.event_log)
    writer = EventWriter(args.event_log, source="trainctl-bridge")
    state = RolloutAggregator()
    reporter = RolloutTrainctlReporter(
        tracker,
        state,
        name=args.name,
        message_mode=args.message_mode,
        message_style=args.message_style,
        notify_task_complete=args.notify_task_complete,
        notify_halfway=args.notify_halfway,
        notify_errors=args.notify_errors,
    )
    initialized = False
    process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), start_new_session=True)

    def stop(signum: int, frame: object) -> None:
        if process.poll() is None:
            os.killpg(process.pid, signum)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        while process.poll() is None:
            for event in reader.read():
                accepted = state.apply(event)
                if not accepted:
                    continue
                if event.get("event") == "rollout_started" and not initialized:
                    reporter.initialize()
                    initialized = True
                if initialized:
                    reporter.process(event)
            time.sleep(0.2)
        code = process.wait()
        events = reader.read()
        for event in events:
            if state.apply(event):
                if event.get("event") == "rollout_started" and not initialized:
                    reporter.initialize()
                    initialized = True
                if initialized:
                    reporter.process(event)
        if code != 0 and state.status not in {"Failed", "Completed"}:
            event = writer.emit(
                "rollout_failed",
                event_id="rollout-failed",
                flush=True,
                error_category="rollout_process_failure",
                message=f"Rollout process exited with code {code}",
                exit_code=code,
                phase=state.phase,
            )
            state.apply(event)
            if initialized:
                reporter.process(event)
        return code
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    namespace, rollout_command = parse_args()
    raise SystemExit(run(namespace, rollout_command))
