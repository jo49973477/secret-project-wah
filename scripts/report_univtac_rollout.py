#!/usr/bin/env python3
"""Generate an offline WAH Rollout Analyzer report from UniVTAC artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.parse import quote
import webbrowser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UNIVTAC_ROOT = REPOSITORY_ROOT / "examples" / "UniVTAC"
sys.path.insert(0, str(UNIVTAC_ROOT))

from univtac_rollout.reporting import generate_report  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Rollout result directory. Repeat to compare 2–5 runs.",
    )
    value.add_argument(
        "--label",
        action="append",
        help="Readable run label. When supplied, provide one per --run in the same order.",
    )
    value.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (or directory). Defaults to RUN/report/index.html for one run.",
    )
    value.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated report in the default browser.",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if len(args.run) > 5:
        raise SystemExit("At most five runs may be compared in one report.")
    if args.label is not None and len(args.label) != len(args.run):
        raise SystemExit("Provide exactly one --label for every --run, or omit all labels.")
    if args.output is None:
        if len(args.run) != 1:
            raise SystemExit("--output is required when comparing multiple runs.")
        output = args.run[0].expanduser().resolve() / "report" / "index.html"
    else:
        output = args.output
    destination = generate_report(args.run, output, labels=args.label)
    print(f"WAH Rollout Analyzer: {destination}")
    resolved_runs = [path.expanduser().resolve() for path in args.run]
    try:
        serve_root = Path(os.path.commonpath([destination.parent, *resolved_runs]))
    except ValueError:
        serve_root = destination.parent
    if serve_root == Path(serve_root.anchor):
        serve_root = destination.parent
    report_url = quote(destination.relative_to(serve_root).as_posix())
    print(
        "If file:// video playback is restricted, serve the common report/run parent with: "
        f"python -m http.server --directory {serve_root} 8000\n"
        f"Then open http://localhost:8000/{report_url}"
    )
    if args.open_browser:
        webbrowser.open(destination.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
