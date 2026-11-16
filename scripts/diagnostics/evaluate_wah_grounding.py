#!/usr/bin/env python3
"""Training-free CLI for WAH experiments 7.1.1 and 7.1.2."""

from pathlib import Path
import sys


# Work from a source checkout without requiring an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gr00t.experiment.wah_diagnostics import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
