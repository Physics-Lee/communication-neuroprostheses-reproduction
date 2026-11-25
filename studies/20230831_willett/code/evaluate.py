"""Compute greedy phoneme error rates for the test split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "eval_btot24.py",
        "RUN_DIRECTORY [--session SESSION] [--output-dir NEW_DIRECTORY]",
    ))
