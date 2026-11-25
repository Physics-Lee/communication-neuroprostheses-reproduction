"""Prepare Willett 2023 competition data for training and evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "convert_btot24.py",
        "[--session SESSION] [--split {train,test}] [--output-dir NEW_DIRECTORY]",
    ))
