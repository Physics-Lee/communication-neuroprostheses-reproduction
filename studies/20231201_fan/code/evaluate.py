"""Replay CORP recalibration and score matched held-out trials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "reproduce_paired_wer.py",
        "--study STUDY_DIRECTORY --output NEW_DIRECTORY [--sessions COUNT] [--steps STEPS] [--backend {tensorflow,torch-cuda}]",
    ))
