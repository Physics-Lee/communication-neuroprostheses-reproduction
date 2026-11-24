"""Prepare the released sentence data for the Kunz models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "convert_kunz.py",
        "[no arguments]",
    ))
