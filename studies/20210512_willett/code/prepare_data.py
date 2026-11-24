"""Prepare the handwriting model's memory-mapped synthetic data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "step3_make_synthetic.py",
        "[--phase {snippets,generate,both}] [--dataDir DIRECTORY] [--cvPart PARTITION] [--nParallel WORKERS]",
    ))
