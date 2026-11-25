"""Train the joint-session typing model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "pytorch_port/train_multisession.py",
        "--participant {t17,t18} [--data-root DIRECTORY] [--out-root RUN_DIRECTORY] [--batches BATCHES]",
    ))
