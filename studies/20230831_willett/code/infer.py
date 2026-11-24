"""Export speech model logits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "dump_btot24_logits.py",
        "RUN_DIRECTORY [LOGITS_FILE]",
    ))
