"""Run the shared indexed WFST decoder in WSL."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "../../../shared/tools/decode_wfst.py",
        "--binary PATH --graph TLG.fst --words WORDS --trials-dir DIRECTORY --meta TSV --output NEW_TSV",
    ))
