"""Decode speech logits with the KenLM 3-gram model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "decode_words_btot24_v2.py",
        "[--alpha WEIGHT] [--beta BONUS] [--beam SIZE] [--limit LIMIT] [--workers WORKERS] [--out CSV]",
    ))
