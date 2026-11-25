"""Evaluate Mel correlations over full synthesized trials."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_commands import run_command
from study_paths import study_path


if __name__ == "__main__":
    raise SystemExit(run_command(
        __file__, "tier2_mel_full_trial.py",
        "[set BCI_RUN_ID for the output directory]",
        cwd=study_path(
            "20250612_wairagkar",
            "open_source_code_for_reference/brain-to-voice-2025",
        ),
    ))
