"""Find self-trained handwriting runs and their standard output directories."""

from datetime import date
import os
import re
from pathlib import Path

from study_paths import release_prefix, run_output, study_checkout

STUDY = "20210512_willett"


def handwriting_run(name, *, for_write=False):
    """Resolve one handwriting training run.

    Parameters
    ----------
    name : str
        Training variant, such as ``HeldOutTrials_synth``, or a dated run ID.
    for_write : bool, optional
        Use the current run ID for newly generated outputs.

    Returns
    -------
    str
        Native runtime directory. Readers also support historical checkouts.
    """
    base = study_checkout(STUDY)
    legacy = base / "data" / STUDY / "runs" / name
    if re.fullmatch(r"\d{8}_[a-zA-Z0-9_-]+", name):
        return str(base / "results" / name / "runtime")
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
        raise ValueError(f"Invalid handwriting run name: {name!r}")
    source_format = "npy" if name.endswith("_synth") else "mat"
    suffix = f"{release_prefix(STUDY)}__{source_format}__char_start__{name.lower()}"
    run_id = f"{date.today():%Y%m%d}_{suffix}"
    if for_write or os.environ.get("BCI_RUN_ID"):
        return str(Path(run_output(STUDY, default_id=run_id)) / "runtime")
    candidates = sorted((base / "results").glob(f"????????_{suffix}/runtime"))
    if candidates:
        return str(candidates[-1])
    if legacy.is_dir():
        return str(legacy)
    return str(base / "results" / run_id / "runtime")
