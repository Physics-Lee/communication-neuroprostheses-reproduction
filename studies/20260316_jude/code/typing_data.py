"""Resolve the derived 20 ms typing data used by the Jude 2026a trainers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import derived_data_path

STUDY = "20260316_jude"


def typing_tfrecord_path(session, *, for_read=False):
    """Return the TFRecord conversion directory for a native session.

    Parameters
    ----------
    session : str
        Participant and session identifier, such as ``t17.2024.04.25``.
    for_read : bool, optional
        Allow the former location beside the source MAT files.

    Returns
    -------
    pathlib.Path
        Directory containing the split shards and the conversion parameters.
    """
    part = Path(session.split(".", 1)[0]) / session
    legacy = Path("data/extracted/Data") / part / "Typing/tfdata_20ms"
    return derived_data_path(STUDY, Path("tfrecord/typing_20ms") / part,
                             legacy=legacy if for_read else None)


def typing_npz_path(session, *, for_read=False, data_root=None):
    """Return the PyTorch NPZ for a native session.

    Parameters
    ----------
    session : str
        Participant and session identifier, such as ``t17.2024.04.25``.
    for_read : bool, optional
        Allow the former location beside the source MAT files.
    data_root : str or pathlib.Path, optional
        Explicit directory containing the participant directories.

    Returns
    -------
    pathlib.Path
        Session NPZ path. Explicit roots accept both directory layouts.
    """
    part = Path(session.split(".", 1)[0]) / session
    filename = "m1_torch_data.npz"
    old_tail = part / "Typing" / filename
    if data_root is not None:
        path = Path(data_root) / part / filename
        return path if path.is_file() else Path(data_root) / old_tail
    legacy = Path("data/extracted/Data") / old_tail
    return derived_data_path(STUDY, Path("npz/typing_20ms") / part / filename,
                             legacy=legacy if for_read else None)
