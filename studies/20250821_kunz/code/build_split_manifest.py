# -*- coding: utf-8 -*-
"""Write the Kunz 2025 split manifests from the exported session files.

Both manifests reference trials by the ``trial_id`` of the common session
files under ``data/processed`` and hold no hand-written rows.

``dryad_gf1vhhn1j__conditions.csv`` has one row per exported trial. The
pipeline splits by file name: ``convert_kunz.py`` writes ``data_val.hdf5``
for a file ending in ``_eval`` and ``data_train.hdf5`` for any other, and
``train_kunz.py`` selects the open-loop files of an arm with::

    train_sessions = [s for s in all_sessions
                      if not s.endswith("_eval")
                      and (arm == "mixed"
                           or (arm == "imagined") == ("_imagined_" in s))]

so ``role`` is ``evaluation`` for the ``_eval`` files, which every arm
validates on, and ``training_arm`` otherwise, with ``arms`` naming the arms
that select the file. ``convert_kunz.py`` also drops trials that its CTC
length filter rejects::

    if f.shape[0] <= 4 * len(ids) + 10 or not np.isfinite(f).all():
        dropped += 1
        continue

such trials get the ``role`` ``dropped`` and the reason in ``drop_reason``.

``dryad_gf1vhhn1j__t16_folds.csv`` has one row per trial of the session
``make_t16_holdout.py`` reads, with the fold that holds the trial's block
out. That script defines the folds with::

    FOLDS = {  # fold -> held-out (val) block
        1: 1,
        2: 4,
        3: 6,
    }

and ``run_t16_holdout.sh`` trains them with ``for fold in 1 2 3``. In fold
``f`` the rows with ``fold == f`` are the validation block and every other
row is training. When the derived fold inputs exist under
``data/derived/dryad_gf1vhhn1j/hdf5/t16_holdout/``, the script reads them
and checks every fold against the manifest before writing it.

Usage (from the repository root):

    python studies/20250821_kunz/code/build_split_manifest.py
"""

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import study_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_t16_holdout  # noqa: E402

STUDY = "20250821_kunz"
CONDITIONS_CSV = "data/splits/dryad_gf1vhhn1j__conditions.csv"
CONDITIONS_COLUMNS = ("session", "trial_id", "participant", "condition", "loop", "block", "role",
                      "arms", "drop_reason")
FOLDS_CSV = "data/splits/dryad_gf1vhhn1j__t16_folds.csv"
FOLDS_COLUMNS = ("session", "trial_id", "block", "fold")


def _convert():
    """Import the study's conversion module on first use.

    Returns
    -------
    module
        ``convert_kunz``, imported lazily because loading its
        grapheme-to-phoneme model takes several seconds.
    """
    import convert_kunz
    return convert_kunz


def exported_sessions():
    """List the sessions exported in the common format.

    Returns
    -------
    list of tuple
        ``(session, path)`` pairs in file-path order, where ``session`` is
        the native identifier stored in the session file.
    """
    sessions = []
    for path in sorted(Path(study_path(STUDY, "data/processed")).glob("*/*.h5")):
        with h5py.File(path, "r") as exported:
            sessions.append((str(exported.attrs["session"]), path))
    return sessions


def trainer_frames(n_frames, bin_ms):
    """Return the length ``convert_kunz.py`` trains on for a trial.

    Parameters
    ----------
    n_frames : int
        Frames of the trial in the source grid.
    bin_ms : int
        Source bin size in milliseconds.

    Returns
    -------
    int
        Frames left after ``convert_kunz.compress_to_20ms``.
    """
    dummy = np.zeros((n_frames, 1), dtype=np.float32)
    return _convert().compress_to_20ms(dummy, dummy, bin_ms)[0].shape[0]


def drop_reason(trial, bin_ms):
    """Return why ``convert_kunz.py`` drops a trial, or ``None`` when it keeps it.

    Parameters
    ----------
    trial : session_format.Trial
        Exported trial.
    bin_ms : int
        Source bin size of its session in milliseconds.

    Returns
    -------
    str or None
        ``"ctc_length_filter"``, ``"non_finite_features"``, or ``None``.

    Notes
    -----
    The filter runs on the 20 ms grid that ``compress_to_20ms`` produces:
    a trial needs more than ``4 * n_phonemes + 10`` frames and finite
    features.
    """
    frames = trainer_frames(int(trial.attrs["n_frames"]), bin_ms)
    if frames <= 4 * len(trial.target("phonemes")) + 10:
        return "ctc_length_filter"
    features = np.concatenate([trial.neural("spike_pow"), trial.neural("tx")], axis=1)
    if not np.isfinite(features.astype(np.float32)).all():
        return "non_finite_features"
    return None


def training_arms(session):
    """Return the arms of ``train_kunz.py`` that train on a file.

    Parameters
    ----------
    session : str
        Native identifier of an open-loop file.

    Returns
    -------
    str
        ``"mixed;imagined"`` or ``"mixed;attempted"``.
    """
    return ";".join(["mixed", "imagined" if "_imagined_" in session else "attempted"])


def conditions_rows():
    """Return the rows of the conditions manifest.

    Returns
    -------
    list of dict
        One row per trial of every exported session, in session and source
        order.
    """
    rows = []
    for session, path in exported_sessions():
        with read_session(path) as exported:
            bin_ms = int(exported.attrs["neural_bin_size_ms"])
            participant = str(exported.attrs["participant"])
            for trial in exported.trials:
                reason = drop_reason(trial, bin_ms)
                if reason:
                    role, arms = "dropped", ""
                elif session.endswith("_eval"):
                    role, arms = "evaluation", ""
                else:
                    role, arms = "training_arm", training_arms(session)
                rows.append({"session": session, "trial_id": trial.trial_id,
                             "participant": participant,
                             "condition": str(trial.attrs["condition"]),
                             "loop": str(trial.attrs["loop"]),
                             "block": int(trial.attrs["source_block"]), "role": role,
                             "arms": arms, "drop_reason": reason or ""})
    return rows


def holdout_trials(conditions):
    """Return the trials ``make_t16_holdout.py`` builds its folds from.

    Parameters
    ----------
    conditions : list of dict
        Rows of the conditions manifest.

    Returns
    -------
    tuple
        ``(session, rows)``, the session ``make_t16_holdout.SRC`` reads and
        its rows with the ``role`` ``training_arm``.

    Raises
    ------
    SystemExit
        When the session is not exported, or when its kept trials are not
        three blocks of 40 trials, the layout ``make_t16_holdout.py`` asserts.
    """
    session = Path(make_t16_holdout.SRC).parent.name
    rows = [row for row in conditions
            if row["session"] == session and row["role"] == "training_arm"]
    if not rows:
        raise SystemExit(f"export the session {session} before building the T16 fold manifest")
    counts = {}
    for row in rows:
        counts[row["block"]] = counts.get(row["block"], 0) + 1
    if sorted(counts) != sorted(make_t16_holdout.FOLDS.values()) or set(counts.values()) != {40}:
        raise SystemExit(f"{session}: unexpected block layout {dict(sorted(counts.items()))}, "
                         "make_t16_holdout.py expects three blocks of 40 trials")
    return session, rows


def fold_rows(conditions):
    """Return the rows of the T16 fold manifest.

    Parameters
    ----------
    conditions : list of dict
        Rows of the conditions manifest.

    Returns
    -------
    tuple
        ``(session, rows)``: the holdout session and one row per kept trial
        with the fold that holds its block out of training.
    """
    session, trials = holdout_trials(conditions)
    held_out_by = {block: fold for fold, block in make_t16_holdout.FOLDS.items()}
    return session, [{"session": session, "trial_id": row["trial_id"], "block": row["block"],
                      "fold": held_out_by[row["block"]]} for row in trials]


def check_folds(session, rows):
    """Compare every fold with the derived fold inputs of ``make_t16_holdout.py``.

    Parameters
    ----------
    session : str
        Native identifier of the holdout session.
    rows : list of dict
        Rows of the T16 fold manifest.

    Returns
    -------
    list of str or None
        One line per derived file when the inputs exist, else ``None``.

    Raises
    ------
    SystemExit
        When a derived file holds other trials than the manifest assigns to
        its side of the fold.
    """
    root = Path(make_t16_holdout.OUT_ROOT)
    if not root.is_dir():
        return None
    exported_path = next(path for name, path in exported_sessions() if name == session)
    with read_session(exported_path) as exported:
        bin_ms = int(exported.attrs["neural_bin_size_ms"])
        source = {}
        for trial in exported.trials:
            frames = trainer_frames(int(trial.attrs["n_frames"]), bin_ms)
            source[trial.trial_id] = (int(trial.attrs["source_block"]), trial.sentence, frames)
    lines = []
    for fold, val_block in sorted(make_t16_holdout.FOLDS.items()):
        for path in sorted((root / f"fold{fold}").glob("*/*.hdf5")):
            is_val = path.name == "data_val.hdf5"
            expected = [source[row["trial_id"]] for row in rows
                        if (row["block"] == val_block) == is_val]
            with h5py.File(path, "r") as derived:
                keys = sorted(key for key in derived.keys() if key.startswith("trial_"))
                found = [(int(derived[key].attrs["block_num"]),
                          str(derived[key].attrs["sentence_label"]),
                          int(derived[key].attrs["n_time_steps"])) for key in keys]
            if found != expected:
                raise SystemExit(f"{path}: holds {len(found)} trials that differ from the "
                                 f"{len(expected)} the manifest assigns to it")
            lines.append(f"fold {fold} {'val' if is_val else 'train'}: {len(found)} trials match "
                         f"{path.relative_to(root).as_posix()}")
    return lines


def write_csv(relative, columns, rows):
    """Write rows to a CSV file in the study's data directory.

    Parameters
    ----------
    relative : str
        Target path relative to the study directory.
    columns : tuple of str
        Column order of the file.
    rows : list of dict
        Rows to write.

    Returns
    -------
    str
        The resolved path that was written.
    """
    target = study_path(STUDY, relative)
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target


def main():
    """Write both manifests after checking the folds against the derived inputs.

    Returns
    -------
    int
        Zero when both manifests were written.
    """
    conditions = conditions_rows()
    print(write_csv(CONDITIONS_CSV, CONDITIONS_COLUMNS, rows=conditions))
    roles = {}
    for row in conditions:
        key = (row["session"], row["role"], row["arms"])
        roles[key] = roles.get(key, 0) + 1
    for (session, role, arms), count in roles.items():
        print(f"    {session}: {count} trials {role}" + (f" ({arms})" if arms else ""))
    for row in conditions:
        if row["role"] == "dropped":
            print(f"    dropped by the pipeline: {row['trial_id']} ({row['drop_reason']})")
    session, folds = fold_rows(conditions)
    checked = check_folds(session, folds)
    if checked is None:
        print(f"T16 fold inputs not found under {make_t16_holdout.OUT_ROOT}, fold check skipped")
    else:
        for line in checked:
            print("    " + line)
    print(write_csv(FOLDS_CSV, FOLDS_COLUMNS, rows=folds))
    for fold in sorted(make_t16_holdout.FOLDS):
        held = sum(row["fold"] == fold for row in folds)
        print(f"    fold {fold}: {held} validation trials (block {make_t16_holdout.FOLDS[fold]}), "
              f"{len(folds) - held} training trials")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
