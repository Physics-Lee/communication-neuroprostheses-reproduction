# -*- coding: utf-8 -*-
"""Build the train/test partition manifest of the Willett 2021 handwriting data.

The publisher ships two cross-validation partitions in
``RNNTrainingSteps/trainTestPartitions_<partition>.mat``. Each holds, per
session, a ``<session>_train`` and a ``<session>_test`` array of 0-based row
indices into the per-sentence arrays of ``Datasets/<session>/sentences.mat``,
which is the ``source_index`` of an exported trial. Three observations fix
the indexing:

* every non-empty ``_train`` array starts at 0, and no index is negative or
  reaches past the last sentence of its session;
* for the eight copy-typing days, ``_train`` and ``_test`` of HeldOutBlocks
  together cover exactly the sentences that ``excludedSentences`` leaves at 0;
* the official ``SummarizeRNNPerformance.ipynb`` gathers
  ``charCounts[0, valIdx]`` with these indices, and the archived evaluation
  rows of HeldOutTrials keep the same numbers as ``archive_trial_id``.

Excluded sentences are in neither role, and neither are the CL Pandarinath
sentences under HeldOutTrials nor the from-memory and free-write trials of the
two free-answer days. The manifest lists the assigned sentences of exported
sessions only: every ``trial_id`` is read from a session file, so it resolves
to an exported trial. Run it again after exporting more sessions.

Usage (from the repository root):

    python studies/20210512_willett/code/build_split_manifest.py
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import release_prefix, study_path  # noqa: E402

STUDY = "20210512_willett"
PARTITIONS = ("HeldOutBlocks", "HeldOutTrials")
ROLES = ("train", "test")
COLUMNS = ("partition", "session", "trial_id", "role", "source_index", "sentence_condition")


def _partition_file(partition):
    """Return the publisher file of one train/test partition.

    Parameters
    ----------
    partition : str
        ``HeldOutBlocks`` or ``HeldOutTrials``.

    Returns
    -------
    pathlib.Path
        ``RNNTrainingSteps/trainTestPartitions_<partition>.mat``.
    """
    name = f"trainTestPartitions_{partition}.mat"
    return Path(study_path(STUDY, "data/extracted/handwritingBCIData/RNNTrainingSteps")) / name


def _indices(matrices, session, role):
    """Return the publisher's sentence indices of one session and role.

    Parameters
    ----------
    matrices : dict
        Variables of one partition file.
    session : str
        Native session identifier such as ``t5.2019.05.08``.
    role : str
        ``train`` or ``test``.

    Returns
    -------
    numpy.ndarray
        The ``<session>_<role>`` array as a flat int64 vector, empty when the
        publisher stored a ``(0, 0)`` array.
    """
    return np.asarray(matrices[f"{session}_{role}"]).ravel().astype(np.int64)


def session_rows(path, partitions):
    """Return the manifest rows of one exported session file.

    Parameters
    ----------
    path : pathlib.Path
        Common session file.
    partitions : dict of str to dict
        Variables of every partition file, by partition name.

    Returns
    -------
    list of dict
        One row per assigned sentence, in partition, role, and index order.

    Raises
    ------
    ValueError
        When an index falls outside the session, names an excluded sentence,
        or is assigned twice.
    """
    rows, assigned = [], set()
    with read_session(path) as session:
        name = session.attrs["session"]
        trials = {int(trial.attrs["source_index"]): trial for trial in session.trials}
        for partition in PARTITIONS:
            for role in ROLES:
                for index in _indices(partitions[partition], name, role):
                    trial = trials.get(int(index))
                    if trial is None:
                        raise ValueError(f"{name}: {partition} {role} index {index} is not a sentence of the session")
                    if Path(trial.attrs["source_file"]).parent.name != name:
                        raise ValueError(f"{name}: trial {trial.trial_id} comes from another file")
                    if bool(trial.attrs["excluded"]):
                        raise ValueError(f"{name}: {partition} {role} assigns the excluded sentence {index}")
                    if (partition, int(index)) in assigned:
                        raise ValueError(f"{name}: sentence {index} is assigned twice in {partition}")
                    assigned.add((partition, int(index)))
                    rows.append({"partition": partition, "session": name, "trial_id": trial.trial_id,
                                 "role": role, "source_index": int(index),
                                 "sentence_condition": trial.attrs["sentence_condition"]})
    return rows


def partition_sessions(partitions):
    """Return the sessions that the partition files name.

    Parameters
    ----------
    partitions : dict of str to dict
        Variables of every partition file, by partition name.

    Returns
    -------
    list of str
        Native session identifiers in file order.
    """
    names = []
    for matrices in partitions.values():
        for key in matrices:
            if not key.startswith("__") and np.asarray(matrices[key]).dtype.kind in "iu":
                name = key.rsplit("_", 1)[0]
                if name not in names:
                    names.append(name)
    return names


def main():
    """Write the manifest from the exported session files and the partition files.

    Returns
    -------
    int
    Zero when a manifest was written, one when no session file is exported.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="write this path instead of data/splits/<release>__train_test_partitions.csv")
    args = parser.parse_args()
    import scipy.io
    partitions = {partition: scipy.io.loadmat(_partition_file(partition)) for partition in PARTITIONS}
    files = sorted(Path(study_path(STUDY, "data/processed")).glob("*/*.h5"))
    if not files:
        print("no exported session file under data/processed; run export_sessions.py first")
        return 1
    rows = [row for path in files for row in session_rows(path, partitions)]
    target = args.output or Path(study_path(STUDY, "data/splits")) / f"{release_prefix(STUDY)}__train_test_partitions.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"written {target}: {len(rows)} rows from {len(files)} session file(s)")
    for partition in PARTITIONS:
        counts = {role: sum(row["partition"] == partition and row["role"] == role for row in rows) for role in ROLES}
        print(f"  {partition:14s} " + ", ".join(f"{role} {count}" for role, count in counts.items()))
    sessions = {row["session"] for row in rows}
    pending = [name for name in partition_sessions(partitions) if name not in sessions]
    if pending:
        print("  partitioned sessions without an exported session file: " + ", ".join(pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
