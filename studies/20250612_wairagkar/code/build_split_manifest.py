# -*- coding: utf-8 -*-
"""Build the Wairagkar 2025 split manifests from released inputs.

Writes two files under ``data/splits/``:

- ``dryad_2280gb64f__evaluation_trials.csv`` lists every trial of every
  recording day with the role the release records for it (``trial_description``)
  and the decoder the day ran with.
- ``dryad_2280gb64f__listener_benchmark.csv`` maps the released listener-test
  indices (``test_idx.mat``) to trials. The release does not state the base of
  those indices. Reading them as zero-based positions in the concatenation of
  all recording days in day order places every one of them on an evaluation
  trial, while a one-based reading puts one random-set index on a training
  trial. The script re-checks that evidence on every run and stops if it no
  longer holds.

Usage (from the repository root):

    python studies/20250612_wairagkar/code/build_split_manifest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_checkout  # noqa: E402

import csv  # noqa: E402
import io  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402

STUDY = "20250612_wairagkar"
TRIALS = "dryad_2280gb64f__evaluation_trials.csv"
BENCHMARK = "dryad_2280gb64f__listener_benchmark.csv"


def released_trials(archive, sessions):
    """Read the role, cue, and decoder of every trial from the publisher ZIP.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open publisher ZIP.
    sessions : list of str
        Session identifiers in day order.

    Returns
    -------
    list of dict
        One record per trial in day order with ``session``, ``local_index``,
        ``description``, ``reference``, and ``decoder``.
    """
    records = []
    for session in sessions:
        neural = sio.loadmat(
            io.BytesIO(archive.read(f"Dryad_T15_all/{session}/{session}_neural_data.mat")),
            squeeze_me=True, variable_names=["trial_description", "cues", "brain2voice_decoder"])
        cues = np.atleast_1d(neural["cues"])
        descriptions = np.atleast_1d(neural["trial_description"])
        decoder = neural.get("brain2voice_decoder")
        decoder = str(decoder).strip() if decoder is not None else ""
        for index, (cue, description) in enumerate(zip(cues, descriptions)):
            records.append({"session": session, "local_index": index,
                            "description": str(description).strip(),
                            "reference": str(cue).strip(), "decoder": decoder})
    return records


def build(splits):
    """Write both manifests after re-checking the index base.

    Parameters
    ----------
    splits : pathlib.Path
        Study ``data/splits`` directory.

    Returns
    -------
    dict
        Trial counts by role, the evaluation days, and listener-test counts.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_sessions import _zip_path, list_sessions
    sessions = [item["session"] for item in list_sessions()]
    with zipfile.ZipFile(_zip_path()) as archive:
        records = released_trials(archive, sessions)
        benchmark = sio.loadmat(io.BytesIO(archive.read("Dryad_T15_all/test_idx.mat")), squeeze_me=True)
    evaluation = np.array(["valuation" in record["description"] for record in records])
    indices = {"random": np.asarray(benchmark["random_test_idx"]).ravel(),
               "video": np.asarray(benchmark["video_test_idx"]).ravel()}
    total = len(records)
    for name, values in indices.items():
        zero_based = int(evaluation[values].sum()) if values.max() < total else -1
        one_based = int(evaluation[values - 1].sum())
        if zero_based != len(values):
            raise SystemExit(
                f"{name} listener-test indices do not all fall on evaluation trials when read as "
                f"zero-based ({zero_based}/{len(values)}, one-based {one_based}/{len(values)}); "
                "the listener manifest would be wrong")
    splits.mkdir(parents=True, exist_ok=True)
    for record in records:
        record["trial_id"] = f"{record['session']}/data/{record['local_index']:04d}"
    with (splits / TRIALS).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["session", "trial_id", "role", "trial_description",
                                                    "brain2voice_decoder", "reference"])
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({"session": record["session"], "trial_id": record["trial_id"],
                             "role": "evaluation" if evaluation[index] else "training",
                             "trial_description": record["description"],
                             "brain2voice_decoder": record["decoder"], "reference": record["reference"]})
    counts = {}
    with (splits / BENCHMARK).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["set", "global_index", "session", "trial_id",
                                                    "trial_description", "reference"])
        writer.writeheader()
        for name, values in indices.items():
            counts[name] = len(values)
            for value in values:
                index = int(value)
                writer.writerow({"set": name, "global_index": index, "session": records[index]["session"],
                                 "trial_id": records[index]["trial_id"],
                                 "trial_description": records[index]["description"],
                                 "reference": records[index]["reference"]})
    return {"trials": total, "evaluation_trials": int(evaluation.sum()),
            "evaluation_days": len({record["session"] for record, flag in zip(records, evaluation) if flag}),
            "listener_random": counts["random"], "listener_video": counts["video"]}


if __name__ == "__main__":
    result = build(Path(study_checkout(STUDY)) / "data" / "splits")
    for key, value in result.items():
        print(f"{key}: {value}")
