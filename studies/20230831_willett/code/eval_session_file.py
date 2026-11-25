# -*- coding: utf-8 -*-
"""Evaluate a BtoT24-trained GRUDecoder on trials read from a common session file.

The session file stores released features. This script applies the block-wise
z-scoring of convert_btot24.py (statistics over the trials that one publisher
file holds for one block) and then follows eval_btot24.py: greedy CTC with the
recording-day index, PER = sum(edit distance) / sum(reference length).

Usage (from the repository root):

    python studies/20230831_willett/code/eval_session_file.py SESSION_FILE RUN_DIRECTORY
        [--partition test] [--day-index N] [--compare ARCHIVED_TRIALS_CSV]
        [--trainer-input SESSION_DIRECTORY] [--output-dir NEW_DIRECTORY]

With --compare, each evaluated trial is checked against the archived per-trial
CSV (sentence, phoneme strings, edit distance, and reference length). With
--trainer-input, the z-scored features and labels of every labeled trial are
checked bit for bit against the ``data_train.hdf5`` and ``data_val.hdf5`` that
convert_btot24.py wrote for the session.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import study_path  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402

import editdistance  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, study_path("20240815_card", "open_source_code_for_reference/nejm-brain-to-text/model_training"))
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME, runSingleDecodingStep  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# transforms must match training (gauss smoothing before the model)
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})
PARTITIONS = ("train", "test", "competitionHoldOut")
TRAINER_FILE = {"train": "data_train.hdf5", "test": "data_val.hdf5"}


def released_features(trial):
    """Build the 512-channel feature matrix of a trial as convert_btot24.py does.

    Parameters
    ----------
    trial : Trial
        Trial of a Willett 2023 session file.

    Returns
    -------
    numpy.ndarray
        ``[spike_pow, tx]`` as float32 in Fortran order, the memory layout of
        the publisher's MATLAB arrays.
    """
    return np.concatenate([np.asfortranarray(trial.neural("spike_pow")),
                           np.asfortranarray(trial.neural("tx").astype(np.float32))], axis=1)


def block_statistics(trials):
    """Z-score features per publisher file and block, as the trainer input does.

    Parameters
    ----------
    trials : list of Trial
        Every trial of the session file.

    Returns
    -------
    dict
        Maps ``(source_file, source_block)`` to ``(mean, standard deviation)``
        arrays of shape ``(1, 512)``, computed over every trial the publisher
        file holds for the block.

    Notes
    -----
    ``convert_btot24.py`` stacks MATLAB arrays, which are column-major, so
    NumPy sums them in Fortran order. Summing the same values in C order
    rounds the means differently and shifts low-variance channels by up to
    about 1e-4. The Fortran layout is kept here to reproduce the trainer input
    bit for bit.
    """
    groups = {}
    for trial in trials:
        key = (trial.attrs["source_file"], int(trial.attrs["source_block"]))
        groups.setdefault(key, []).append(released_features(trial))
    statistics = {}
    for key, blocks in groups.items():
        stack = np.concatenate(blocks, axis=0)
        mean = stack.mean(axis=0, keepdims=True)
        deviation = stack.std(axis=0, keepdims=True)
        deviation[deviation == 0] = 1.0
        statistics[key] = (mean, deviation)
    return statistics


def model_input(trial, statistics):
    """Return the z-scored feature matrix of one trial.

    Parameters
    ----------
    trial : Trial
        Trial of a Willett 2023 session file.
    statistics : dict
        Result of ``block_statistics``.

    Returns
    -------
    numpy.ndarray
        ``(frames, 512)`` float32 input of the decoder.
    """
    mean, deviation = statistics[(trial.attrs["source_file"], int(trial.attrs["source_block"]))]
    return (released_features(trial) - mean) / deviation


def check_trainer_input(trials, statistics, directory):
    """Compare session-file features and labels with the converted trainer input.

    Parameters
    ----------
    trials : list of Trial
        Every trial of the session file.
    statistics : dict
        Result of ``block_statistics``.
    directory : pathlib.Path
        Trainer-input directory of the session.

    Returns
    -------
    dict
        Counts of trials with identical features and identical labels, the
        number checked, and the largest absolute feature difference.
    """
    result = {"features_identical": 0, "labels_identical": 0, "checked": 0, "max_feature_difference": 0.0}
    for trial in trials:
        partition = trial.attrs.get("source_partition")
        if partition not in TRAINER_FILE:
            continue
        key = f"trial_{int(trial.attrs['source_index']):04d}"
        with h5py.File(directory / TRAINER_FILE[partition], "r") as handle:
            group = handle[key]
            features = model_input(trial, statistics)
            stored = group["input_features"][:]
            result["checked"] += 1
            result["features_identical"] += np.array_equal(features, stored)
            result["max_feature_difference"] = max(
                result["max_feature_difference"], float(np.abs(features - stored).max()))
            length = int(group.attrs["seq_len"])
            text = trial.target("transcription")
            result["labels_identical"] += (
                group.attrs["sentence_label"] == trial.sentence
                and np.array_equal(group["seq_class_ids"][:length], trial.target("phonemes"))
                and np.array_equal(group["transcription"][:len(text)], text))
    return result


def load_model(run_dir, device):
    """Rebuild the evaluation model and load a run's best checkpoint.

    Parameters
    ----------
    run_dir : pathlib.Path
        Model run holding ``checkpoint/best_checkpoint``.
    device : torch.device
        Device for inference.

    Returns
    -------
    tuple
        ``(model, checkpoint path)``.
    """
    model = GRUDecoder(
        neural_dim=512, n_units=768, n_days=24, n_classes=41,
        rnn_dropout=0.4, input_dropout=0.2, n_layers=5,
        patch_size=14, patch_stride=4,
    )
    ckpt_path = run_dir / "checkpoint" / "best_checkpoint"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    state = ckpt["model_state_dict"]
    for key in list(state.keys()):
        state[key.replace("module.", "").replace("_orig_mod.", "")] = state.pop(key)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt_path


def day_index(session_name):
    """Return the recording-day index of a session in sorted session order.

    Parameters
    ----------
    session_name : str
        Native session identifier such as ``t12.2022.05.19``.

    Returns
    -------
    int
        Zero-based position that ``eval_btot24.py`` passes to the decoder.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_sessions import list_sessions
    return [item["session"] for item in list_sessions()].index(session_name)


def main():
    """Evaluate one session file and check it against the archived pipeline.

    Returns
    -------
    None
        Prints per-trial totals; with ``--output-dir`` writes ``trials.csv``,
        ``summary.json``, and ``run.yaml`` into a new directory.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_file", type=Path, help="common session .h5 file")
    parser.add_argument("run_dir", type=Path, help="model run with checkpoint/best_checkpoint")
    parser.add_argument("--partition", choices=PARTITIONS, default="test",
                        help="publisher partition to evaluate (default: test)")
    parser.add_argument("--day-index", type=int, help="override the recording-day index")
    parser.add_argument("--compare", type=Path, help="archived per-trial CSV to check against")
    parser.add_argument("--trainer-input", type=Path,
                        help="converted session directory to compare features and labels with")
    parser.add_argument("--output-dir", type=Path, help="new directory for trials.csv, summary.json, run.yaml")
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, ckpt_path = load_model(args.run_dir, device)
    rows, checks = [], {}
    with read_session(args.session_file) as session:
        session_name = session.attrs["session"]
        index = args.day_index if args.day_index is not None else day_index(session_name)
        trials = session.trials
        statistics = block_statistics(trials)
        if args.trainer_input is not None:
            checks["trainer_input"] = check_trainer_input(trials, statistics, args.trainer_input)
            print(f"trainer input comparison: {checks['trainer_input']}")
        for trial in [trial for trial in trials if trial.attrs.get("source_partition") == args.partition]:
            if trial.sentence is None:
                continue
            features = model_input(trial, statistics)
            x = torch.tensor(features[None], device=device, dtype=torch.bfloat16)
            logits = runSingleDecodingStep(x, index, model, MODEL_ARGS, device)[0]
            pred = np.argmax(logits, axis=-1)
            pred = torch.unique_consecutive(torch.tensor(pred), dim=-1).numpy()
            pred = np.array([i for i in pred if i != 0])
            true = trial.target("phonemes")
            distance = editdistance.eval(pred.tolist(), true.tolist())
            rows.append({
                "trial_id": trial.trial_id,
                "session": session_name,
                "block_num": int(trial.attrs["source_block"]),
                "sentence_label": trial.sentence,
                "true_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in true),
                "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                "edit_distance": int(distance), "true_len": int(len(true)),
            })
        dataset_id = session.attrs["dataset_id"]

    errors = sum(row["edit_distance"] for row in rows)
    reference = sum(row["true_len"] for row in rows)
    print(f"day index {index} | {args.partition} partition: {len(rows)} trials, "
          f"{errors}/{reference} phonemes wrong = {100 * errors / reference:.2f}% PER")
    if args.compare:
        with args.compare.open(encoding="utf-8-sig", newline="") as stream:
            archived = [row for row in csv.DictReader(stream) if row["session"] == session_name]
        for ours, theirs in zip(rows, archived):
            if not all(str(ours[name]) == theirs[name] for name in
                       ("block_num", "sentence_label", "true_phonemes", "pred_phonemes",
                        "edit_distance", "true_len")):
                raise SystemExit(f"Archive mismatch at {ours['trial_id']}:\n  ours:   {ours}\n  archive: {theirs}")
        if len(rows) != len(archived):
            raise SystemExit(f"Archive holds {len(archived)} trials, session file evaluated {len(rows)}")
        archive_errors = sum(int(row["edit_distance"]) for row in archived)
        archive_reference = sum(int(row["true_len"]) for row in archived)
        checks["archive"] = {"trials_identical": len(archived), "phoneme_errors": archive_errors,
                             "reference_phonemes": archive_reference, "source": str(args.compare)}
        print(f"archive comparison: {len(rows)}/{len(archived)} trials identical row by row; "
              f"archived total {archive_errors}/{archive_reference}")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        with (args.output_dir / "trials.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        per = errors / reference
        (args.output_dir / "summary.json").write_text(
            json.dumps({"trials": len(rows), "phoneme_errors": errors,
                        "reference_phonemes": reference, "per": per, "day_index": index,
                        "checks": checks}, indent=2) + "\n", encoding="utf-8")
        run = {"study": "20230831_willett", "dataset_id": dataset_id, "release_id": None,
               "input_format": "hdf5", "split": args.partition, "session": session_name,
               "decoder": "ctc_greedy", "rescorer": None, "vocoder": None,
               "checkpoint": str(ckpt_path.resolve()),
               "session_file": str(args.session_file.resolve()), "day_index": index,
               "trials": "trials.csv", "summary": "summary.json", "command": sys.argv}
        (args.output_dir / "run.yaml").write_text(
            yaml.safe_dump(run, sort_keys=False), encoding="utf-8")
        print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
