# -*- coding: utf-8 -*-
"""Check a Wairagkar session file's neural and acoustic time axes.

Reads an exported session file without the publisher's ZIP and verifies that

- every neural feature and every time-aligned acoustic target of a trial has
  the trial's number of 10 ms frames,
- every word and syllable interval lies inside its trial, and
- the silence intervals (frames outside every word interval) agree with the
  silent frames of the acoustic targets. A target frame is silent when its
  first LPCNet coefficient, the log energy, sits on the session's energy
  floor.

Usage (from the repository root):

    python studies/20250612_wairagkar/code/check_session_timing.py SESSION_FILE
        [--output-dir NEW_DIRECTORY]

With --output-dir the counts are written to ``summary.json`` in a new
``results/<run_id>/`` directory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

BIN_MS = 10


def covered(table, frames):
    """Mark the frames a released interval table covers.

    Parameters
    ----------
    table : numpy.ndarray
        ``(n, 2)`` onset and offset frames as released. Onsets are zero-based
        and the offset frame is normally still voiced, so both bounds count.
    frames : int
        Number of frames in the trial.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(frames,)``.

    Notes
    -----
    On day 195 the frame before each onset is silent in 1,694 of 1,697 word
    intervals and the frame at each offset is voiced in all of them, which is
    why the released bounds are read as zero-based and inclusive. The offset
    is not a last-voiced-frame marker; the frame after it is still voiced in
    1,666 intervals.
    """
    mask = np.zeros(frames, dtype=bool)
    for onset, offset in np.atleast_2d(table).reshape(-1, 2):
        mask[max(0, int(onset)):min(frames, int(offset) + 1)] = True
    return mask


def check(path):
    """Compare the time axes of one exported session.

    Parameters
    ----------
    path : str or pathlib.Path
        Exported Wairagkar session file.

    Returns
    -------
    dict
        Trial and role counts, per-axis frame agreement, interval counts, and
        the frame-level agreement between word-interval silence and the
        energy floor of the acoustic targets.
    """
    with read_session(path) as session:
        trials = session.trials
        report = {
            "session": session.attrs["session"], "trials": len(trials), "bin_ms": BIN_MS,
            "roles": dict(collections.Counter(trial.attrs["trial_description"] for trial in trials)),
            "neural_frames_match": 0, "acoustic_frames_match": 0, "intervals_inside_trial": 0,
            "words": 0, "syllables": 0, "closed_loop_audio_trials": 0,
        }
        masks, energies = [], []
        for trial in trials:
            frames = int(trial.attrs["n_frames"])
            neural = trial.neural()
            targets = {name: trial.target(name) for name in ("lpcnet_word", "lpcnet_syllable")}
            report["neural_frames_match"] += all(
                array.shape[0] == frames for array in neural.values())
            report["acoustic_frames_match"] += all(
                array.shape[0] == frames for array in targets.values())
            word, syllable = trial.timing("word"), trial.timing("syllable")
            report["intervals_inside_trial"] += all(
                table.ndim == 2 and table.shape[1] == 2 and (table[:, 0] >= 0).all()
                and (table[:, 1] < frames).all() and (table[:, 1] >= table[:, 0]).all()
                for table in (word, syllable))
            report["words"] += len(word)
            report["syllables"] += len(syllable)
            report["closed_loop_audio_trials"] += int(trial.attrs.get("closed_loop_audio_samples", 0) > 0)
            masks.append(covered(word, frames))
            energies.append(targets["lpcnet_word"][:, 0])
        inside = np.concatenate(masks)
        energy = np.concatenate(energies)
        silent = energy == energy.min()
        report.update({
            "frames": int(inside.size),
            "silent_target_frames": int(silent.sum()),
            "silent_frames_outside_words": int((~inside).sum()),
            "silent_target_frames_inside_words": int((silent & inside).sum()),
            "active_target_frames_outside_words": int((~silent & ~inside).sum()),
        })
        report["silence_agreement"] = float((silent == ~inside).mean())
        report["ok"] = bool(
            report["neural_frames_match"] == report["acoustic_frames_match"]
            == report["intervals_inside_trial"] == len(trials))
    return report


def main():
    """Check one session file and print the result.

    Returns
    -------
    int
        Zero when every axis check passes, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_file", type=Path, help="common session .h5 file")
    parser.add_argument("--output-dir", type=Path, help="new directory for summary.json")
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    report = check(args.session_file)
    for key, value in report.items():
        print(f"{key}: {value}")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {args.output_dir}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
