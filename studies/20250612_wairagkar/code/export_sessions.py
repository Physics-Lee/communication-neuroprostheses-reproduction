# -*- coding: utf-8 -*-
"""Export Wairagkar 2025 voice-synthesis sessions to the common session format.

The publisher ships one ZIP with a folder per recording day. Each day holds a
neural file, a speech-index file, and a target speech-feature file whose
per-trial arrays share one 10 ms time axis. One session file holds every
trial of one day, in the order of those arrays.

Usage (from the repository root):

    python studies/20250612_wairagkar/code/export_sessions.py --list
    python studies/20250612_wairagkar/code/export_sessions.py --session t15_day195
"""

import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20250612_wairagkar"
DATASET = "https://doi.org/10.5061/dryad.2280gb64f"
BIN_MS = 10
NEURAL = "_neural_data.mat"
INDICES = "_speech_indices.mat"
FEATURES = "_target_speech_feats.mat"
ROOT = "Dryad_T15_all"
_CACHE = {"key": None, "day": None}


def _zip_path():
    """Return the publisher ZIP.

    Returns
    -------
    pathlib.Path
        Path of ``t15_neural_data.zip`` in the study's data directory.
    """
    return Path(study_path(STUDY, "data/raw/t15_neural_data.zip"))


def _relative(path):
    """Return a publisher file path relative to the study directory.

    Parameters
    ----------
    path : pathlib.Path
        Resolved publisher file.

    Returns
    -------
    str
        POSIX-style path such as
        ``data/raw/dryad_2280gb64f/dryad_2280gb64f__t15_neural_data.zip``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _member(day, suffix):
    """Return the ZIP member name of one file of a day.

    Parameters
    ----------
    day : str
        Session identifier such as ``t15_day195``.
    suffix : str
        File-name suffix of the member.

    Returns
    -------
    str
        Member name inside the ZIP.
    """
    return f"{ROOT}/{day}/{day}{suffix}"


def _day(source_file, day):
    """Load the three files of one day, keeping the last day in memory.

    Parameters
    ----------
    source_file : str
        Publisher ZIP path relative to the study directory.
    day : str
        Session identifier.

    Returns
    -------
    dict
        The per-trial arrays of the neural, speech-index, and target files.
    """
    import scipy.io as sio
    key = (source_file, day)
    if _CACHE["key"] != key:
        with zipfile.ZipFile(study_path(STUDY, source_file)) as archive:
            loaded = [sio.loadmat(io.BytesIO(archive.read(_member(day, suffix))), squeeze_me=True)
                      for suffix in (NEURAL, INDICES, FEATURES)]
        neural, indices, feats = loaded
        _CACHE["key"], _CACHE["day"] = key, {
            "spike_pow": np.atleast_1d(neural["spikepow_trials"]),
            "tx": np.atleast_1d(neural["threshcross_trials"]),
            "cues": np.atleast_1d(neural["cues"]),
            "description": np.atleast_1d(neural["trial_description"]),
            "audio": np.atleast_1d(neural.get("brain2voice_pred_audio_16kHz", [])),
            "word": np.atleast_1d(indices["word_indices"]),
            "syllable": np.atleast_1d(indices["syl_indices"]),
            "lpcnet_word": np.atleast_1d(feats["target_speech_feats_word"]),
            "lpcnet_syllable": np.atleast_1d(feats["target_speech_feats_syl"]),
        }
    return _CACHE["day"]


def list_sessions():
    """List the recording days that hold neural, speech-index, and target files.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in day order.
    """
    sessions = []
    with zipfile.ZipFile(_zip_path()) as archive:
        names = set(archive.namelist())
        days = sorted({match.group(1) for name in names
                       if (match := re.match(rf"{ROOT}/(t15_day\d+)/", name))})
    source = _relative(_zip_path())
    for day in days:
        if all(_member(day, suffix) in names for suffix in (NEURAL, INDICES, FEATURES)):
            sessions.append({"session": day, "participant": "t15", "sources": [source]})
    return sessions


def extra_attrs(session):
    """Return the root attributes a session shares by its recording day.

    Parameters
    ----------
    session : str
        Session identifier.

    Returns
    -------
    dict
        ``brain2voice_decoder``, the released decoder the session ran with
        when it was an evaluation day, and the neural bin size.
    """
    import scipy.io as sio
    with zipfile.ZipFile(_zip_path()) as archive:
        neural = sio.loadmat(io.BytesIO(archive.read(_member(session, NEURAL))),
                             squeeze_me=True, variable_names=["brain2voice_decoder"])
    attrs = {"neural_bin_size_ms": BIN_MS}
    decoder = neural.get("brain2voice_decoder")
    if decoder is not None and str(decoder).strip():
        attrs["brain2voice_decoder"] = str(decoder).strip()
    return attrs


def _spec(session, loaded, source_file, index):
    """Build the trial record at one position of a day's arrays.

    Parameters
    ----------
    session : str
        Session identifier such as ``t15_day195``.
    loaded : dict
        Per-trial arrays of the recording day (see ``_day``).
    source_file : str
        Publisher ZIP path relative to the study directory.
    index : int
        Zero-based trial position in the arrays.

    Returns
    -------
    TrialSpec
        The trial with its neural features, acoustic targets, and timings.
    """
    neural = {
        "spike_pow": Field(np.ascontiguousarray(loaded["spike_pow"][index]), bin_ms=BIN_MS,
                           source_name="spikepow_trials",
                           description="Spike-band power per electrode in the publisher's array order."),
        "tx": Field(np.ascontiguousarray(loaded["tx"][index]), bin_ms=BIN_MS,
                    source_name="threshcross_trials",
                    description="Threshold-crossing features per electrode in the publisher's array order."),
    }
    targets = {
        "sentence": Field(str(loaded["cues"][index]).strip(), source_name="cues"),
        "lpcnet_word": Field(np.ascontiguousarray(loaded["lpcnet_word"][index]), bin_ms=BIN_MS,
                             source_name="target_speech_feats_word",
                             description="Synthetic target LPCNet features (18 BFCC, pitch, correlation) "
                                         "aligned to the neural frames with the word timings."),
        "lpcnet_syllable": Field(np.ascontiguousarray(loaded["lpcnet_syllable"][index]), bin_ms=BIN_MS,
                                 source_name="target_speech_feats_syl",
                                 description="Synthetic target LPCNet features (18 BFCC, pitch, correlation) "
                                             "aligned to the neural frames with the syllable timings."),
    }
    timing = {
        "word": Field(np.ascontiguousarray(np.atleast_2d(loaded["word"][index])), bin_ms=BIN_MS,
                      source_name="word_indices",
                      description="Attempted-speech onset and offset frame of each cued word, as released."),
        "syllable": Field(np.ascontiguousarray(np.atleast_2d(loaded["syllable"][index])), bin_ms=BIN_MS,
                          source_name="syl_indices",
                          description="Attempted-speech onset and offset frame of each syllable, as released."),
    }
    audio = loaded["audio"][index] if index < len(loaded["audio"]) else np.zeros(0)
    meta = {"trial_description": str(loaded["description"][index]).strip(),
            "closed_loop_audio_samples": int(len(audio)) if audio is not None else 0,
            "source_member_neural": _member(session, NEURAL),
            "source_member_speech_indices": _member(session, INDICES),
            "source_member_target_speech_feats": _member(session, FEATURES)}
    return TrialSpec(trial_id=trial_id(session, "data", index), source_file=source_file,
                     source_index=index, neural=neural, targets=targets, timing=timing, meta=meta)


def iter_trials(session):
    """Yield every trial of one recording day in array order.

    Parameters
    ----------
    session : str
        Session identifier such as ``t15_day195``.

    Yields
    ------
    TrialSpec
        Trials in the order of the publisher's per-trial arrays.
    """
    source = _relative(_zip_path())
    loaded = _day(source, session)
    for index in range(len(loaded["cues"])):
        yield _spec(session, loaded, source, index)


def read_source_trial(attrs):
    """Rebuild one trial from the publisher ZIP named in its attributes.

    Parameters
    ----------
    attrs : dict
        Attributes stored on an exported trial.

    Returns
    -------
    TrialSpec
        The trial as the publisher's files hold it.
    """
    session = attrs["source_member_neural"].split("/")[1]
    loaded = _day(attrs["source_file"], session)
    return _spec(session, loaded, attrs["source_file"], int(attrs["source_index"]))


def label_sets():
    """Return an empty mapping: this study stores no label sequences.

    Returns
    -------
    dict
        Empty.
    """
    return {}


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="zip", extra_attrs=extra_attrs,
        description="Export Wairagkar 2025 voice-synthesis sessions to the common session format."))
