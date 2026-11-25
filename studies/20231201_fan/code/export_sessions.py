# -*- coding: utf-8 -*-
"""Export Fan 2023 handwriting sessions to the common session format.

The publisher ships one ZIP with a TFRecord folder per session. Eleven seed
sessions hold the trials the released decoder was trained on, split into a
``train`` and a ``test`` shard; their features were already normalized by the
publisher. Ten follow-up sessions hold the online evaluation in two arms,
``no_recalibration`` and ``recalibration``, whose ``train`` shards carry the
features unnormalized and whose ``test`` shards are empty. One session file
holds every trial of one session, unit by unit: ``train`` then ``test`` for a
seed session, and the two arms in the order they ran for a follow-up session.

The ZIP also holds one MAT file per unit with the sentence, the block number,
and the recorded online decodes (``rnn_decoded``, ``lm_decoded``) of every
trial. Each exported trial stores the member and the zero-based row of its MAT
entry (``source_member_mat``, ``mat_index``), so the online transcripts can be
read without joining on sentence text. ``inputFeatures`` is stored as
released, a time-major ``(nTimeSteps, 192)`` float32 array.
``classLabelsOneHot`` is all-zero in every released example, so it is stored
as the released matrix instead of a per-frame class. ``ceMask`` and
``newClassSignal`` are left out: only the package's cross-entropy loss reads
them, and they are constant in the release.

Usage (from the repository root):

    python studies/20231201_fan/code/export_sessions.py --list
    python studies/20231201_fan/code/export_sessions.py --session t5.2022.09.29
"""

import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20231201_fan"
DATASET = "https://doi.org/10.5061/dryad.hqbzkh1p6"
# Publisher layout under data/extracted/<release>/ and inside the ZIP.
RELEASE = "CORP_data_release"
SEED_TREE = "seed_model_training_data"
EVAL_TREE = "online_evaluation_data"
# Follow-up arms in the order they ran: their block numbers increase.
ARMS = ("no_recalibration", "recalibration")
PARTITIONS = ("train", "test")
N_FEATURES = 192
N_CLASSES = 31
# The release states no bin duration. 20 ms follows the repository's
# evaluation guide, which calls each tx_feats row one 20 ms bin. At 20 ms the
# released trial lengths give about 75 characters per minute, close to the
# paper's reported writing speed; 10 ms would give twice that.
BIN_MS = 20
DESCRIPTIONS = {
    SEED_TREE: "Binned threshold-crossing counts per electrode, one row per bin. The publisher "
               "already normalized them: a per-channel affine rescaling with one slope and offset "
               "per block.",
    EVAL_TREE: "Binned threshold-crossing counts per electrode, one row per bin, not normalized: "
               "identical to tx_feats in the release's MAT file.",
}


def _release():
    """Return the extracted publisher tree.

    Returns
    -------
    pathlib.Path
        ``data/extracted/<release>/CORP_data_release`` inside the study.
    """
    return Path(study_path(STUDY, f"data/extracted/{RELEASE}"))


def _zip_path():
    """Return the publisher ZIP.

    Returns
    -------
    pathlib.Path
        Path of ``dryad_hqbzkh1p6__CORP_data_release.zip`` in the study's data
        directory.
    """
    return Path(study_path(STUDY, "data/raw/CORP_data_release.zip"))


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
        ``data/extracted/dryad_hqbzkh1p6/CORP_data_release/seed_model_training_data/tfrecords/t5.2022.05.18/train/chunk_0.tfrecord``.
    """
    return Path(path).resolve().relative_to(Path(study_checkout(STUDY)).resolve()).as_posix()


def _shards(directory):
    """Return the non-empty shards of one publisher partition in name order.

    Parameters
    ----------
    directory : pathlib.Path
        Publisher partition directory, such as ``<session>/train``.

    Returns
    -------
    list of pathlib.Path
        Shards holding at least one byte; the follow-up ``test`` shards are
        empty files.
    """
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.glob("*.tfrecord"), key=str) if path.stat().st_size]


def _units(session):
    """Return the source units of one session in file order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t5.2022.09.29``.

    Returns
    -------
    list of tuple
        ``(unit, directory, member)`` per unit with a non-empty shard, where
        ``member`` names the MAT file of the unit inside the ZIP. A seed
        session has the units ``train`` and ``test``; a follow-up session has
        ``<arm>/train`` for each arm.
    """
    release = _release()
    found = []
    for partition in PARTITIONS:
        found.append((partition, release / SEED_TREE / "tfrecords" / session / partition,
                      f"{RELEASE}/{SEED_TREE}/mat/{session}.mat"))
    for arm in ARMS:
        for partition in PARTITIONS:
            found.append((f"{arm}/{partition}", release / EVAL_TREE / arm / "tfrecords" / session / partition,
                          f"{RELEASE}/{EVAL_TREE}/{arm}/mat/{session}.mat"))
    return [unit for unit in found if _shards(unit[1])]


def _mat_rows(archive, member):
    """Read the sentence and block rows of one MAT member.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Open publisher ZIP.
    member : str
        MAT member inside the ZIP.

    Returns
    -------
    tuple of list or None
        Sentences and block numbers in file order, or ``None`` when the ZIP
        lacks the member.
    """
    import scipy.io as sio
    if member not in archive.namelist():
        return None
    mat = sio.loadmat(io.BytesIO(archive.read(member)), squeeze_me=True,
                      variable_names=["sentences", "blocks"])
    return ([str(sentence) for sentence in np.atleast_1d(mat["sentences"])],
            [int(block) for block in np.atleast_1d(mat["blocks"]).ravel()])


def label_sets():
    """Return the label vocabulary of the character targets.

    Returns
    -------
    dict of str to list of str
        ``chars``: the 31 handwriting classes of the released decoder in class
        order (``a`` to ``z``, then the word separator ``>``, ``,``, ``'``,
        ``~``, and ``?``). ``char_class_ids`` are zero-based positions in it;
        the released ``seqClassIDs`` are the same ids plus one, with 0 as
        padding.
    """
    sys.path.insert(0, str(Path(study_path(STUDY, "open_source_code_for_reference/CORP/NeuralDecoder"))))
    from neuralDecoder.datasets.handwritingDataset import CHAR_DEF
    return {"chars": list(CHAR_DEF)}


def _sentence(example):
    """Decode the sentence of one example.

    Parameters
    ----------
    example : tensorflow.train.Example
        Parsed publisher example.

    Returns
    -------
    str
        Characters of ``transcription`` up to ``nSeqElements``, with the
        publisher's ``>`` separators and sentence markers.
    """
    features = example.features.feature
    n_seq = int(features["nSeqElements"].int64_list.value[0])
    return "".join(chr(int(code)) for code in features["transcription"].int64_list.value[:n_seq])


def _display(text):
    """Convert the released sentence symbols to display text.

    Parameters
    ----------
    text : str
        Sentence decoded from ``transcription``.

    Returns
    -------
    str
        Stripped text with ``>`` separators as spaces, ``~`` as periods, and
        ``#`` removed, the convention of the archived transcripts.
    """
    return text.strip().replace(">", " ").replace("~", ".").replace("#", "").strip()


def _link(rows, text, expected):
    """Find the MAT row of one trial.

    Parameters
    ----------
    rows : tuple of list or None
        Sentences and block numbers of the unit's MAT file.
    text : str
        Sentence decoded from the trial's ``transcription``.
    expected : int
        Row the trial has when TFRecord and MAT orders agree. Examples are
        counted per MAT file, so the ``test`` examples of a seed session
        follow its ``train`` examples.

    Returns
    -------
    int
        Zero-based row, or -1 when no row holds the sentence or several do.
    """
    if rows is None:
        return -1
    sentences = rows[0]
    if expected < len(sentences) and sentences[expected] == text:
        return expected
    hits = [index for index, sentence in enumerate(sentences) if sentence == text]
    return hits[0] if len(hits) == 1 else -1


def _spec(example, session, unit, relative, source_index, index, link):
    """Build the trial record of one TFRecord example.

    Parameters
    ----------
    example : tensorflow.train.Example
        Parsed publisher example.
    session : str
        Native session identifier such as ``t5.2022.09.29``.
    unit : str
        Source unit name such as ``recalibration/train``.
    relative : str
        Shard path relative to the study directory.
    source_index : int
        Zero-based example position in the shard.
    index : int
        Zero-based trial position in the unit, across its shards.
    link : dict or None
        ``member`` and zero-based ``index`` of the trial's MAT row and its
        ``block`` number, or ``None`` when the trial has no MAT row.

    Returns
    -------
    TrialSpec
        The trial with released features, targets, and MAT reference.
    """
    features = example.features.feature
    n_time = int(features["nTimeSteps"].int64_list.value[0])
    n_seq = int(features["nSeqElements"].int64_list.value[0])
    codes = np.asarray(features["transcription"].int64_list.value, dtype=np.int64)
    ids = np.asarray(features["seqClassIDs"].int64_list.value, dtype=np.int64)
    neural_array = np.asarray(features["inputFeatures"].float_list.value, dtype=np.float32)
    one_hot = np.asarray(features["classLabelsOneHot"].float_list.value, dtype=np.float32)
    where = trial_id(session, unit, index)
    if codes[n_seq:].any() or ids[n_seq:].any():
        raise ValueError(f"{where}: labels continue past nSeqElements={n_seq}")
    if neural_array.size != n_time * N_FEATURES or one_hot.size != n_time * N_CLASSES:
        raise ValueError(f"{where}: feature lengths {neural_array.size} and {one_hot.size} do not match "
                         f"nTimeSteps={n_time}")
    if n_seq and (ids[:n_seq].min() < 1 or ids[:n_seq].max() > N_CLASSES):
        raise ValueError(f"{where}: seqClassIDs run from {ids[:n_seq].min()} to {ids[:n_seq].max()}, "
                         f"not over the 1..{N_CLASSES} classes")
    tree = EVAL_TREE if unit.split("/", 1)[0] in ARMS else SEED_TREE
    neural = {"input_features": Field(neural_array.reshape(n_time, N_FEATURES), bin_ms=BIN_MS,
                                      source_name="inputFeatures", description=DESCRIPTIONS[tree])}
    targets = {
        "sentence": Field(_display(_sentence(example)), source_name="transcription",
                          description="Reference sentence decoded from the released character codes."),
        "transcription": Field(codes[:n_seq], source_name="transcription",
                               description="Character codes of the sentence, cut at nSeqElements."),
        # The common format indexes a label set from zero, so the released ids lose one.
        "char_class_ids": Field(ids[:n_seq] - 1, label_set="chars", source_name="seqClassIDs",
                                description="Zero-based positions in the label set chars, cut at "
                                            "nSeqElements: the released one-based seqClassIDs minus one."),
        "char_class_onehot": Field(one_hot.reshape(n_time, N_CLASSES), source_name="classLabelsOneHot",
                                   description="Released per-frame label matrix. All-zero in every released "
                                               "example, so it is not one-hot and holds no labels."),
    }
    meta = {"source_example_index": index, "n_time_steps": n_time, "n_seq_elements": n_seq}
    if tree == EVAL_TREE:
        meta["source_arm"] = unit.split("/", 1)[0]
    if link is not None:
        meta["source_member_mat"] = link["member"]
        meta["mat_index"] = link["index"]
    return TrialSpec(trial_id=where, source_file=relative, source_index=source_index, neural=neural,
                     source_partition=unit.rsplit("/", 1)[-1],
                     source_block=link["block"] if link else -1, targets=targets, meta=meta)


def list_sessions():
    """List the sessions that have a publisher shard.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in date order. The seed sessions sit directly under
        ``seed_model_training_data/tfrecords``; the follow-up sessions sit one
        level deeper, under the arm they ran in.

    Notes
    -----
    A follow-up session is listed when either arm has a non-empty shard.
    """
    release = _release()
    folders = [release / SEED_TREE / "tfrecords"] + [release / EVAL_TREE / arm / "tfrecords" for arm in ARMS]
    names = sorted({path.name for folder in folders if folder.is_dir()
                    for path in folder.glob("*") if path.is_dir()})
    sessions = []
    for name in names:
        sources = [_relative(shard) for _, directory, _ in _units(name) for shard in _shards(directory)]
        if sources:
            sessions.append({"session": name, "participant": name.split(".")[0], "sources": sources})
    return sessions


def iter_trials(session):
    """Yield every trial of one session in unit and shard order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t5.2022.09.29``.

    Yields
    ------
    TrialSpec
        Trials of each unit in turn, each in shard order.

    Notes
    -----
    Shards are streamed one at a time. Only the sentence and block rows of
    the current MAT member stay in memory; its feature arrays are never read.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    with zipfile.ZipFile(_zip_path()) as archive:
        seen = {}
        for unit, directory, member in _units(session):
            rows = _mat_rows(archive, member)
            count, unlinked = seen.get(member, 0), 0
            for shard in _shards(directory):
                relative = _relative(shard)
                for source_index, record in enumerate(tf.data.TFRecordDataset(str(shard)).as_numpy_iterator()):
                    example = tf.train.Example.FromString(bytes(record))
                    row = _link(rows, _sentence(example), count)
                    if row < 0:
                        unlinked += 1
                        link = None
                    else:
                        link = {"member": member, "index": row, "block": rows[1][row]}
                    yield _spec(example, session, unit, relative, source_index, count, link)
                    count += 1
            seen[member] = count
            if unlinked:
                print(f"note      {session} {unit}: {unlinked} trials have no row in {member}",
                      file=sys.stderr, flush=True)


def _example_at(path, index):
    """Return one TFRecord example without reading the examples before it.

    Parameters
    ----------
    path : pathlib.Path
        Publisher shard.
    index : int
        Zero-based example position in the shard.

    Returns
    -------
    tensorflow.train.Example
        The parsed example.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    with open(path, "rb") as stream:
        for _ in range(index):
            header = stream.read(12)
            if len(header) < 12:
                raise IndexError(f"example {index} is not in {path}")
            # each record is a 12-byte header, its body, and a 4-byte CRC
            stream.seek(int.from_bytes(header[:8], "little") + 4, 1)
        header = stream.read(12)
        if len(header) < 12:
            raise IndexError(f"example {index} is not in {path}")
        record = stream.read(int.from_bytes(header[:8], "little"))
    return tf.train.Example.FromString(record)


def read_source_trial(attrs):
    """Rebuild one trial from the publisher shard named in its attributes.

    Parameters
    ----------
    attrs : dict
        Attributes stored on an exported trial.

    Returns
    -------
    TrialSpec
        The trial as the publisher's shard holds it, with the MAT reference
        recorded at export time.
    """
    session, rest = attrs["trial_id"].split("/", 1)
    unit, number = rest.rsplit("/", 1)
    link = None
    if "source_member_mat" in attrs:
        link = {"member": attrs["source_member_mat"], "index": int(attrs["mat_index"]),
                "block": int(attrs["source_block"])}
    example = _example_at(Path(study_path(STUDY, attrs["source_file"])), int(attrs["source_index"]))
    return _spec(example, session, unit, attrs["source_file"], int(attrs["source_index"]), int(number), link)


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="tfrecord",
        description="Export Fan 2023 handwriting sessions to the common session format."))
