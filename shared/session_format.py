"""Common per-session HDF5 format for the eight open-data BCI studies.

Each study exports one ``.h5`` file per recording session in the same layout
(schema ``bci-session``, version 1), so readers, validators, and experiments do
not need to know the publisher's container. The module provides the record
types (``Field``, ``TrialSpec``), the ``SessionWriter``, the ``read_session``
reader, ``validate_session``, and ``export_cli``. A study exporter supplies
``TrialSpec`` objects and keeps the knowledge of its native format.

Layout
------
::

    <root>                 attrs: schema, schema_version, study, dataset_id, participant,
                           session, source_format, exporter, created_utc, n_trials,
                           dataset and release_id when known
    /trial_ids             (n,) str      stable source trial ids, in /trials key order
    /label_sets/<name>     (k,) str      vocabulary referenced by a target's label_set
    /trials/t0000          one group per trial, in source order
        attrs: trial_id, source_file, source_index, source_block,
               source_onset_frame, source_offset_frame, n_frames,
               source_partition when the publisher labels partitions,
               plus other scalar fields recorded by the publisher
        /neural/<feature>  (n_frames, channels) numeric   attrs: bin_ms, source_name, ...
        /targets/<name>    scalar str or numeric sequence attrs: label_set, source_name, ...
        /timing/<name>     (n, 2) int   [onset, offset) in frames from the trial start

A numeric target with a ``label_set`` attribute holds zero-based positions in
that vocabulary, whatever convention the publisher uses (some release
one-based ids with 0 as padding); the exporter shifts them and says so in the
target's description. A publisher's own normalization, bin size, and channel
order of the neural features are kept as released.

Trial ids follow ``<session>/<unit>/<source_index:04d>``. ``<unit>`` names the
source unit when one session spans several, such as a publisher partition
(``test``) or a block (``block05``), and is ``data`` otherwise. ``source_file``
is relative to the study directory. Trials carved from a continuous recording
store the half-open range ``[source_onset_frame, source_offset_frame)``;
trials the publisher stores one by one leave both at -1.

Split assignments (training, validation, folds, calibration, and adaptation
order) are not stored in the session file. They live in
``data/splits/<release>__<split>.csv`` manifests that reference ``trial_id``,
so different experiments can share the same exported trials.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_paths import release_prefix, session_export_path, study_path  # noqa: E402

SCHEMA = "bci-session"
SCHEMA_VERSION = 1
ROOT_ATTRS = ("schema", "schema_version", "study", "dataset_id", "participant", "session",
              "exporter", "created_utc", "n_trials")
TRIAL_ATTRS = ("trial_id", "source_file", "source_index", "source_block",
               "source_onset_frame", "source_offset_frame", "n_frames")
GROUPS = ("neural", "targets", "timing")
TEXT = h5py.string_dtype(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# Standard feature names shared by several studies. An exporter may store a
# feature the publisher defines without a counterpart elsewhere; the validator
# then reports it as a non-standard name.
NEURAL_FEATURES = {
    "spike_pow": "Spike-band power binned per electrode.",
    "tx": "Threshold-crossing counts binned per electrode.",
    "neural_activity": "Binned neural activity released per sentence.",
    "input_features": "Feature vector combined by the publisher before release.",
}
# Free space an export requires before each session file is written.
MIN_FREE_BYTES = 2 << 30
# Errors or warnings listed per report before the rest are counted.
MAX_DETAIL_LINES = 12


class Field:
    """One dataset of a trial and the attributes stored with it.

    Parameters
    ----------
    data : numpy.ndarray or str
        Numeric array with time on the first axis, or a text value.
    **attrs
        Dataset attributes such as ``bin_ms``, ``source_name``, ``label_set``,
        or ``description``.
    """

    def __init__(self, data, **attrs):
        self.data = data
        self.attrs = attrs


@dataclass
class TrialSpec:
    """Everything one exported trial records.

    Parameters
    ----------
    trial_id : str
        Stable source trial id.
    source_file : str
        Publisher file holding the trial, relative to the study directory.
    source_index : int
        Zero-based position of the trial in that file.
    neural : dict of str to Field
        Neural feature arrays, each ``(n_frames, channels)``.
    source_partition : str, optional
        Publisher partition label such as ``train`` or ``test``.
    source_block : int, optional
        Publisher block number, or -1 when the source has no blocks.
    source_onset_frame, source_offset_frame : int, optional
        Half-open frame range in the source recording, both -1 when the
        publisher stores trials one by one.
    targets : dict of str to Field, optional
        Task targets: a ``sentence`` text, label sequences, or time-aligned
        acoustic features.
    timing : dict of str to Field, optional
        ``(n, 2)`` onset and offset frames relative to the trial start.
    meta : dict, optional
        Other scalar fields recorded by the publisher.
    """

    trial_id: str
    source_file: str
    source_index: int
    neural: dict
    source_partition: str = ""
    source_block: int = -1
    source_onset_frame: int = -1
    source_offset_frame: int = -1
    targets: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def trial_id(session, unit, index):
    """Build the stable trial id of one source trial.

    Parameters
    ----------
    session : str
        Native session identifier, for example ``t12.2022.05.19``.
    unit : str
        Source unit name, or ``data`` when the session has a single unit.
    index : int
        Zero-based index of the trial inside the unit.

    Returns
    -------
    str
        Identifier such as ``t12.2022.05.19/test/0000``.
    """
    return f"{session}/{unit}/{index:04d}"


def _write_field(group, name, fld, level):
    """Write one ``Field`` as a dataset under ``group``.

    Parameters
    ----------
    group : h5py.Group
        Parent ``neural``, ``targets``, or ``timing`` group.
    name : str
        Dataset name.
    fld : Field
        Value and attributes to store.
    level : int
        gzip compression level; the shuffle filter is added for multi-byte
        types.

    Returns
    -------
    None
    """
    if isinstance(fld.data, str):
        dataset = group.create_dataset(name, data=fld.data, dtype=TEXT)
    else:
        array = np.asarray(fld.data)
        if array.dtype.kind not in "fiub" or not array.dtype.isnative:
            raise TypeError(f"{name}: expected a native numeric array, got {array.dtype}")
        dataset = group.create_dataset(
            name, data=array, compression="gzip", compression_opts=level,
            shuffle=array.dtype.itemsize > 1, chunks=True)
    for key, value in fld.attrs.items():
        dataset.attrs[key] = value


def _scalar(value):
    """Return ``value`` as a type HDF5 attributes accept, or raise.

    Parameters
    ----------
    value : object
        Candidate attribute value.

    Returns
    -------
    str, int, float, or bool
        The same value as a plain Python scalar.

    Raises
    ------
    TypeError
        When the value is not a scalar.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"trial attributes must be scalars, got {type(value).__name__}")


def read_dataset(dataset):
    """Read a whole dataset, returning text datasets as ``str``.

    Parameters
    ----------
    dataset : h5py.Dataset
        Dataset of a trial.

    Returns
    -------
    numpy.ndarray, str, or numpy scalar
        The stored content.
    """
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return dataset.asstr()[()]
    return dataset[()]


class SessionWriter:
    """Write one session file; the file appears only after a successful close.

    Parameters
    ----------
    path : str or pathlib.Path
        Target ``.h5`` path. The parent directory is created when missing.
    study : str
        Study directory name, for example ``20230831_willett``.
    dataset_id : str
        Identifier that prefixes generated file names, for example
        ``dryad_x69p8czpq``.
    participant : str
        Participant code such as ``t12``.
    session : str
        Native session identifier.
    exporter : str
        Repository-relative path of the exporting script.
    dataset : str, optional
        Dataset DOI or URL.
    release_id : str, optional
        Verified provider release; omitted while the local version is
        unverified.
    source_format : str, optional
        Publisher container family, such as ``mat`` or ``hdf5``.
    label_sets : dict of str to list of str, optional
        Vocabularies that targets reference by name.
    overwrite : bool, optional
        Replace an existing file. Default is False.
    compression : int, optional
        gzip level. Default is 4.

    Notes
    -----
    Use as a context manager. Output goes to ``<path>.tmp`` and is renamed to
    ``path`` when the block finishes without error, so an interrupted export
    never leaves a partial session file behind.
    """

    def __init__(self, path, *, study, dataset_id, participant, session, exporter,
                 dataset="", release_id="", source_format="", label_sets=None,
                 overwrite=False, compression=4):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"Session file already exists (pass overwrite to replace): {self.path}")
        self._tmp = self.path.with_name(self.path.name + ".tmp")
        self._file = h5py.File(self._tmp, "w")
        self._level = compression
        self._ids = []
        self._n = 0
        attrs = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "study": study,
                 "dataset_id": dataset_id, "participant": participant, "session": session,
                 "exporter": exporter, "source_format": source_format,
                 "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if dataset:
            attrs["dataset"] = dataset
        if release_id:
            attrs["release_id"] = release_id
        for key, value in attrs.items():
            self._file.attrs[key] = value
        self._file.create_group("trials")
        self._file.create_group("label_sets")
        for name, labels in (label_sets or {}).items():
            self._file["label_sets"].create_dataset(
                name, data=np.asarray(list(labels), dtype=TEXT))

    def __enter__(self):
        """Return the writer.

        Returns
        -------
        SessionWriter
            The writer itself.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the file, keeping the target only after a successful block.

        Parameters
        ----------
        exc_type, exc, tb
            Exception information supplied by the ``with`` statement.

        Returns
        -------
        None
        """
        self._file.close()
        if exc_type is None:
            os.replace(self._tmp, self.path)
        else:
            self._tmp.unlink(missing_ok=True)

    @property
    def attrs(self):
        """h5py.AttributeManager: root attributes of the file being written.

        Session-level facts a study wants to record, such as the decoder a
        recording day ran with, go here.
        """
        return self._file.attrs

    def add_trial(self, spec):
        """Append one trial as the next ``/trials/tNNNN`` group.

        Parameters
        ----------
        spec : TrialSpec
            Trial to write.

        Returns
        -------
        str
            Group key of the new trial, such as ``t0007``.

        Raises
        ------
        ValueError
            If a neural array is not two-dimensional, the arrays of one trial
            disagree on the number of frames, or ``meta`` reuses a reserved
            attribute name.
        """
        if not spec.neural:
            raise ValueError(f"{spec.trial_id}: a trial needs at least one neural feature")
        frames = {}
        for name, fld in spec.neural.items():
            array = np.asarray(fld.data)
            if array.ndim != 2:
                raise ValueError(f"{spec.trial_id}: neural feature {name} must be (frames, channels)")
            frames[name] = array.shape[0]
        if len(set(frames.values())) != 1:
            raise ValueError(f"{spec.trial_id}: neural features disagree on frames: {frames}")
        clash = set(spec.meta) & set(TRIAL_ATTRS + ("source_partition",))
        if clash:
            raise ValueError(f"{spec.trial_id}: meta reuses reserved attributes {sorted(clash)}")
        meta = {name: _scalar(value) for name, value in spec.meta.items()}
        key = f"t{self._n:04d}"
        group = self._file["trials"].create_group(key)
        attrs = {"trial_id": spec.trial_id, "source_file": spec.source_file,
                 "source_index": int(spec.source_index), "source_block": int(spec.source_block),
                 "source_onset_frame": int(spec.source_onset_frame),
                 "source_offset_frame": int(spec.source_offset_frame),
                 "n_frames": int(next(iter(frames.values())))}
        if spec.source_partition:
            attrs["source_partition"] = spec.source_partition
        for name, value in attrs.items():
            group.attrs[name] = value
        for name in GROUPS:
            contents = getattr(spec, name)
            if contents:
                group.create_group(name)
                for field_name, fld in contents.items():
                    _write_field(group[name], field_name, fld, self._level)
        for name, value in meta.items():
            if isinstance(value, float) and np.isnan(value):
                continue
            group.attrs[name] = value
        self._ids.append(spec.trial_id)
        self._n += 1
        return key

    def close(self):
        """Finish the file: store trial ids and the trial count.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            When two trials share an id.
        """
        if len(set(self._ids)) != len(self._ids):
            raise ValueError("trial ids are not unique")
        self._file.create_dataset("trial_ids", data=np.asarray(self._ids, dtype=TEXT))
        self._file.attrs["n_trials"] = self._n


class Trial:
    """Lazy view of one exported trial.

    Parameters
    ----------
    session : Session
        Session the trial belongs to.
    key : str
        Group key under ``/trials``, such as ``t0007``.
    attrs : dict
        Attributes stored on the trial group.

    Notes
    -----
    Arrays are read on demand. The parent session must stay open, so keep the
    ``Session`` alive (or use it as a context manager) while using its trials.
    """

    def __init__(self, session, key, attrs):
        self.session = session
        self.key = key
        self.attrs = dict(attrs)
        self.trial_id = self.attrs["trial_id"]

    @property
    def group(self):
        """h5py.Group: the trial's ``/trials/<key>`` group."""
        return self.session.file["trials"][self.key]

    def __getitem__(self, name):
        """Read a dataset by path relative to the trial.

        Parameters
        ----------
        name : str
            Dataset path such as ``neural/spike_pow`` or ``targets/phonemes``.

        Returns
        -------
        numpy.ndarray or str
            The stored value.
        """
        return read_dataset(self.group[name])

    def names(self, group):
        """List the dataset names under ``neural``, ``targets``, or ``timing``.

        Parameters
        ----------
        group : str
            One of ``neural``, ``targets``, ``timing``.

        Returns
        -------
        list of str
            Dataset names, empty when the group is absent.
        """
        return sorted(self.group[group].keys()) if group in self.group else []

    def neural(self, name=None):
        """Read one neural feature or all of them.

        Parameters
        ----------
        name : str, optional
            Feature name. When omitted, every feature is read.

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray
            A ``(frames, channels)`` array, or a mapping from name to array.
        """
        if name is not None:
            return self[f"neural/{name}"]
        return {feature: self[f"neural/{feature}"] for feature in self.names("neural")}

    def target(self, name=None):
        """Read one target or all of them.

        Parameters
        ----------
        name : str, optional
            Target name. When omitted, every target is read.

        Returns
        -------
        numpy.ndarray, str, or dict
            The stored value, or a mapping from name to value.
        """
        if name is not None:
            return self[f"targets/{name}"]
        return {target: self[f"targets/{target}"] for target in self.names("targets")}

    def timing(self, name=None):
        """Read one timing table or all of them.

        Parameters
        ----------
        name : str, optional
            Timing name such as ``word``. When omitted, every table is read.

        Returns
        -------
        numpy.ndarray or dict of str to numpy.ndarray
            An ``(n, 2)`` array of onset and offset frames, or a mapping.
        """
        if name is not None:
            return self[f"timing/{name}"]
        return {table: self[f"timing/{table}"] for table in self.names("timing")}

    @property
    def sentence(self):
        """str or None: the ``sentence`` target when the trial has one."""
        return self["targets/sentence"] if "sentence" in self.names("targets") else None


class Session:
    """An open common session file.

    Parameters
    ----------
    path : str or pathlib.Path
        Session ``.h5`` file to open read-only.

    Notes
    -----
    ``read_session`` returns this object; use it as a context manager so the
    file closes on exit. Only the trial index is read up front and arrays load
    when a ``Trial`` is accessed, which keeps one session at a time within the
    memory budget.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        self.attrs = dict(self.file.attrs)
        self.label_sets = {name: list(ds.asstr()[:])
                           for name, ds in self.file["label_sets"].items()}
        self._keys = sorted(self.file["trials"].keys())
        self._ids = list(self.file["trial_ids"].asstr()[:])

    def __enter__(self):
        """Return the session.

        Returns
        -------
        Session
            The session itself.
        """
        return self

    def __exit__(self, *exc):
        """Close the file.

        Parameters
        ----------
        *exc
            Exception information supplied by the ``with`` statement.

        Returns
        -------
        None
        """
        self.close()

    def close(self):
        """Close the underlying file.

        Returns
        -------
        None
        """
        self.file.close()

    def __len__(self):
        """Return the number of trials.

        Returns
        -------
        int
            Number of trials in the session.
        """
        return len(self._keys)

    def trial(self, key):
        """Return one trial by position, group key, or trial id.

        Parameters
        ----------
        key : int or str
            Zero-based position, a key such as ``t0007``, or a trial id.

        Returns
        -------
        Trial
            The matching trial.

        Raises
        ------
        KeyError
            When no trial matches ``key``.
        """
        if isinstance(key, int):
            group_key = self._keys[key]
        elif key in self._ids:
            group_key = self._keys[self._ids.index(key)]
        elif key in self._keys:
            group_key = key
        else:
            raise KeyError(f"no trial {key!r} in {self.path.name}")
        return Trial(self, group_key, self.file["trials"][group_key].attrs)

    @property
    def trials(self):
        """list of Trial: every trial in source order (lazy handles)."""
        return [self.trial(index) for index in range(len(self))]

    def select(self, **attrs):
        """Return the trials whose attributes equal every given value.

        Parameters
        ----------
        **attrs
            Attribute values to match, for example ``source_partition="test"``.

        Returns
        -------
        list of Trial
            Matching trials in source order.
        """
        return [trial for trial in self.trials
                if all(trial.attrs.get(name) == value for name, value in attrs.items())]

    def where(self, predicate):
        """Return the trials for which ``predicate(trial)`` is true.

        Parameters
        ----------
        predicate : callable
            Function from ``Trial`` to bool.

        Returns
        -------
        list of Trial
            Matching trials in source order.
        """
        return [trial for trial in self.trials if predicate(trial)]


def read_session(path):
    """Open a common session file.

    Parameters
    ----------
    path : str or pathlib.Path
        Session ``.h5`` file.

    Returns
    -------
    Session
        The opened session; close it or use it in a ``with`` block.
    """
    return Session(path)


@dataclass
class Report:
    """Outcome of validating one session file.

    Parameters
    ----------
    path : str
        Validated file.
    errors : list of str
        Failures that make the file invalid.
    warnings : list of str
        Findings that do not invalidate the file.
    summary : dict
        Study, participant, session, counts, and source tracing results.
    """

    path: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self):
        """bool: True when no error was recorded."""
        return not self.errors


def load_exporter(exporter):
    """Import a study exporter from its repository-relative path.

    Parameters
    ----------
    exporter : str
        Path recorded in the session file, relative to the repository root.

    Returns
    -------
    module
        The imported exporter module.
    """
    path = Path(exporter) if Path(exporter).is_absolute() else ROOT / exporter
    if not path.is_file():
        raise FileNotFoundError(f"exporter not found: {path}")
    name = "bci_exporter_" + re.sub(r"\W+", "_", Path(exporter).with_suffix("").as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _same(stored, source):
    """Compare a stored value with the value re-read from the source.

    Parameters
    ----------
    stored, source : numpy.ndarray or str
        Values to compare.

    Returns
    -------
    bool
        True when both hold the same content.
    """
    if isinstance(stored, str) or isinstance(source, str):
        return stored == source
    stored, source = np.asarray(stored), np.asarray(source)
    return stored.shape == source.shape and np.array_equal(stored, source, equal_nan=True)


def validate_session(path, source="files"):
    """Check a session file's structure and trace its trials to their source.

    Parameters
    ----------
    path : str or pathlib.Path
        Session ``.h5`` file.
    source : str, optional
        Depth of the source check. ``none`` checks structure only, ``files``
        also requires every recorded source file to exist, and ``trials``
        re-reads each trial from the publisher's file through the study
        exporter and requires identical content. Default is ``files``.

    Returns
    -------
    Report
        Errors, warnings, and a summary of the file.

    Notes
    -----
    For ``trials``, the exporter named in the file must expose
    ``read_source_trial(attrs)``, which rebuilds a ``TrialSpec`` from the
    attributes stored on a trial.
    """
    report = Report(str(path))
    errors, warnings, summary = report.errors, report.warnings, report.summary
    try:
        session = Session(path)
    except OSError as exc:
        errors.append(f"cannot open as a common session file: {exc}")
        return report
    with session:
        attrs = session.attrs
        for name in ROOT_ATTRS:
            if name not in attrs:
                errors.append(f"missing root attribute {name}")
        if attrs.get("schema") != SCHEMA:
            errors.append(f"schema is {attrs.get('schema')!r}, expected {SCHEMA!r}")
        if int(attrs.get("schema_version", -1)) != SCHEMA_VERSION:
            errors.append(f"schema_version is {attrs.get('schema_version')!r}, expected {SCHEMA_VERSION}")
        summary.update({name: attrs.get(name) for name in
                        ("study", "dataset_id", "participant", "session")})
        keys = session._keys
        ids = [session.trial(key).trial_id for key in keys]
        if attrs.get("n_trials") != len(keys) or session._ids != ids:
            errors.append("n_trials, /trial_ids, and /trials disagree")
        if keys != [f"t{index:04d}" for index in range(len(keys))]:
            errors.append("trial groups are not t0000..t{n-1} without gaps")
        if len(set(ids)) != len(ids):
            errors.append("trial ids are not unique")
        summary["trials"] = len(keys)
        sources, features, targets, timings = {}, {}, {}, {}
        nonstandard = []
        for trial in session.trials:
            where = trial.trial_id
            if any(name not in trial.attrs for name in TRIAL_ATTRS):
                errors.append(f"{where}: missing one of {TRIAL_ATTRS}")
                continue
            n_frames = int(trial.attrs["n_frames"])
            group = trial.group
            for name in GROUPS:
                if name in group and not len(group[name]):
                    errors.append(f"{where}: empty {name} group")
            names = trial.names("neural")
            if not names:
                errors.append(f"{where}: no neural feature")
            for name in names:
                dataset = group["neural"][name]
                if dataset.ndim != 2 or dataset.shape[0] != n_frames:
                    errors.append(f"{where}: neural/{name} shape {dataset.shape} does not start with {n_frames} frames")
                if dataset.dtype.kind not in "fiub":
                    errors.append(f"{where}: neural/{name} is not numeric")
                if int(dataset.attrs.get("bin_ms", 0)) <= 0:
                    errors.append(f"{where}: neural/{name} has no positive bin_ms")
                if name not in NEURAL_FEATURES and name not in nonstandard:
                    nonstandard.append(name)
                features.setdefault(name, []).append(dataset.shape[1])
            for name in trial.names("targets"):
                dataset = group["targets"][name]
                targets[name] = targets.get(name, 0) + 1
                label_set = dataset.attrs.get("label_set")
                if label_set is not None and not isinstance(read_dataset(dataset), str):
                    values = dataset[:]
                    if label_set not in session.label_sets:
                        errors.append(f"{where}: targets/{name} references unknown label set {label_set!r}")
                    elif values.size and (values.min() < 0 or values.max() >= len(session.label_sets[label_set])):
                        errors.append(f"{where}: targets/{name} holds ids outside label set {label_set!r}")
            for name in trial.names("timing"):
                dataset = group["timing"][name]
                timings[name] = timings.get(name, 0) + 1
                table = dataset[:]
                if dataset.ndim != 2 or dataset.shape[1] != 2 or dataset.dtype.kind not in "iu":
                    errors.append(f"{where}: timing/{name} must be an integer (n, 2) table")
                elif table.size and ((table[:, 0] < 0).any() or (table[:, 1] < table[:, 0]).any()):
                    errors.append(f"{where}: timing/{name} has a negative onset or an offset before its onset")
                elif table.size and (table[:, 1] > n_frames).any() and "timing passes the trial end" not in warnings:
                    warnings.append("timing passes the trial end in at least one trial")
            onset, offset = int(trial.attrs["source_onset_frame"]), int(trial.attrs["source_offset_frame"])
            if (onset < 0) != (offset < 0):
                errors.append(f"{where}: only one of the source frame limits is set")
            if onset >= 0 and offset - onset != n_frames:
                errors.append(f"{where}: source frame range spans {offset - onset} frames, trial has {n_frames}")
            if int(trial.attrs["source_index"]) < 0:
                errors.append(f"{where}: negative source_index")
            source_file = str(trial.attrs["source_file"])
            if Path(source_file).is_absolute() or ".." in Path(source_file).parts:
                errors.append(f"{where}: source_file must be relative to the study directory")
            sources[source_file] = sources.get(source_file, 0) + 1
        summary.update(neural={name: {"channels": sorted(set(v)), "trials": len(v)} for name, v in features.items()},
                       targets=targets, timing=timings, sources=sources)
        if nonstandard:
            warnings.append("feature names not shared across studies: " + ", ".join(nonstandard))
        if source == "none":
            return report
        for source_file, count in sources.items():
            try:
                location = study_path(attrs.get("study"), source_file)
            except (ValueError, TypeError) as exc:
                errors.append(f"cannot locate {source_file}: {exc}")
                continue
            if not Path(location).exists():
                errors.append(f"source file not found ({count} trials): {source_file}")
        if source != "trials" or errors:
            return report
        try:
            exporter = load_exporter(str(attrs.get("exporter")))
        except (OSError, KeyError, ImportError, AttributeError) as exc:
            errors.append(f"cannot load exporter {attrs.get('exporter')!r}: {exc}")
            return report
        traced = 0
        for trial in session.trials:
            where = trial.trial_id
            try:
                spec = exporter.read_source_trial(trial.attrs)
            except Exception as exc:  # noqa: BLE001 - report it and keep checking other trials
                errors.append(f"{where}: source could not be read: {type(exc).__name__}: {exc}")
                continue
            if spec.trial_id != where:
                errors.append(f"{where}: source reader returned {spec.trial_id!r}")
            provenance = {"source_index": int(spec.source_index), "source_block": int(spec.source_block),
                          "source_onset_frame": int(spec.source_onset_frame),
                          "source_offset_frame": int(spec.source_offset_frame)}
            provenance.update({name: getattr(spec, name) for name in ("source_partition",) if getattr(spec, name)})
            for name, value in provenance.items():
                if trial.attrs.get(name) != value and not (
                        isinstance(value, float) and np.isnan(value)):
                    errors.append(f"{where}: {name} is {trial.attrs.get(name)!r}, source reader gives {value!r}")
            for name, value in spec.meta.items():
                if name not in trial.attrs:
                    errors.append(f"{where}: source reader records {name}, the file does not")
                elif trial.attrs[name] != value and not (isinstance(value, float) and np.isnan(value)):
                    errors.append(f"{where}: {name} is {trial.attrs[name]!r}, source reader gives {value!r}")
            stored = {name: trial.names(name) for name in GROUPS}
            expected = {name: sorted(getattr(spec, name)) for name in GROUPS}
            if stored != expected:
                errors.append(f"{where}: stored {stored} differs from source {expected}")
                continue
            for name in GROUPS:
                for item in expected[name]:
                    dataset = trial.group[name][item]
                    original = getattr(spec, name)[item].data
                    value = read_dataset(dataset)
                    if not _same(value, original):
                        errors.append(f"{where}: {name}/{item} differs from the source")
                    elif not isinstance(value, str) and dataset.dtype != np.asarray(original).dtype:
                        warnings.append(f"{where}: {name}/{item} stored as {dataset.dtype}, source is {np.asarray(original).dtype}")
            traced += 1
        summary["traced"] = traced
    return report


def format_report(report):
    """Render a validation report as printable lines.

    Parameters
    ----------
    report : Report
        Report returned by ``validate_session``.

    Returns
    -------
    list of str
        One status line followed by indented detail lines.
    """
    s = report.summary
    lines = [f"[{'PASS' if report.ok else 'FAIL'}] {report.path}"]
    if s:
        lines.append(f"    study={s.get('study')} dataset={s.get('dataset_id')} participant={s.get('participant')} "
                     f"session={s.get('session')} trials={s.get('trials')}")
        for name, info in s.get("neural", {}).items():
            lines.append(f"    neural/{name}: channels={info['channels']} in {info['trials']} trials")
        for group in ("targets", "timing"):
            if s.get(group):
                lines.append(f"    {group}: " + ", ".join(f"{k} ({v})" for k, v in s[group].items()))
        for source, count in s.get("sources", {}).items():
            lines.append(f"    source ({count} trials): {source}")
        if "traced" in s:
            lines.append(f"    source trace: {s['traced']}/{s.get('trials')} trials re-read and identical")
    for sign, items in (("!", report.errors), ("?", report.warnings)):
        lines.extend(f"    {sign} {item}" for item in items[:MAX_DETAIL_LINES])
        if len(items) > MAX_DETAIL_LINES:
            lines.append(f"    {sign} ... {len(items) - MAX_DETAIL_LINES} more")
    return lines


def _required_free(directory):
    """Return the free space an export needs before writing the next session.

    Parameters
    ----------
    directory : pathlib.Path
        Directory that will hold the session file.

    Returns
    -------
    int
        Bytes to keep free: at least ``MIN_FREE_BYTES``, and twice the largest
        session file already exported beside it.
    """
    reserve = MIN_FREE_BYTES
    for existing in directory.glob("*.h5"):
        reserve = max(reserve, 2 * existing.stat().st_size)
    return reserve


def export_cli(study, list_sessions, iter_trials, read_source_trial, label_sets, entry,
               dataset="", source_format="", description=None, extra_attrs=None):
    """Run a study exporter from the command line.

    Parameters
    ----------
    study : str
        Study directory name.
    list_sessions : callable
        Returns the exportable sessions as dicts with ``session``, optional
        ``participant``, and ``sources`` keys, in source order.
    iter_trials : callable
        Takes a native session identifier and yields ``TrialSpec`` objects.
    read_source_trial : callable
        Rebuilds one ``TrialSpec`` from stored trial attributes.
    label_sets : callable
        Returns the label vocabularies used by the study's targets.
    entry : str
        Path of the exporter script (``__file__``).
    dataset : str, optional
        Dataset DOI or URL.
    source_format : str, optional
        Publisher container family.
    description : str, optional
        One-line summary shown by ``--help``.
    extra_attrs : callable, optional
        Takes a native session identifier and returns additional root
        attributes, such as the decoder a session ran with.

    Returns
    -------
    int
        Process exit code: zero when every requested session was written or
        already present, one otherwise.

    Notes
    -----
    Arguments are ``--list``, ``--session ID`` (repeatable), ``--all``,
    ``--output PATH`` (single session only), and ``--overwrite``. Files go to
    ``data/processed/<participant>/<dataset_id>_<date>.h5`` inside the study
    directory. Sessions are written one at a time and each write is preceded
    by a free-space check.
    """
    parser = argparse.ArgumentParser(description=description or __doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print the exportable sessions and exit")
    parser.add_argument("--session", action="append", default=[], help="native session identifier (repeatable)")
    parser.add_argument("--all", action="store_true", help="export every session")
    parser.add_argument("--output", type=Path, help="write this path instead of data/processed (one session only)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing session files")
    args = parser.parse_args()
    catalog = list_sessions()
    known = {item["session"]: item for item in catalog}
    if args.list:
        for item in catalog:
            target = session_export_path(study, item["session"])
            state = "exported" if target.exists() else "pending"
            print(f"{state:9s} {item['session']}  ->  {target}")
        return 0
    chosen = [item["session"] for item in catalog] if args.all else args.session
    if not chosen:
        parser.error("choose --list, --session ID, or --all")
    unknown = [name for name in chosen if name not in known]
    if unknown:
        parser.error(f"unknown session(s) {unknown}; run --list")
    if args.output is not None and len(chosen) != 1:
        parser.error("--output needs exactly one --session")
    vocab = label_sets()
    try:
        exporter = Path(entry).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        exporter = str(Path(entry).resolve())
    failed = 0
    for name in chosen:
        target = args.output or session_export_path(study, name)
        if target.exists() and not args.overwrite:
            print(f"skip      {name}: {target} exists")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        free, required = shutil.disk_usage(target.parent).free, _required_free(target.parent)
        if free < required:
            print(f"STOP      {name}: {free / 2**30:.1f} GiB free under {target.parent}, "
                  f"{required / 2**30:.1f} GiB required before each session")
            return 1
        count = 0
        try:
            with SessionWriter(target, study=study, dataset_id=release_prefix(study),
                               participant=known[name].get("participant") or re.split(r"[._]", name)[0],
                               session=name, exporter=exporter, dataset=dataset,
                               source_format=source_format, label_sets=vocab,
                               overwrite=args.overwrite) as writer:
                for key, value in (extra_attrs(name) if extra_attrs else {}).items():
                    writer.attrs[key] = value
                for spec in iter_trials(name):
                    writer.add_trial(spec)
                    count += 1
                writer.close()
        except Exception as exc:  # noqa: BLE001 - report the session and stop the batch
            print(f"FAIL      {name}: {type(exc).__name__}: {exc}")
            failed += 1
            break
        print(f"written   {name}: {count} trials, {target.stat().st_size / 2**20:.1f} MiB -> {target}", flush=True)
    return 1 if failed else 0
