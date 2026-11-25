"""Resolve active study code, external inputs, and dated run output paths."""

import os
import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAMES = {
    "20210512_willett": "20261009_reproduce_20210512_willett",
    "20230831_willett": "20261008_reproduce_20230831_willett",
    "20231201_fan": "20261022_reproduce_20231201_fan",
    "20240815_card": "20261008_reproduce_20240815_card",
    "20250612_wairagkar": "20261009_reproduce_20250612_wairagkar",
    "20250821_kunz": "20261008_reproduce_20250821_kunz",
    "20260316_jude": "20261009_reproduce_20260316_jude",
    "20260428_jude": "20261008_reproduce_20260428_jude",
}

UPSTREAM_ALIASES = {
    "20210512_willett": ("handwritingBCI", "open_source_code_for_reference/20210512_willett_handwritingBCI"),
    "20231201_fan": ("CORP", "open_source_code_for_reference/20231201_fan_CORP"),
    "20240815_card": ("nejm-brain-to-text", "nejm-brain-to-text"),
    "20250612_wairagkar": ("brain-to-voice-2025", "open_source_code_for_reference/20250612_wairagkar_brain-to-voice-2025"),
    "20260316_jude": ("typingBCI", "open_source_code_for_reference/20260316_jude_typingBCI"),
}

# Keep the old names resolvable for external data trees and historical checkouts.
DATA_ALIASES = {
    "20210512_willett": {
        "data/20210512_willett/handwritingBCIData": "data/extracted/handwritingBCIData",
    },
    "20230831_willett": {
        "data/20230616_willett_BtoT24/competitionData": "data/extracted/competitionData",
        "data/20230616_willett_BtoT24/hdf5": "data/processed/hdf5",
        "data/20230616_willett_BtoT24/btot24_description.csv": "data/processed/session_description.csv",
        "data/raw/20230616_willett_BtoT24/languageModel_work/languageModel": "data/language_models/kenlm_3gram",
        "data/raw/20230616_willett_BtoT24/languageModel_5gram_work": "data/language_models/wfst_5gram",
    },
    "20231201_fan": {
        "data/20231201_fan/CORP_data_release": "data/extracted/CORP_data_release",
        "data/raw/20231201_fan/CORP_data_release.zip": "data/raw/CORP_data_release.zip",
    },
    "20240815_card": {
        "data/20241009_card_BtoT25/hdf5_data_final": "data/extracted/hdf5_data_final",
        "data/20241009_card_BtoT25/t15_copyTaskData_description.csv": "data/extracted/t15_copyTaskData_description.csv",
        "data/20241009_card_BtoT25/t15_pretrained_rnn_baseline": "data/checkpoints/t15_pretrained_rnn_baseline",
        "data/raw/20241009_card_BtoT25/t15_copyTask.pkl": "data/raw/t15_copyTask.pkl",
    },
    "20250612_wairagkar": {
        "data/raw/20250612_wairagkar/t15_neural_data.zip": "data/raw/t15_neural_data.zip",
    },
    "20250821_kunz": {
        "data/20250821_kunz/sentenceData": "data/extracted/sentenceData",
        "data/20250821_kunz/hdf5": "data/processed/hdf5",
        "data/20250821_kunz/hdf5_t16holdout": "data/processed/hdf5_t16holdout",
        "data/20250821_kunz/kunz_description.csv": "data/processed/session_description.csv",
    },
    "20260316_jude": {
        "data/20260316_jude/Data": "data/extracted/Data",
        "data/20260316_jude/LanguageModels/Typing5gramPunc": "data/language_models/wfst_5gram",
    },
    "20260428_jude": {
        "data/20260428_jude/Data": "data/extracted/Data",
        "data/20260428_jude/hdf5": "data/processed/hdf5",
        "data/20260428_jude/jude_description.csv": "data/processed/session_description.csv",
    },
}

# The local data use the verified dataset IDs until provider versions are known.
# Each path template is relative to the study and is keyed by its previous path.
RELEASE_LAYOUT = {
    "20210512_willett": ("dryad_wh70rxwmv", {
        "data/extracted/handwritingBCIData": "data/extracted/{release}/handwritingBCIData",
    }),
    "20230831_willett": ("dryad_x69p8czpq", {
        "data/extracted/competitionData": "data/extracted/{release}/competitionData",
        "data/processed/hdf5": "data/derived/{release}/hdf5/trainer_input",
        "data/processed/session_description.csv": "data/derived/{release}/hdf5/{release}__session_description.csv",
    }),
    "20231201_fan": ("dryad_hqbzkh1p6", {
        "data/extracted/CORP_data_release": "data/extracted/{release}/CORP_data_release",
        "data/raw/CORP_data_release.zip": "data/raw/{release}/{release}__CORP_data_release.zip",
    }),
    "20240815_card": ("dryad_dncjsxm85", {
        "data/extracted/hdf5_data_final": "data/extracted/{release}/hdf5_data_final",
        "data/extracted/t15_copyTaskData_description.csv": "data/extracted/{release}/t15_copyTaskData_description.csv",
        "data/checkpoints/t15_pretrained_rnn_baseline": "data/checkpoints/{release}/t15_pretrained_rnn_baseline",
        "data/raw/t15_copyTask.pkl": "data/raw/{release}/{release}__t15_copyTask.pkl",
    }),
    "20250612_wairagkar": ("dryad_2280gb64f", {
        "data/raw/t15_neural_data.zip": "data/raw/{release}/{release}__t15_neural_data.zip",
    }),
    "20250821_kunz": ("dryad_gf1vhhn1j", {
        "data/extracted/sentenceData": "data/extracted/{release}/sentenceData",
        "data/processed/hdf5": "data/derived/{release}/hdf5/trainer_input",
        "data/processed/hdf5_t16holdout": "data/derived/{release}/hdf5/t16_holdout",
        "data/processed/session_description.csv": "data/derived/{release}/hdf5/{release}__session_description.csv",
    }),
    "20260316_jude": ("dryad_cz8w9gjjk", {
        "data/extracted/Data": "data/extracted/{release}/Data",
    }),
    "20260428_jude": ("dryad_vq83bk481", {
        "data/extracted/Data": "data/extracted/{release}/Data",
        "data/processed/hdf5": "data/derived/{release}/hdf5/trainer_input",
        "data/processed/session_description.csv": "data/derived/{release}/hdf5/{release}__session_description.csv",
    }),
}

# Native files keep the filenames expected by each trainer inside the run.
RUNTIME_RUNS = {
    "20230831_willett": {
        "run_60k": "20260831_dryad_x69p8czpq__hdf5__ctc_greedy__run_60k",
    },
    "20240815_card": {
        "run_120k": "20260902_dryad_dncjsxm85__hdf5__ctc_greedy__run_120k",
        "run_excl5_40k": "20260903_dryad_dncjsxm85__hdf5__ctc_greedy__excl5_40k",
        "run_p4_20261013": "20261013_dryad_dncjsxm85__hdf5__ctc_greedy__p4",
        "run_calib_20261014": "20261014_dryad_dncjsxm85__hdf5__ctc_greedy__online_calibration",
    },
    "20250821_kunz": {
        **{
            f"run_{participant}_{arm}": f"{started}_dryad_gf1vhhn1j__hdf5__ctc_greedy__{participant}_{arm}"
            for participant in ("t12", "t15", "t16")
            for arm, started in (("mixed", "20260901"), ("imagined", "20260902"), ("attempted", "20260902"))
        },
        **{f"run_t16_holdout_fold{fold}": f"20260902_dryad_gf1vhhn1j__hdf5__ctc_greedy__t16_holdout_fold{fold}"
           for fold in (1, 2, 3)},
        "vac": "20261001_dryad_gf1vhhn1j__npy__wfst_5gram__decoder_sweep",
    },
    "20260428_jude": {
        "run_25k": "20260901_dryad_vq83bk481__hdf5__ctc_greedy__run_25k",
        "vac": "20261002_dryad_vq83bk481__npy__wfst_5gram__decoder_sweep",
    },
}
RUNTIME_LEGACY_DIRS = {
    "20230831_willett": "20230616_willett_BtoT24",
    "20240815_card": "20241009_card_BtoT25",
    "20250821_kunz": "20250821_kunz",
    "20260428_jude": "20260428_jude",
}

DEFAULT_WORKFLOWS = {
    "20210512_willett": ("mat", "wfst_bigram", "handwriting_decode"),
    "20230831_willett": ("csv", "none", "benchmark_plots"),
    "20231201_fan": ("tfrecord", "char_5gram_beam", "corp_replay"),
    "20240815_card": ("hdf5", "ctc_greedy", "checkpoint_eval"),
    "20250612_wairagkar": ("zip", "none", "voice_synthesis"),
    "20250821_kunz": ("csv", "none", "ablation_figures"),
    "20260316_jude": ("npz", "ctc_greedy", "multisession"),
    "20260428_jude": ("hdf5", "ctc_greedy", "checkpoint_eval"),
}
RUN_TOKEN = r"[a-z0-9]+(?:_[a-z0-9]+)*"
RUN_PATTERN = re.compile(
    rf"(?P<date>\d{{8}})_(?P<release>{RUN_TOKEN})__(?P<input_format>{RUN_TOKEN})"
    rf"__(?P<decoder>{RUN_TOKEN})__(?P<experiment>{RUN_TOKEN})"
)


def runtime_path(study_id, run, *parts, checkout=None):
    """Resolve an existing study runtime package or one of its files.

    Parameters
    ----------
    study_id : str
        Study directory name.
    run : str
        Original run name used by the study commands.
    *parts : str or pathlib.Path
        Path components inside the native run package.
    checkout : pathlib.Path, optional
        Explicit study directory.

    Returns
    -------
    pathlib.Path
        Canonical runtime or log path. An external checkout can still
        supply the original run directory.
    """
    base = Path(checkout) if checkout is not None else study_checkout(study_id)
    legacy = base / "data" / RUNTIME_LEGACY_DIRS[study_id] / run
    run_id = RUNTIME_RUNS[study_id][run]
    current = base / "results" / run_id / "runtime"
    tail = Path(*parts)
    if tail == Path("training_log"):
        log = base / "log" / run_id / tail
        return log if log.is_file() or not legacy.is_dir() else legacy / tail
    return current / tail if current.is_dir() or not legacy.is_dir() else legacy / tail


def study_checkout(study_id):
    """Locate a study's local data and upstream dependencies.

    Parameters
    ----------
    study_id : str
        Study directory name.

    Returns
    -------
    pathlib.Path
        Active study directory or the explicitly configured data directory.

    Notes
    -----
    ``BCI_DATA_ROOT`` uses the public layout with one directory per study.
    ``BCI_ARCHIVE_ROOT`` selects the historical checkout naming scheme.
    """
    if study_id not in ARCHIVE_NAMES:
        raise ValueError(f"Unknown study: {study_id}")
    if data_root := os.environ.get("BCI_DATA_ROOT"):
        return Path(data_root).expanduser() / study_id
    if archive_root := os.environ.get("BCI_ARCHIVE_ROOT"):
        return Path(archive_root).expanduser() / ARCHIVE_NAMES[study_id]
    return ROOT / "studies" / study_id


def build_run_id(study_id, input_format, decoder, experiment):
    """Name a run using the study's release and today's local date.

    Parameters
    ----------
    study_id : str
        Study directory name.
    input_format, decoder, experiment : str
        Lowercase run components. Use ``none`` for a run without a text decoder.

    Returns
    -------
    str
        Run ID in the repository's release, format, decoder, and experiment layout.
    """
    run_id = f"{date.today():%Y%m%d}_" + "__".join(
        (release_prefix(study_id), input_format, decoder, experiment))
    if not RUN_PATTERN.fullmatch(run_id):
        raise ValueError(f"Invalid standard run name: {run_id}")
    return run_id


def run_output(study_id, *, default_id=None, workflow=None, log=False):
    """Return the output directory for a new dated study run.

    Parameters
    ----------
    study_id : str
        Study directory name.
    default_id : str, optional
        Run ID to use when BCI_RUN_ID is unset.
    workflow : tuple of str, optional
        Input format, decoder, and experiment for the default name.
    log : bool, optional
        Return the corresponding log directory.

    Returns
    -------
    str
        Writable path for the selected standard run ID. Result directories
        receive a run record on first use.
    """
    if study_id not in ARCHIVE_NAMES:
        raise ValueError(f"Unknown study: {study_id}")
    default_id = default_id or build_run_id(study_id, *(workflow or DEFAULT_WORKFLOWS[study_id]))
    run_id = os.environ.get("BCI_RUN_ID", default_id)
    match = RUN_PATTERN.fullmatch(run_id)
    if not match:
        short = re.fullmatch(r"(\d{8})_(?!import_)([a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*)", run_id)
        if short and (default := RUN_PATTERN.fullmatch(default_id)):
            prefix = "__".join(default.group(key) for key in ("release", "input_format", "decoder"))
            run_id = f"{short[1]}_{prefix}__{short[2].lower().replace('-', '_')}"
            match = RUN_PATTERN.fullmatch(run_id)
    if not match or match["release"] != release_prefix(study_id):
        raise ValueError("BCI_RUN_ID must use this study's release and the dated run naming format")
    target = ROOT / "studies" / study_id / ("log" if log else "results") / run_id
    target.mkdir(parents=True, exist_ok=True)
    if not log:
        manifest = (target.parent.parent / "data" / "sources.yaml")
        source = yaml.safe_load(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        record = {
            "id": run_id,
            "study": study_id,
            "dataset_id": source.get("dataset_id", release_prefix(study_id)),
            "release_id": source.get("release_id"),
            "input_format": match["input_format"],
            "decoder": None if match["decoder"] == "none" else match["decoder"],
            "experiment": match["experiment"],
        }
        try:
            with (target / "run.yaml").open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(record, stream, sort_keys=False)
        except FileExistsError:
            pass
    return str(target)


def study_data_path(study_id, relative, checkout=None):
    """Resolve a preferred data path or its equivalent historical path.

    Parameters
    ----------
    study_id : str
        Study directory name.
    relative : str or pathlib.Path
        Path within a study, including the leading data directory.
    checkout : pathlib.Path, optional
        Explicit study directory if the command accepts one.

    Returns
    -------
    pathlib.Path
        Path in the preferred layout, or in a historical layout that exists.
    """
    base = Path(checkout) if checkout is not None else study_checkout(study_id)
    relative = Path(relative)
    runtime_dir = RUNTIME_LEGACY_DIRS.get(study_id)
    if runtime_dir and relative.parts[:2] == ("data", runtime_dir) and len(relative.parts) >= 3:
        _, _, run, *tail = relative.parts
        if run in RUNTIME_RUNS[study_id]:
            return runtime_path(study_id, run, *tail, checkout=base)
    release, templates = RELEASE_LAYOUT.get(study_id, ("", {}))
    for original, previous in DATA_ALIASES.get(study_id, {}).items():
        preferred = templates.get(previous, previous).format(release=release)
        candidates = tuple(dict.fromkeys(map(Path, (preferred, previous, original))))
        for alias in candidates:
            if relative.is_relative_to(alias):
                tail = relative.relative_to(alias)
                location = next(
                    (base / candidate for candidate in candidates if (base / candidate).exists()),
                    base / candidates[0],
                )
                return location / tail
    return base / relative


def release_prefix(study_id):
    """Return the identifier that prefixes generated data file names.

    Parameters
    ----------
    study_id : str
        Study directory name.

    Returns
    -------
    str
        The verified provider release recorded in ``data/sources.yaml``, or
        the dataset ID while the local version is unverified.

    Notes
    -----
    The manifest is read from the repository copy of the study, which Git
    tracks, so the rule does not depend on a local data tree.
    """
    if study_id not in ARCHIVE_NAMES:
        raise ValueError(f"Unknown study: {study_id}")
    prefix = RELEASE_LAYOUT.get(study_id, ("", {}))[0]
    manifest = ROOT / "studies" / study_id / "data" / "sources.yaml"
    if manifest.is_file():
        with manifest.open(encoding="utf-8") as stream:
            recorded = yaml.safe_load(stream) or {}
        prefix = str(recorded.get("release_id") or recorded.get("dataset_id") or prefix)
    return prefix


def derived_data_path(study_id, relative, *, legacy=None):
    """Resolve one trainer artifact in the release-scoped derived tree.

    Parameters
    ----------
    study_id : str
        Study directory name.
    relative : str or pathlib.Path
        Path below the release directory, starting with the file format.
    legacy : str or pathlib.Path, optional
        Previous study-relative input path. Writers omit this argument.

    Returns
    -------
    pathlib.Path
        The canonical path, or the existing legacy artifact if needed.
    """
    path = study_checkout(study_id) / "data" / "derived" / release_prefix(study_id) / relative
    if legacy is not None and not path.exists():
        previous = study_data_path(study_id, legacy)
        if previous.exists():
            return previous
    return path


def session_file_name(study_id, session):
    """Name the common session file of one native session identifier.

    Parameters
    ----------
    study_id : str
        Study directory name.
    session : str
        Native session identifier such as ``t12.2022.05.19`` or ``t15_day195``.

    Returns
    -------
    tuple of str
        ``(participant, file name)``. The participant is the code before the
        first ``.`` or ``_``; the file name joins the release prefix and the
        remaining identifier with dots replaced by dashes, for example
        ``dryad_x69p8czpq_2022-05-19.h5``.

    Raises
    ------
    ValueError
        When the identifier does not start with a participant code.
    """
    parts = re.split(r"[._]", session, maxsplit=1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError(f"Session {session!r} does not start with a participant code such as t12.")
    return parts[0], f"{release_prefix(study_id)}_{parts[1].replace('.', '-')}.h5"


def session_export_path(study_id, session, checkout=None):
    """Return where a study exports one session in the common HDF5 format.

    Parameters
    ----------
    study_id : str
        Study directory name.
    session : str
        Native session identifier.
    checkout : pathlib.Path, optional
        Explicit study directory if the command accepts one.

    Returns
    -------
    pathlib.Path
        ``data/processed/<participant>/<release>_<date>.h5`` inside the study,
        for example ``data/processed/t12/dryad_x69p8czpq_2022-05-19.h5``.
    """
    participant, name = session_file_name(study_id, session)
    return study_data_path(study_id, Path("data", "processed", participant, name), checkout)


def study_path(study_id, relative=""):
    """Resolve a path from the original study checkout layout.

    Parameters
    ----------
    study_id : str
        Study directory name.
    relative : str
        Path within the study's data, upstream code, or runtime outputs.

    Returns
    -------
    str
        Platform-native absolute path.
    """
    parts = Path(relative).parts
    if parts and parts[0] == "code":
        return str(ROOT.joinpath("studies", study_id, *parts))
    if parts and parts[0] == "result":
        tail = parts[1:]
        workflow = None
        if tail and study_id == "20250612_wairagkar" and tail[0].endswith(".json"):
            workflow = ("zip", "none", Path(tail[0]).stem)
            tail = ("summary.json", *tail[1:])
        elif tail and study_id == "20231201_fan" and tail[0].endswith(".json"):
            stage = Path(tail[0]).stem
            if stage == "tier2_seed_model_inference":
                workflow = ("tfrecord", "ctc_greedy", stage)
            elif stage == "rescore_official_outputs":
                workflow = ("mat", "char_5gram_beam", stage)
            else:
                workflow = ("tfrecord", "char_5gram_beam", stage)
            tail = ("summary.json", *tail[1:])
        elif tail and study_id == "20240815_card":
            if tail[0] == "val_greedy_per_trial.csv":
                tail = ("trials.csv", *tail[1:])
            elif tail[0] == "val_greedy_summary.json":
                tail = ("summary.json", *tail[1:])
        return str(Path(run_output(study_id, workflow=workflow)).joinpath(*tail))
    if parts and parts[0] == "data":
        return str(study_data_path(study_id, relative))
    checkout = study_checkout(study_id)
    if study_id in UPSTREAM_ALIASES:
        name, legacy = UPSTREAM_ALIASES[study_id]
        preferred = Path("open_source_code_for_reference", name)
        previous = Path(legacy)
        requested = Path(relative)
        for alias in (preferred, previous):
            if requested.is_relative_to(alias):
                suffix = requested.relative_to(alias)
                for base in (checkout, ROOT / "studies" / study_id):
                    for candidate in (preferred, previous):
                        if (base / candidate).is_dir():
                            return str(base / candidate / suffix)
                return str(ROOT / "studies" / study_id / preferred / suffix)
    return str(checkout.joinpath(*parts))
