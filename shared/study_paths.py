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


def run_output(study_id):
    """Return the output directory for a new dated study run.

    Parameters
    ----------
    study_id : str
        Study directory name.

    Returns
    -------
    str
        Writable path for the run ID supplied in BCI_RUN_ID, or today's local run.
    """
    if study_id not in ARCHIVE_NAMES:
        raise ValueError(f"Unknown study: {study_id}")
    run_id = os.environ.get("BCI_RUN_ID", date.today().strftime("%Y%m%d") + "_local")
    if not re.fullmatch(r"\d{8}_(?!import_)[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id):
        raise ValueError("BCI_RUN_ID must be a dated run name such as 20261124_gru_seed42")
    target = ROOT / "studies" / study_id / "results" / run_id
    target.mkdir(parents=True, exist_ok=True)
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
        return str(Path(run_output(study_id)).joinpath(*parts[1:]))
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
