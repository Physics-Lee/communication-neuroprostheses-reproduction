"""Run the new-study onboarding example from start to a validated session file.

The example adds a small synthetic study in a temporary directory, following
the steps of docs/20261124_repository_migration/onboarding_a_study.md:

1. register the study in the path helper (in memory, standing in for the two
   dictionary edits a real study makes in ``shared/study_paths.py``),
2. write a publisher file and an exporter,
3. export one session with ``export_cli``,
4. validate the session file with ``validate_session``, re-reading every trial.

Nothing is written inside the repository and the temporary directory is
removed at the end.

Usage (from the repository root):

    python shared/tools/onboarding_example.py [--keep DIRECTORY]
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_paths  # noqa: E402
from session_format import export_cli, format_report, validate_session  # noqa: E402

STUDY = "20990101_example"
DATASET_ID = "example_ds001"
EXPORTER = '''# -*- coding: utf-8 -*-
"""Export Example 2000 sessions to the common session format.

One publisher file holds one session. Every trial is stored with its neural
feature, its sentence, and the file and position it came from.

Usage (from the repository root):

    python studies/20990101_example/code/export_sessions.py --list
    python studies/20990101_example/code/export_sessions.py --session x1.2000.01.01
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, {shared!r})
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20990101_example"
DATASET_ID = "example_ds001"
DATASET = "https://doi.org/10.0000/example.dataset"
RAW_DIR = f"data/raw/{{DATASET_ID}}"
BIN_MS = 20
_CACHE = {{"path": None, "mat": None}}


def _relative(path):
    """Return the publisher file path relative to the study directory."""
    return Path(path).resolve().relative_to(Path(study_checkout(STUDY)).resolve()).as_posix()


def _load(path):
    """Load one publisher file, keeping the last file in memory."""
    if _CACHE["path"] != str(path):
        _CACHE.update(path=str(path), mat=loadmat(path, squeeze_me=True))
    return _CACHE["mat"]


def list_sessions():
    """List the publisher files, one per session."""
    directory = Path(study_path(STUDY, RAW_DIR))
    return [{{"session": path.stem.removeprefix(f"{{DATASET_ID}}__"), "sources": [_relative(path)]}}
            for path in sorted(directory.glob(f"{{DATASET_ID}}__*.mat"))]


def _spec(mat, session, relative, index):
    """Build the trial record at one position of a publisher file."""
    features = np.ascontiguousarray(mat["features"][index])
    return TrialSpec(
        trial_id=trial_id(session, "data", index), source_file=relative, source_index=index,
        source_block=int(mat["blocks"][index]),
        neural={{"spike_pow": Field(features, bin_ms=BIN_MS, source_name="features",
                                 description="Spike-band power per electrode.")}},
        targets={{"sentence": Field(str(mat["sentences"][index]).strip(), source_name="sentences")}})


def iter_trials(session):
    """Yield every trial of one session in file order."""
    path = Path(study_path(STUDY, f"{{RAW_DIR}}/{{DATASET_ID}}__{{session}}.mat"))
    relative, mat = _relative(path), _load(path)
    for index in range(mat["sentences"].shape[0]):
        yield _spec(mat, session, relative, index)


def read_source_trial(attrs):
    """Rebuild one trial from the publisher file named in its attributes."""
    return _spec(_load(Path(study_path(STUDY, attrs["source_file"]))),
                 Path(attrs["source_file"]).stem.removeprefix(f"{{DATASET_ID}}__"),
                 attrs["source_file"], int(attrs["source_index"]))


def label_sets():
    """Return an empty mapping: this study stores no label sequences."""
    return {{}}


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat",
        description="Export Example 2000 sessions to the common session format."))
'''


def write_publisher_file(path, seed):
    """Write one synthetic publisher file with three sentence trials.

    Parameters
    ----------
    path : pathlib.Path
        MAT file to create; the parent directory is made when missing.
    seed : int
        Seed of the random feature generator.

    Returns
    -------
    None
    """
    import numpy as np
    from scipy.io import savemat
    generator = np.random.default_rng(seed)
    sentences = ["Is it going outside?", "What is my need?", "I hope you like it up."]
    lengths = [120, 95, 140]
    features = np.empty((1, len(sentences)), dtype=object)
    for index, length in enumerate(lengths):
        features[0, index] = generator.normal(size=(length, 8)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"features": features,
                   "sentences": np.array(sentences, dtype=object).reshape(1, -1),
                   "blocks": np.arange(1, len(sentences) + 1).reshape(-1, 1)}, do_compression=True)


def run(directory):
    """Register, export, and validate the example study.

    Parameters
    ----------
    directory : pathlib.Path
        Empty directory that receives the example study.

    Returns
    -------
    session_format.Report
        The validation report of the exported session file.
    """
    study = directory / STUDY
    raw_dir = Path("data", "raw", DATASET_ID)
    for seed, session in enumerate(("x1.2000.01.01", "x1.2000.01.02")):
        write_publisher_file(study / raw_dir / f"{DATASET_ID}__{session}.mat", seed=seed)
    (study / "data" / "sources.yaml").write_text(
        f"study: {STUDY}\ndataset: https://doi.org/10.0000/example.dataset\n"
        f"dataset_id: {DATASET_ID}\nrelease_id: null\nexternal_input_path: {raw_dir.as_posix()}\n",
        encoding="utf-8",
    )
    entry = study / "code" / "export_sessions.py"
    entry.parent.mkdir(parents=True)
    entry.write_text(EXPORTER.format(shared=str(Path(__file__).resolve().parents[1])), encoding="utf-8")

    study_paths.ARCHIVE_NAMES[STUDY] = STUDY
    study_paths.RELEASE_LAYOUT[STUDY] = (DATASET_ID, {})
    os.environ["BCI_DATA_ROOT"] = str(directory)

    sys.path.insert(0, str(entry.parent))
    sys.argv = [entry.name, "--list"]
    namespace = {"__name__": "onboarding_exporter", "__file__": str(entry)}
    exec(compile(entry.read_text(encoding="utf-8"), str(entry), "exec"), namespace)
    print("--- export_sessions.py --list")
    namespace["export_cli"](STUDY, namespace["list_sessions"], namespace["iter_trials"],
                            namespace["read_source_trial"], namespace["label_sets"], str(entry),
                            dataset=namespace["DATASET"], source_format="mat")
    sys.argv = [entry.name, "--session", "x1.2000.01.01"]
    print("--- export_sessions.py --session x1.2000.01.01")
    status = namespace["export_cli"](STUDY, namespace["list_sessions"], namespace["iter_trials"],
                                     namespace["read_source_trial"], namespace["label_sets"], str(entry),
                                     dataset=namespace["DATASET"], source_format="mat")
    if status:
        raise RuntimeError(f"export failed with status {status}")
    target = study_paths.session_export_path(STUDY, "x1.2000.01.01")
    print(f"--- validate_session.py {target} --source trials")
    return validate_session(target, "trials")


def main():
    """Run the example in a temporary directory and report the outcome.

    Returns
    -------
    int
        Zero when the exported session file passes the source trace, one
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", type=Path, help="use this directory and leave the files in place")
    args = parser.parse_args()
    if args.keep is not None:
        args.keep.mkdir(parents=True, exist_ok=True)
        report = run(args.keep)
    else:
        with tempfile.TemporaryDirectory(prefix="bci_onboarding_") as name:
            report = run(Path(name))
    print("\n".join(format_report(report)))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
