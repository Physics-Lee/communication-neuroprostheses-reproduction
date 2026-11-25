# Onboarding a new study

2026-11-24

A new study joins the repository in six steps. The last step is the same
validation command the eight existing studies use. To watch the whole
sequence run on a small synthetic study in a temporary directory, execute

```text
python shared/tools/onboarding_example.py
```

It ends with `[PASS]` and `source trace: 3/3 trials re-read and identical`
and leaves nothing behind.

## 1. Create the study directory

```text
studies/<yyyymmdd>_<author>/
├── README.md
├── AGENTS.md
├── tutorial.ipynb
├── code/
│   ├── README.md
│   └── export_sessions.py
├── data/
│   ├── sources.yaml
│   ├── raw/<dataset_id>/<dataset_id>__<publisher file>
│   └── splits/
└── results/<run_id>/
```

The [repository layout](README.md#repository-layout) explains each entry.

## 2. Register the source

`data/sources.yaml` names the dataset for readers:

```yaml
study: 20990101_example
dataset: https://doi.org/10.0000/example.dataset
dataset_id: example_ds001
release_id: null
external_input_path: data/raw/example_ds001/example_ds001__example_release.mat
```

`shared/study_paths.py` names it for the code. Add one entry to
`ARCHIVE_NAMES` and one to `RELEASE_LAYOUT`:

```python
ARCHIVE_NAMES["20990101_example"] = "20990101_example"
RELEASE_LAYOUT["20990101_example"] = ("example_ds001", {
    "data/raw/example_release": "data/raw/example_release",
})
```

The dataset ID is used in paths and file names until the provider's release
version is verified. Record the verified version as `release_id` when it is.

## 3. Place the publisher's file

Save the file where `sources.yaml` says, under `data/raw/<dataset_id>/`, with
the dataset ID as the file-name prefix. Scripts reach it through
`study_path("<study>", "<path>")`, so a reader can keep the data on another
disk by setting `BCI_DATA_ROOT`.

## 4. Write the exporter

`code/export_sessions.py` is the only file that needs to know the publisher's
format. It has four parts:

| Part | Returns |
| --- | --- |
| `list_sessions()` | The sessions the study can export, in source order |
| `iter_trials(session)` | One `TrialSpec` per trial, in source order |
| `read_source_trial(attrs)` | The same `TrialSpec` again, rebuilt from the attributes stored on a trial |
| `label_sets()` | The vocabularies that label targets refer to |

Three rules hold for every exporter. Neural features are stored as released,
without normalization or resampling. A numeric target that names a
`label_set` holds zero-based positions in that vocabulary; when the publisher
numbers its classes from one, the exporter shifts them and says so in the
target's description. Trial roles, folds, and calibration order stay out of
the session file, in a split list under `data/splits/` that
`code/build_split_manifest.py` writes from the exported files.

This is the complete exporter of the example study:

```python
STUDY = "20990101_example"
DATASET = "https://doi.org/10.0000/example.dataset"
BIN_MS = 20
_CACHE = {"path": None, "mat": None}


def _relative(path):
    return Path(path).resolve().relative_to(Path(study_checkout(STUDY)).resolve()).as_posix()


def _load(path):
    if _CACHE["path"] != str(path):
        _CACHE.update(path=str(path), mat=loadmat(path, squeeze_me=True))
    return _CACHE["mat"]


def list_sessions():
    directory = Path(study_path(STUDY, "data/raw/example_release"))
    return [{"session": path.stem, "sources": [_relative(path)]}
            for path in sorted(directory.glob("*.mat"))]


def _spec(mat, session, relative, index):
    features = np.ascontiguousarray(mat["features"][index])
    return TrialSpec(
        trial_id=trial_id(session, "data", index), source_file=relative, source_index=index,
        source_block=int(mat["blocks"][index]),
        neural={"spike_pow": Field(features, bin_ms=BIN_MS, source_name="features",
                                 description="Spike-band power per electrode.")},
        targets={"sentence": Field(str(mat["sentences"][index]).strip(), source_name="sentences")})


def iter_trials(session):
    path = Path(study_path(STUDY, f"data/raw/example_release/{session}.mat"))
    relative, mat = _relative(path), _load(path)
    for index in range(mat["sentences"].shape[0]):
        yield _spec(mat, session, relative, index)


def read_source_trial(attrs):
    return _spec(_load(Path(study_path(STUDY, attrs["source_file"]))),
                 Path(attrs["source_file"]).stem, attrs["source_file"], int(attrs["source_index"]))


def label_sets():
    return {}


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat"))
```

Four rules keep the export verifiable:

- Store features as released, with the publisher's dtype and values.
- Record where each trial came from: `source_file` relative to the study
  directory, `source_index`, `source_block`, and for a trial cut out of a
  continuous recording `source_onset_frame` and `source_offset_frame`.
- Build `read_source_trial` on the same code as `iter_trials`. The validator
  compares what it returns with what the file holds.
- Keep splits out of the file. Training, validation, and fold assignments go
  into `data/splits/<release>__<split>.csv` lists that name `trial_id`.

The layout of the file is defined in
[`shared/session_format.py`](../../shared/session_format.py).

## 5. Export one session

```text
python studies/20990101_example/code/export_sessions.py --list
python studies/20990101_example/code/export_sessions.py --session x1.2000.01.01
```

The file is written to `data/processed/x1/example_ds001_2000-01-01.h5`. The
name joins the dataset ID and the session identifier without its participant
code. `--all` exports every session, one at a time, after a free-space check.

## 6. Verify with the shared command

```text
python shared/tools/validate_session.py studies/20990101_example/data/processed/x1/example_ds001_2000-01-01.h5 --source trials
```

The command checks the file structure and re-reads every trial from the
publisher's file. A file passes when the two agree.

## After the first session

- Add the study's split lists under `data/splits/` and describe them in
  `data/splits/README.md`.
- Describe the commands in `code/README.md` and put the data setup at the top
  of `tutorial.ipynb`, as the eight studies do.
- Recompute one result through the study's commands before adding the study to
  `figures/reproduction_benchmark.yaml`.
- Run `python shared/tools/check_study_tutorials.py --run` and
  `python shared/tools/session_export_inventory.py --write`.
