"""Run an independent Card GRU on the recoverable Figure 2 neural recordings."""

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


def ordered_matches(valid):
    """Return only pairs forced in every maximum-cardinality ordered alignment.

    Parameters
    ----------
    valid : ndarray of bool
        Candidate recording mask, indexed by online row and neural row.
    """
    m, n = valid.shape
    left = np.zeros((m + 1, n + 1), dtype=int)
    right = left.copy()
    for i in range(m):
        for j in range(n):
            left[i + 1, j + 1] = max(left[i, j + 1], left[i + 1, j],
                                      left[i, j] + int(valid[i, j]))
    for i in reversed(range(m)):
        for j in reversed(range(n)):
            right[i, j] = max(right[i + 1, j], right[i, j + 1],
                              right[i + 1, j + 1] + int(valid[i, j]))
    best = left[m, n]
    forced = {}
    for i in range(m):
        possible = [j for j in range(n)
                    if valid[i, j] and left[i, j] + 1 + right[i + 1, j + 1] == best]
        can_omit = np.any(left[i] + right[i + 1] == best)
        if not can_omit and len(possible) == 1:
            forced[i] = possible[0]
    return forced


def prepare(replay, source, neural, run):
    """Freeze the neural test linkage and write isolated training/validation views.

    Parameters
    ----------
    replay : Path
        Completed Card online-logit replay directory, supplying the frozen cohort.
    source : Path
        Trusted Dryad pickle, used here only for the count of output frames.
    neural : Path
        Native hdf5_data_final directory.
    run : Path
        New directory for this end-to-end experiment.
    """
    with source.open("rb") as stream:
        online = pickle.load(stream)
    with (replay / "cohort.csv").open(encoding="utf-8", newline="") as stream:
        cohort = {row["key"]: row for row in csv.DictReader(stream)}
    with (replay / "neural_linkage.csv").open(encoding="utf-8", newline="") as stream:
        by_day = defaultdict(list)
        for row in csv.DictReader(stream):
            by_day[row["day"]].append(row)
    neural = neural.resolve()
    records = {}
    for path in sorted(neural.glob("t15.*/data_*.hdf5")):
        with h5py.File(path) as h:
            for i, key in enumerate(sorted(h)):
                a = h[key].attrs
                ident = (a["session"], int(a["block_num"]), int(a["trial_num"]))
                if ident in records:
                    raise ValueError(f"Duplicate recording: {ident}")
                records[ident] = (path, i, key, int(a["n_time_steps"]))
    excluded_blocks = set()
    mapping = {}
    for day, rows in sorted(by_day.items()):
        session = "t15." + day.replace("-", ".")
        ordered = sorted(k for k in records if k[0] == session)
        positions = {k: i for i, k in enumerate(ordered)}
        anchors = [(i, (session, int(row["source_block"]), int(row["trial_num"])))
                   for i, row in enumerate(rows) if row["source_file"]]
        for i, ident in anchors:
            path, index, _, _ = records[ident]
            if (path.relative_to(neural).as_posix(), index) != (
                rows[i]["source_file"], int(rows[i]["source_index"])
            ):
                raise ValueError(f"Anchor provenance mismatch: {ident}")
            mapping[rows[i]["key"]] = (ident, "cue_and_order")
            excluded_blocks.add(ident[:2])
        # The authors switched from a 40 ms to an 80 ms RNN step. Offsets
        # are calibrated on the labeled anchors, separately for block starts.
        implant_day = online["post_implant_day"][int(rows[0]["key"])]
        stride = 2 if implant_day <= 34 else 4
        prev_i, prev_pos = -1, -1
        for i, pos in [(i, positions[k]) for i, k in anchors] + [(len(rows), len(ordered))]:
            gap = rows[prev_i + 1:i]
            candidates = ordered[prev_pos + 1:pos]
            if gap:
                # Do not let plausible recordings influence fit/selection, even
                # when the corresponding online row cannot be assigned uniquely.
                # Labeled rows with different cues are not timing candidates.
                unknown = [k for k in candidates if records[k][0].stem == "data_test"]
                excluded_blocks.update(k[:2] for k in unknown)
                valid = np.zeros((len(gap), len(unknown)), dtype=bool)
                for j, row in enumerate(gap):
                    length = len(online["decoded_logits"][int(row["key"])])
                    for k, ident in enumerate(unknown):
                        offset = records[ident][-1] - stride * length
                        low, high = ((17, 25) if ident[-1] == 0 else
                                     (75, 81) if stride == 2 else (65, 79))
                        valid[j, k] = low <= offset <= high
                for j, k in ordered_matches(valid).items():
                    mapping[gap[j]["key"]] = (unknown[k], "frame_count_and_order")
            prev_i, prev_pos = i, pos
    if not set(mapping) <= cohort.keys() or len({k for k, _ in mapping.values()}) != len(mapping):
        raise ValueError("Trial mapping is not one-to-one")
    run.mkdir(parents=True)
    view = run / "runtime" / "neural_view"
    grouped = defaultdict(list)
    for ident, (path, _, key, _) in sorted(records.items()):
        if ident[:2] not in excluded_blocks and path.stem in {"data_train", "data_val"}:
            grouped[path].append((ident, key))
    train_days = {p.parent.name for p in grouped if p.stem == "data_train"}
    if {k[0] for k, _ in mapping.values()} - train_days:
        raise ValueError("A target recording day has no training data for its input layer")
    counts = Counter()
    with (run / "partitions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("role", "session", "block", "trial", "source_file", "source_key"))
        for path, items in grouped.items():
            relative = path.relative_to(neural)
            destination = view / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            role = path.stem.removeprefix("data_")
            with h5py.File(destination, "x") as out:
                for index, (ident, key) in enumerate(items):
                    out[f"trial_{index:04d}"] = h5py.ExternalLink(path.as_posix(), f"/{key}")
                    writer.writerow((role, *ident, relative.as_posix(), key))
                    counts[role] += 1
                assert out["trial_0000"]["input_features"].shape[1] == 512
    with (run / "cohort.csv").open("x", encoding="utf-8", newline="") as stream, \
            (run / "unresolved.csv").open("x", encoding="utf-8", newline="") as missing:
        fields = (*next(iter(cohort.values())), "linkage", "session", "block", "trial",
                  "source_file", "source_key")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        missing_writer = csv.DictWriter(missing, fieldnames=list(next(iter(cohort.values()))))
        missing_writer.writeheader()
        for key, row in cohort.items():
            if key not in mapping:
                missing_writer.writerow(row)
                continue
            ident, linkage = mapping[key]
            path, _, group, _ = records[ident]
            writer.writerow(dict(row, linkage=linkage, session=ident[0], block=ident[1],
                                 trial=ident[2], source_file=path.relative_to(neural).as_posix(),
                                 source_key=group))
            counts[linkage] += 1
    print(dict(counts), "unresolved:", len(cohort) - len(mapping), flush=True)
    (run / "split_summary.json").write_text(
        json.dumps(dict(counts, unresolved=len(cohort) - len(mapping)), indent=2) + "\n",
        encoding="utf-8")


def main():
    """Execute the requested preparation stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("neural", type=Path)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    prepare(args.replay, args.source, args.neural, args.run)


if __name__ == "__main__":
    main()
