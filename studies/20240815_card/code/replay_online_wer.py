"""Prepare or score a same-trial WFST replay of Card's released online logits."""

import argparse
import csv
import itertools
import json
import pickle
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import editdistance
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "figures/code"))
from extract_online_trials import CARD_DAYS, words

EXPECTED_COUNT = 1628
FIELDS = ("key", "day", "reference", "online_hypothesis")


def prepare(source, run):
    """Freeze the released cohort and export model-order log probabilities.

    Parameters
    ----------
    source : Path
        Trusted Dryad online-output pickle.
    run : Path
        New run directory to create.
    """
    # Only load the authors' trusted Dryad pickle.
    with source.open("rb") as stream:
        data = pickle.load(stream)
    phones = (
        "BLANK SIL AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K "
        "L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH"
    ).split()
    selected = []
    for i, (day, vocab, ref, hyp, logits, raw) in enumerate(zip(
        data["post_implant_day"], data["vocab_size"], data["cue_sentence"],
        data["decoded_sentence"], data["decoded_logits"], data["decoded_phonemes_raw"],
        strict=True,
    )):
        if day not in CARD_DAYS or vocab != 125000:
            continue
        if logits.ndim != 2 or logits.shape[1] != 41 or not np.isfinite(logits).all():
            raise ValueError(f"Invalid logits at pickle row {i}")
        # Check the original online class order against the saved greedy phones.
        greedy = (phones[k] for k, _ in itertools.groupby(logits.argmax(axis=1)) if k)
        if [p for p, _ in itertools.groupby(greedy)] != list(raw):
            raise ValueError(f"Released greedy phones disagree with logits at row {i}")
        if not words(ref) or hyp is None or any(c in ref for c in "\t\r\n"):
            raise ValueError(f"Invalid reference or missing online output at row {i}")
        selected.append((f"{i:04d}", date(2023, 7, 17) + timedelta(days=day), ref, hyp, logits))
    if len(selected) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} trials, got {len(selected)}")
    run.mkdir(parents=True)
    trials = run / "runtime" / "logits"
    trials.mkdir(parents=True)
    with (run / "cohort.csv").open("x", encoding="utf-8", newline="") as cohort, \
            (trials.parent / "meta.tsv").open("x", encoding="utf-8", newline="\n") as meta:
        writer = csv.writer(cohort)
        writer.writerow(FIELDS)
        for key, day, ref, hyp, logits in selected:
            # Published order is BLANK,SIL,phones. C++ expects BLANK,phones,SIL.
            ordered = np.concatenate((logits[:, :1], logits[:, 2:], logits[:, 1:2]), axis=1)
            logp = ordered - np.logaddexp.reduce(ordered, axis=1, keepdims=True)
            np.save(trials / f"{key}.npy", logp.astype(np.float32, copy=False))
            writer.writerow((key, day, ref, hyp))
            meta.write(f"{key}\t{ref}\n")


def audit_neural(run, neural):
    """Audit whether online rows can be joined to the neural release.

    Parameters
    ----------
    run : Path
        Prepared run directory containing the frozen cohort.
    neural : Path
        Root of the extracted native HDF5 release (hdf5_data_final).

    Notes
    -----
    Unique cue matches are diagnostic anchors, not official trial IDs.
    Between consecutive anchors, list the unlabeled recordings that could
    account for unmatched rows if their chronological orders agree.
    """
    import h5py

    by_day = defaultdict(list)
    with (run / "cohort.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_day[row["day"]].append(row)
    with (run / "neural_linkage.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("key", "day", "status", "source_file", "source_index", "source_block",
                         "trial_num", "unmatched_rows_in_gap", "unlabeled_candidates_in_gap"))
        counts = Counter()
        for day, online in sorted(by_day.items()):
            session = "t15." + day.replace("-", ".")
            found = []
            labeled = defaultdict(list)
            for split in ("train", "val", "test"):
                path = neural / session / f"data_{split}.hdf5"
                with h5py.File(path, "r") as source:
                    for index, key in enumerate(sorted(source)):
                        a = source[key].attrs
                        if a["session"] != session:
                            raise ValueError(f"Wrong session: {path}/{key}")
                        label = a.get("sentence_label")
                        found.append((int(a["block_num"]), int(a["trial_num"]), split, index,
                                      None if label is None else tuple(words(label))))
            found.sort()
            if len({item[:2] for item in found}) != len(found):
                raise ValueError(f"Duplicate source block/trial IDs: {session}")
            for pos, (*_, cue) in enumerate(found):
                if cue is not None:
                    labeled[cue].append(pos)
            cues = [tuple(words(r["reference"])) for r in online]
            online_counts = Counter(cues)
            anchors = [(j, labeled[cue][0]) for j, cue in enumerate(cues)
                       if online_counts[cue] == len(labeled[cue]) == 1]
            left = (-1, -1)
            for j, pos in anchors + [(len(online), len(found))]:
                if pos <= left[1]:
                    raise ValueError(f"Diagnostic anchors are out of order: {session}")
                gap = j - left[0] - 1
                unknowns = sum(item[-1] is None for item in found[left[1] + 1:pos])
                for k in range(left[0] + 1, j):
                    status = "unresolved"
                    writer.writerow((online[k]["key"], day, status, "", "", "", "", gap, unknowns))
                    counts[status] += 1
                if j < len(online):
                    block, trial, split, index, _ = found[pos]
                    status = f"{split}_cue_anchor"
                    writer.writerow((online[j]["key"], day, status, f"{session}/data_{split}.hdf5",
                                     index, block, trial, "", ""))
                    counts[status] += 1
                left = j, pos
    print(dict(counts))


def score(run, expected_count=EXPECTED_COUNT, arm="wfst_replay"):
    """Require a one-to-one key and reference match and score both decoders.

    Parameters
    ----------
    run : Path
        Prepared run directory with completed WFST output in final.tsv.
    expected_count : int
        Expected size of the predeclared test set.
    arm : str
        Label for the reproduced hypotheses in the summary.
    """
    with (run / "cohort.csv").open(encoding="utf-8", newline="") as stream:
        cohort = list(csv.DictReader(stream))
    with (run / "final.tsv").open(encoding="utf-8", newline="") as stream:
        outputs = {}
        for key, ref, hyp, *_ in csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            if key == "key":
                continue
            if key in outputs:
                raise ValueError(f"Duplicate decoded trial {key}")
            outputs[key] = (ref, hyp)
    ids = {r["key"] for r in cohort}
    if len(ids) != len(cohort) or len(ids) != expected_count or ids != outputs.keys():
        raise ValueError(f"Wrong trial coverage: {len(ids - outputs.keys())} missing, "
                         f"{len(outputs.keys() - ids)} extra")
    rows = []
    daily = {}
    for row in cohort:
        ref, hyp = outputs[row["key"]]
        if ref != row["reference"]:
            raise ValueError(f"Reference mismatch for trial {row['key']}")
        tokens = words(ref)
        edits = [editdistance.eval(tokens, words(text)) for text in (row["online_hypothesis"], hyp)]
        counts = (1, len(tokens), *edits)
        daily[row["day"]] = [a + b for a, b in zip(daily.get(row["day"], [0] * 4), counts)]
        rows.append((*[row[k] for k in FIELDS], hyp, len(tokens), *edits))
    with (run / "trials.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((*FIELDS, "replay_hypothesis", "reference_words", "online_edits", "replay_edits"))
        writer.writerows(rows)
    with (run / "daily.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("day", "trials", "reference_words", "online_edits", "replay_edits",
                         "online_wer_pct", "replay_wer_pct"))
        for day, (n, length, online, replay) in sorted(daily.items()):
            writer.writerow((day, n, length, online, replay, 100 * online / length, 100 * replay / length))
    totals = [sum(col) for col in zip(*daily.values())]
    summary = {"trials": totals[0], "days": len(daily), "reference_words": totals[1]}
    for idx, name in enumerate(("authors_online", arm), start=2):
        rates = [100 * counts[idx] / counts[1] for counts in daily.values()]
        summary[name] = {
            "edits": totals[idx], "pooled_wer_pct": 100 * totals[idx] / totals[1],
            "mean_daily_wer_pct": statistics.mean(rates), "sample_sd_pp": statistics.stdev(rates),
        }
    with (run / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps(summary, indent=2))


def main():
    """Select the preparation or scoring stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    inputs = commands.add_parser("prepare")
    inputs.add_argument("source", type=Path)
    inputs.add_argument("run", type=Path)
    audit = commands.add_parser("audit-neural")
    audit.add_argument("run", type=Path)
    audit.add_argument("neural", type=Path)
    scoring = commands.add_parser("score")
    scoring.add_argument("run", type=Path)
    scoring.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    scoring.add_argument("--arm", default="wfst_replay")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.source, args.run)
    elif args.stage == "audit-neural":
        audit_neural(args.run, args.neural)
    else:
        score(args.run, args.expected_count, args.arm)


if __name__ == "__main__":
    main()
