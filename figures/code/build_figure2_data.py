"""Summarize or rebuild the daily input for Figure 2(c,d).

Notes
-----
Run without arguments to verify and summarize the committed daily CSV.
Pass --sources with a manifest of the public inputs listed in README.md to
regenerate the CSV. The renderer uses the same daily summary functions.
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import date
from math import isfinite
from pathlib import Path
import re
from statistics import mean, stdev


if __package__:
    from .figure_paths import PUBLISHED, ROOT
else:
    from figure_paths import PUBLISHED, ROOT

DAILY_CSV = PUBLISHED / "figure2cd_daily.csv"
COUNT_FIELDS = (
    "series", "day", "released_trials", "wer_trials", "word_errors", "reference_words",
    "speed_trials", "speed_count", "speed_unit", "seconds", "source",
)
RATE_FIELDS = ("wer_pct", "speed_wpm")
FIELDS = COUNT_FIELDS + RATE_FIELDS
SPEED_DIVISORS = {"words": 1, "characters": 5}


def daily_rates_from_counts(row, context="daily record"):
    """Validate counts and durations and calculate the daily rates.

    Parameters
    ----------
    row : dict
        Daily trial counts, word edits, reference counts, and timing fields.
    context : str, optional
        Record identifier used in validation errors.

    Returns
    -------
    wer_pct, speed_wpm : float or None
        WER as a percentage and reference speed in words per minute. A
        missing metric is None.

    Raises
    ------
    ValueError
        If the counts or duration are invalid, or both metrics are missing.
    """
    released = int(row["released_trials"])
    wer = speed = None
    if row["wer_trials"]:
        trials, errors, words = (int(row[name]) for name in
                                 ("wer_trials", "word_errors", "reference_words"))
        if not 0 < trials <= released or errors < 0 or words < trials:
            raise ValueError(f"Invalid WER counts at {context}")
        wer = 100 * errors / words
    elif row["word_errors"] or row["reference_words"]:
        raise ValueError(f"Incomplete WER counts at {context}")
    if row["speed_trials"]:
        trials, count = int(row["speed_trials"]), int(row["speed_count"])
        elapsed = float(row["seconds"])
        if (not 0 < trials <= released or count < trials or not isfinite(elapsed)
                or elapsed <= 0 or row["speed_unit"] not in SPEED_DIVISORS):
            raise ValueError(f"Invalid speed counts at {context}")
        speed = 60 * count / (elapsed * SPEED_DIVISORS[row["speed_unit"]])
    elif any(row[name] for name in ("speed_count", "speed_unit", "seconds")):
        raise ValueError(f"Incomplete speed counts at {context}")
    if wer is None and speed is None:
        raise ValueError(f"No eligible metrics at {context}")
    return wer, speed


def format_daily_rates(wer_pct, speed_wpm):
    """Format the daily rate columns for CSV export.

    Parameters
    ----------
    wer_pct, speed_wpm : float or None
        Daily WER as a percentage and speed in words per minute.

    Returns
    -------
    dict of str
        Rate columns rounded to six decimal places. Missing rates are blank.
    """
    return {field: "" if value is None else f"{value:.6f}"
            for field, value in zip(RATE_FIELDS, (wer_pct, speed_wpm), strict=True)}


def read_daily(path):
    """Validate daily records and calculate rates from their denominators.

    Parameters
    ----------
    path : path-like
        Path to a daily CSV with the column names in FIELDS. Earlier CSVs
        with only the count fields are also supported.

    Returns
    -------
    dict of dict
        Mapping from series identifiers to date-keyed daily tuples.
        Each tuple contains WER as a percentage, speed in words per minute,
        and the reference unit used for speed. Missing rates are None.

    Raises
    ------
    ValueError
        If the schema is invalid, a series contains duplicate dates, or a
        record has missing metadata, incomplete counts, or an invalid
        denominator, or rate columns inconsistent with their counts.
    """
    series = defaultdict(dict)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if fields not in (FIELDS, COUNT_FIELDS):
            raise ValueError(f"Unexpected column names in {path}")
        for row in reader:
            key, day = row["series"], row["day"]
            context = f"{path}:{reader.line_num} ({key}, {day})"
            date.fromisoformat(day)
            if not key or not row["source"] or day in series[key]:
                raise ValueError(f"Missing source or duplicate day at {context}")
            wer, speed = daily_rates_from_counts(row, context)
            if fields == FIELDS:
                for field, expected in format_daily_rates(wer, speed).items():
                    actual = row[field]
                    if actual != expected and (
                        not actual or not expected or float(actual) != float(expected)
                    ):
                        raise ValueError(f"{field} does not match daily counts at {context}")
            series[key][day] = (wer, speed, row["speed_unit"])
    return series


def summarize_daily(path):
    """Summarize equally weighted daily WER and speed by series.

    Parameters
    ----------
    path : path-like
        Path to the daily CSV.

    Returns
    -------
    dict of dict
        Per-series means, sample standard deviations, included dates, and
        the speed count unit. Dates are sorted and listed separately for
        each metric. A metric without daily observations has no mean, and
        sample standard deviation requires at least two days.

    Raises
    ------
    ValueError
        If the daily CSV fails validation or a series mixes speed count units.
    """
    result = {}
    for key, days in read_daily(path).items():
        summary = {}
        for index, metric in enumerate(("wer", "speed_wpm")):
            included = {day: row[index] for day, row in sorted(days.items())
                        if row[index] is not None}
            values = list(included.values())
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else None
            summary[f"{metric}_dates"] = list(included)
        units = {row[2] for row in days.values() if row[2]}
        if len(units) > 1:
            raise ValueError(f"Inconsistent speed units for {key}")
        summary["speed_count_unit"] = next(iter(units), None)
        result[key] = summary
    return result


def apply_daily_metrics(data, yaml_path):
    """Populate the Figure 2(c,d) panel inputs from the daily CSV.

    Parameters
    ----------
    data : dict
        Parsed benchmark configuration. Rows with a daily-series identifier
        receive the computed statistics and plot labels in place. A
        configuration without a daily CSV reference requires no updates.
    yaml_path : path-like
        Benchmark YAML path used to resolve the relative daily CSV path.

    Raises
    ------
    KeyError
        If a configured daily-series identifier has no summary in the CSV.
    ValueError
        If a series exceeds the configured number of released days, the CSV
        contains unused series, or the daily summary fails validation.
    """
    if "panel_cd_daily_csv" not in data:
        return
    configured_path = Path(data["panel_cd_daily_csv"])
    base = ROOT if configured_path.parts[0] == "figures" else Path(yaml_path).parent
    daily = summarize_daily(base / configured_path)
    for row in data["panel_cd_online_wer_speed"]:
        if "daily_series" not in row:
            continue
        stats = daily.pop(row["daily_series"])
        row.update(stats)
        for metric, label in (("wer", "wer_caliber"), ("speed_wpm", "speed_caliber")):
            n = len(stats[f"{metric}_dates"])
            if n > row["released_days"]:
                raise ValueError(f"{row['label']} has more evaluated than released days")
            if n == 0:
                row[label] = "n/r"
            else:
                basis = "single day" if n == 1 else "mean±SD"
                row[label] = f"{basis}\n{n} of {row['released_days']} d"
    if daily:
        raise ValueError(f"Unused Figure 2(c,d) series: {', '.join(sorted(daily))}")


def apply_online_wer_anchors(data):
    """Use panel (c) as the online WER reference for panel (f).

    Parameters
    ----------
    data : dict
        Loaded benchmark with panel (c) rates already resolved. Updates the
    panel (f) comparator and its whisker for rows with an online reference.

    Raises
    ------
    ValueError
        If the two panels differ in row order or a text comparator is missing.
    """
    online = data["panel_cd_online_wer_speed"]
    reproduction = data["panel_f_repro_full_wer"]
    if [row["label"] for row in online] != [row["label"] for row in reproduction]:
        raise ValueError("Figure 2(c) and (f) study rows do not align")
    for source, target in zip(online, reproduction, strict=True):
        if target["ours"] is None and target["paper"] is None:
            continue
        if source["wer_mean"] is None:
            raise ValueError(f"Figure 2(c) is missing online WER for {source['label']}")
        target["paper"] = source["wer_mean"]
        target["paper_sd"] = source["wer_sd"]


def _reproduction_trials(source):
    """Read archived references and reproduced transcripts with session IDs.

    Parameters
    ----------
    source : dict
        Archive format, repository-relative paths, and optional participant
        or vocabulary selection from the benchmark configuration.

    Yields
    ------
    tuple of str
        Recording session, reference text, and reproduced text. TSV rows
        are linked to metadata only after checking their reference order.
    """
    from build_session_per_vs_wer import norm_words

    path = ROOT / source["reproduction_source"]
    if source["format"] == "typing_json":
        with path.open(encoding="utf-8") as stream:
            sessions = json.load(stream)[source["participant"]]
        for session, record in sessions.items():
            if len(record["trials"]) != record["n_trials"]:
                raise ValueError(f"Typing trial-count mismatch in {session}")
            for trial in record["trials"]:
                yield session, trial["ref"], trial["hyp"]
    elif source["format"] == "handwriting_csv":
        seen = set()
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["session"], row["sidx"]
                if key in seen:
                    raise ValueError(f"Duplicate handwriting trial {key}")
                seen.add(key)
                yield row["session"], row["trueText"], row["decText"]
    elif source["format"] == "decode_tsv":
        metadata = ROOT / source["metadata_source"]
        with path.open(encoding="utf-8", newline="") as decoded, metadata.open(
            encoding="utf-8", newline=""
        ) as meta:
            seen = set()
            for row, trial in zip(csv.DictReader(decoded, delimiter="\t"),
                                  csv.DictReader(meta), strict=True):
                if row["key"] in seen:
                    raise ValueError(f"Duplicate decode key {row['key']} in {path}")
                seen.add(row["key"])
                if norm_words(row["ref"]) != norm_words(trial["sentence_label"]):
                    raise ValueError(f"Reproduction/reference order mismatch in {path}")
                if source.get("vocab") and trial["vocab"] != source["vocab"]:
                    continue
                yield trial["session"], row["ref"], row["hyp"]
    else:
        raise ValueError(f"Unknown Figure 2f archive format: {source['format']}")


def apply_reproduction_metrics(data):
    """Recompute Figure 2f offline WER from archived word-error counts.

    Parameters
    ----------
    data : dict
        Benchmark configuration with ``figure2_bar_sources['2f']`` and
        ``panel_f_repro_full_wer``. Reproduction values are updated in place.

    Notes
    -----
    References and hypotheses use the same normalization as online WER.
    Word errors are pooled within each day; bars and whiskers are equal-day
    means and sample SDs. A single day has no SD. Each available reproduction
    must cover exactly the online evaluation dates; trial sets may differ.
    Missing full-pipeline reproductions remain unavailable.
    Daily counts, dates, and pooled WER are retained for source inspection.
    """
    from build_session_per_vs_wer import norm_words, word_ed

    sources = data.get("figure2_bar_sources", {}).get("2f")
    if sources is None:
        return
    online = summarize_daily(ROOT / data["panel_cd_daily_csv"])
    for row in data["panel_f_repro_full_wer"]:
        source = sources[row["label"]]
        if not source["reproduction_source"]:
            row.update(ours=None, ours_sd=None, ours_days=0, ours_trials=0,
                       ours_daily=[], ours_pooled=None,
                       note=source.get("display_note", source["selection"]))
            continue
        if not source.get("online_series"):
            raise ValueError(f"Missing online evaluation date binding for {row['label']}")
        selected_dates = set(online[source["online_series"]]["wer_dates"])
        counts = defaultdict(lambda: {"trials": 0, "word_errors": 0,
                                      "reference_words": 0})
        for session, reference, hypothesis in _reproduction_trials(source):
            match = re.search(r"(\d{4})[.-](\d{2})[.-](\d{2})", session)
            if match is None:
                raise ValueError(f"Missing recording date in {session}")
            day = "-".join(match.groups())
            if day not in selected_dates:
                continue
            if not norm_words(reference):
                raise ValueError(f"Empty normalized reproduction reference in {session}")
            errors, words = word_ed(reference, hypothesis)
            count = counts[day]
            count["trials"] += 1
            count["word_errors"] += errors
            count["reference_words"] += words
        if not counts:
            raise ValueError(f"No selected reproduction trials for {row['label']}")
        if set(counts) != selected_dates:
            missing = sorted(selected_dates - set(counts))
            raise ValueError(f"Reproduction dates do not match online dates for "
                             f"{row['label']}; missing: {missing}")
        daily = [{"day": day, **count,
                  "wer_pct": 100 * count["word_errors"] / count["reference_words"]}
                 for day, count in sorted(counts.items())]
        rates = [day["wer_pct"] for day in daily]
        row.update(
            ours=mean(rates), ours_sd=stdev(rates) if len(rates) > 1 else None,
            ours_days=len(daily), ours_trials=sum(d["trials"] for d in daily),
            ours_daily=daily,
            ours_pooled=100 * sum(d["word_errors"] for d in daily)
            / sum(d["reference_words"] for d in daily),
            note=source["selection"],
        )


def main():
    """Print the validated daily summary as JSON.

    Notes
    -----
    The --csv option selects the daily table. The optional --sources option
    supplies a manifest of public inputs and rebuilds that table before
    summarizing it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, help="JSON map of public source paths")
    parser.add_argument("--csv", type=Path, default=DAILY_CSV)
    args = parser.parse_args()
    if args.sources:
        if __package__:
            from .extract_online_trials import rebuild
        else:
            from extract_online_trials import rebuild
        rebuild(args.sources, args.csv)
    print(json.dumps(summarize_daily(args.csv), indent=2))


if __name__ == "__main__":
    main()
