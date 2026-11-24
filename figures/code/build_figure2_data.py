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
