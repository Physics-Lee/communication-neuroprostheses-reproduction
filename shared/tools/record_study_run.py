"""Save already scored study outputs under the public results filenames."""

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_paths import ROOT, study_path


def provenance_path(path):
    """Return a repository-relative path for an in-repository artifact.

    Parameters
    ----------
    path : pathlib.Path
        Source artifact path.

    Returns
    -------
    str
        Repository-relative or external absolute path.
    """
    absolute = path.resolve()
    return absolute.relative_to(ROOT).as_posix() if absolute.is_relative_to(ROOT) else str(absolute)


def main():
    """Copy scored artifacts and record their study and provenance.

    Notes
    -----
    The output directory must be new. CSV inputs retain their original
    columns and values. TSV inputs are written with CSV delimiters.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", help="Study directory name, such as 20230831_willett")
    parser.add_argument("output_dir", type=Path, nargs="?", help="Explicit new run directory")
    parser.add_argument("--trials", type=Path, help="Scored CSV or TSV to publish as trials.csv")
    parser.add_argument("--summary", type=Path, help="Existing summary JSON to publish as summary.json")
    parser.add_argument("--split", help="Split identifier or manifest path")
    parser.add_argument("--input-format", help="Format the run actually read, for example hdf5")
    parser.add_argument("--decoder")
    parser.add_argument("--rescorer")
    parser.add_argument("--vocoder")
    parser.add_argument("--command", help="Command used to produce the original evaluation")
    parser.add_argument("--label", help="Experiment label for an automatically named run directory")
    args = parser.parse_args()
    if not (args.trials or args.summary):
        parser.error("Provide --trials or --summary")
    if args.trials and (not args.trials.is_file() or args.trials.suffix.lower() not in (".csv", ".tsv")):
        parser.error("--trials must be an existing CSV or TSV file")
    if args.summary:
        with args.summary.open(encoding="utf-8-sig") as file:
            json.load(file)
    source = ROOT / "studies" / args.study / "data" / "sources.yaml"
    study_path(args.study)
    dataset = yaml.safe_load(source.read_text(encoding="utf-8"))
    release = dataset.get("release_id") or dataset["dataset_id"]
    output_dir = args.output_dir
    if output_dir is None:
        components = (release, args.input_format, args.decoder or "none", args.label)
        if not all(part and re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", part) for part in components):
            parser.error("Automatic run naming requires lowercase --input-format and --label, and a valid decoder if supplied")
        run_id = date.today().strftime("%Y%m%d_") + "__".join(components)
        output_dir = source.parents[1] / "results" / run_id
    output_dir.mkdir(parents=True)

    artifacts = {}
    if args.trials:
        destination = output_dir / "trials.csv"
        if args.trials.suffix.lower() == ".tsv":
            with (args.trials.open(encoding="utf-8-sig", newline="") as original,
                  destination.open("x", encoding="utf-8", newline="") as output):
                csv.writer(output).writerows(csv.reader(original, delimiter="\t"))
        else:
            shutil.copyfile(args.trials, destination)
        artifacts[destination.name] = provenance_path(args.trials)
    if args.summary:
        destination = output_dir / "summary.json"
        shutil.copyfile(args.summary, destination)
        artifacts[destination.name] = provenance_path(args.summary)
    run = {
        "study": args.study, "dataset_id": dataset["dataset_id"],
        "release_id": dataset.get("release_id"), "input_format": args.input_format,
        "split": args.split,
        "decoder": args.decoder, "rescorer": args.rescorer, "vocoder": args.vocoder,
        "command": args.command, "artifacts": artifacts,
    }
    (output_dir / "run.yaml").write_text(yaml.safe_dump(run, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
