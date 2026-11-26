"""Decode one Card checkpoint's logits and publish a separate indexed WFST run."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "shared"))
from study_paths import build_run_id, runtime_path, study_path

STUDY = "20240815_card"
LOGITS = {
    "p2": ("run_120k", "t15_logits.hdf5"),
    "p4": ("run_p4_20261013", "t15_logits.hdf5"),
    "p4_fixed": ("run_p4_20261013", "t15_logits_fixed.hdf5"),
}


def main():
    """Export the logits, run the mmap decoder, and record the new results.

    Returns
    -------
    None
        The new result directory is printed after the run is recorded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=(*LOGITS, "bigru", "ensemble"))
    parser.add_argument("--logits", type=Path, help="Override the input HDF5 logits")
    parser.add_argument("--binary", type=Path,
                        default=Path(os.environ.get("BCI_WFST_BINARY", Path.home() / "bin/wenet_lattice")))
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--label", help="Use a distinct experiment name for another run on the same date")
    parser.add_argument("--dry-run", action="store_true", help="Show the input and output paths without decoding")
    args = parser.parse_args()

    logits = args.logits
    if logits is None:
        if args.variant not in LOGITS:
            parser.error(f"{args.variant} requires --logits with its HDF5 file")
        logits = runtime_path(STUDY, *LOGITS[args.variant])
    if not logits.is_file():
        parser.error(f"Missing input logits: {logits}")
    logits = logits.resolve()
    graph_root = Path(study_path(
        "20230831_willett", "data/language_models/wfst_5gram/speech_5gram/lang_test"))
    graph, words = graph_root / "TLG.fst", graph_root / "words.txt"
    run_id = build_run_id(STUDY, "npy", "wfst_5gram", args.label or f"{args.variant}_decode")
    result = ROOT / "studies" / STUDY / "results" / run_id
    if args.dry_run:
        for path in (logits, graph, words, result):
            print(path)
        return
    if result.exists():
        parser.error(f"Result already exists: {result}. Choose --label to create another run.")
    for path in (args.binary, graph, words):
        if not path.is_file():
            parser.error(f"Missing decoder input: {path}")

    result.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="card-wfst-", dir=result.parent) as temporary:
        work = Path(temporary)
        trials, meta = work / "trials", work / "meta.tsv"
        final, nbest = work / "final.tsv", work / "nbest.tsv"
        subprocess.run([sys.executable, str(ROOT / "shared/tools/export_wfst_trials.py"),
                        str(logits), str(trials), str(meta)], check=True)
        command = [sys.executable, str(ROOT / "shared/tools/decode_wfst.py"),
                   "--binary", str(args.binary), "--graph", str(graph), "--words", str(words),
                   "--trials-dir", str(trials), "--meta", str(meta), "--output", str(final),
                   "--nbest-prefix", str(nbest.with_suffix(""))]
        if args.build_index:
            command.append("--build-index")
        subprocess.run(command, check=True)

        staged = work / run_id
        subprocess.run([sys.executable, str(ROOT / "shared/tools/record_study_run.py"), STUDY,
                        str(staged), "--input-format", "npy", "--decoder", "wfst_5gram",
                        "--split", "validation", "--trials", str(final)], check=True)
        native = staged / "runtime"
        native.mkdir()
        shutil.copyfile(final, native / final.name)
        shutil.copyfile(nbest, native / nbest.name)

        record_file = staged / "run.yaml"
        record = yaml.safe_load(record_file.read_text(encoding="utf-8"))
        record["artifacts"]["trials.csv"] = "runtime/final.tsv"
        record.update(source_logits=str(logits), graph=str(graph), words=str(words),
                      acoustic_scale=0.325, blank_penalty=90, nbest=100)
        record_file.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
        staged.rename(result)
    print(result)


if __name__ == "__main__":
    main()
