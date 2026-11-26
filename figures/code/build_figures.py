"""Rebuild the benchmark figures and the tables used to inspect their values."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from figure_paths import BENCHMARK, FIGURES, PUBLISHED, ROOT, load_benchmark


def write_table(path, rows):
    """Save the selected panel rows as a consistently ordered CSV.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV.
    rows : list of dict
        Rows with a consistent column schema across the panel.
    """
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output, benchmark):
    """Export bar and point values from the same inputs as the renderers.

    Parameters
    ----------
    output : pathlib.Path
        Figure build directory.
    benchmark : dict
        Canonical benchmark configuration and source paths.
    """
    resolved = json.loads((output / "reproduction_benchmark.json").read_text(encoding="utf-8"))
    bars = []
    for panel, key in (("2e", "panel_e_repro_acoustic"), ("2f", "panel_f_repro_full_wer")):
        for row in resolved[key]:
            evidence = benchmark.get("figure2_bar_sources", {}).get(panel, {}).get(row["label"], {})
            bars.append({
                "panel": panel, **{key: value for key, value in row.items()
                                  if key != "ours_daily"},
                "reproduction_source": evidence.get("reproduction_source", ""),
                "reproduction_selection": evidence.get("selection", ""),
                "comparison_source": benchmark["panel_cd_daily_csv"] if panel == "2f"
                else evidence.get("paper_source", ""),
            })
    write_table(output / "figure2ef_bars.csv", bars)
    write_table(output / "figure2f_daily.csv", [
        {"label": row["label"], **day}
        for row in resolved["panel_f_repro_full_wer"]
        for day in row.get("ours_daily", [])
    ])
    points = []
    for axis, panel in (("per", "3a"), ("cer", "3b")):
        sessions = json.loads((output / (f"session_{axis}_vs_wer.json")).read_text(encoding="utf-8"))
        for name, sources in benchmark[f"figure3_{axis}_sources"].items():
            if not sources.get("plotted", True):
                continue
            for row in sessions[name]["sessions"]:
                points.append({
                    "panel": panel, "arm": name, **row, "selection": sources["selection"],
                    "raw_source": sources.get("greedy", sources.get("greedy_counts", sources.get("archive"))),
                    "word_source": sources.get("decode", sources.get("archive")),
                })
    write_table(output / "figure3ab_session_points.csv", points)
    memory = [
        {"panel": panel, "resource": resource, **row,
         "source": str(BENCHMARK.relative_to(ROOT).as_posix()) + "#" + key}
        for panel, resource, key in (
            ("3c", "host RAM", "panel_g_host_ram_walls"),
            ("3d", "GPU VRAM", "panel_h_gpu_vram_walls"),
        )
        for row in benchmark[key]
    ]
    write_table(output / "figure3cd_memory.csv", memory)


def write_receipt(output, benchmark):
    """Record the paths and hashes of files used in a figure build.

    Parameters
    ----------
    output : pathlib.Path
        Figure build directory.
    benchmark : dict
        Canonical benchmark configuration and source paths.
    """
    inputs = {BENCHMARK, *(FIGURES / "code").glob("*.py")}
    for collection in ("figure3_per_sources", "figure3_cer_sources", "figure2_bar_sources"):
        def source_files(mapping):
            for value in mapping.values():
                if isinstance(value, dict):
                    yield from source_files(value)
                elif isinstance(value, str) and value.startswith("studies/"):
                    yield ROOT / value
        inputs.update(source_files(benchmark.get(collection, {})))
    inputs.add(ROOT / benchmark["panel_cd_daily_csv"])
    inputs.update(PUBLISHED.glob("figure2cd_paired_*.csv"))
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    receipt = {
        "base_commit": result.stdout.strip() if result.returncode == 0 else None,
        "command": "python figures/code/build_figures.py --output-dir <figure-build-directory>",
        "inputs": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(inputs)
        },
    }
    (output / "build.yaml").write_text(
        yaml.safe_dump(receipt, sort_keys=False), encoding="utf-8", newline="\n",
    )


def main():
    """Build the figures into a new release or refresh the published bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--publish", action="store_true",
                        help="Refresh the configured published figure bundle")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if args.publish and destination != PUBLISHED:
        parser.error("--publish requires the configured published directory")
    if destination.exists() and not args.publish:
        parser.error("Output directory already exists. Choose a new figure build directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as stage:
        output = Path(stage) / "build"
        output.mkdir()
        benchmark = load_benchmark()
        shutil.copyfile(ROOT / benchmark["panel_cd_daily_csv"], output / "figure2cd_daily.csv")
        for path in PUBLISHED.glob("figure2cd_paired_*.csv"):
            shutil.copyfile(path, output / path.name)
        commands = (
            ("build_session_per_vs_wer.py", "--output-dir", str(output)),
            ("build_session_cer_vs_wer.py", "--output-dir", str(output)),
            ("render_figure2.py", "--out", str(output / "figure2")),
            ("render_figure3.py", "--results-dir", str(output)),
        )
        for script, *options in commands:
            subprocess.run([sys.executable, "-B", str(FIGURES / "code" / script), *options],
                           cwd=ROOT, check=True)
        export_tables(output, benchmark)
        write_receipt(output, benchmark)
        if args.publish:
            for artifact in output.iterdir():
                artifact.replace(destination / artifact.name)
        else:
            output.rename(destination)
    print(f"Figure build written to {destination}")


if __name__ == "__main__":
    main()
