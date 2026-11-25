"""Run the repository's indexed WeNet lattice decoder in WSL."""

import argparse
import subprocess
from pathlib import Path


def main():
    """Parse public options and invoke the memory-mapped decoder.

    Returns
    -------
    int
        Decoder exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="Compiled shared/tools/wenet_lattice binary")
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--words", type=Path, required=True)
    parser.add_argument("--trials-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output TSV")
    parser.add_argument("--acoustic-scale", type=float, default=0.325)
    parser.add_argument("--blank-penalty", type=float, default=90)
    parser.add_argument("--beam", type=float, default=17)
    parser.add_argument("--lattice-beam", type=float, default=8)
    parser.add_argument("--max-active", type=int, default=7000)
    parser.add_argument("--min-active", type=int, default=200)
    parser.add_argument("--nbest", type=int, default=100)
    parser.add_argument("--nbest-prefix", default="-")
    parser.add_argument("--build-index", action="store_true", help="Build the graph offset sidecar if needed")
    parser.add_argument("--skip-threshold", type=float, default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.binary.resolve(), args.graph, args.words, args.trials_dir, args.meta, args.output,
        args.acoustic_scale, args.blank_penalty, args.beam, args.lattice_beam, args.max_active,
        args.min_active, args.nbest, args.nbest_prefix, int(args.build_index), args.skip_threshold,
    ]
    return subprocess.run([str(value) for value in command], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
