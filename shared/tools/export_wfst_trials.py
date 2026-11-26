"""Export speech-logit HDF5 trials and their references for the indexed decoder."""

import argparse
from pathlib import Path

import h5py
import numpy as np


def main():
    """Write one float32 NPY per trial and a key/reference TSV without a header."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logits", type=Path, help="HDF5 containing session/trial/log_probs")
    parser.add_argument("trials_dir", type=Path)
    parser.add_argument("meta", type=Path, help="New key/reference TSV")
    args = parser.parse_args()
    args.trials_dir.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(args.logits, "r") as source, args.meta.open("x", encoding="utf-8", newline="\n") as meta:
        for session in source.values():
            for trial in session.values():
                log_probs = trial["log_probs"]
                if log_probs.ndim != 2 or log_probs.shape[1] != 41:
                    raise ValueError(f"Expected 41 speech classes in {log_probs.name}, got {log_probs.shape}")
                label = trial.attrs["sentence_label"]
                if isinstance(label, bytes):
                    label = label.decode("utf-8")
                key = f"{count:04d}"
                np.save(args.trials_dir / f"{key}.npy", log_probs[...].astype(np.float32))
                meta.write(f"{key}\t{label}\n")
                count += 1
    print(f"Exported {count} trials to {args.trials_dir} and {args.meta}")


if __name__ == "__main__":
    main()
