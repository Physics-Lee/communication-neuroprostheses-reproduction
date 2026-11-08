# -*- coding: utf-8 -*-
"""Build the T16 attempted-speech holdout splits (review A.4 task 3).

Source: D:\\repositories\\20250821_kunz\\data\\20250821_kunz\\hdf5\\t16.2024.04.29_attempted_50words\\data_train.hdf5
        (120 trials, blocks 1/4/6 x 40 trials, one recording day, 50-word vocab)

Design: 3-fold BLOCK-level cross-validation. Each fold holds out one 40-trial
block as val and trains on the other two blocks (80 trials). Block-level split
keeps the per-block z-scoring statistics disjoint between train and val (no
weak leakage), mirroring the "val = separate block" convention of the pipeline.
A separate dataset root (hdf5_t16holdout) isolates these fold sessions from the
main kunz hdf5 tree so day-index bookkeeping of the existing runs is untouched.

Layout per fold F with held-out block V and train blocks (A, B), each fold in
its own sub-root so train-time session globbing (and day indexing) sees exactly
the two sessions of that fold and no other fold's blocks can leak in:
  hdf5_t16holdout\\foldF\\t16.2024.04.29_attempted_50words_blkAB\\data_train.hdf5
  hdf5_t16holdout\\foldF\\t16.2024.04.29_attempted_50words_blkVval\\data_val.hdf5
Trial groups are copied verbatim (datasets + attrs); trial keys are re-indexed
sequentially per output file.

Usage: make_t16_holdout.py            (idempotent: refuses to overwrite)
"""
import os

import h5py

SRC = r".\20250821_kunz\data\20250821_kunz\hdf5\t16.2024.04.29_attempted_50words\data_train.hdf5"
OUT_ROOT = r".\20250821_kunz\data\20250821_kunz\hdf5_t16holdout"
FOLDS = {  # fold -> held-out (val) block
    1: 1,
    2: 4,
    3: 6,
}


def main():
    """Build the three block-level T16 holdout fold directories.

    Groups the source trials by ``block_num``, sanity-checks the expected
    3 x 40-trial block layout, and for each fold writes the train
    (two blocks) and val (one block) session directories under
    ``hdf5_t16holdout/foldF``.

    Returns
    -------
    None
        Per-fold split summaries are printed to stdout.

    Raises
    ------
    AssertionError
        If the source block layout differs from blocks 1/4/6 with 40
        trials each.
    SystemExit
        If a fold output directory already exists (idempotent guard).
    """
    with h5py.File(SRC, "r") as src:
        by_block = {}
        for key in sorted(src.keys()):
            g = src[key]
            by_block.setdefault(int(g.attrs["block_num"]), []).append((key, g))

    blocks = sorted(by_block)
    assert blocks == [1, 4, 6] and all(len(v) == 40 for v in by_block.values()), \
        f"unexpected block layout: {[(b, len(by_block[b])) for b in blocks]}"

    for fold, val_blk in FOLDS.items():
        train_blks = [b for b in blocks if b != val_blk]
        fold_root = os.path.join(OUT_ROOT, f"fold{fold}")
        tr_dir = os.path.join(fold_root, f"t16.2024.04.29_attempted_50words_blk{train_blks[0]}{train_blks[1]}")
        va_dir = os.path.join(fold_root, f"t16.2024.04.29_attempted_50words_blk{val_blk}val")
        for d in (tr_dir, va_dir):
            if os.path.exists(d):
                raise SystemExit(f"refusing to overwrite existing {d}")
            os.makedirs(d)
        _write(os.path.join(tr_dir, "data_train.hdf5"),
               [k for b in train_blks for k, _ in by_block[b]], os.path.basename(tr_dir))
        _write(os.path.join(va_dir, "data_val.hdf5"),
               [k for k, _ in by_block[val_blk]], os.path.basename(va_dir))
        print(f"fold {fold}: val=block {val_blk} (40) | train=blocks {train_blks} (80)")


def _write(out_path, keys, session_name):
    """Copy selected source trial groups into a new hdf5 file.

    Datasets are copied verbatim; the ``session`` attribute is rewritten to
    ``session_name`` and trial keys are re-indexed sequentially.

    Parameters
    ----------
    out_path : str
        Path of the hdf5 file to create.
    keys : list of str
        Source trial group names, in output order.
    session_name : str
        Session label to store in each trial's ``session`` attribute.

    Returns
    -------
    None
        The written path and trial count are printed to stdout.
    """
    with h5py.File(SRC, "r") as src, h5py.File(out_path, "w") as dst:
        for i, key in enumerate(keys):
            g = src[key]
            ng = dst.create_group(f"trial_{i:04d}")
            for ds in ("input_features", "seq_class_ids", "transcription"):
                src.copy(g[ds], ng, name=ds)
            for attr, val in g.attrs.items():
                if attr == "session":
                    val = session_name
                ng.attrs[attr] = val
    print(f"  wrote {out_path} ({len(keys)} trials)")


if __name__ == "__main__":
    main()
