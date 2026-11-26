"""Reader acceptance: compare scratch conversion outputs with the shared tree."""

import filecmp
import os
import sys

import numpy as np

base_s = r"D:\repositories\reader_acceptance_scratch"
base_e = r"D:\repositories\reader_acceptance_clean\studies"


def compare_tree(rel_top):
    """Byte-compare every file under the same relative top of both roots."""
    top_s = os.path.join(base_s, rel_top)
    top_e = os.path.join(base_e, rel_top)
    same, diff, missing = [], [], []
    for root, _, files in os.walk(top_s):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), top_s)
            path_s = os.path.join(root, name)
            path_e = os.path.join(top_e, rel)
            if not os.path.exists(path_e):
                missing.append(rel)
            elif filecmp.cmp(path_s, path_e, shallow=False):
                same.append(rel)
            else:
                diff.append(rel)
    return same, diff, missing


if __name__ == "__main__":
    rel_top = sys.argv[1]
    same, diff, missing = compare_tree(rel_top)
    print(f"tree {rel_top}: identical {len(same)} | differ {len(diff)} | missing-in-shared {len(missing)}")
    for name in diff:
        print("  DIFF", name)
    for name in missing:
        print("  MISSING", name)
    npz_rel = sys.argv[2] if len(sys.argv) > 2 else None
    if npz_rel:
        a = np.load(os.path.join(base_s, npz_rel), allow_pickle=True)
        b = np.load(os.path.join(base_e, npz_rel), allow_pickle=True)
        keys_ok = set(a.files) == set(b.files)

        def equal(x, y):
            if isinstance(x, np.ndarray) and x.dtype == object:
                return len(x) == len(y) and all(equal(xi, yi) for xi, yi in zip(x, y))
            return np.array_equal(x, y)

        ok = keys_ok and all(equal(a[k], b[k]) for k in a.files)
        print(f"npz {npz_rel}: arrays identical {ok} | keys {len(a.files)}")
