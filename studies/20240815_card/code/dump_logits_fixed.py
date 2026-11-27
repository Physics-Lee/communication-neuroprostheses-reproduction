# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs with CORRECT day indexing (2026-10-16 fix).

BUG BEING FIXED: dump_t15_logits.py / dump_p4_logits.py enumerate the
val-having subset and pass that counter as the day index, but day layers are
positional over the FULL 45-session list whose first entry (t15.2023.08.11)
has no val split -- so every val trial was decoded through the PREVIOUS
day's input layer. The official evaluate_model.py uses
sessions.index(session) (full list); this script matches official semantics.
Impact measured 2026-10-16: logits-greedy PER 08.13 20.9% vs correct 6.4%,
2025.01.10 65.7% vs 24.4% -- all prior WER numbers were computed on the
wrong-layer logits.

Usage: dump_logits_fixed.py <run_dir> [model=gru|bigru]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_path

import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, study_path("20240815_card", 'open_source_code_for_reference/nejm-brain-to-text/model_training'))
sys.path.insert(0, study_path("20240815_card", 'code'))

from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep  # noqa: E402

DATA_DIR = study_path("20240815_card", 'data/extracted/hdf5_data_final')
CSV_PATH = study_path("20240815_card", 'data/extracted/t15_copyTaskData_description.csv')


def dump_manifest(path, output, model, model_args, device, neural_root):
    """Export WFST inputs on a frozen manifest, preserving the online row keys.

    Parameters
    ----------
    path : Path
        Frozen CSV mapping online IDs to native neural trial IDs.
    output : Path
        New directory for NPY files and meta.tsv.
    model : torch.nn.Module
        Fitted decoder in evaluation mode.
    model_args : DictConfig
        Checkpoint configuration including the authoritative session order.
    device : torch.device
        CUDA inference device.
    neural_root : Path
        Native HDF5 release root.
    """
    trials = output / "logits"
    trials.mkdir(parents=True)
    with path.open(encoding="utf-8", newline="") as stream, \
            (output / "meta.tsv").open("x", encoding="utf-8", newline="\n") as meta:
        seen = set()
        for row in csv.DictReader(stream):
            key = row["key"]
            if not key.isdecimal() or key in seen:
                raise ValueError(f"Invalid or duplicate trial key: {key}")
            seen.add(key)
            with h5py.File(neural_root / row["source_file"], "r") as source:
                g = source[row["source_key"]]
                if (g.attrs["session"], int(g.attrs["block_num"]), int(g.attrs["trial_num"])) != (
                    row["session"], int(row["block"]), int(row["trial"])
                ):
                    raise ValueError(f"Neural trial identity mismatch: {key}")
                x = torch.as_tensor(g["input_features"][:], device=device, dtype=torch.bfloat16)[None]
            day_idx = model_args.dataset.sessions.index(row["session"])
            logits = runSingleDecodingStep(x, day_idx, model, model_args, device)[0]
            logp = torch.from_numpy(logits).log_softmax(dim=-1).numpy()
            if not len(logp) or not np.isfinite(logp).all():
                raise ValueError(f"Invalid inference logits: {key}")
            np.save(trials / f"{key}.npy", logp)
            meta.write(f"{key}\t{row['reference']}\n")
    print(f"Exported {len(seen)} neural inference trials to {output}", flush=True)


def main():
    """Decode all val trials with correct day indexing and dump CTC log-probs.

    Loads the run's best checkpoint, iterates the full session list from the
    run's args.yaml (using ``sessions.index(session)`` as the day index, i.e.
    official evaluate_model.py semantics), converts per-trial logits to
    log-probs, and writes them to ``RUN/t15_logits_fixed.hdf5`` grouped as
    ``<session>/tr<NNNN>``.

    Returns
    -------
    None
        Results are written to ``OUT``; progress is printed to stdout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("model", choices=("gru", "bigru"), nargs="?", default="gru")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--neural-root", type=Path, default=Path(DATA_DIR))
    cli = parser.parse_args()
    if bool(cli.manifest) != bool(cli.manifest_output):
        parser.error("--manifest and --manifest-output must be used together")
    if cli.model == "bigru":
        from calib_models import BiGRUDecoder as Decoder
    else:
        from rnn_model import GRUDecoder as Decoder
    output = cli.run / "t15_logits_fixed.hdf5"
    model_args = OmegaConf.load(cli.run / "checkpoint" / "args.yaml")
    ds = model_args["dataset"]
    device = torch.device("cuda:0")
    model = Decoder(
        neural_dim=model_args["model"]["n_input_features"],
        n_units=model_args["model"]["n_units"],
        n_days=len(ds["sessions"]),
        n_classes=ds["n_classes"],
        rnn_dropout=model_args["model"]["rnn_dropout"],
        input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
        n_layers=model_args["model"]["n_layers"],
        patch_size=model_args["model"]["patch_size"],
        patch_stride=model_args["model"]["patch_stride"],
    )
    ckpt = torch.load(cli.run / "checkpoint" / "best_checkpoint",
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in list(sd.keys()):
        sd[key.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(key)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"checkpoint loaded | val_PER in ckpt: {ckpt.get('val_PER')}")
    if cli.manifest:
        dump_manifest(cli.manifest, cli.manifest_output, model, model_args, device, cli.neural_root)
        return

    b2txt_csv_df = pd.read_csv(CSV_PATH)
    n = 0
    t0 = time.time()
    with h5py.File(output, "w") as out:
        for session in ds["sessions"]:
            vp = cli.neural_root / session / "data_val.hdf5"
            if not os.path.exists(vp):
                continue
            day_idx = ds["sessions"].index(session)  # official semantics
            data = load_h5py_file(vp, b2txt_csv_df)
            for trial in range(len(data["neural_features"])):
                x = np.expand_dims(data["neural_features"][trial], axis=0)
                x = torch.tensor(x, device=device, dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, day_idx, model, model_args, device)[0]
                logp = torch.log_softmax(
                    torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
                g = out.create_group(f"{session}/tr{trial:04d}")
                g.create_dataset("log_probs", data=logp.astype(np.float32),
                                 compression="gzip")
                g.attrs["sentence_label"] = data["sentence_label"][trial]
                g.attrs["corpus"] = data["corpus"][trial]
                sl = int(data["seq_len"][trial])
                g.attrs["seq_len"] = sl
                g.create_dataset("seq_class_ids",
                                 data=np.asarray(data["seq_class_ids"][trial][:sl]))
                n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {session} (day {day_idx}) done "
                  f"({n} trials, {time.time()-t0:.0f}s)", flush=True)
    print(f"saved {output} ({n} trials)")


if __name__ == "__main__":
    main()
