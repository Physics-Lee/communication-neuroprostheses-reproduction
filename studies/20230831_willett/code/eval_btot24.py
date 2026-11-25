# -*- coding: utf-8 -*-
"""Evaluate a BtoT24-trained GRUDecoder checkpoint on the competitionData test partition.

Same greedy-CTC/PER semantics as the official trainer validation:
argmax -> unique_consecutive -> drop blank; aggregate PER = sum(ed)/sum(len).
Usage: eval_btot24.py <run_dir>   (expects <run_dir>/checkpoint/best_checkpoint)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_path

import argparse
import json
import os
import time

import editdistance
import h5py
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, study_path("20240815_card", 'open_source_code_for_reference/nejm-brain-to-text/model_training'))
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME, runSingleDecodingStep  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DATA_DIR = study_path("20230831_willett", 'data/processed/hdf5')
# transforms must match training (gauss smoothing before the model)
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})


def main():
    """Evaluate a BtoT24 checkpoint on the test partition with greedy CTC.

    Rebuilds the GRUDecoder with the training configuration, loads
    <run_dir>/checkpoint/best_checkpoint, decodes every data_val.hdf5
    trial with greedy CTC (argmax -> unique_consecutive -> drop blank),
    aggregates PER = sum(edit distance) / sum(reference length), writes a
    per-trial csv into run_dir and prints the best/worst 5 sessions.
    """
    parser = argparse.ArgumentParser(description="Evaluate the Willett 2023 test partition.")
    parser.add_argument("run_dir", type=Path, help="Existing model run with checkpoint/best_checkpoint")
    parser.add_argument("--session", help="Only evaluate this test session")
    parser.add_argument("--output-dir", type=Path, help="New directory for trials.csv, summary.json, and run.yaml")
    args = parser.parse_args()
    if args.session and args.output_dir is None:
        parser.error("--session requires a new --output-dir")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")
    ckpt_path = args.run_dir / "checkpoint" / "best_checkpoint"
    sessions = sorted(d for d in os.listdir(DATA_DIR) if d.startswith("t12."))
    if args.session and args.session not in sessions:
        parser.error(f"Unknown test session {args.session!r} under {DATA_DIR}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model config identical to train_btot24.py (official rnn_args.yaml model section)
    model = GRUDecoder(
        neural_dim=512, n_units=768, n_days=24, n_classes=41,
        rnn_dropout=0.4, input_dropout=0.2, n_layers=5,
        patch_size=14, patch_stride=4,
    )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for k in list(sd.keys()):
        sd[k.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"loaded {ckpt_path} | ckpt val_PER = {ckpt.get('val_PER')}")

    rows = []
    total_ed = total_len = n_trials = 0
    t0 = time.time()
    for si, sess in enumerate(sessions):
        if args.session and sess != args.session:
            continue
        vf = os.path.join(DATA_DIR, sess, "data_val.hdf5")
        with h5py.File(vf, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                x = torch.tensor(g["input_features"][:][None], device=device,
                                 dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, si, model, MODEL_ARGS, device)[0]
                pred = np.argmax(logits, axis=-1)
                pred = torch.unique_consecutive(torch.tensor(pred), dim=-1).numpy()
                pred = np.array([i for i in pred if i != 0])
                seq_len = int(g.attrs["seq_len"])
                true = np.asarray(g["seq_class_ids"][:][:seq_len])
                ed = editdistance.eval(pred.tolist(), true.tolist())
                total_ed += ed
                total_len += seq_len
                n_trials += 1
                rows.append({
                    "session": sess,
                    "block_num": int(g.attrs["block_num"]),
                    "sentence_label": g.attrs["sentence_label"],
                    "true_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in true),
                    "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                    "edit_distance": ed, "true_len": seq_len,
                })
        print(f"[{time.strftime('%H:%M:%S')}] {sess} done ({n_trials} trials, "
              f"{time.time()-t0:.0f}s)", flush=True)

    per = total_ed / total_len
    print(f"\n=== BtoT24 test-partition greedy PER: {total_ed}/{total_len} "
          f"= {100*per:.2f}% over {n_trials} trials ===")
    df = pd.DataFrame(rows)
    out_csv = args.run_dir / "test_partition_greedy_per_trial.csv"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        out_csv = args.output_dir / "trials.csv"
        summary = {"trials": n_trials, "phoneme_errors": total_ed,
                   "reference_phonemes": total_len, "per": per}
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        run = {"study": "20230831_willett", "dataset_id": "dryad_x69p8czpq",
               "release_id": None, "input_format": "hdf5", "split": "test",
               "session": args.session,
               "decoder": "ctc_greedy", "rescorer": None, "vocoder": None,
               "checkpoint": str(ckpt_path.resolve()), "data_path": DATA_DIR,
               "trials": "trials.csv", "summary": "summary.json", "command": sys.argv}
        (args.output_dir / "run.yaml").write_text(
            yaml.safe_dump(run, sort_keys=False), encoding="utf-8",
        )
    df.to_csv(out_csv, index=False)
    day = df.groupby("session").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(),
        include_groups=False).sort_values()
    print("\nbest 5 days:"); print(day.head(5).apply(lambda v: f"{100*v:.2f}%").to_string())
    print("\nworst 5 days:"); print(day.tail(5).apply(lambda v: f"{100*v:.2f}%").to_string())
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
