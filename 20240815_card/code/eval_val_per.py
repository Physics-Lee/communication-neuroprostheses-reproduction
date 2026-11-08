# -*- coding: utf-8 -*-
"""BtoT25 Phase-1 evaluation: pretrained RNN inference + greedy CTC phoneme decode.

Mirrors nejm-brain-to-text semantics exactly:
- model construction: evaluate_model.py lines 59-83 (GRUDecoder from checkpoint/args.yaml)
- inference: evaluate_model_helpers.runSingleDecodingStep (gauss_smooth + autocast bf16)
- greedy decode: rnn_trainer.py validation (argmax -> unique_consecutive -> drop blank)
- aggregate PER = sum(editdistance) / sum(true_seq_len)   (official aggregation)

Stops BEFORE the language-model step (no redis / n-gram / OPT rescoring).
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd
import editdistance
import torch
from omegaconf import OmegaConf

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import (  # noqa: E402
    load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME,
)

MODEL_PATH = r".\20240815_card\nejm-brain-to-text\data\t15_pretrained_rnn_baseline"
DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
CSV_PATH = r".\20240815_card\nejm-brain-to-text\data\t15_copyTaskData_description.csv"
OUT_CSV = r".\20240815_card\result\val_greedy_per_trial.csv"
OUT_SUMMARY = r".\20240815_card\result\val_greedy_summary.json"


def main():
    """Evaluate the pretrained RNN baseline on the val set with greedy CTC decode.

    Rebuilds the GRUDecoder from the checkpoint's args.yaml, decodes every
    trial of every session that has a data_val.hdf5 split via the official
    runSingleDecodingStep AMP path, applies the greedy CTC phoneme decode
    (argmax -> unique_consecutive -> drop blank), aggregates PER as
    sum(editdistance)/sum(true_seq_len), and writes per-trial and summary
    outputs. The day index passed to the decoder is the position within the
    val-having session subset.

    Returns
    -------
    None
        Writes ``OUT_CSV`` and ``OUT_SUMMARY``; per-session and per-corpus
        PER are printed to stdout.
    """
    model_args = OmegaConf.load(os.path.join(MODEL_PATH, "checkpoint/args.yaml"))
    ds = model_args["dataset"]
    print("dataset keys (minus sessions):",
          {k: v for k, v in ds.items() if k != "sessions"})

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| torch:", torch.__version__)

    model = GRUDecoder(
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
    ckpt = torch.load(os.path.join(MODEL_PATH, "checkpoint/best_checkpoint"),
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in list(sd.keys()):
        sd[key.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(key)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"checkpoint loaded | saved val_PER in ckpt: {ckpt.get('val_PER')}")

    b2txt_csv_df = pd.read_csv(CSV_PATH)

    rows = []
    total_ed = 0
    total_len = 0
    t0 = time.time()
    n_trials = 0

    for si, session in enumerate(ds["sessions"]):
        val_file = os.path.join(DATA_DIR, session, "data_val.hdf5")
        if not os.path.exists(val_file):
            continue
        data = load_h5py_file(val_file, b2txt_csv_df)
        input_layer = si
        s_ed, s_len = 0, 0
        for trial in range(len(data["neural_features"])):
            x = np.expand_dims(data["neural_features"][trial], axis=0)
            x = torch.tensor(x, device=device, dtype=torch.bfloat16)
            logits = runSingleDecodingStep(x, input_layer, model, model_args, device)
            logits = logits[0]  # (T_out, n_classes)

            # official greedy CTC decode (rnn_trainer.py validation semantics)
            pred = np.argmax(logits, axis=-1)
            pred = torch.unique_consecutive(torch.tensor(pred), dim=-1).numpy()
            pred = np.array([i for i in pred if i != 0])

            true = np.asarray(data["seq_class_ids"][trial][0:data["seq_len"][trial]])

            ed = editdistance.eval(pred.tolist(), true.tolist())
            s_ed += ed
            s_len += len(true)
            total_ed += ed
            total_len += len(true)
            n_trials += 1

            rows.append({
                "session": session,
                "block_num": int(data["block_num"][trial]),
                "trial_num": int(data["trial_num"][trial]),
                "corpus": data["corpus"][trial],
                "sentence_label": data["sentence_label"][trial],
                "true_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in true),
                "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                "edit_distance": ed,
                "true_len": len(true),
            })
        print(f"[{time.strftime('%H:%M:%S')}] {session}: {len(data['neural_features'])} trials, "
              f"day PER {s_ed}/{s_len} = {100 * s_ed / max(s_len, 1):.2f}% "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)

    per = total_ed / total_len
    print(f"\n=== aggregate val greedy PER: {total_ed}/{total_len} = {100 * per:.2f}% "
          f"over {n_trials} trials ===")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    by_corpus = df.groupby("corpus").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(), include_groups=False
    ).sort_values()
    print("\nPER by corpus:")
    print(by_corpus.apply(lambda v: f"{100 * v:.2f}%").to_string())

    day = df.groupby("session").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(), include_groups=False
    ).sort_values()
    print("\nBest 5 sessions:"); print(day.head(5).apply(lambda v: f"{100*v:.2f}%").to_string())
    print("\nWorst 5 sessions:"); print(day.tail(5).apply(lambda v: f"{100*v:.2f}%").to_string())

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_PER": per,
            "aggregate_PCT": 100 * per,
            "total_edit_distance": int(total_ed),
            "total_true_len": int(total_len),
            "n_trials": n_trials,
            "n_sessions_with_val": int(df["session"].nunique()),
            "per_by_corpus": {k: float(v) for k, v in by_corpus.items()},
            "checkpoint_saved_val_PER": ckpt.get("val_PER"),
            "greedy_spec": "argmax->unique_consecutive->drop blank; agg=sum(ed)/sum(len)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_CSV}\nsaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
