# -*- coding: utf-8 -*-
"""P0.2 + P0.3 of the calibration plan (20261014).

P0.2  CTC loss scale calibration: per-sentence CTC NLL (the trainer's
      reduction='none' scale, i.e. the scale the training log and Table S5's
      "loss < 0.5-0.6" live on) for 2 good days + 2 bad 2025 days (val split,
      val-mode transform = smoothing only, 'same' padding), PLUS data_train
      sentences of 2 bad days (quantifies P4's memorization = the generous
      starting point of the P1 simulation).

P0.3  checkpoint smoke: greedy-decode 5 trials of t15.2023.08.18 with
      eval_val_per.py semantics ('valid' smoothing) and compare edit distances
      to run_p4_20261013/val_greedy_per_trial.csv row by row.

Usage: python calib_p0_loss_scale.py
"""
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = r".\20240815_card"
P4 = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013")
DATA_DIR = os.path.join(REPO, r"nejm-brain-to-text\data\hdf5_data_final")

sys.path.insert(0, os.path.join(REPO, r"nejm-brain-to-text\model_training"))
from rnn_model import GRUDecoder  # noqa: E402
from data_augmentations import gauss_smooth  # noqa: E402

GOOD = ["t15.2023.08.18", "t15.2023.11.04"]
BAD = ["t15.2025.01.10", "t15.2025.03.14"]
TRAINPROBE = ["t15.2025.01.10", "t15.2025.04.13"]


def load_trials(session, split):
    import h5py
    path = os.path.join(DATA_DIR, session, f"{split}.hdf5")
    trials = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            trials.append({
                "x": g["input_features"][:].astype(np.float32),
                "y": g["seq_class_ids"][:].astype(np.int64),
                "seq_len": int(g.attrs["seq_len"]),
                "block": int(g.attrs["block_num"]),
                "trial": int(g.attrs["trial_num"]),
            })
    return trials


@torch.no_grad()
def sentence_losses(model, args, trials, day_idx, batch=16, transform="val"):
    """Per-sentence CTC NLL on the trainer's reduction='none' scale.

    transform='val': smoothing only ('same'), trainer validation semantics.
    """
    device = next(model.parameters()).device
    ta = args["dataset"]["data_transforms"]
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    out = []
    for i in range(0, len(trials), batch):
        chunk = trials[i:i + batch]
        T = max(t["x"].shape[0] for t in chunk)
        x = torch.zeros(len(chunk), T, 512)
        lens = torch.zeros(len(chunk), dtype=torch.long)
        for j, t in enumerate(chunk):
            x[j, :t["x"].shape[0]] = torch.from_numpy(t["x"])
            lens[j] = t["x"].shape[0]
        y = torch.zeros(len(chunk), 500, dtype=torch.long)
        ylens = torch.zeros(len(chunk), dtype=torch.long)
        for j, t in enumerate(chunk):
            y[j, :t["seq_len"]] = torch.from_numpy(t["y"][:t["seq_len"]])
            ylens[j] = t["seq_len"]
        x, lens, y, ylens = x.to(device), lens.to(device), y.to(device), ylens.to(device)
        day = torch.tensor([day_idx] * len(chunk), device=device)
        with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
            x = gauss_smooth(x, device, ta["smooth_kernel_std"], ta["smooth_kernel_size"], "same")
            adj = ((lens - args["model"]["patch_size"]) / args["model"]["patch_stride"] + 1).to(torch.int32)
            logits = model(x, day)
            losses = ctc(torch.permute(logits.log_softmax(2), [1, 0, 2]), y, adj, ylens)
        out.extend(losses.float().cpu().numpy().tolist())
    return np.array(out)


@torch.no_grad()
def greedy_decode(model, args, trial, day_idx):
    """eval_val_per.py semantics: bf16 input, gauss_smooth 'valid', greedy."""
    device = next(model.parameters()).device
    ta = args["dataset"]["data_transforms"]
    x = torch.tensor(np.expand_dims(trial["x"], 0), device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
        x = gauss_smooth(x, device, ta["smooth_kernel_std"], ta["smooth_kernel_size"], "valid")
        logits, _ = model(x, torch.tensor([day_idx], device=device), None, True)
    pred = torch.argmax(logits[0].float(), dim=-1).cpu()
    pred = torch.unique_consecutive(pred, dim=-1).numpy()
    return np.array([i for i in pred if i != 0])


def main():
    args = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions = list(args["dataset"]["sessions"])
    device = torch.device("cuda:0")
    model = GRUDecoder(
        neural_dim=args["model"]["n_input_features"], n_units=args["model"]["n_units"],
        n_days=len(sessions), n_classes=args["dataset"]["n_classes"],
        rnn_dropout=args["model"]["rnn_dropout"],
        input_dropout=args["model"]["input_network"]["input_layer_dropout"],
        n_layers=args["model"]["n_layers"], patch_size=args["model"]["patch_size"],
        patch_stride=args["model"]["patch_stride"])
    ckpt = torch.load(os.path.join(P4, "checkpoint", "best_checkpoint"),
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in list(sd.keys()):
        sd[key.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(key)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"checkpoint loaded, saved val_PER={ckpt.get('val_PER'):.4f}")

    # ---- P0.2 loss distributions ----
    print("\n== P0.2 per-sentence CTC NLL (reduction='none' scale) ==")
    for s in GOOD + BAD:
        tr = load_trials(s, "data_val")
        L = sentence_losses(model, args, tr, sessions.index(s))
        print(f"{s}  VAL   n={len(L):3d}  median {np.median(L):7.2f}  "
              f"q25 {np.percentile(L,25):7.2f}  q75 {np.percentile(L,75):7.2f}  "
              f"min {L.min():6.2f}  max {L.max():7.2f}")
    for s in TRAINPROBE:
        tr = load_trials(s, "data_train")
        L = sentence_losses(model, args, tr, sessions.index(s))
        print(f"{s}  TRAIN n={len(L):3d}  median {np.median(L):7.2f}  "
              f"q25 {np.percentile(L,25):7.2f}  q75 {np.percentile(L,75):7.2f}  "
              f"min {L.min():6.2f}  max {L.max():7.2f}")

    # ---- P0.3 greedy smoke vs csv ----
    print("\n== P0.3 greedy smoke (t15.2023.08.18, first 5 trials) ==")
    import editdistance
    s = "t15.2023.08.18"
    tr = load_trials(s, "data_val")
    import csv as csvmod
    with open(os.path.join(P4, "val_greedy_per_trial.csv"), newline="") as f:
        csv_rows = [r for r in csvmod.DictReader(f) if r["session"] == s][:5]
    ok = 0
    for i in range(5):
        pred = greedy_decode(model, args, tr[i], sessions.index(s))
        true = tr[i]["y"][:tr[i]["seq_len"]]
        ed = editdistance.eval(pred.tolist(), true.tolist())
        ref = int(csv_rows[i]["edit_distance"]), int(csv_rows[i]["true_len"])
        match = (ed, len(true)) == ref
        ok += match
        print(f"trial {i}: recomputed ed={ed} len={len(true)}  csv ed={ref[0]} len={ref[1]}  {'OK' if match else 'MISMATCH'}")
    print(f"smoke: {ok}/5 match" if ok == 5 else f"SMOKE FAILED: {ok}/5")


if __name__ == "__main__":
    main()
