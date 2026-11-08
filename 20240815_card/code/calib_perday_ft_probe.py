# -*- coding: utf-8 -*-
"""T1 probe (64h block A): dedicated per-day fine-tune from the P4 soup.

Competition-style 'same-day specialization': fine-tune P4 on the target day's
data_train split (all trials, offline batch training, augmentation + 40%
history replay) and check whether that day's data_val PER moves. P0.2 predicts
~neutral (soup already memorizes the train split; val gap is generalization);
this closes the question empirically with a proper offline regimen (vs P1's
32-step online-sim bursts).

Usage: python calib_perday_ft_probe.py <session> [lr=0.001] [steps=1500]
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_online_sim import (  # noqa: E402
    build_model, load_trials, collate, train_transform, decode_trial,
    make_optimizer, REPO, P4, DATA_DIR,
)
from omegaconf import OmegaConf  # noqa: E402
import editdistance  # noqa: E402
import pandas as pd  # noqa: E402


def day_per(model, args, trials, day_idx, device):
    """Compute greedy PER over a list of trials for one day.

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder (put into eval mode for the scan).
    args : omegaconf.DictConfig
        Training arguments passed through to ``decode_trial``.
    trials : list of dict
        Trial dicts with keys ``x`` and ``y``/``seq_len``.
    day_idx : int
        Recording-day index used for the day layer.
    device : torch.device
        Device the computation runs on.

    Returns
    -------
    tuple of (float, int, int)
        ``(PER_percent, total_edit_distance, total_reference_length)``.
    """
    tot_ed = tot_len = 0
    model.eval()
    for t in trials:
        pred = decode_trial(model, args, t, day_idx, device)
        true = t["y"][:t["seq_len"]]
        tot_ed += editdistance.eval(pred.tolist(), true.tolist())
        tot_len += len(true)
    return 100 * tot_ed / tot_len, tot_ed, tot_len


def main():
    """Fine-tune the P4 soup on one day's train split and report val PER.

    Loads the P4 checkpoint, evaluates the target day's data_val PER, runs a
    competition-style offline fine-tune (batch 64 = 38 current-day + 26
    replay trials, official augmentation, gradient accumulation micro 32 x 2)
    for the requested number of steps, re-evaluates val PER, and writes a
    JSON summary under ``run_calib_20261014``.

    Returns
    -------
    None
        Progress and the summary JSON are printed to stdout; results are
        saved to the output JSON file.
    """
    session = sys.argv[1]
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    seed = 10
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda:0")

    args = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions = list(args["dataset"]["sessions"])
    s_idx = sessions.index(session)
    model = build_model(args, sessions, device)

    train = load_trials(session, "data_train")
    val = load_trials(session, "data_val")
    replay_pool = []
    for d in range(0, s_idx):
        tr = load_trials(sessions[d], "data_train")
        per_day = min(len(tr), 60)
        for i in rng.choice(len(tr), size=per_day, replace=False):
            replay_pool.append((tr[int(i)], d))
    print(f"[{session}] day {s_idx}: {len(train)} train / {len(val)} val / "
          f"{len(replay_pool)} replay", flush=True)

    per0, ed0, ln0 = day_per(model, args, val, s_idx, device)
    print(f"[before] val PER {per0:.2f}% ({ed0}/{ln0})", flush=True)

    opt = make_optimizer(model, lr, args)
    ta = args["dataset"]["data_transforms"]
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    model.train()
    t0 = time.time()
    losses = []
    for step in range(steps):
        opt.zero_grad()
        batch = [(train[rng.integers(len(train))], s_idx) for _ in range(38)] + \
                [replay_pool[rng.integers(len(replay_pool))] for _ in range(26)]
        rng.shuffle(batch)
        micro_losses = []
        for m in range(2):
            micro = batch[m * 32:(m + 1) * 32]
            x, lens, y, ylens, day = collate(micro, device)
            with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
                x, lens = train_transform(x, lens, ta, device)
                adj = ((lens - args["model"]["patch_size"]) / args["model"]["patch_stride"] + 1).to(torch.int32)
                logits = model(x, day)
                loss = torch.mean(ctc(torch.permute(logits.log_softmax(2), [1, 0, 2]), y, adj, ylens)) / 2
            loss.backward()
            micro_losses.append(loss.detach().item() * 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args["grad_norm_clip_value"],
                                       error_if_nonfinite=True, foreach=True)
        opt.step()
        losses.append(float(np.mean(micro_losses)))
        if (step + 1) % 250 == 0:
            print(f"  step {step+1}: batch loss {np.mean(losses[-250:]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    per1, ed1, ln1 = day_per(model, args, val, s_idx, device)
    out = {"session": session, "lr": lr, "steps": steps,
           "val_PER_before": per0, "val_PER_after": per1,
           "delta": per1 - per0, "final_batch_loss": float(np.mean(losses[-50:])),
           "first_batch_loss": float(np.mean(losses[:20])), "wall_min": (time.time() - t0) / 60}
    print(json.dumps(out, indent=2), flush=True)
    outdir = os.path.join(REPO, "data", "20241009_card_BtoT25", "run_calib_20261014")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"t1_perdayft_{session.split('.')[-2]}{session.split('.')[-1]}_lr{lr}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
