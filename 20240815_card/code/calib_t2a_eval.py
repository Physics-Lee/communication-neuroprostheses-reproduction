# -*- coding: utf-8 -*-
"""T2a evaluation: zero-shot + cross-fit per-day fine-tune on 2025.01.10.

T2a = GRU trained on 44 sessions (2025.01.10 fully excluded). Its day layers
are positional over the 44-list; this script remaps them into a 45-slot model
(slot 39 = 01.10 left fresh: eye = official new-day mechanism), then:
  1. zero-shot PER on 01.10 val with slot-39 eye layer and with copy-of-38;
  2. sanity PER on a few early days (should match P4-level);
  3. cross-fit fine-tune on 01.10's 106 train sentences (replay 40%) with a
     small LR x steps grid; val PER after each.

Usage: python calib_t2a_eval.py [--ft]   (--ft runs the sweep, ~1.5h)
"""
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_online_sim import (  # noqa: E402
    REPO, P4, DATA_DIR, load_trials, collate, train_transform, decode_trial,
    make_optimizer, build_model,
)
import editdistance  # noqa: E402

T2A = os.path.join(REPO, "data", "20241009_card_BtoT25", "run_t2a_excl0110_20261016")
TARGET = "t15.2025.01.10"


def load_45slot(device):
    """Build a 45-day GRUDecoder and load the 44-session T2a checkpoint with
    slot 39 (01.10) left fresh identity/zero."""
    args45 = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions45 = list(args45["dataset"]["sessions"])
    model = build_model(args45, sessions45, device)  # P4 arch, 45 slots
    args44 = OmegaConf.load(os.path.join(T2A, "checkpoint", "args.yaml"))
    s44 = list(args44["dataset"]["sessions"])
    assert len(s44) == 44 and s44 == [s for s in sessions45 if s != TARGET]
    ckpt = torch.load(os.path.join(T2A, "checkpoint", "best_checkpoint"),
                      weights_only=False, map_location="cpu")["model_state_dict"]
    for key in list(ckpt.keys()):
        ckpt[key.replace("module.", "").replace("_orig_mod.", "")] = ckpt.pop(key)
    with torch.no_grad():
        for i in range(45):
            src = i if i <= 38 else i - 1
            model.day_weights[i].copy_(ckpt[f"day_weights.{src}"])
            model.day_biases[i].copy_(ckpt[f"day_biases.{src}"])
        # non-day params: identical names in both archs
        sd = {k: v for k, v in ckpt.items() if not k.startswith("day_")}
        missing = model.load_state_dict(sd, strict=False)
        assert not missing.unexpected_keys
    return model, args45, sessions45


def day_per(model, args, trials, day_idx, device):
    ed = ln = 0
    model.eval()
    for t in trials:
        pred = decode_trial(model, args, t, day_idx, device)
        true = t["y"][:t["seq_len"]]
        ed += editdistance.eval(pred.tolist(), true.tolist())
        ln += len(true)
    return 100 * ed / ln


def main():
    do_ft = "--ft" in sys.argv
    device = torch.device("cuda:0")
    torch.manual_seed(10)
    rng = np.random.default_rng(10)
    model, args, sessions = load_45slot(device)
    t_idx = sessions.index(TARGET)

    val = load_trials(TARGET, "data_val")
    train = load_trials(TARGET, "data_train")
    per_eye = day_per(model, args, val, t_idx, device)
    with torch.no_grad():  # copy-of-38 variant
        w39 = model.day_weights[t_idx].clone()
        b39 = model.day_biases[t_idx].clone()
        model.day_weights[t_idx].copy_(model.day_weights[t_idx - 1])
        model.day_biases[t_idx].copy_(model.day_biases[t_idx - 1])
        per_copy = day_per(model, args, val, t_idx, device)
        model.day_weights[t_idx].copy_(w39)
        model.day_biases[t_idx].copy_(b39)
    print(f"[zero-shot] 01.10 val PER: eye {per_eye:.2f}% | copy-prev {per_copy:.2f}%", flush=True)
    for s in ["t15.2023.08.13", "t15.2023.08.18", "t15.2024.07.28", "t15.2025.04.13"]:
        di = sessions.index(s)
        print(f"[sanity] {s}: {day_per(model, args, load_trials(s, 'data_val'), di, device):.2f}%", flush=True)

    if not do_ft:
        return
    # cross-fit fine-tune sweep on the fresh model
    replay = []
    for d in range(0, t_idx):
        tr = load_trials(sessions[d], "data_train")
        for i in rng.choice(len(tr), size=min(len(tr), 60), replace=False):
            replay.append((tr[int(i)], d))
    ta = args["dataset"]["data_transforms"]
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    results = []
    for lr, steps in [(0.001, 300), (0.001, 1000), (0.004, 300), (0.004, 1000), (0.001, 3000)]:
        model2, _, _ = load_45slot(device)  # fresh reload each config
        opt = make_optimizer(model2, lr, args)
        model2.train()
        t0 = time.time()
        for step in range(steps):
            opt.zero_grad()
            batch = [(train[rng.integers(len(train))], t_idx) for _ in range(38)] + \
                    [replay[rng.integers(len(replay))] for _ in range(26)]
            rng.shuffle(batch)
            for m in range(2):
                micro = batch[m * 32:(m + 1) * 32]
                x, lens, y, ylens, day = collate(micro, device)
                with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
                    x, lens = train_transform(x, lens, ta, device)
                    adj = ((lens - args["model"]["patch_size"]) / args["model"]["patch_stride"] + 1).to(torch.int32)
                    logits = model2(x, day)
                    loss = torch.mean(ctc(torch.permute(logits.log_softmax(2), [1, 0, 2]), y, adj, ylens)) / 2
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=10, error_if_nonfinite=True, foreach=True)
            opt.step()
        per = day_per(model2, args, val, t_idx, device)
        results.append({"lr": lr, "steps": steps, "val_PER": per})
        print(f"[ft] lr={lr} steps={steps}: 01.10 val PER {per:.2f}% ({time.time()-t0:.0f}s)", flush=True)
        del model2
        torch.cuda.empty_cache()
    print(results)


if __name__ == "__main__":
    main()
