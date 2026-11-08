# -*- coding: utf-8 -*-
"""S2.03 sentence-by-sentence online fine-tuning, simulated offline (plan P1).

Faithful to the paper's Table S5 recipe on the official trainer's semantics:
- decode each val trial of the target session in file order (chronological),
  greedy PER via eval_val_per.py semantics (bf16, gauss_smooth 'valid');
- after K disclosed trials, fine-tune after every trial: batch 64 =
  60% disclosed current-day trials (with replacement) + 40% replay from all
  previous sessions' data_train; day layer + GRU updated; augmentation =
  official (white 1.0 / offset 0.2 / cut 3 / smoothing 'same' inside the
  trainer's transform_data); Adam rebuilt fresh per day, constant LR;
  per trigger: min-steps..max-steps optimizer steps, early stop when the
  step's batch loss < loss-stop (0.5-0.6 in Table S5 = trainer's per-sentence
  NLL scale, calibrated in calib_p0_loss_scale.py);
- day layer init per --arm: copy (duplicate most recent session's day layer,
  S2.03), eye (official offline-code mechanism), keep (P4's own layer).

Memory: replay pool is preloaded once as fp16 (subsampled to --replay-pool
trials); current-day trials stay in fp32. micro 32 x accum 2 == official 64.

Usage:
  calib_online_sim.py t15.2025.01.10 0.004 [--arm copy] [--k 5] ...
  calib_online_sim.py t15.2025.01.10 0.004 --smoke 3 8   (3 trials, 8 steps)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = r".\20240815_card"
P4 = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013")
DATA_DIR = os.path.join(REPO, r"nejm-brain-to-text\data\hdf5_data_final")
OUT_ROOT = os.path.join(REPO, r"data\20241009_card_BtoT25\run_calib_20261014")

sys.path.insert(0, os.path.join(REPO, r"nejm-brain-to-text\model_training"))
from rnn_model import GRUDecoder  # noqa: E402
from data_augmentations import gauss_smooth  # noqa: E402

import editdistance  # noqa: E402


def load_trials(session, split):
    """Load all trials of one session split from the official HDF5 release.

    Parameters
    ----------
    session : str
        Session directory name, e.g. ``t15.2025.01.10``.
    split : str
        HDF5 split filename stem, e.g. ``data_val`` or ``data_train``.

    Returns
    -------
    list of dict
        One dict per trial (sorted by key) with keys ``x`` (float32 input
        features, shape ``(time, 512)``), ``y`` (int64 class-id sequence),
        ``seq_len`` (int), ``block`` (int) and ``trial`` (int).
    """
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


def build_model(args, sessions, device):
    """Instantiate the official GRUDecoder and load the P4 best checkpoint.

    Parameters
    ----------
    args : omegaconf.DictConfig
        Training arguments loaded from the P4 run's ``args.yaml`` (used for
        model hyperparameters and dataset metadata).
    sessions : list of str
        Ordered session list defining the day-layer count.
    device : torch.device
        Device the model is moved to.

    Returns
    -------
    rnn_model.GRUDecoder
        The model in eval-ready state on ``device`` with P4 weights loaded.
    """
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
    return model.to(device)


def make_optimizer(model, lr, args):
    """Trainer's param groups (bias wd 0, day wd 0, rest wd 0.001) but with a
    single constant LR for every group (Table S5 gives one online LR).

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder whose parameters are grouped.
    lr : float
        Constant learning rate applied to every parameter group.
    args : omegaconf.DictConfig
        Training arguments supplying optimizer hyperparameters (weight
        decays, betas, epsilon).

    Returns
    -------
    torch.optim.AdamW
        Fresh optimizer with ``bias``/``day_layer``/``other`` parameter
        groups (``fused=True``).
    """
    bias = [p for n, p in model.named_parameters() if "gru.bias" in n or "out.bias" in n]
    day = [p for n, p in model.named_parameters() if "day_" in n]
    other = [p for n, p in model.named_parameters()
             if "day_" not in n and "gru.bias" not in n and "out.bias" not in n]
    return torch.optim.AdamW(
        [{"params": bias, "weight_decay": 0, "group_type": "bias"},
         {"params": day, "lr": lr, "weight_decay": args["weight_decay_day"], "group_type": "day_layer"},
         {"params": other, "weight_decay": args["weight_decay"], "group_type": "other"}],
        lr=lr, betas=(args["beta0"], args["beta1"]), eps=args["epsilon"],
        weight_decay=args["weight_decay"], fused=True)


def train_transform(features, n_time_steps, ta, device):
    """rnn_trainer.transform_data(mode='train'), official order and scales;
    static_gain / random_walk are 0 in rnn_args.yaml and skipped like upstream.

    Parameters
    ----------
    features : torch.Tensor
        Input batch of shape ``(batch, time, channels)``.
    n_time_steps : torch.Tensor
        Per-sample sequence lengths (adjusted for the random cut).
    ta : omegaconf.DictConfig or dict
        Transform hyperparameters (white_noise_std, constant_offset_std,
        random_cut, smooth_kernel_std, smooth_kernel_size).
    device : torch.device
        Device on which random noise tensors are created.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor)
        The augmented features and the updated sequence lengths.
    """
    b, _, c = features.shape
    features = features + torch.randn(features.shape, device=device) * ta["white_noise_std"]
    features = features + torch.randn((b, 1, c), device=device) * ta["constant_offset_std"]
    cut = int(np.random.randint(0, ta["random_cut"]))
    if cut > 0:
        features = features[:, cut:, :]
        n_time_steps = n_time_steps - cut
    features = gauss_smooth(features, device, ta["smooth_kernel_std"],
                            ta["smooth_kernel_size"], "same")
    return features, n_time_steps


def collate(samples, device):
    """Pad and stack trials into batched training tensors.

    Parameters
    ----------
    samples : list of tuple
        Each element is ``(trial_dict, day_index)`` where ``trial_dict`` has
        keys ``x``, ``y``, ``seq_len``.
    device : torch.device
        Target device for the returned tensors.

    Returns
    -------
    tuple of torch.Tensor
        ``(x, lens, y, ylens, day)`` — zero-padded inputs, input lengths,
        zero-padded (to 500) class-id sequences, label lengths, and day
        indices, all on ``device``.
    """
    T = max(s["x"].shape[0] for s, _ in samples)
    x = torch.zeros(len(samples), T, 512)
    lens = torch.zeros(len(samples), dtype=torch.long)
    y = torch.zeros(len(samples), 500, dtype=torch.long)
    ylens = torch.zeros(len(samples), dtype=torch.long)
    day = torch.zeros(len(samples), dtype=torch.long)
    for j, (s, d) in enumerate(samples):
        x[j, :s["x"].shape[0]] = torch.from_numpy(s["x"]).float()
        lens[j] = s["x"].shape[0]
        y[j, :s["seq_len"]] = torch.from_numpy(s["y"][:s["seq_len"]])
        ylens[j] = s["seq_len"]
        day[j] = d
    return (x.to(device), lens.to(device), y.to(device), ylens.to(device), day.to(device))


@torch.no_grad()
def decode_trial(model, args, trial, day_idx, device):
    """Greedy-decode a single trial with eval_val_per.py semantics.

    Applies bf16 autocast and 'valid'-mode Gaussian smoothing before the
    forward pass, then collapses repeats and removes blanks to yield the CTC
    prediction.

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder in eval mode.
    args : omegaconf.DictConfig
        Training arguments (transform parameters, use_amp flag).
    trial : dict
        Trial dict with key ``x`` (input features).
    day_idx : int
        Recording-day index used for the day layer.
    device : torch.device
        Device the computation runs on.

    Returns
    -------
    numpy.ndarray
        Array of predicted non-blank class ids for the trial.
    """
    ta = args["dataset"]["data_transforms"]
    x = torch.tensor(np.expand_dims(trial["x"], 0), device=device, dtype=torch.bfloat16)
    with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
        x = gauss_smooth(x, device, ta["smooth_kernel_std"], ta["smooth_kernel_size"], "valid")
        logits, _ = model(x, torch.tensor([day_idx], device=device), None, True)
    pred = torch.argmax(logits[0].float(), dim=-1).cpu()
    pred = torch.unique_consecutive(pred, dim=-1).numpy()
    return np.array([i for i in pred if i != 0])


def finetune_trigger(model, opt, args, disclosed, day_idx, replay, rng, cfg, device):
    """One post-trial update burst: fresh 60/40 batches per optimizer step,
    micro 32 x accum 2, min..max steps, early stop below loss-stop.

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder being fine-tuned (must be in train mode).
    opt : torch.optim.AdamW
        Optimizer to step.
    args : omegaconf.DictConfig
        Training arguments (transform params, grad-norm clip, patching).
    disclosed : list of dict
        Current-day trials disclosed so far (sampled with replacement).
    day_idx : int
        Day index assigned to current-day trials.
    replay : list of tuple
        Replay pool of ``(trial_dict, day_index)`` pairs from history
        sessions.
    rng : numpy.random.Generator
        RNG driving batch sampling and shuffling.
    cfg : argparse.Namespace
        CLI config with ``batch``, ``cur_frac``, ``accum``, ``min_steps``,
        ``max_steps`` and ``loss_stop``.
    device : torch.device
        Device the computation runs on.

    Returns
    -------
    list of float
        Mean (per-sentence NLL scale) loss of each optimizer step taken.
    """
    ta = args["dataset"]["data_transforms"]
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    n_cur = int(round(cfg.batch * cfg.cur_frac))
    n_rep = cfg.batch - n_cur
    step_losses = []
    for step in range(cfg.max_steps):
        opt.zero_grad()
        micro_losses = []
        cur_pick = [(disclosed[rng.integers(len(disclosed))], day_idx) for _ in range(n_cur)]
        rep_pick = [replay[rng.integers(len(replay))] for _ in range(n_rep)]
        batch = cur_pick + rep_pick
        rng.shuffle(batch)
        for m in range(cfg.accum):
            micro = batch[m * (cfg.batch // cfg.accum):(m + 1) * (cfg.batch // cfg.accum)]
            x, lens, y, ylens, day = collate(micro, device)
            with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
                x, lens = train_transform(x, lens, ta, device)
                adj = ((lens - args["model"]["patch_size"]) / args["model"]["patch_stride"] + 1).to(torch.int32)
                logits = model(x, day)
                loss = ctc(torch.permute(logits.log_softmax(2), [1, 0, 2]), y, adj, ylens)
                loss = torch.mean(loss) / cfg.accum
            loss.backward()
            micro_losses.append(loss.detach().item() * cfg.accum)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args["grad_norm_clip_value"],
                                       error_if_nonfinite=True, foreach=True)
        opt.step()
        step_losses.append(float(np.mean(micro_losses)))
        if step + 1 >= cfg.min_steps and step_losses[-1] < cfg.loss_stop:
            break
    return step_losses


def main():
    """Run the offline simulation of sentence-by-sentence online fine-tuning.

    Decodes each val trial of the target session before its label is
    disclosed, fine-tunes after every trial once ``--k`` trials are
    accumulated, then writes per-trial online PER, endpoint re-decode,
    a steps log, summary JSON and the final checkpoint under
    ``OUT_ROOT/<session>_lr<lr>_arm<arm>``.

    Returns
    -------
    None
        Results are printed to stdout and written to the output directory.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("lr", type=float)
    ap.add_argument("--arm", default="copy", choices=["copy", "eye", "keep"])
    ap.add_argument("--k", type=int, default=5, help="trials accumulated before updates start")
    ap.add_argument("--cur-frac", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--min-steps", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--loss-stop", type=float, default=0.55)
    ap.add_argument("--replay-pool", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--smoke", nargs=2, type=int, default=None, metavar=("NTRIALS", "NSTEPS"))
    cfg = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device("cuda:0")

    args = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions = list(args["dataset"]["sessions"])
    s_idx = sessions.index(cfg.session)
    prev_idx = s_idx - 1

    tag = f"{cfg.session}_lr{cfg.lr}_arm{cfg.arm}" + ("_smoke" if cfg.smoke else "")
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(args, sessions, device)

    # day-layer init
    if cfg.arm == "copy":
        with torch.no_grad():
            model.day_weights[s_idx].copy_(model.day_weights[prev_idx])
            model.day_biases[s_idx].copy_(model.day_biases[prev_idx])
        print(f"[init] day layer {s_idx} <- {prev_idx} ({sessions[prev_idx]})", flush=True)
    elif cfg.arm == "eye":
        with torch.no_grad():
            model.day_weights[s_idx].copy_(torch.eye(512))
            model.day_biases[s_idx].zero_()
        print(f"[init] day layer {s_idx} <- eye/zero", flush=True)

    val_trials = load_trials(cfg.session, "data_val")
    if cfg.smoke:
        cfg.k = 2
        cfg.min_steps = cfg.max_steps = cfg.smoke[1]
        val_trials = val_trials[:cfg.smoke[0]]
        replay_days = [prev_idx]
    else:
        replay_days = list(range(0, s_idx))

    # replay pool: subsample from history sessions' data_train, fp16 in RAM
    t0 = time.time()
    replay = []
    per_day = max(1, cfg.replay_pool // len(replay_days))
    for d in replay_days:
        tr = load_trials(sessions[d], "data_train")
        take = min(len(tr), per_day)
        for t in rng.choice(len(tr), size=take, replace=False):
            replay.append((tr[int(t)], d))
    for item, _ in replay:
        item["x"] = item["x"].astype(np.float16)
    print(f"[pool] replay {len(replay)} trials from {len(replay_days)} history sessions "
          f"({time.time()-t0:.0f}s)", flush=True)

    # snapshot untouched day layers for the smoke check
    w_before = model.day_weights[0].detach().clone()

    opt = make_optimizer(model, cfg.lr, args)

    rows = []
    total_steps = 0
    steps_log = open(os.path.join(out_dir, "steps_log.jsonl"), "w")
    t_start = time.time()
    for i, trial in enumerate(val_trials):
        model.eval()
        pred = decode_trial(model, args, trial, s_idx, device)
        true = trial["y"][:trial["seq_len"]]
        ed = editdistance.eval(pred.tolist(), true.tolist())
        rows.append({"i": i, "block": trial["block"], "trial": trial["trial"],
                     "ed": ed, "len": int(trial["seq_len"]),
                     "per": ed / max(1, trial["seq_len"]), "steps": 0, "last_loss": float("nan")})
        cum = sum(r["ed"] for r in rows) / max(1, sum(r["len"] for r in rows))
        print(f"[trial {i:3d}] block {trial['block']} ed {ed}/{trial['seq_len']} "
              f"cum PER {100*cum:.2f}%", flush=True)

        if i + 1 >= cfg.k:
            model.train()
            t_ft = time.time()
            losses = finetune_trigger(model, opt, args, val_trials[:i + 1], s_idx,
                                      replay, rng, cfg, device)
            total_steps += len(losses)
            rows[-1]["steps"] = len(losses)
            rows[-1]["last_loss"] = losses[-1]
            steps_log.write(json.dumps({"trial": i, "losses": losses}) + "\n")
            steps_log.flush()
            print(f"  [ft] {len(losses)} steps, loss {losses[0]:.2f} -> {losses[-1]:.2f} "
                  f"({time.time()-t_ft:.0f}s)", flush=True)

    # final re-decode of all trials with endpoint weights
    model.eval()
    fin_rows = []
    for i, trial in enumerate(val_trials):
        pred = decode_trial(model, args, trial, s_idx, device)
        true = trial["y"][:trial["seq_len"]]
        ed = editdistance.eval(pred.tolist(), true.tolist())
        fin_rows.append({"i": i, "ed": ed, "len": int(trial["seq_len"])})

    online_per = sum(r["ed"] for r in rows) / sum(r["len"] for r in rows)
    endpoint_per = sum(r["ed"] for r in fin_rows) / sum(r["len"] for r in fin_rows)
    w_changed = (model.day_weights[0] - w_before).abs().max().item()
    summary = {
        "session": cfg.session, "lr": cfg.lr, "arm": cfg.arm, "k": cfg.k,
        "cur_frac": cfg.cur_frac, "min_steps": cfg.min_steps, "max_steps": cfg.max_steps,
        "loss_stop": cfg.loss_stop, "n_trials": len(rows),
        "online_PER": online_per, "endpoint_PER": endpoint_per,
        "total_ft_steps": total_steps,
        "day0_weight_max_delta (0 expected in smoke only)": w_changed,
        "wall_min": (time.time() - t_start) / 60,
    }
    print(json.dumps(summary, indent=2), flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_trial_online.csv"), index=False)
    pd.DataFrame(fin_rows).to_csv(os.path.join(out_dir, "final_decode.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model_state_dict": model.state_dict()},
               os.path.join(out_dir, "final_checkpoint.pt"))
    steps_log.close()
    print(f"[done] outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
