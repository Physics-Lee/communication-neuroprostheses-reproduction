# -*- coding: utf-8 -*-
"""T7 probe (64h block A): noise-perturbation TTA on P4 (Yoon 2026
'pseudoensembling', single decoder + Gaussian noise at inference + average).

For each val trial of the selected days: K forward passes with additive
Gaussian noise (std s) on raw features BEFORE smoothing (matching the trainer's
augmentation order), average log-softmax, greedy decode. Baseline = standard
single pass. PER per day, before vs after.

Usage: python calib_tta_probe.py [K=8] [stds=0.3,1.0]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_online_sim import build_model, load_trials, REPO, P4  # noqa: E402
from data_augmentations import gauss_smooth  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import editdistance  # noqa: E402

DAYS = ["t15.2025.01.10", "t15.2025.01.12", "t15.2025.03.14",
        "t15.2025.03.16", "t15.2025.03.30", "t15.2025.04.13",
        "t15.2023.08.13", "t15.2023.08.18"]


@torch.no_grad()
def decode_pass(model, args, x_np, day_idx, device, noise_std):
    """Run a single (optionally noise-perturbed) forward pass on one trial.

    Additive Gaussian noise is applied to the raw features BEFORE smoothing,
    matching the trainer's augmentation order.

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder in eval mode.
    args : omegaconf.DictConfig
        Training arguments (transform params, use_amp flag).
    x_np : numpy.ndarray
        Raw input features of shape ``(time, 512)``.
    day_idx : int
        Recording-day index used for the day layer.
    device : torch.device
        Device the computation runs on.
    noise_std : float
        Standard deviation of the additive Gaussian noise; 0 for a clean
        baseline pass.

    Returns
    -------
    torch.Tensor
        Log-softmax probabilities of shape ``(time, n_classes)``.
    """
    ta = args["dataset"]["data_transforms"]
    x = torch.tensor(np.expand_dims(x_np, 0), device=device, dtype=torch.bfloat16)
    if noise_std > 0:
        x = x + (torch.randn(x.shape, device=device, dtype=torch.bfloat16) * noise_std)
    with torch.autocast(device_type="cuda", enabled=args["use_amp"], dtype=torch.bfloat16):
        x = gauss_smooth(x, device, ta["smooth_kernel_std"], ta["smooth_kernel_size"], "valid")
        logits, _ = model(x, torch.tensor([day_idx], device=device), None, True)
    return torch.from_numpy(logits[0].float().cpu().numpy()).log_softmax(-1)


def per_from_preds(model, args, trials, day_idx, device, K, std):
    """Greedy-decode trials and return the corpus-level PER.

    With ``std > 0``, runs ``K`` noise-perturbed passes per trial and greedy-
    decodes from the averaged log-softmax; with ``std == 0``, a single clean
    pass (baseline).

    Parameters
    ----------
    model : torch.nn.Module
        The GRUDecoder (put into eval mode for the scan).
    args : omegaconf.DictConfig
        Training arguments passed through to ``decode_pass``.
    trials : list of dict
        Trial dicts with keys ``x`` and ``y``/``seq_len``.
    day_idx : int
        Recording-day index used for the day layer.
    device : torch.device
        Device the computation runs on.
    K : int
        Number of noise-perturbed passes averaged per trial.
    std : float
        Noise standard deviation; 0 selects the single-pass baseline.

    Returns
    -------
    float
        PER in percent (total edit distance / total reference length).
    """
    tot_ed = tot_len = 0
    model.eval()
    for t in trials:
        if std == 0:
            lp = decode_pass(model, args, t["x"], day_idx, device, 0)
        else:
            lp = None
            for _ in range(K):
                one = decode_pass(model, args, t["x"], day_idx, device, std)
                lp = one if lp is None else lp + one
            lp = lp / K
        pred = torch.argmax(lp, dim=-1)
        pred = torch.unique_consecutive(pred, dim=-1).numpy()
        pred = np.array([i for i in pred if i != 0])
        true = t["y"][:t["seq_len"]]
        tot_ed += editdistance.eval(pred.tolist(), true.tolist())
        tot_len += len(true)
    return 100 * tot_ed / tot_len


def main():
    """Run the noise-perturbation TTA probe on the P4 model.

    For each day in ``DAYS``, prints the baseline single-pass val PER next
    to the TTA PER for each requested noise std (K averaged passes).

    Returns
    -------
    None
        The per-day comparison table is printed to stdout.
    """
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    stds = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.3,1.0").split(",")]
    device = torch.device("cuda:0")
    args = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions = list(args["dataset"]["sessions"])
    model = build_model(args, sessions, device)

    print(f"day  baseline  " + "  ".join(f"TTA(s={s},K={K})" for s in stds))
    for s in DAYS:
        trials = load_trials(s, "data_val")
        di = sessions.index(s)
        base = per_from_preds(model, args, trials, di, device, 1, 0.0)
        row = [base] + [per_from_preds(model, args, trials, di, device, K, sd) for sd in stds]
        print(f"{s}  " + "  ".join(f"{v:7.2f}" for v in row), flush=True)


if __name__ == "__main__":
    main()
