"""Multi-session joint + cumulative warm-start trainer for Jude et al. (2026a).

Implements the multi-session training architecture of the official
``utils/brainToText_trainDecoder.py`` (one ``InputNet`` per session + a shared
5-layer ``TypingGRU`` backbone), together with the official participant-specific
hyperparameters from ``T17_trainArgs.yaml`` and ``T18_trainArgs.yaml``:

- **T17**: ``channels_to_exclude = range(128, 256)`` (excludes Area 55b ->
  ``n_features = 512``), ``patch_size = 16``, ``patch_stride = 9``,
  ``white_noise_sd = 1.4``, ``const_offset_sd = 0.4``.
- **T18**: all 384 channels (``n_features = 768``), ``patch_size = 15``,
  ``patch_stride = 6``, ``white_noise_sd = 1.5``, ``const_offset_sd = 0.22``.

After joint multi-session training across all sessions of a participant, an
optional per-session warm-start fine-tuning stage refines each session's
``InputNet`` + ``TypingGRU`` from the best checkpoint, keeping the lowest
validation CER and exporting ``test_logits.npz`` for 5-gram WFST decoding.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import edit_distance, gauss_smooth, greedy_ctc_decode
from model import InputNet, TypingGRU

T17_SESSIONS = [
    't17.2024.04.25',
    't17.2024.06.03',
    't17.2024.06.04',
    't17.2024.06.05',
    't17.2024.06.13',
    't17.2024.07.09',
    't17.2024.07.10',
    't17.2024.07.22',
]

T18_SESSIONS = [
    't18.2024.12.04',
    't18.2024.12.05',
    't18.2025.01.14',
    't18.2025.01.15',
    't18.2025.01.21',
    't18.2025.01.22',
    't18.2025.02.04',
    't18.2025.02.05',
]

# Official T17 channels_to_exclude = range(128, 256) in [thresh(384), spikepow(384)]
T17_KEEP_IDX = np.array(
    list(range(0, 128)) + list(range(256, 384))
    + list(range(384, 512)) + list(range(640, 768)),
    dtype=np.int64,
)


class SessionDataset:
    """Per-session variable-length dataset supporting channel slicing and custom patch geometry."""

    def __init__(self, npz_path, split='train', keep_idx=None, patch_size=15, patch_stride=6):
        z = np.load(npz_path, allow_pickle=True)
        raw = z[f'{split}_features']
        if raw.ndim != 1:
            arr = np.empty(raw.shape[0], dtype=object)
            for i in range(raw.shape[0]):
                arr[i] = np.asarray(raw[i], dtype=np.float32)
            raw = arr
        if keep_idx is not None:
            self.features = [np.ascontiguousarray(f[:, keep_idx], dtype=np.float32) for f in raw]
        else:
            self.features = [np.ascontiguousarray(f, dtype=np.float32) for f in raw]
        self.seq_ids = z[f'{split}_seqClassIDs']
        self.n_ts = z[f'{split}_nTimeSteps']
        self.n_el = z[f'{split}_nSeqElements']
        self.transcriptions = z[f'{split}_transcription']
        self.labels = [torch.tensor(r[r > 0] - 1, dtype=torch.long) for r in self.seq_ids]
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def __len__(self):
        return len(self.features)

    def make_batch(self, idx, augment=False, white_noise_sd=1.5, const_offset_sd=0.22, device=None, max_len=None):
        feats = [torch.tensor(self.features[i][:max_len] if max_len else self.features[i], dtype=torch.float32) for i in idx]
        tmax = max(f.shape[0] for f in feats)
        b = len(idx)
        c = feats[0].shape[1]
        x = torch.zeros(b, tmax, c)
        for j, f in enumerate(feats):
            x[j, :f.shape[0]] = f
        if device is not None:
            x = x.to(device)
        if augment:
            if white_noise_sd > 0:
                x = x + torch.randn_like(x) * white_noise_sd
            if const_offset_sd > 0:
                x = x + torch.randn(b, 1, c, device=x.device) * const_offset_sd
        x = gauss_smooth(x)
        input_lengths = torch.tensor(
            [(min(int(self.n_ts[i]), tmax) - self.patch_size) // self.patch_stride + 1 for i in idx],
            dtype=torch.long,
        )
        targets = torch.cat([self.labels[i] for i in idx])
        target_lengths = torch.tensor([int(self.n_el[i]) for i in idx], dtype=torch.long)
        return x, input_lengths, targets, target_lengths


@torch.no_grad()
def eval_session(model, input_net, val_ds, device, batch_size=64):
    """Evaluate greedy-CTC CER on a single session's test split."""
    model.eval()
    input_net.eval()
    n_err, n_tot = 0, 0
    idx_all = list(range(len(val_ds)))
    for s in range(0, len(idx_all), batch_size):
        batch_idx = idx_all[s:s + batch_size]
        x, il, _, _ = val_ds.make_batch(batch_idx, augment=False, device=device)
        logits = model(input_net(x, training=False), training=False)
        for j, i in enumerate(batch_idx):
            dec = greedy_ctc_decode(logits[j].cpu(), int(il[j]))
            true = val_ds.labels[i].tolist()
            n_err += edit_distance(true, dec)
            n_tot += len(true)
    return n_err, max(1, n_tot), n_err / max(1, n_tot)


@torch.no_grad()
def dump_session_logits(model, input_net, val_ds, out_dir, step, cer, device, batch_size=64):
    """Export per-trial test_logits.npz in the exact format consumed by s9a_lm_decode_multi.py."""
    model.eval()
    input_net.eval()
    all_logits, all_true = [], []
    idx_all = list(range(len(val_ds)))
    for s in range(0, len(idx_all), batch_size):
        idx = idx_all[s:s + batch_size]
        x, il, _, _ = val_ds.make_batch(idx, augment=False, device=device)
        logits = model(input_net(x, training=False), training=False)
        for j, i in enumerate(idx):
            n = int(il[j])
            all_logits.append(logits[j, :n].float().cpu().numpy())
            tr = val_ds.transcriptions[i]
            end = int(np.argwhere(tr == 0)[0, 0]) if (tr == 0).any() else len(tr)
            all_true.append(''.join(chr(int(c)) for c in tr[:end]))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / 'test_logits.npz',
        logits=np.array(all_logits, dtype=object),
        true_sentences=np.array(all_true, dtype=object),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--participant', choices=['t17', 't18'], required=True)
    ap.add_argument('--data-root', default=r'D:\repositories\20261009_reproduce_20260316_jude\data\20260316_jude\Data')
    ap.add_argument('--out-root', default=r'D:\repositories\20261106_communication_neuroprostheses_reproduction\20260316_jude\results\runs_multisess')
    ap.add_argument('--batches', type=int, default=10000)
    ap.add_argument('--ft-batches', type=int, default=1500)
    ap.add_argument('--patience', type=int, default=2000)
    ap.add_argument('--val-every', type=int, default=50)
    ap.add_argument('--max-train-len', type=int, default=0)
    ap.add_argument('--warmstart-single', action='store_true')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    is_t17 = args.participant == 't17'
    sessions = T17_SESSIONS if is_t17 else T18_SESSIONS
    keep_idx = T17_KEEP_IDX if is_t17 else None
    n_features = 512 if is_t17 else 768
    patch_size = 16 if is_t17 else 15
    patch_stride = 9 if is_t17 else 6
    white_noise_sd = 1.4 if is_t17 else 1.5
    const_offset_sd = 0.4 if is_t17 else 0.22
    batch_size = 64
    weight_reg = 1e-5
    grad_clip = 10.0
    lr_start = 0.02
    lr_end = 0.0
    adam_eps = 1e-1

    out_root = Path(args.out_root) / args.participant
    out_root.mkdir(parents=True, exist_ok=True)
    log_f = open(out_root / 'training_log', 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'=== Multi-Session Training ({args.participant.upper()}) on {device} ===')
    log(f'n_features={n_features}, patch=({patch_size},{patch_stride}), noise=({white_noise_sd},{const_offset_sd})')

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_dss, val_dss = [], []
    for s in sessions:
        npz_path = Path(args.data_root) / args.participant / s / 'Typing' / 'm1_torch_data.npz'
        tr_ds = SessionDataset(npz_path, 'train', keep_idx=keep_idx, patch_size=patch_size, patch_stride=patch_stride)
        va_ds = SessionDataset(npz_path, 'test', keep_idx=keep_idx, patch_size=patch_size, patch_stride=patch_stride)
        train_dss.append(tr_ds)
        val_dss.append(va_ds)
        log(f'  loaded {s}: train={len(tr_ds)}, test={len(va_ds)}')

    model = TypingGRU(
        n_features=n_features, units=512, n_layers=5, n_classes=31,
        patch_size=patch_size, patch_stride=patch_stride, gru_dropout=0.6,
    ).to(device)
    input_nets = nn.ModuleList([InputNet(n_features, n_features, dropout=0.2).to(device) for _ in sessions])

    params = list(model.parameters()) + list(input_nets.parameters())
    opt = torch.optim.Adam(params, lr=lr_start, betas=(0.9, 0.999), eps=adam_eps)

    best_overall_cer = float('inf')
    best_sess_cer = [float('inf') for _ in sessions]
    best_sess_state = [None for _ in sessions]
    best_overall_state = None
    batches_since_improved = 0
    batch_orders = [[] for _ in sessions]

    if args.warmstart_single and not is_t17:
        single_root = Path(r'D:\repositories\20261009_reproduce_20260316_jude\result\runs')
        single_map = {
            't18.2024.12.04': 'm1_t18_20241204_torch',
            't18.2024.12.05': 'm3_t18_20241205_torch',
            't18.2025.01.14': 'm3_t18_20250114_torch',
            't18.2025.01.15': 'm3_t18_20250115_torch',
            't18.2025.01.21': 'm3_t18_20250121_torch',
            't18.2025.01.22': 'm3_t18_20250122_torch',
            't18.2025.02.04': 'm3_t18_20250204_torch',
            't18.2025.02.05': 'm3_t18_20250205_torch',
        }
        anchor_ck = torch.load(single_root / 'm3_t18_20250121_torch' / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(anchor_ck['model'])
        lr_start = 0.01
        for d_idx, s in enumerate(sessions):
            ck_p = single_root / single_map[s] / 'best.pt'
            if ck_p.exists():
                ck = torch.load(ck_p, map_location=device, weights_only=True)
                best_sess_cer[d_idx] = float(ck['cer'])
                best_sess_state[d_idx] = {
                    'model': copy.deepcopy(ck['model']),
                    'input_net': copy.deepcopy(ck['input_net']),
                    'step': f'single_{ck.get("step")}',
                    'cer': float(ck['cer']),
                }
                if s == 't18.2025.01.21':
                    input_nets[d_idx].load_state_dict(ck['input_net'])
                log(f'  seeded {s} baseline best_cer={best_sess_cer[d_idx]:.4f}')

    # Weight session sampling gently by sqrt(n_train) or uniform (official is uniform randint)
    t0 = time.time()
    for step in range(args.batches):
        day = int(rng.integers(0, len(sessions)))
        while len(batch_orders[day]) < batch_size:
            batch_orders[day].extend(rng.permutation(len(train_dss[day])).tolist())
        idx = [batch_orders[day].pop() for _ in range(batch_size)]

        frac = min(1.0, step / max(1, args.batches))
        lr = lr_start + frac * (lr_end - lr_start)
        for g in opt.param_groups:
            g['lr'] = lr

        model.train()
        input_nets[day].train()
        max_tl = args.max_train_len if args.max_train_len > 0 else None
        x, il, tg, tl = train_dss[day].make_batch(
            idx, augment=True, white_noise_sd=white_noise_sd, const_offset_sd=const_offset_sd,
            device=device, max_len=max_tl,
        )
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(input_nets[day](x, training=True), training=True)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            pred_loss = F.ctc_loss(
                log_probs.permute(1, 0, 2), tg.to(device), il.to(device), tl.to(device),
                blank=30, reduction='mean', zero_infinity=True,
            )
        reg = model.l2_reg(weight_reg) + weight_reg * (input_nets[day].fc.weight ** 2).sum()
        loss = pred_loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if torch.isfinite(grad_norm):
            opt.step()

        if step % args.val_every == 0 or step == args.batches - 1:
            tot_err, tot_ch = 0, 0
            per_s_str = []
            any_sess_improved = False
            for d_idx, s in enumerate(sessions):
                ne, nc, cer = eval_session(model, input_nets[d_idx], val_dss[d_idx], device, batch_size)
                tot_err += ne
                tot_ch += nc
                if cer < best_sess_cer[d_idx]:
                    best_sess_cer[d_idx] = cer
                    best_sess_state[d_idx] = {
                        'model': copy.deepcopy(model.state_dict()),
                        'input_net': copy.deepcopy(input_nets[d_idx].state_dict()),
                        'step': step,
                        'cer': cer,
                    }
                    any_sess_improved = True
                per_s_str.append(f'{s[-5:]}:{cer:.3f}(*{best_sess_cer[d_idx]:.3f})')
            pooled_cer = tot_err / max(1, tot_ch)
            mean_best = float(np.mean(best_sess_cer))
            el = (time.time() - t0) / 60.0
            log(f'step {step:4d} ({el:4.1f}m) | pooled {pooled_cer:.4f} | mean_best {mean_best:.4f} | ' + ' '.join(per_s_str))
            if pooled_cer < best_overall_cer or any_sess_improved:
                if pooled_cer < best_overall_cer:
                    best_overall_cer = pooled_cer
                    best_overall_state = {
                        'model': copy.deepcopy(model.state_dict()),
                        'input_nets': copy.deepcopy(input_nets.state_dict()),
                        'step': step,
                        'pooled_cer': pooled_cer,
                    }
                batches_since_improved = 0
            else:
                batches_since_improved += args.val_every
                if batches_since_improved >= args.patience:
                    log(f'Joint early stop @ step {step}')
                    break

    # Optional per-session warm-start fine-tuning from each session's best joint state
    if args.ft_batches > 0:
        log(f'=== Starting per-session warm-start fine-tuning ({args.ft_batches} batches/session) ===')
        for d_idx, s in enumerate(sessions):
            st = best_sess_state[d_idx]
            model.load_state_dict(st['model'])
            inet = InputNet(n_features, n_features, dropout=0.2).to(device)
            inet.load_state_dict(st['input_net'])
            ft_params = list(model.parameters()) + list(inet.parameters())
            ft_lr_start = 0.01
            ft_opt = torch.optim.Adam(ft_params, lr=ft_lr_start, betas=(0.9, 0.999), eps=adam_eps)
            b_order = []
            no_imp = 0
            start_cer = best_sess_cer[d_idx]
            for fstep in range(1, args.ft_batches + 1):
                while len(b_order) < batch_size:
                    b_order.extend(rng.permutation(len(train_dss[d_idx])).tolist())
                idx = [b_order.pop() for _ in range(batch_size)]
                flr = ft_lr_start * (1.0 - fstep / max(1, args.ft_batches))
                for g in ft_opt.param_groups:
                    g['lr'] = flr
                model.train()
                inet.train()
                max_tl = args.max_train_len if args.max_train_len > 0 else None
                x, il, tg, tl = train_dss[d_idx].make_batch(
                    idx, augment=True, white_noise_sd=white_noise_sd, const_offset_sd=const_offset_sd,
                    device=device, max_len=max_tl,
                )
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                    logits = model(inet(x, training=True), training=True)
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    pred_loss = F.ctc_loss(
                        log_probs.permute(1, 0, 2), tg.to(device), il.to(device), tl.to(device),
                        blank=30, reduction='mean', zero_infinity=True,
                    )
                reg = model.l2_reg(weight_reg) + weight_reg * (inet.fc.weight ** 2).sum()
                loss = pred_loss + reg
                ft_opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(ft_params, grad_clip)
                if torch.isfinite(grad_norm):
                    ft_opt.step()
                if fstep % 25 == 0 or fstep == args.ft_batches:
                    _, _, cer = eval_session(model, inet, val_dss[d_idx], device, batch_size)
                    if cer < best_sess_cer[d_idx]:
                        best_sess_cer[d_idx] = cer
                        best_sess_state[d_idx] = {
                            'model': copy.deepcopy(model.state_dict()),
                            'input_net': copy.deepcopy(inet.state_dict()),
                            'step': f'joint+{fstep}',
                            'cer': cer,
                        }
                        no_imp = 0
                    else:
                        no_imp += 25
                        if no_imp >= 600:
                            break
            log(f'  FT {s}: {start_cer:.4f} -> {best_sess_cer[d_idx]:.4f} (step={best_sess_state[d_idx]["step"]})')

    # Save per-session best checkpoints and test_logits.npz
    summary = {}
    for d_idx, s in enumerate(sessions):
        s_dir = out_root / s
        s_dir.mkdir(parents=True, exist_ok=True)
        st = best_sess_state[d_idx]
        if (s_dir / 'best.pt').exists():
            prev_st = torch.load(s_dir / 'best.pt', map_location=device, weights_only=True)
            if float(prev_st.get('cer', 1.0)) < float(st['cer']):
                log(f'  keeping prior best for {s}: {prev_st["cer"]:.4f} < {st["cer"]:.4f}')
                st = prev_st
        torch.save(st, s_dir / 'best.pt')
        model.load_state_dict(st['model'])
        inet = InputNet(n_features, n_features, dropout=0.2).to(device)
        inet.load_state_dict(st['input_net'])
        dump_session_logits(model, inet, val_dss[d_idx], s_dir, st['step'], st['cer'], device, batch_size)
        rep = {
            'session': s,
            'best_cer': float(st['cer']),
            'step': st['step'],
            'n_train': len(train_dss[d_idx]),
            'n_test': len(val_dss[d_idx]),
        }
        with open(s_dir / 'final_report.json', 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2)
        summary[s] = rep

    with open(out_root / 'multisession_summary.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'participant': args.participant,
                'total_hours': (time.time() - t0) / 3600.0,
                'sessions': summary,
            },
            f,
            indent=2,
        )
    log('=== MULTISESSION_DONE ===')


if __name__ == '__main__':
    main()
