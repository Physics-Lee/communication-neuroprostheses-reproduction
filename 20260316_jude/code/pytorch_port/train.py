# train.py — 官方 brainToText_trainDecoder 训练循环的 PyTorch 移植（GPU 版，2026-10-10）
# 超参默认值逐项对照 code/m1_t18_20241204_args.yaml（官方 notebook 25 的 T18_trainArgs.yaml 翻版）。
# 运行环境：.\20260910_reproduce_20260629_levy\.venv312\Scripts\python.exe（torch 2.7.1+cu128）
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import TypingDataset, greedy_ctc_decode, edit_distance
from model import InputNet, TypingGRU

NPZ = Path(r'.\20260316_jude\data\20260316_jude\Data\t18\t18.2024.12.04\Typing\m1_torch_data.npz')

# 官方 yaml 逐项抄录（改动须在 README 偏差表登记）
HP = dict(
    batch_size=64, n_batches=10000, learn_rate_start=0.02, learn_rate_end=0.0,
    learn_rate_decay_steps=10000, grad_clip=10.0, weight_reg=1e-5,
    adam_eps=1e-1,                      # 官方 epsilon=1e-01（异常大，但必须照抄）
    white_noise_sd=1.5, const_offset_sd=0.22,
    batches_per_val=50, early_stop_batches=1000, seed=1234,
)


def lr_at(step, hp):
    # 官方 PolynomialDecay(power=1)：线性 0.02 -> 0
    frac = min(1.0, step / hp['learn_rate_decay_steps'])
    return hp['learn_rate_start'] + frac * (hp['learn_rate_end'] - hp['learn_rate_start'])


@torch.no_grad()
def validate(model, input_net, val_ds, device):
    model.eval()
    input_net.eval()
    n_err, n_tot = 0, 0
    idx = list(range(len(val_ds)))
    bsz = HP['batch_size']
    for s in range(0, len(idx), bsz):
        batch_idx = idx[s:s + bsz]
        x, il, _, tl = val_ds.make_batch(batch_idx, augment=False)
        logits = model(input_net(x.to(device), training=False), training=False)
        for j, i in enumerate(batch_idx):
            dec = greedy_ctc_decode(logits[j].cpu(), int(il[j]))
            true = val_ds.labels[i].tolist()
            n_err += edit_distance(true, dec)
            n_tot += len(true)
    return n_err / max(1, n_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0, help='>0 则只跑 N 批冒烟并报告速度')
    ap.add_argument('--batches', type=int, default=HP['n_batches'])
    ap.add_argument('--npz', default=str(NPZ))
    ap.add_argument('--n-features', type=int, default=768, help='T18=768, T17=512')
    ap.add_argument('--patience', type=int, default=HP['early_stop_batches'], help='早停耐心（批数）；CER 长期不降的会话可调大')
    ap.add_argument('--out', default=r'.\20260316_jude\result\runs\m1_t18_20241204_torch')
    args = ap.parse_args()
    npz_path = Path(args.npz)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / 'training_log', 'a', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + '\n')
        log_f.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'device: {device} ({torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})')
    torch.manual_seed(HP['seed'])
    torch.backends.cudnn.benchmark = False  # 变长序列下 benchmark 每个新形状都重调优，反而极慢；用启发式
    torch.set_float32_matmul_precision('high')  # TF32，仅提速不影响收敛量级
    np.random.seed(HP['seed'])

    train_ds = TypingDataset(npz_path, 'train')
    val_ds = TypingDataset(npz_path, 'test')
    log(f'trials: train={len(train_ds)} test={len(val_ds)}')

    model = TypingGRU(n_features=args.n_features).to(device)
    input_net = InputNet(args.n_features, args.n_features).to(device)  # 单会话：一个输入网（多会话时按官方建多个）

    params = list(model.parameters()) + list(input_net.parameters())
    # 官方 epsilon=0.1 的 Adam
    opt = torch.optim.Adam(params, lr=HP['learn_rate_start'], betas=(0.9, 0.999), eps=HP['adam_eps'])

    n_batches = args.smoke if args.smoke > 0 else args.batches
    best_cer = 1.0
    batches_since_improved = 0
    rng = np.random.default_rng(HP['seed'])
    t0 = time.time()
    batch_order = []
    log(f'begin training: {n_batches} batches, batch_size={HP["batch_size"]}')

    for step in range(n_batches):
        # 官方为流式 shuffle(buffer 1024)+repeat；此处每 epoch 打乱循环（偏差见 README）
        if not batch_order:
            batch_order = rng.permutation(len(train_ds)).tolist()
        idx = [batch_order.pop() for _ in range(min(HP['batch_size'], len(batch_order)))]

        for g in opt.param_groups:
            g['lr'] = lr_at(step, HP)
        model.train()
        input_net.train()
        x, il, tg, tl = train_ds.make_batch(idx, augment=True,
                                            white_noise_sd=HP['white_noise_sd'],
                                            const_offset_sd=HP['const_offset_sd'], device=device)
        # bf16 混合精度（已记录偏差，官方为 fp32）：显存减半（fp32 会顶满 16GB 导致分配器颠簸），
        # 且 cuDNN RNN 的 bf16 核更快；log_softmax/ctc_loss 在 autocast 下自动以 fp32 执行。
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            logits = model(input_net(x, training=True), training=True)
            log_probs = F.log_softmax(logits.float(), dim=-1)  # (B,T',31)
            pred_loss = F.ctc_loss(log_probs.permute(1, 0, 2), tg.to(device), il.to(device), tl.to(device),
                                   blank=30, reduction='mean', zero_infinity=True)
        reg = model.l2_reg(HP['weight_reg']) + HP['weight_reg'] * (input_net.fc.weight ** 2).sum()
        loss = pred_loss + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, HP['grad_clip'])
        if torch.isfinite(grad_norm):
            opt.step()

        if step % 10 == 0 or step == n_batches - 1:
            el = time.time() - t0
            log(f'batch {step}: loss {float(loss):.2f} (pred {float(pred_loss):.2f}) gradNorm {float(grad_norm):.2f} '
                f'lr {lr_at(step, HP):.5f} elapsed {el / 60:.1f}min')

        if args.smoke == 0 and step % HP['batches_per_val'] == 0 and step > 0:
            cer = validate(model, input_net, val_ds, device)
            log(f'val @ {step}: CER {cer:.4f} (best {best_cer:.4f}, since_improved {batches_since_improved})')
            if cer < best_cer:
                best_cer = cer
                batches_since_improved = 0
                torch.save({'model': model.state_dict(), 'input_net': input_net.state_dict(),
                            'step': step, 'cer': cer}, out_dir / 'best.pt')
            else:
                batches_since_improved += HP['batches_per_val']
                if batches_since_improved >= args.patience:
                    log(f'early stop @ {step}（{args.patience} 批无改进）')
                    break

    if args.smoke > 0:
        per_batch = (time.time() - t0) / n_batches
        log(f'SMOKE_DONE: {n_batches} 批均耗时 {per_batch:.2f}s/批 -> 10000 批预计 {per_batch * 10000 / 3600:.2f} 小时')
    else:
        cer = validate(model, input_net, val_ds, device)
        log(f'FINAL val CER {cer:.4f} | best {best_cer:.4f} | total {(time.time() - t0) / 3600:.2f} h')
        with open(out_dir / 'final_report.json', 'w', encoding='utf-8') as f:
            json.dump({'final_cer': cer, 'best_cer': best_cer,
                       'n_batches_trained': step + 1, 'hours': (time.time() - t0) / 3600}, f, indent=2)
        log('TRAIN_DONE')


if __name__ == '__main__':
    main()
