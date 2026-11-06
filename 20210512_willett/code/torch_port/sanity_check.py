# -*- coding: utf-8 -*-
"""torch_port 移植的自检脚本：不依赖真实数据，用合成张量验证模型/损失/增强/GPU 链路。
运行：/c/Users/user/.conda/envs/reproduce_20210512_willett/python.exe sanity_check.py
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CharSeqRNN, char_seq_loss, l2_cost, build_gauss_kernel  # noqa: E402
from augmentation import extract_sentence_snippet, add_mean_noise  # noqa: E402

PASS = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    PASS.append(cond)
    print(f'[{status}] {name} {detail}')


# 1) 高斯核
k = build_gauss_kernel(2)
check('高斯核长度为奇数且对称', len(k) % 2 == 1 and np.allclose(k, k[::-1], atol=1e-7),
      f'len={len(k)}, sum={k.sum():.6f}')

# 2) CPU 前向（单向，官方默认维度 192 输入 / 32 输出 / 10 输入层）
torch.manual_seed(313793)
model = CharSeqRNN(n_inputs=192, n_outputs=32, n_units=512, n_input_layers=10)
x = torch.randn(2, 1200, 192)
y = model(x, input_layer_idx=3)
check('CPU 前向形状 (2,1200,32) 且有限', y.shape == (2, 1200, 32) and torch.isfinite(y).all().item())

# 3) 损失 + 反向：梯度覆盖所有可训练参数
targ = torch.zeros(2, 1200, 32)
targ[:, :, 0] = 1.0                                   # 假装全类 0
targ[:, ::100, -1] = 1.0                              # 周期性起始信号
ews = torch.ones(2, 1200)
total, ce_t, tr_t = char_seq_loss(y, targ, ews)
total.backward()
# 单批只走被选中的输入层（官方 switch_case 语义），其余输入层无梯度属正常
expected_grad = ['input_W.3', 'input_b.3', 'readout_W', 'readout_b', 'start_state']
missing = [n for n, p in model.named_parameters()
           if (n in expected_grad or n.startswith(('gru1.', 'gru2.'))) and p.requires_grad and p.grad is None]
check('损失有限且反向完成（含被选输入层与全部 GRU）', torch.isfinite(total).item() and not missing,
      f'total={total.item():.4f} (ce={ce_t.item():.4f}, trans={tr_t.item():.4f}), 缺梯度={missing}')
model.zero_grad()

# 4) 参数量分解（对齐官方结构预期）
gru1_p = sum(p.numel() for p in model.gru1.parameters())
gru2_p = sum(p.numel() for p in model.gru2.parameters())
inp_p = sum(p.numel() for p in model.input_W) + sum(p.numel() for p in model.input_b)
ro_p = model.readout_W.numel() + model.readout_b.numel()
n_all = sum(p.numel() for p in model.parameters())
check('参数量分解', gru1_p == 3 * 512 * (192 + 512) + 2 * 3 * 512 and gru2_p == 3 * 512 * (512 + 512) + 2 * 3 * 512,
      f'GRU1={gru1_p/1e6:.2f}M GRU2={gru2_p/1e6:.2f}M 输入层={inp_p/1e6:.3f}M 读出={ro_p/1e6:.3f}M 共={n_all/1e6:.2f}M')

# 5) 双向模式前向
bi = CharSeqRNN(n_inputs=192, n_outputs=32, n_units=512, directionality='bidirectional', n_input_layers=1)
ybi = bi(x, input_layer_idx=0)
check('双向前向形状', ybi.shape == (2, 1200, 32) and torch.isfinite(ybi).all().item())

# 6) 增强：截句形状与 errWeight 语义（首字符前清零）
rng = np.random.default_rng(0)
T, N, C = 4000, 192, 32
inp = rng.normal(size=(T, N)).astype(np.float32)
tg = np.zeros((T, C), dtype=np.float32)
tg[:, 0] = 1.0
tg[500, -1] = 1.0
tg[1500, -1] = 1.0
ew = np.ones(T, dtype=np.float32)
a, b, c = extract_sentence_snippet(inp, tg, ew, 1500, 1200, 'unidirectional', rng)
first_char = int(np.argwhere(b[1:, -1] - b[:-1, -1] >= 0.1)[0, 0])
check('截句形状 1200', a.shape == (1200, N) and b.shape == (1200, C) and c.shape == (1200,))
check('errWeight 首字符前为 0', c[:first_char].sum() == 0 and c[first_char:].sum() > 0,
      f'firstChar={first_char}')
z = add_mean_noise(np.zeros((10, 4)), 0.6, 0.02, rng, axis_mode='official')
check('随机游走噪声(official 轴向)可用', z.shape == (10, 4) and np.isfinite(z).all())

# 7) GPU 全链路：batch 64 前向+反向+Adam 步，官方超参 (eps=0.1)
if torch.cuda.is_available():
    dev = 'cuda'
    m = CharSeqRNN(n_inputs=192, n_outputs=32, n_units=512, n_input_layers=10).to(dev)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, betas=(0.9, 0.999), eps=1e-01)
    xx = torch.randn(64, 1200, 192, device=dev)
    tt = torch.zeros(64, 1200, 32, device=dev)
    tt[:, :, 0] = 1
    bb = torch.ones(64, 1200, device=dev)
    for _ in range(3):  # 预热
        loss, _, _ = char_seq_loss(m(xx), tt, bb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    n_it = 20
    for _ in range(n_it):
        loss, _, _ = char_seq_loss(m(xx), tt, bb)
        cost = loss + l2_cost(m) * 1e-5
        opt.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(ps, 10.0)
        opt.step()
    torch.cuda.synchronize()
    sec_it = (time.time() - t0) / n_it
    check('GPU 训练步可跑', True,
          f'{sec_it*1000:.0f} ms/it -> 100k 批约 {sec_it*100000/3600:.1f} h（不含数据增强采样开销）, loss={loss.item():.4f}')
else:
    print('[SKIP] 无 CUDA')

print(f"\n结果：{sum(PASS)}/{len(PASS)} 通过" + ('' if all(PASS) else '，存在失败项！'))
