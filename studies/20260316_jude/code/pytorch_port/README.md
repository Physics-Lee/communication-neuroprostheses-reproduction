# pytorch_port/ — 官方 TF 2.14 训练管线的 PyTorch 移植（2026-10-10）

## 为什么有这个目录

官方训练是 TF 2.14：Windows 无 GPU 版；捆绑的 CUDA 12.2 不支持本机 RTX 5060 Ti（Blackwell）。
CPU 实测 110–215 s/批 → 官方 10000 批 ≈ 13+ 天，不可行。本移植用机器上已验证可用的
torch 2.7.1+cu128（借 levy 仓 venv）在同一块卡上训练，解开算力死结。

## 移植范围与对照（官方 = utils/brainToText_trainDecoder.py + utils/models.py）

| 组件 | 官方（TF） | 本移植 | 保真度 |
|---|---|---|---|
| 数据 | tfrecord（已 z-score） | **同一 tfrecord 一次性导出 npz**（export_tfrecord_to_npz.py，TF 环境跑） | 数据零改动 |
| 归一化层 | Keras Normalization，**从未 adapt → 恒等**（归一化在上游已做） | 省略 | 等价 |
| 输入网/会话 | Dense 768→768，softsign，**单位阵初始化**，Dropout 0.2，L2 1e-5 | InputNet 同构同初始化 | 等价 |
| patch 化 | extract_patches(size 15, stride 6, VALID) | tensor.unfold 同参 | 等价（patch 内元素排列序不同——从头训练无影响，不可载 TF 权重） |
| GRU | 5×512，dropout 0.6（含第 1 层输入），初始状态=可学习(glorot,仅层0)+其余零 | nn.GRU + 手动输入 dropout + 同构 init_states | 等价（见 train.py 注释） |
| 输出头 | Dense(31) logits | nn.Linear(512,31) | 等价 |
| 增广 | 白噪 1.5 + 常数偏移 0.22（每 trial 每通道），然后高斯平滑 SD=2（SAME） | dataset.py 同参同顺序 | 等价 |
| 损失 | CTC（blank=31 类中最后一个）+ Keras L2（=1e-5×Σw²，GRU 输入核+输入网核） | F.ctc_loss(blank=30, zero_infinity) + 手工同款 L2 | 等价（zero_infinity 为容错差异） |
| 优化器 | Adam(eps=**0.1**!)，lr 0.02→0 线性衰减（10000 步），梯度全局范数裁剪 10，非有限跳过 | 同 | 等价 |
| 验证 | 每 50 批 greedy CTC + 编辑距离 CER，存最优，早停 1000 | 同 | 等价 |

已知偏差（结果解读时计入）：①patch 内元素排列序与 TF 不同（只影响能否互载权重，不影响从零训练等价性）；
②shuffle 机制为"每 epoch 打乱循环"，非官方 1024 buffer 流式 shuffle（采样分布差异可忽略）；③框架数值细节
（cuDNN GRU 实现等）决定本移植与 TF 逐批 loss 不逐点一致，只追求同量级收敛。

## 怎么跑

```bash
# 1) 导出 npz（用 reproduce_20260316_jude 环境，一次性）
/c/Users/user\.conda\envs\reproduce_20260316_jude\python.exe export_tfrecord_to_npz.py

# 2) 训练（用 levy 仓的 cu128 torch；--smoke 30 为快速冒烟）
.\20260910_reproduce_20260629_levy\.venv312\Scripts\python.exe train.py --batches 10000
.\20260910_reproduce_20260629_levy\.venv312\Scripts\python.exe train.py --smoke 30
```

产物：`result/runs/m1_t18_20241204_torch/`（best.pt、training_log、最终 CER 报告）。
