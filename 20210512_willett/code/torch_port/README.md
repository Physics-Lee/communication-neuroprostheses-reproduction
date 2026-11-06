# torch_port — 官方 TF1 RNN 的 PyTorch 移植

官方 `charSeqRNN.py`（TF1.15 + `tf.contrib.cudnn_rnn.CudnnGRU`，本机不可装）的忠实 PyTorch 重写。
对齐口径：**指标级**（两划分 raw CER 与官方同数量级），非逐权重复刻（官方 checkpoint 为 TF1 格式，本就不加载）。

## 文件与官方映射

| 本目录 | 官方对应 | 状态 |
|---|---|---|
| `model.py` `CharSeqRNN` | `charSeqRNN.py` RNN 图（平滑/输入层/双层 GRU/读出/上采样） | ✅ 移植完成，sanity 9/9 |
| `model.py` `char_seq_loss`/`l2_cost` | 官方损失（CE + 5×起始 MSE，除以 timeSteps）与 L2 权重集合 | ✅ |
| `model.py` `compute_frame_accuracy` | `charSeqRNN.py computeFrameAccuracy` | ✅ |
| `augmentation.py` | `extractSentenceSnippet`/`addMeanNoise`/`addWhiteNoise`/`parseDataset` 噪声 | ✅（cumsum 轴向可配置） |
| `train.py` | `charSeqRNN.train()` 训练循环 + `dataPreprocessing.prepareDataCubesForRNN`（直接 import 官方模块） | ✅ 真数据链路；合成流待 Step3 移植 |
| `sanity_check.py` | — | ✅ 9/9 通过（2026-10-10） |
| Step5 推理脚本 | `charSeqRNN.inference()` + `rnnEval.py` | ⬜ 待写 |

## 结构（与官方逐项对应）

平滑（SD=4/rnnBinSize=2，截断>0.01，TF SAME）→ **每 session 专属线性输入层**（W=恒等、b=0 初始化，10 层）→
GRU1（512，全 1200 步）→ GRU2（512，每 `skipLen=5` 步取 1）→ 读出 → `repeat_interleave(5)` 上采样。
输出 32 列 = 31 字符 logit + 1 字符起始 logit（数据已核实：charProbTarget 31 列 + charStartTarget 1 列，nInputs=192）。
官方超参见 `data/20210512_willett/handwritingBCIData/RNNTrainingSteps/Step4_RNNTraining/*/args.p`。

## 忠实点（照抄官方）

- Adam **eps=0.1**、betas (0.9,0.999)；lr 线性衰减 0.01→0；梯度全局范数裁剪 10；**非有限梯度整批跳过**
- 损失按 timeSteps=1200 归一（不除有效帧数）；起始信号权重系数 5
- L2 集合 = readout_W + 各输入层 W + GRU 全部参数（含 bias，对应 cuDNN 融合权重 blob）；**不含** readout_b/输入 b/start_state
- 单批单日（switch_case 语义：只更新当日输入层）；日选择 multinomial(dayProbability)；合成(24)+真(40)拼批、验证批不加噪
- 两层 GRU 共享同一个可训练零初始化 start_state

## 已知偏差/怪癖（记录在案）

1. **cumsum 轴向**：官方随机游走噪声 `cumsum(axis=1)` 沿**特征轴**（疑似上游笔误但即论文结果出处）。默认 `axis_mode='official'` 照抄；`'time'` 供消融。
2. **GRU 初始化**：权重用 xavier_uniform 近似 cuDNN 默认 glorot（bias 严格置零）；分布级差异，10 万步训练后影响有限。
3. GRU 数学：PyTorch `nn.GRU` 与 cuDNN GRU 同约定（reset 在隐层矩阵乘之后，即 Keras `reset_after=True`），CUDA 上即 cuDNN 内核。
4. HeldOutTrials 官方 args 的 `timeSeriesFile_*` 误指向 HeldOutBlocks 目录（官方笔误，`labelsFile_*` 才是实际使用的标签源，不受影响）。

## 性能（RTX 5060 Ti，torch 2.11.0+cu128）

batch 64×1200×192 前向+反向+Adam 步 ≈ **974 ms/it** → 10 万批 ≈ **27 h**（两划分约 2.3 天 GPU 时间）。
要点：GRU2 输入需 `.contiguous()`（步进切片否则掉出 cuDNN 快速路径，慢 50%）；`cudnn.benchmark=True`。

## 用法

```bash
PY=/c/Users/user/.conda/envs/reproduce_20210512_willett/python.exe
$PY code/torch_port/sanity_check.py                    # 自检（无数据依赖）
$PY code/torch_port/train.py --partition HeldOutBlocks --nBatches 500   # 真数据短跑冒烟（需先解压全部 Step2 标签与数据）
```

## 下一步

1. Step3 合成数据移植（`makeSyntheticSentences.py` 算法照搬，输出改 npz 分片；~100GB 磁盘风险见计划 R2/D3）
2. 真数据全量训练（先 HeldOutTrials，synth 可选）+ Step5 推理脚本（含 `rnnOutputToKaldiMatrices` 接口）
3. 指标对齐官方 README 表
