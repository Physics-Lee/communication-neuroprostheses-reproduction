# -*- coding: utf-8 -*-
"""Step4 RNN 训练的 PyTorch 移植入口（官方 charSeqRNN.py train() 循环的等价实现）。

用法：
    python train.py --partition HeldOutBlocks [--synth] [--argsFile xxx.p] [--outputDir xxx]
数据依赖：
    - 真数据：data/20210512_willett/handwritingBCIData/ 下 Datasets/ + Step2_HMMLabels/ + trainTestPartitions_*
      （经官方 dataPreprocessing.prepareDataCubesForRNN 装载，本脚本只读引用官方模块，不改上游）
    - 合成数据：code/step3_make_synthetic.py 产出的 bat_*_{inputs,labels,errW}.npy（mmap 顺序流 +
      官方同款噪声；官方弱 shuffle(4) 近似顺序读，统计等价，已记录）。
忠实点：Adam(eps=0.1, betas 0.9/0.999)、lr 线性衰减、梯度全局范数裁剪 10、非有限梯度跳过更新、
损失 = CE + 5*起始信号 MSE（除以 timeSteps）、L2 = 官方权重集合、日选择 multinomial(dayProbability)、
合成(前) + 真数据(后) 拼 batch、验证批不加噪、统计与 .mat 文件名对齐官方。
"""
import argparse
import os
import pickle
import queue
import sys
import random
import threading
from datetime import datetime

import numpy as np
import scipy.io
import torch

REPO = r'.\20210512_willett'
OFFICIAL_REPO = os.path.join(REPO, r'open_source_code_for_reference\20210512_willett_handwritingBCI')
sys.path.insert(0, OFFICIAL_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataPreprocessing import prepareDataCubesForRNN  # noqa: E402  官方模块，只读引用
from model import CharSeqRNN, char_seq_loss, l2_cost, compute_frame_accuracy  # noqa: E402
from augmentation import extract_sentence_snippet, apply_train_noise  # noqa: E402

SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def build_default_args(partition, synth_batch_size):
    """官方 getDefaultRNNArgs + 数据包 args.p 的超参，路径改到本仓解压区。

    Parameters
    ----------
    partition : str
        Cross-validation partition name (``'HeldOutBlocks'`` or
        ``'HeldOutTrials'``); selects Step2 label and partition files and
        the run-directory suffix.
    synth_batch_size : int
        Synthetic portion of each batch (0 disables synthetic data and
        gives a ``_real`` run directory suffix).

    Returns
    -------
    dict
        Full training-args dictionary mirroring the official args.p keys
        (model, augmentation, optimisation and per-session file paths)
        plus this port's ``randomWalkAxisMode`` key.
    """
    root = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
    args = {
        'mode': 'train', 'nUnits': 512, 'rnnBinSize': 2, 'smoothInputs': 1, 'skipLen': 5,
        'outputDelay': 50, 'directionality': 'unidirectional',
        'constantOffsetSD': 0.6, 'randomWalkSD': 0.02, 'whiteNoiseSD': 1.2,
        'l2scale': 1e-5, 'learnRateStart': 0.01, 'learnRateEnd': 0.0,
        'trainableInput': 1, 'trainableBackEnd': 1, 'seed': 313793,
        'nCheckToKeep': 1, 'batchesPerSave': 200, 'batchesPerVal': 50, 'batchesPerModelSave': 5000,
        'nBatchesToTrain': 100000, 'timeSteps': 1200, 'batchSize': 64,
        'synthBatchSize': synth_batch_size, 'inputScale': 1.0,
        'inferenceOutputFileName': 'None', 'inferenceInputLayer': 0,
        'outputDir': os.path.join(REPO, 'data', '20210512_willett', 'runs', partition + ('_synth' if synth_batch_size else '_real')),
        'randomWalkAxisMode': 'official',  # 本移植新增：'official'(照抄 cumsum 特征轴) / 'time'(直觉修正)
    }
    for x, sess in enumerate(SESSIONS):
        args['sessionName_' + str(x)] = sess
        args['sentencesFile_' + str(x)] = os.path.join(root, 'Datasets', sess, 'sentences.mat')
        args['singleLettersFile_' + str(x)] = os.path.join(root, 'Datasets', sess, 'singleLetters.mat')
        args['labelsFile_' + str(x)] = os.path.join(root, 'RNNTrainingSteps', 'Step2_HMMLabels', partition, sess + '_timeSeriesLabels.mat')
        args['cvPartitionFile_' + str(x)] = os.path.join(root, 'RNNTrainingSteps', 'trainTestPartitions_' + partition + '.mat')
        args['syntheticDatasetDir_' + str(x)] = os.path.join(root, 'RNNTrainingSteps', 'Step3_SyntheticSentences', partition, sess + '_syntheticSentences')
    args['dayToLayerMap'] = str(list(range(len(SESSIONS))))
    args['dayProbability'] = str([1.0 / len(SESSIONS)] * len(SESSIONS))
    return args


class RealDataBatcher:
    """真数据流：随机截句 + 训练噪声（验证时 add_noise=False），一次产一个 batch 的 numpy 数组。

    Parameters
    ----------
    neural : numpy.ndarray
        Per-sentence neural cubes for the split, shape ``(S, T, N)``.
    targets : numpy.ndarray
        Per-sentence targets, shape ``(S, T, C)``.
    err_w : numpy.ndarray
        Per-sentence error weights, shape ``(S, T)``.
    bins_per_trial : numpy.ndarray
        Per-sentence bin counts, shape ``(S,)``.
    args : dict
        Training args (timeSteps, directionality, noise SDs, seed,
        randomWalkAxisMode).
    add_noise : bool
        Whether to apply training noise to each snippet (False for
        validation batches).
    """

    def __init__(self, neural, targets, err_w, bins_per_trial, args, add_noise):
        """Store the sentence arrays and seed a private RNG.

        Parameters
        ----------
        neural : numpy.ndarray
            Per-sentence neural cubes, shape ``(S, T, N)``.
        targets : numpy.ndarray
            Per-sentence targets, shape ``(S, T, C)``.
        err_w : numpy.ndarray
            Per-sentence error weights, shape ``(S, T)``.
        bins_per_trial : numpy.ndarray
            Per-sentence bin counts, shape ``(S,)``.
        args : dict
            Training args used by ``next``.
        add_noise : bool
            Whether ``next`` applies training noise; the RNG seed is offset
            by 10,000,000 in the noise-free (validation) case so train and
            val streams never share draws.

        Returns
        -------
        None
        """
        self.neural, self.targets, self.err_w = neural, targets, err_w
        self.bins = bins_per_trial
        self.args, self.add_noise = args, add_noise
        self.rng = np.random.default_rng(args['seed'] + (0 if add_noise else 10_000_000))

    def next(self, batch_size):
        """Sample one training/validation batch of random sentence snippets.

        Draws ``batch_size`` sentence indices, extracts a random
        ``timeSteps``-long snippet from each (recomputing error weights per
        the official rule), optionally applies training noise, and stacks
        the results into torch tensors.

        Parameters
        ----------
        batch_size : int
            Number of sentence snippets in the batch.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            ``(inputs, targets, err_weights)`` float32 tensors of shapes
            ``(B, timeSteps, N)``, ``(B, timeSteps, C)`` and
            ``(B, timeSteps)``.
        """
        n_steps = self.args['timeSteps']
        idx = self.rng.integers(0, self.neural.shape[0], size=batch_size)
        inp, targ, ews = [], [], []
        for i in idx:
            a, b, c = extract_sentence_snippet(self.neural[i], self.targets[i], self.err_w[i],
                                               int(self.bins[i]), n_steps,
                                               self.args['directionality'], self.rng)
            if self.add_noise:
                a, b, c = apply_train_noise(a, b, c, None, self.args, self.rng,
                                            axis_mode=self.args.get('randomWalkAxisMode', 'official'))
            inp.append(a)
            targ.append(b)
            ews.append(c)
        return (torch.from_numpy(np.asarray(inp, dtype=np.float32)),
                torch.from_numpy(np.asarray(targ, dtype=np.float32)),
                torch.from_numpy(np.asarray(ews, dtype=np.float32)))


class PrefetchBatcher:
    """固定批大小的预取双缓冲（等价官方 tf.data 的 prefetch 语义）。

    背景：合成数据从 D 盘（机械盘）逐批流式 mmap 读 ~22MB，单线程读盘让 GPU 挨饿
    （实测任务管理器 GPU 3%、D 盘 100%）。后台线程提前产出下 1-2 批，与 GPU 计算重叠。
    rng 只被工作线程访问，消耗顺序与串行调用一致，不改变数据分布。

    Parameters
    ----------
    inner : object
        Wrapped batcher exposing ``next(batch_size)`` (RealDataBatcher or
        SynthDataBatcher).
    batch_size : int
        Fixed batch size requested from ``inner`` on every fill.
    depth : int, optional
        Queue capacity, i.e. how many batches may be produced ahead of
        consumption (default 2).
    """

    def __init__(self, inner, batch_size, depth=2):
        """Start the background producer thread.

        Parameters
        ----------
        inner : object
            Wrapped batcher exposing ``next(batch_size)``.
        batch_size : int
            Fixed batch size used for every prefetch fill.
        depth : int, optional
            Maximum number of prefetched batches (default 2).

        Returns
        -------
        None
        """
        self.inner, self.bs = inner, batch_size
        self._q = queue.Queue(maxsize=depth)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        """Producer thread body: endlessly fill the queue with batches.

        Blocks on the bounded queue when the consumer is slow, which caps
        memory at ``depth`` batches.

        Returns
        -------
        None
        """
        while True:
            self._q.put(self.inner.next(self.bs))

    def next(self, batch_size):
        """Consume the next prefetched batch.

        Parameters
        ----------
        batch_size : int
            Requested batch size; must equal the fixed size given at
            construction.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            The ``(inputs, targets, err_weights)`` batch produced by the
            background thread.

        Raises
        ------
        AssertionError
            If ``batch_size`` differs from the construction-time size.
        """
        assert batch_size == self.bs, 'PrefetchBatcher 只支持固定批大小'
        return self._q.get()


class SynthDataBatcher:
    """合成数据流（Step3 移植产物 bat_*_inputs/labels/errW.npy）。

    官方语义：当日全部 20 个 bat 文件顺序流式读取 + 弱 shuffle(buffer=4) + batch 24 + 无限 repeat，
    每句加与真数据同款噪声（whiteNoiseSD/constantOffsetSD/randomWalkSD，cumsum 特征轴）。
    本移植：mmap 顺序读（跳过弱 shuffle——官方 buffer=4 的洗牌近似于顺序，统计上等价），文件间循环。

    Parameters
    ----------
    synth_dir : str
        Directory holding the Step3 ``bat_*_inputs/labels/errW.npy``
        triplets for one session.
    args : dict
        Training args (timeSteps, noise SDs, seed, randomWalkAxisMode).

    Raises
    ------
    FileNotFoundError
        If ``synth_dir`` contains no ``bat_*_inputs.npy`` files.
    """

    def __init__(self, synth_dir, args):
        """Discover and sort the batch files, seed the noise RNG.

        Parameters
        ----------
        synth_dir : str
            Directory holding the Step3 ``bat_*_{inputs,labels,errW}.npy``
            triplets for one session.
        args : dict
            Training args; the RNG is seeded with ``args['seed'] +
            20_000_000`` so this stream never shares draws with the real
            data streams.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If ``synth_dir`` contains no ``bat_*_inputs.npy`` files.
        """
        import glob
        import re
        self.args = args
        files = glob.glob(os.path.join(synth_dir, 'bat_*_inputs.npy'))
        self.files = sorted(files, key=lambda p: int(re.search(r'bat_(\d+)_inputs', p).group(1)))
        if not self.files:
            raise FileNotFoundError(f'{synth_dir} 下没有 bat_*_inputs.npy（先跑 code/step3_make_synthetic.py）')
        self.rng = np.random.default_rng(args['seed'] + 20_000_000)
        self.file_idx = 0
        self.row_idx = 0
        self._mmaps = {}

    def _mmap(self, suffix):
        """Return the memoised mmap view of the current batch file's array.

        Parameters
        ----------
        suffix : str
            Which member of the triplet to load: ``'inputs'``, ``'labels'``
            or ``'errW'``.

        Returns
        -------
        numpy.ndarray
            Memory-mapped array for the current ``file_idx`` and suffix,
            cached in ``self._mmaps``.
        """
        key = (self.file_idx, suffix)
        if key not in self._mmaps:
            path = self.files[self.file_idx].replace('_inputs.npy', f'_{suffix}.npy')
            self._mmaps[key] = np.load(path, mmap_mode='r')
        return self._mmaps[key]

    def next(self, batch_size):
        """Produce the next noise-augmented batch from the mmap stream.

        Reads rows sequentially from the current batch file, wraps to the
        next file (cyclically) when exhausted, and applies the official
        training noise to every sentence as it is read.

        Parameters
        ----------
        batch_size : int
            Number of sentences in the batch; the stream may span file
            boundaries to fill it.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
            ``(inputs, targets, err_weights)`` float32 tensors of shapes
            ``(B, timeSteps, N)``, ``(B, timeSteps, C)`` and
            ``(B, timeSteps)``.
        """
        inp, targ, ews = [], [], []
        n_steps = self.args['timeSteps']
        while len(inp) < batch_size:
            rows_in = self._mmap('inputs')
            take = min(batch_size - len(inp), rows_in.shape[0] - self.row_idx)
            sl = slice(self.row_idx, self.row_idx + take)
            for i in range(sl.start, sl.stop):
                a = np.asarray(rows_in[i], dtype=np.float32)
                b = np.asarray(self._mmap('labels')[i], dtype=np.float32)
                c = np.asarray(self._mmap('errW')[i], dtype=np.float32)
                a, b, c = apply_train_noise(a, b, c, None, self.args, self.rng,
                                            axis_mode=self.args.get('randomWalkAxisMode', 'official'))
                inp.append(a)
                targ.append(b)
                ews.append(c)
            self.row_idx += take
            if self.row_idx >= rows_in.shape[0]:
                self.row_idx = 0
                self.file_idx = (self.file_idx + 1) % len(self.files)
        return (torch.from_numpy(np.asarray(inp, dtype=np.float32)),
                torch.from_numpy(np.asarray(targ, dtype=np.float32)),
                torch.from_numpy(np.asarray(ews, dtype=np.float32)))





def load_all_days(args):
    """Load and prepare data cubes for all 10 sessions via the official loader.

    Parameters
    ----------
    args : dict
        Training args providing per-session file paths
        (``sentencesFile_<d>``, ``singleLettersFile_<d>``, ``labelsFile_<d>``,
        ``cvPartitionFile_<d>``, ``sessionName_<d>``) plus ``rnnBinSize``
        and ``timeSteps``.

    Returns
    -------
    tuple of (list of dict, int, int)
        ``(cubes, n_inputs, n_outputs)``: one dict per session with keys
        ``'neural'``, ``'targets'``, ``'err_w'``, ``'bins'`` and ``'cv'``
        (train/test indices from the official loader), followed by the
        input feature count and target channel count.
    """
    cubes = []
    for d in range(len(SESSIONS)):
        neural, targets, err_w, bins, cv = prepareDataCubesForRNN(
            args['sentencesFile_' + str(d)], args['singleLettersFile_' + str(d)],
            args['labelsFile_' + str(d)], args['cvPartitionFile_' + str(d)],
            args['sessionName_' + str(d)], args['rnnBinSize'], args['timeSteps'], True)
        cubes.append({'neural': neural, 'targets': targets, 'err_w': err_w, 'bins': bins, 'cv': cv})
    n_inputs = cubes[0]['neural'].shape[2]
    n_outputs = cubes[0]['targets'].shape[2]
    return cubes, n_inputs, n_outputs


def main():
    """Train the CharSeqRNN following the official Step4 training loop.

    Builds/loads the training args, loads all 10 session cubes, constructs
    the model and the real/synthetic/val batchers (with prefetching), then
    runs the official-equivalent optimisation loop: multinomial day
    selection, synthetic-then-real batch concatenation, Adam(eps=0.1) with
    linearly decaying LR, global grad-norm clip 10, update skipping on
    non-finite gradients, periodic validation snapshots/stats and rotating
    checkpoints. Saves ``model_step{n}.pt`` plus stats .mat files to the
    output directory.

    Parameters
    ----------
    None
        Options come from the command line: ``--partition``,
        ``--synth`` (enable the synthetic stream), ``--argsFile``
        (official args pickle, defaults filled in), ``--nBatches``
        (override batch count) and ``--outputDir``.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', choices=['HeldOutBlocks', 'HeldOutTrials'], default='HeldOutBlocks')
    ap.add_argument('--synth', action='store_true', help='使用合成数据（依赖 Step3 移植，未完成）')
    ap.add_argument('--argsFile', default=None, help='可选：官方/自定义 args pickle（优先于默认）')
    ap.add_argument('--nBatches', type=int, default=None, help='覆盖训练批数（调试用）')
    ap.add_argument('--outputDir', default=None)
    cli = vars(ap.parse_args())

    if cli['argsFile']:
        with open(cli['argsFile'], 'rb') as f:
            args = pickle.load(f)
        for k, v in build_default_args(cli['partition'], 0).items():  # 补齐本移植新增键
            args.setdefault(k, v)
    else:
        args = build_default_args(cli['partition'], 24 if cli['synth'] else 0)
    if cli['nBatches']:
        args['nBatchesToTrain'] = cli['nBatches']
    if cli['outputDir']:
        args['outputDir'] = cli['outputDir']

    os.makedirs(args['outputDir'], exist_ok=True)
    seed = args['seed'] if args['seed'] != -1 else datetime.now().microsecond
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    rng_global = np.random.default_rng(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    print('加载数据（10 sessions）...', flush=True)
    cubes, n_inputs, n_outputs = load_all_days(args)
    day_to_layer = eval(args['dayToLayerMap'])
    day_prob = eval(args['dayProbability'])

    model = CharSeqRNN(n_inputs=n_inputs, n_outputs=n_outputs, n_units=args['nUnits'],
                       directionality=args['directionality'],
                       n_input_layers=len(set(day_to_layer)), skip_len=args['skipLen'],
                       rnn_bin_size=args['rnnBinSize'], smooth_inputs=bool(args['smoothInputs']),
                       trainable_input=bool(args['trainableInput']),
                       trainable_backend=bool(args['trainableBackEnd'])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'nInputs={n_inputs} nOutputs={n_outputs} 参数量={n_params/1e6:.2f}M device={device}', flush=True)

    train_batchers = [PrefetchBatcher(
        RealDataBatcher(c['neural'][c['cv']['trainIdx']], c['targets'][c['cv']['trainIdx']],
                        c['err_w'][c['cv']['trainIdx']], c['bins'][c['cv']['trainIdx']],
                        args, add_noise=True),
        args['batchSize'] - (args['synthBatchSize'] if args['synthBatchSize'] > 0 else 0))
        for c in cubes]
    if args['synthBatchSize'] > 0:
        synth_batchers = [PrefetchBatcher(SynthDataBatcher(args['syntheticDatasetDir_' + str(d)], args),
                                          args['synthBatchSize'])
                          for d in range(len(SESSIONS))]
        print(f'合成数据流开启：synthBatchSize={args["synthBatchSize"]} '
              f'+ 真数据 {args["batchSize"] - args["synthBatchSize"]}', flush=True)
    else:
        synth_batchers = [None] * len(SESSIONS)
    val_days = [d for d, c in enumerate(cubes) if len(c['cv']['testIdx']) > 0]
    val_batchers = {d: RealDataBatcher(cubes[d]['neural'][cubes[d]['cv']['testIdx']],
                                       cubes[d]['targets'][cubes[d]['cv']['testIdx']],
                                       cubes[d]['err_w'][cubes[d]['cv']['testIdx']],
                                       cubes[d]['bins'][cubes[d]['cv']['testIdx']],
                                       args, add_noise=False) for d in val_days}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, betas=(0.9, 0.999), eps=1e-01)  # 官方 epsilon=0.1
    ckpts = []

    n_train = args['nBatchesToTrain']
    batch_train_stats = np.zeros([n_train, 6])
    n_val_slots = int(np.ceil(n_train / args['batchesPerVal']))
    batch_val_stats = np.zeros([n_val_slots, 4])

    for i in range(n_train):
        t0 = datetime.now()
        lr = args['learnRateStart'] * (1 - i / n_train) + args['learnRateEnd'] * (i / n_train)
        for g in opt.param_groups:
            g['lr'] = lr

        day = int(np.argwhere(rng_global.multinomial(1, day_prob))[0][0])
        synth_bs = args['synthBatchSize']
        if synth_bs > 0:  # 官方 combineSynthAndReal：合成在前、真数据在后拼接
            s_inp, s_targ, s_ews = synth_batchers[day].next(synth_bs)
            r_inp, r_targ, r_ews = train_batchers[day].next(args['batchSize'] - synth_bs)
            inp = torch.cat([s_inp, r_inp])
            targ = torch.cat([s_targ, r_targ])
            ews = torch.cat([s_ews, r_ews])
        else:
            inp, targ, ews = train_batchers[day].next(args['batchSize'])
        inp, targ, ews = inp.to(device), targ.to(device), ews.to(device)

        logits = model(inp, input_layer_idx=day_to_layer[day])
        err, ce_t, tr_t = char_seq_loss(logits, targ, ews, args['timeSteps'], args['outputDelay'])
        cost = err + l2_cost(model) * args['l2scale'] if args['l2scale'] > 0 else err
        opt.zero_grad(set_to_none=True)
        cost.backward()

        grads_finite = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        grad_norm = float('nan')
        if grads_finite:  # 官方：非有限梯度整批跳过更新
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 10.0).item()
            opt.step()

        train_acc = compute_frame_accuracy(logits.detach().cpu().numpy(), targ.cpu().numpy(),
                                           ews.cpu().numpy(), args['outputDelay'])
        dt = (datetime.now() - t0).total_seconds()
        batch_train_stats[i, :] = [i, float(err.detach()), grad_norm, train_acc, dt, day]

        if i % args['batchesPerVal'] == 0:
            vd = val_days[int(rng_global.integers(len(val_days)))]
            v_inp, v_targ, v_ews = val_batchers[vd].next(args['batchSize'])
            with torch.no_grad():
                v_logits = model(v_inp.to(device), input_layer_idx=day_to_layer[vd])
                v_err, _, _ = char_seq_loss(v_logits, v_targ.to(device), v_ews.to(device),
                                            args['timeSteps'], args['outputDelay'])
                v_acc = compute_frame_accuracy(v_logits.cpu().numpy(), v_targ.numpy(),
                                               v_ews.numpy(), args['outputDelay'])
            batch_val_stats[i // args['batchesPerVal'], :] = [i, float(v_err), grad_norm, v_acc]
            print(f'批 {i}/{n_train} valErr={float(v_err):.4f} trainErr={float(err):.4f} '
                  f'valAcc={v_acc:.4f} trainAcc={train_acc:.4f} lr={lr:.5f} {dt:.2f}s', flush=True)
            snapshot = {'inputs': v_inp[0].numpy(), 'charProbOutput': v_logits[0, :, :-1].cpu().numpy(),
                        'charStartOutput': torch.sigmoid(v_logits[0, args['outputDelay']:, -1]).cpu().numpy(),
                        'charProbTarget': v_targ[0, :, :-1].numpy(), 'charStartTarget': v_targ[0, :, -1].numpy(),
                        'errorWeight': v_ews[0].numpy()}
            scipy.io.savemat(os.path.join(args['outputDir'], 'outputSnapshot'), snapshot)

        if i >= args['batchesPerSave'] - 1 and i % args['batchesPerSave'] == 0:
            scipy.io.savemat(os.path.join(args['outputDir'], 'intermediateOutput'),
                             {'batchTrainStats': batch_train_stats, 'batchValStats': batch_val_stats})
        if i % args['batchesPerModelSave'] == 0:
            ckpt = os.path.join(args['outputDir'], f'model_step{i}.pt')
            torch.save({'step': i, 'model': model.state_dict(), 'args': args}, ckpt)
            ckpts.append(ckpt)
            while len(ckpts) > args['nCheckToKeep']:
                os.remove(ckpts.pop(0))

    scipy.io.savemat(os.path.join(args['outputDir'], 'finalOutput'),
                     {'batchTrainStats': batch_train_stats, 'batchValStats': batch_val_stats})
    torch.save({'step': n_train, 'model': model.state_dict(), 'args': args},
               os.path.join(args['outputDir'], f'model_step{n_train}.pt'))
    print('训练完成，产物在', args['outputDir'])


if __name__ == '__main__':
    main()
