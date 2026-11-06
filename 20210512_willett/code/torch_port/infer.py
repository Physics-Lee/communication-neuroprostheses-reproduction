# -*- coding: utf-8 -*-
"""Step5 RNN 推理的 PyTorch 移植（官方 Step5_rnnInference.ipynb + charSeqRNN.inference()）。

流程：
1. 载入 train.py 存的 checkpoint（model_step{N}.pt）
2. 逐 session：prepareDataCubesForRNN(isTraining=False, timeSteps=7500) → 整段前向（无噪声）
   → logits 存 <outDir>/<session>_inferenceOutputs.mat['outputs']
3. 官方 rnnEval.evaluateRNNOutput（直接 import 官方模块）算 errCounts/decSentences
   → 存 <outDir>/<session>_errCounts.mat（口径与包内 Step5 errCounts 相同，可无缝进 M1a 汇总器）

用法：
    python infer.py --partition HeldOutTrials --runDir <runs 目录> [--step 100000] [--outDir <输出目录>]
"""
import argparse
import os
import sys

import numpy as np
import scipy.io
import torch

REPO = r'.\20210512_willett'
OFFICIAL_REPO = os.path.join(REPO, r'open_source_code_for_reference\20210512_willett_handwritingBCI')
sys.path.insert(0, OFFICIAL_REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataPreprocessing import prepareDataCubesForRNN  # noqa: E402  官方模块，只读引用
from characterDefinitions import getHandwritingCharacterDefinitions  # noqa: E402
from rnnEval import evaluateRNNOutput  # noqa: E402  官方模块，只读引用
from model import CharSeqRNN  # noqa: E402

ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partition', choices=['HeldOutBlocks', 'HeldOutTrials'], required=True)
    ap.add_argument('--runDir', required=True, help='train.py 的 outputDir（含 model_step*.pt）')
    ap.add_argument('--step', type=int, default=None, help='checkpoint 步数（默认取目录里最大者）')
    ap.add_argument('--outDir', default=None, help='推理输出目录（默认 <runDir>/step5_inference）')
    ap.add_argument('--batchSize', type=int, default=4)
    ap.add_argument('--timeSteps', type=int, default=7500)
    cli = vars(ap.parse_args())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    if cli['step'] is None:
        cands = [int(f[len('model_step'):-3]) for f in os.listdir(cli['runDir'])
                 if f.startswith('model_step') and f.endswith('.pt')]
        cli['step'] = max(cands)
    ckpt_path = os.path.join(cli['runDir'], f'model_step{cli["step"]}.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    targs = ckpt['args']

    out_dir = cli['outDir'] or os.path.join(cli['runDir'], 'step5_inference')
    os.makedirs(out_dir, exist_ok=True)

    model = CharSeqRNN(n_inputs=192, n_outputs=32, n_units=targs['nUnits'],
                       directionality=targs['directionality'], n_input_layers=10,
                       skip_len=targs['skipLen'], rnn_bin_size=targs['rnnBinSize'],
                       smooth_inputs=bool(targs['smoothInputs']),
                       trainable_input=bool(targs['trainableInput']),
                       trainable_backend=bool(targs['trainableBackEnd'])).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    char_def = getHandwritingCharacterDefinitions()
    part = cli['partition']

    for day, sess in enumerate(SESSIONS):
        neural, targets, err_w, bins, cv = prepareDataCubesForRNN(
            os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'),
            os.path.join(ROOT, 'Datasets', sess, 'singleLetters.mat'),
            os.path.join(ROOT, 'RNNTrainingSteps', 'Step2_HMMLabels', part, sess + '_timeSeriesLabels.mat'),
            os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + part + '.mat'),
            sess, targs['rnnBinSize'], cli['timeSteps'], False)
        n_in = neural.shape[2]
        assert n_in == 192, f'nInputs={n_in}'

        outs = []
        with torch.no_grad():
            for i in range(0, neural.shape[0], cli['batchSize']):
                inp = torch.from_numpy(np.asarray(neural[i:i + cli['batchSize']], dtype=np.float32)).to(device)
                logits = model(inp, input_layer_idx=day)
                outs.append(logits.cpu().numpy())
        outputs = np.concatenate(outs, axis=0)[:neural.shape[0]].astype(np.float32)

        scipy.io.savemat(os.path.join(out_dir, sess + '_inferenceOutputs.mat'), {'outputs': outputs})

        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'))
        err_counts, dec_sentences = evaluateRNNOutput(
            outputs, sentence_dat['numTimeBinsPerSentence'] / 2 + 50,
            sentence_dat['sentencePrompt'], char_def,
            charStartThresh=0.3, charStartDelay=15)
        scipy.io.savemat(os.path.join(out_dir, sess + '_errCounts.mat'),
                         dict(decSentences=dec_sentences,
                              trueSentences=sentence_dat['sentencePrompt'], **err_counts))

        cer = 100 * np.sum(err_counts['charErrors']) / np.sum(err_counts['charCounts'])
        wer = 100 * np.sum(err_counts['wordErrors']) / np.sum(err_counts['wordCounts'])
        print(f'{sess}: 全集 raw CER={cer:.2f}% WER={wer:.2f}% '
              f'(val 子集指标用 step5_eval_from_errcounts.py 汇总)', flush=True)

    print('推理完成 ->', out_dir)


if __name__ == '__main__':
    main()
