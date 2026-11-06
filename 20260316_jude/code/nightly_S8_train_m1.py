# S8 夜间流水线：M1 训练 + 推理 + greedy raw CER（2026-10-09）
# 严格复刻官方 notebook 25 T18_train_model.ipynb CELL 3/4：
#   decoder = brainToText_decoder(OmegaConf.load(yaml)); train(); 新实例 inference()
# raw CER：对 inference 输出 logits 做 greedy CTC（blank=30），与真句做字符级编辑距离（utils.lmutils.calc_wer）。
import sys
import json
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REF = Path(r'.\20260316_jude\open_source_code_for_reference\20260316_jude_typingBCI')
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF / 'utils'))

from utils.brainToText_trainDecoder import brainToText_decoder  # noqa: E402

# calc_wer 原样复制自 utils/lmutils.py（该模块顶层 import lm_decoder，仅 WSL 编译成功后可用，此处用不到它）
def calc_wer(r, h):
    import numpy
    d = numpy.zeros((len(r) + 1) * (len(h) + 1), dtype=numpy.uint8)
    d = d.reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]

yaml_path = sys.argv[1]
marker = sys.argv[2] if len(sys.argv) > 2 else 'S8_DONE'
args = OmegaConf.load(yaml_path)
out_dir = Path(args.outputDir)

CHAR_DEF = ['>', ',', '?', '.',
            'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']  # 30 类；模型第 31 类（index 30）= blank


def greedy_decode(logits):
    seq = np.argmax(logits, axis=-1)
    chars, prev = [], None
    for s in seq:
        if s != prev and s != 30:
            chars.append(CHAR_DEF[s])
        prev = s
    return ''.join(chars)


def extract_true(trans_row):
    end_idx = np.argwhere(trans_row == 0)[0, 0]
    return ''.join(chr(int(c)) for c in trans_row[:end_idx])


def strip_punct(s):
    for p in ['.', '?', ',']:
        s = s.replace(p, '')
    return s.strip()


t0 = time.time()
decoder = brainToText_decoder(args)
inf_out, stats = decoder.train()
train_time_s = time.time() - t0
print('train_time_s:', round(train_time_s, 1), flush=True)

decoder2 = brainToText_decoder(args)  # 官方 CELL 4 模式：新实例从 checkpoint 加载后推理
out, out_by_day = decoder2.inference()
print('inference keys:', list(out.keys()), flush=True)

results = []
for i in range(len(out['logits'])):
    logit_len = out['logitLengths'][i]
    dec = greedy_decode(out['logits'][i][:logit_len])
    true = extract_true(out['transcriptions'][i])
    n_err = calc_wer(list(true), list(dec))
    results.append({'true': true, 'decoded': dec, 'char_err': n_err, 'char_total': len(true)})

n_char_err = sum(r['char_err'] for r in results)
n_char = sum(r['char_total'] for r in results)
cer_raw = n_char_err / n_char
# 去标点口径（官方 LM 评估的常用后处理；两口径都记）
errs_sp, tot_sp = 0, 0
for r in results:
    d, t = strip_punct(r['decoded']), strip_punct(r['true'])
    errs_sp += calc_wer(list(t), list(d))
    tot_sp += len(t)
cer_raw_stripped = errs_sp / tot_sp

np.save(out_dir / 'inference_logits.npy', np.asarray(out['logits'], dtype=object), allow_pickle=True)
np.save(out_dir / 'inference_transcriptions.npy', np.asarray(out['transcriptions']), allow_pickle=True)
report = {
    'yaml': str(yaml_path),
    'n_trials': len(results),
    'nBatchesToTrain': int(args.nBatchesToTrain),
    'train_time_s': round(train_time_s, 1),
    'cer_raw_greedy': round(float(cer_raw), 4),
    'cer_raw_greedy_stripped': round(float(cer_raw_stripped), 4),
    'per_trial': results,
}
with open(out_dir / 'm1_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in report.items() if k != 'per_trial'}, ensure_ascii=False))
print(marker)
