# S8a 报告修复 v2：重跑 inference，按 logitLengths 截断后保存并正确计算 raw CER（2026-10-10 07:05）
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import sys
REF = Path(r'.\20260316_jude\open_source_code_for_reference\20260316_jude_typingBCI')
sys.path.insert(0, str(REF))
from utils.brainToText_trainDecoder import brainToText_decoder  # noqa: E402

OUT_DIR = Path(r'.\20260316_jude\result\runs\m1_t18_20241204_smoke')
args = OmegaConf.load(r'.\20260316_jude\code\m1_t18_20241204_args_smoke.yaml')

decoder = brainToText_decoder(args)
out, out_by_day = decoder.inference()
print('n inference:', len(out['logits']), '| decoder 自带 editDistances 均值:', float(np.mean(out['editDistances'])) if len(out['editDistances']) else 'n/a')

CHAR_DEF = ['>', ',', '?', '.',
            'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


def calc_wer(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.int32).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return int(d[len(r)][len(h)])


def greedy_decode(logits):
    seq = np.argmax(logits, axis=-1)
    chars, prev = [], None
    for s in seq:
        s = int(s)
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


# 保存截断后的 logits（供 S9a 用）
trunc = [np.asarray(out['logits'][i][:int(out['logitLengths'][i])], dtype=np.float32) for i in range(len(out['logits']))]
np.save(OUT_DIR / 'inference_logits_trunc.npy', np.array(trunc, dtype=object), allow_pickle=True)

results = []
for i in range(len(trunc)):
    dec = greedy_decode(trunc[i])
    true = extract_true(out['transcriptions'][i])
    n_err = calc_wer(list(true), list(dec))
    results.append({'true': true, 'decoded': dec, 'char_err': n_err, 'char_total': len(true)})

cer_raw = sum(r['char_err'] for r in results) / max(1, sum(r['char_total'] for r in results))
errs_sp, tot_sp = 0, 0
for r in results:
    d, t = strip_punct(r['decoded']), strip_punct(r['true'])
    errs_sp += calc_wer(list(t), list(d))
    tot_sp += len(t)
cer_sp = errs_sp / max(1, tot_sp)

report = {
    'note': '200/10000 批欠收敛；评估集=官方划分的 test trial（num_test_trials=1，样本极小）；decoder 验证 CER 同为 0.421',
    'n_trials': len(results),
    'nBatchesToTrain': 200,
    'decoder_val_CER': 0.421,
    'cer_raw_greedy': round(cer_raw, 4),
    'cer_raw_greedy_stripped': round(cer_sp, 4),
    'per_trial': results,
}
with open(OUT_DIR / 'm1_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('cer_raw_greedy:', round(cer_raw, 4), '| stripped:', round(cer_sp, 4))
for t in results[:4]:
    print('  true:', t['true'][:38], '| dec:', t['decoded'][:38], '| err', t['char_err'], '/', t['char_total'])
print('REPORT_V2_DONE')
