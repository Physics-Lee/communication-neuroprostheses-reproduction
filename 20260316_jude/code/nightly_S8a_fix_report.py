# S8a 报告修复：从已落盘的 inference npy 重算 raw CER（uint8→int 的 json 序列化坑），补写 m1_report.json（2026-10-10）
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(r'.\20260316_jude\result\runs\m1_t18_20241204_smoke')

CHAR_DEF = ['>', ',', '?', '.',
            'A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']  # 30 类；index 30 = blank


def calc_wer(r, h):
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
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
        if s != prev and s != 30:
            chars.append(CHAR_DEF[int(s)])
        prev = s
    return ''.join(chars)


def extract_true(trans_row):
    end_idx = np.argwhere(trans_row == 0)[0, 0]
    return ''.join(chr(int(c)) for c in trans_row[:end_idx])


def strip_punct(s):
    for p in ['.', '?', ',']:
        s = s.replace(p, '')
    return s.strip()


logits_all = np.load(OUT_DIR / 'inference_logits.npy', allow_pickle=True)
trans_all = np.load(OUT_DIR / 'inference_transcriptions.npy', allow_pickle=True)
print('n trials:', len(logits_all))

results = []
for i in range(len(logits_all)):
    # 逐 trial 变长：train 脚本保存时可能已截断到 logitLen；防御性按 transcription 长度对齐
    dec = greedy_decode(logits_all[i])
    true = extract_true(trans_all[i])
    n_err = calc_wer(list(true), list(dec))
    results.append({'true': true, 'decoded': dec, 'char_err': n_err, 'char_total': len(true)})

cer_raw = sum(r['char_err'] for r in results) / sum(r['char_total'] for r in results)
errs_sp, tot_sp = 0, 0
for r in results:
    d, t = strip_punct(r['decoded']), strip_punct(r['true'])
    errs_sp += calc_wer(list(t), list(d))
    tot_sp += len(t)
cer_sp = errs_sp / tot_sp

report = {
    'note': '200/10000 批欠收敛口径；本 CER 对全部 57 个训练 trial 评估（拟合优度口径，非 held-out——官方 num_test_trials=1）',
    'n_trials': len(results),
    'nBatchesToTrain': 200,
    'cer_raw_greedy': round(cer_raw, 4),
    'cer_raw_greedy_stripped': round(cer_sp, 4),
    'per_trial': results,
}
with open(OUT_DIR / 'm1_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('cer_raw_greedy:', round(cer_raw, 4), '| stripped:', round(cer_sp, 4))
for t in results[:4]:
    print('  true:', t['true'][:38], '| dec:', t['decoded'][:38], '| err', t['char_err'], '/', t['char_total'])
print('REPORT_FIXED')
