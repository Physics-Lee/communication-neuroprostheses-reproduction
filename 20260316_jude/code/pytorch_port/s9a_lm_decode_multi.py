#!/usr/bin/env python3
# s9a_lm_decode_multi.py — 一次 TLG 加载，批量解码多个会话的 test logits（2026-10-11 00:40）
import sys
import time
from pathlib import Path

import numpy as np

REF = Path('./20260316_jude/open_source_code_for_reference/20260316_jude_typingBCI')
sys.path.insert(0, str(REF / 'utils'))

from lmutils import build_lm_decoder, lm_decode, calc_wer  # noqa: E402

import json as _json
from pathlib import Path as _P
_RJ = _P('./20260316_jude/code/pytorch_port/s9_runs.json')
RUNS = _json.loads(_RJ.read_text()) if _RJ.exists() else {
    't18.2024.12.05': './20260316_jude/result/runs/m3_t18_20241205_torch',
}
LM_PATH = './20260316_jude/data/20260316_jude/LanguageModels/Typing5gramPunc'

t0 = time.time()
decoder = build_lm_decoder(LM_PATH)
print(f'LM loaded in {time.time()-t0:.0f}s', flush=True)

grand = {}

def strip_punct(s):
    return s.replace(".", "").replace("?", "").replace(",", "").strip()


for sess, run_dir in RUNS.items():
    z = np.load(Path(run_dir) / 'test_logits.npz', allow_pickle=True)
    logits_list, trues = list(z['logits']), [str(s) for s in z['true_sentences']]
    ce_sum = ch_sum = we_sum = w_sum = 0
    for i, lg in enumerate(logits_list):
        lg = np.asarray(lg, dtype=np.float64)
        rear = np.concatenate([lg[:, -1:], lg[:, :-1]], axis=1)  # blank 挪 index 0（官方 rearrange）
        dec = lm_decode(decoder, rear, blankPenalty=1.0)
        dec_c = strip_punct(dec)
        true_c = strip_punct(trues[i])
        ce = calc_wer(list(true_c), list(dec_c))
        we = calc_wer(true_c.split(' '), dec_c.split(' '))
        ce_sum += ce
        ch_sum += len(true_c)
        we_sum += we
        w_sum += len(true_c.split())
    cer, wer = ce_sum / max(1, ch_sum), we_sum / max(1, w_sum)
    grand[sess] = {'n': len(logits_list), 'cer': round(cer, 4), 'wer': round(wer, 4)}
    print(f"{sess}: n={len(logits_list)} post-LM CER {cer:.4f} WER {wer:.4f}", flush=True)

print('MULTI_DONE', grand, flush=True)
