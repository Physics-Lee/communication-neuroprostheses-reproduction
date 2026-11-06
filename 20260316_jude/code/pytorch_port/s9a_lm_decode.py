#!/usr/bin/env python3
# S9a：用编译好的 lm_decoder 对 S10b 的 test logits 做 5-gram 解码（2026-10-10 22:35，WSL py310lm）
# 官方 lmutils.build_lm_decoder + lm_decode；logits 原生序 blank=30，需 rearrange（blank 挪 0）。
import sys
import time
from pathlib import Path

import numpy as np

REF = Path('./20260316_jude/open_source_code_for_reference/20260316_jude_typingBCI')
sys.path.insert(0, str(REF / 'utils'))

from lmutils import build_lm_decoder, lm_decode, calc_wer  # noqa: E402

CKPT = Path('./20260316_jude/result/runs/m1_t18_20241204_torch')
LM_PATH = './20260316_jude/data/20260316_jude/LanguageModels/Typing5gramPunc'

z = np.load(CKPT / 'test_logits.npz', allow_pickle=True)
logits_list = list(z['logits'])
trues = [str(s) for s in z['true_sentences']]
print(f'{len(logits_list)} trials loaded', flush=True)

t0 = time.time()
decoder = build_lm_decoder(LM_PATH)  # 只加载 TLG（G_no_prune 不存在→自动跳过 rescore）
print(f'LM loaded in {time.time()-t0:.0f}s', flush=True)

CHAR_DEF = ['>', ',', '?', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

total_err = total_char = 0
word_err = total_word = 0
for i, logits in enumerate(logits_list):
    logits = np.asarray(logits, dtype=np.float64)
    # 官方 rearrange_speech_logits 语义：blank（末列）挪到 index 0
    rear = np.concatenate([logits[:, -1:], logits[:, :-1]], axis=1)
    t1 = time.time()
    decoded = lm_decode(decoder, rear, blankPenalty=1.0)
    dt = time.time() - t1
    dec_clean = decoded.replace('.', '').replace('?', '').replace(',', '').strip()
    true_clean = trues[i].replace('.', '').replace('?', '').replace(',', '').strip()
    ce = calc_wer(list(true_clean), list(dec_clean))
    we = calc_wer(true_clean.split(' '), dec_clean.split(' '))
    total_err += ce
    total_char += len(true_clean)
    word_err += we
    total_word += len(true_clean.split(' '))
    print(f'trial {i}: decode {dt:.1f}s | true: {true_clean[:45]!r} | lm: {dec_clean[:45]!r} | char_err {ce}/{len(true_clean)} word_err {we}/{len(true_clean.split())}', flush=True)

print(f'S9A_DONE post-LM CER {total_err/max(1,total_char):.4f} WER {word_err/max(1,total_word):.4f} ({len(logits_list)} trials)', flush=True)
