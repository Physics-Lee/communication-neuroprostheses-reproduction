#!/usr/bin/env python3
"""Batch 5-gram WFST language-model decoder for Jude et al. (2026a) multi-session runs.

Loads the official ``Typing5gramPunc`` TLG FST once in WSL (``py310lm`` env),
decodes ``test_logits.npz`` for all 8 T17 and 8 T18 sessions from
``results/runs_multisess/``, and writes ``lm_decode_summary.json`` containing
per-session raw CER, post-LM CER, and post-LM WER.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REF = Path('/mnt/d/repositories/20261009_reproduce_20260316_jude/open_source_code_for_reference/20260316_jude_typingBCI')
sys.path.insert(0, str(REF / 'utils'))

from lmutils import build_lm_decoder, calc_wer, lm_decode  # noqa: E402

LM_PATH = '/mnt/d/repositories/20261009_reproduce_20260316_jude/data/20260316_jude/LanguageModels/Typing5gramPunc'
MULTISESS_ROOT = Path('/mnt/d/repositories/20261106_communication_neuroprostheses_reproduction/20260316_jude/results/runs_multisess')
SINGLE_ROOT = Path('/mnt/d/repositories/20261009_reproduce_20260316_jude/result/runs')

SINGLE_MAP = {
    't17.2024.04.25': 'm3_t1720240425_torch',
    't17.2024.06.03': 'm3_t17_20240603_torch',
    't17.2024.06.04': 'm3_t17_20240604_torch',
    't17.2024.06.05': 'm3_t1720240605_torch',
    't17.2024.06.13': 'm3_t1720240613_torch',
    't17.2024.07.09': 'm3_t17_20240709_torch',
    't17.2024.07.10': 'm3_t1720240710_torch',
    't17.2024.07.22': 'm3_t1720240722_torch',
    't18.2024.12.04': 'm1_t18_20241204_torch',
    't18.2024.12.05': 'm3_t18_20241205_torch',
    't18.2025.01.14': 'm3_t18_20250114_torch',
    't18.2025.01.15': 'm3_t18_20250115_torch',
    't18.2025.01.21': 'm3_t18_20250121_torch',
    't18.2025.01.22': 'm3_t18_20250122_torch',
    't18.2025.02.04': 'm3_t18_20250204_torch',
    't18.2025.02.05': 'm3_t18_20250205_torch',
}


def strip_punct(s):
    """Strip sentence punctuation before scoring CER/WER."""
    return s.replace('.', '').replace('?', '').replace(',', '').strip()


def decode_npz(decoder, npz_path):
    """Decode all trials in a test_logits.npz file with the 5-gram TLG decoder."""
    z = np.load(npz_path, allow_pickle=True)
    logits_list = list(z['logits'])
    trues = [str(s) for s in z['true_sentences']]
    ce_sum = ch_sum = we_sum = w_sum = 0
    trials = []
    for i, lg in enumerate(logits_list):
        lg = np.asarray(lg, dtype=np.float64)
        rear = np.concatenate([lg[:, -1:], lg[:, :-1]], axis=1)
        dec = lm_decode(decoder, rear, blankPenalty=1.0)
        dec_c = strip_punct(dec)
        true_c = strip_punct(trues[i])
        ce = int(calc_wer(list(true_c), list(dec_c)))
        we = int(calc_wer(true_c.split(' '), dec_c.split(' ')))
        nc = len(true_c)
        nw = len(true_c.split())
        ce_sum += ce
        ch_sum += nc
        we_sum += we
        w_sum += nw
        trials.append({'ref': true_c, 'hyp': dec_c, 'ce': ce, 'nc': nc, 'we': we, 'nw': nw})
    return {
        'n_trials': len(logits_list),
        'n_chars': ch_sum,
        'n_words': w_sum,
        'char_errs': ce_sum,
        'word_errs': we_sum,
        'post_lm_cer': round(ce_sum / max(1, ch_sum), 4),
        'post_lm_wer': round(we_sum / max(1, w_sum), 4),
        'trials': trials,
    }


def main():
    t0 = time.time()
    print('Loading 5-gram TLG decoder...', flush=True)
    decoder = build_lm_decoder(LM_PATH)
    print(f'LM loaded in {time.time() - t0:.1f}s', flush=True)

    results = {'t17': {}, 't18': {}}
    for part in ['t17', 't18']:
        part_dir = MULTISESS_ROOT / part
        if not part_dir.exists():
            continue
        for s_dir in sorted(part_dir.iterdir()):
            if not s_dir.is_dir():
                continue
            sess = s_dir.name
            npz_ms = s_dir / 'test_logits.npz'
            rep_ms = s_dir / 'final_report.json'
            if not npz_ms.exists():
                continue
            raw_cer_ms = json.loads(rep_ms.read_text())['best_cer'] if rep_ms.exists() else None
            res_ms = decode_npz(decoder, npz_ms)
            res_ms['raw_cer'] = round(float(raw_cer_ms), 4) if raw_cer_ms is not None else None

            # Also check single-session run for comparison
            single_dir = SINGLE_ROOT / SINGLE_MAP.get(sess, '')
            rep_sg = single_dir / 'final_report.json'
            raw_cer_sg = json.loads(rep_sg.read_text())['best_cer'] if rep_sg.exists() else None
            res_ms['single_session_raw_cer'] = round(float(raw_cer_sg), 4) if raw_cer_sg is not None else None

            results[part][sess] = res_ms
            print(
                f'{sess}: raw_CER={res_ms["raw_cer"]:.4f} (single={res_ms["single_session_raw_cer"]}) '
                f'| post-LM CER={res_ms["post_lm_cer"]:.4f} | post-LM WER={res_ms["post_lm_wer"]:.4f}',
                flush=True,
            )

    out_file = MULTISESS_ROOT / 'lm_decode_summary.json'
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'SAVED: {out_file}', flush=True)


if __name__ == '__main__':
    main()
