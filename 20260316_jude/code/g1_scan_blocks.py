"""G1 round 2: scan CPM across all blocks of all T18 sessions (sentence-copy-block convention).

Sentence-copy blocks: the typed content is the cue sentence and the duration
is trial_end_redis_time - go_cue_redis_time. Prints per-block mean/median/
peak CPM to locate the origin of the paper's 110 cpm figure.

Notes
-----
Written 2026-10-11. Run with the ``reproduce_20260316_jude`` environment.
"""
import glob
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

BASE = Path(r'.\20260316_jude\data\20260316_jude\Data\t18')
sessions = sorted(p.name for p in BASE.iterdir() if p.is_dir())
print('session | block | n | cpm_mean | cpm_median | cpm_peak')
all_blocks = []
for s in sessions:
    for f in sorted((BASE / s / 'Typing' / 'RedisMat').glob('*.mat')):
        m = re.search(r'\((\d+)\)', f.name)
        if not m:
            continue
        block = int(m.group(1))
        d = sio.loadmat(f)
        if 'cue' not in d or 'go_cue_redis_time' not in d:
            continue
        cpms = []
        for x in range(len(d['cue'])):
            cue = str(d['cue'][x]).strip()
            if len(cue) < 5:
                continue  # 跳过空/极短（非句子复制 trial）
            dur = (d['trial_end_redis_time'][0][x] - d['go_cue_redis_time'][0][x]) / 1000.0
            if dur > 3:
                cpms.append(60 * len(cue) / dur)
        if cpms:
            print(f'{s} | b{block} | {len(cpms)} | {np.mean(cpms):.1f} | {np.median(cpms):.1f} | {np.max(cpms):.1f}')
            all_blocks.append((s, block, np.mean(cpms), np.max(cpms), len(cpms)))
print('\n按块均值排序 top5:')
for row in sorted(all_blocks, key=lambda r: -r[2])[:5]:
    print(f'  {row[0]} b{row[1]}: mean {row[2]:.1f} peak {row[3]:.1f} (n={row[4]})')
