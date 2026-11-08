"""G1: recompute the online system's typing speed and WER from free-typing-block .mat files.

Speed convention: cpm = 60 * len(ngram_decoder_final_output) /
(trial_end_redis_time - go_cue_redis_time, in seconds). Multiple conventions
(with/without spaces, mean/peak) are all printed, for comparison against the
paper's 110 cpm / 22 wpm / WER 1.6%. WER compares the LM final output with
the human-confirmed true sentence (cue).

Notes
-----
Written 2026-10-11 12:35. Run with the ``reproduce_20260316_jude`` environment.
"""
import glob
import sys

import numpy as np
import scipy.io as sio


def calc_wer(r, h):
    """Compute the token-level Levenshtein distance between two sequences.

    Parameters
    ----------
    r : sequence of str
        Reference tokens (words of the true cue sentence).
    h : sequence of str
        Hypothesis tokens (words of the decoded sentence).

    Returns
    -------
    int
        Minimum number of insertions, deletions and substitutions needed to
        turn ``h`` into ``r``.
    """
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int64)
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = d[i - 1][j - 1] if r[i - 1] == h[j - 1] else min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return int(d[len(r)][len(h)])


def get_string(el):
    """元素可能是 [[0]]（该 tick 无输出）或 array(['句子'])。

    Extract the decoded string from one per-decoding-tick element of the
    object arrays stored in the ``ngram_decoder_*`` fields of the .mat file.

    Parameters
    ----------
    el : numpy.ndarray
        One element of a per-tick object array from the .mat file.

    Returns
    -------
    str
        The string decoded at that tick, or ``''`` if the element carries no
        text output.
    """
    try:
        a = el.reshape(-1)
        if a.size and isinstance(a[0], str):
            return str(a[0])
    except Exception:
        pass
    return ''


def analyze(pattern, label):
    """Analyze every .mat matching ``pattern`` and print per-block CPM/WER statistics.

    Per trial, the final LM sentence is recovered from the per-tick partial
    outputs when the block stores them (windowed by the redis clock between
    go_cue and trial_end), and otherwise taken from
    ``ngram_decoder_final_output``; typed speed and token-level WER against
    the cue sentence are then aggregated per block.

    Parameters
    ----------
    pattern : str
        Glob pattern selecting the RedisMat .mat files of one block.
    label : str
        Session/block label prefixed to the printed report lines.

    Returns
    -------
    None
        Statistics are printed to stdout only.
    """
    for f in sorted(glob.glob(pattern)):
        d = sio.loadmat(f)
        fname = f.split('\\')[-1]
        n = len(d['cue'])
        # 自由打字块：LM 输出按解码 tick 存（长 9081 类对象数组），按 redis 时钟窗口取每 trial 的最终句
        tick_clock = d['decoder_output_redis_clock'][0] if 'decoder_output_redis_clock' in d else None
        partial = d['ngram_decoder_partial_output'][0] if tick_clock is not None and d['ngram_decoder_partial_output'].ndim == 2 and d['ngram_decoder_partial_output'].shape[0] == 1 else None
        rows = []
        tot_err = tot_word = 0
        for x in range(n):
            cue = str(d['cue'][x]).strip()  # numpy <U 字符串本身即 str，再 [0] 会取首字符
            if partial is not None:
                mask = (tick_clock >= d['go_cue_redis_time'][0][x]) & (tick_clock <= d['trial_end_redis_time'][0][x])
                idx = np.where(mask)[0]
                cands = [get_string(partial[i]) for i in idx] if len(idx) else []
                cands = [c for c in cands if c]
                out = max(cands, key=len).strip() if cands else ''
                # 打字速度口径：从首个非空 partial 的时间戳到 trial 结束（排除起步延迟，论文"用户自定速"）
                first_tick = next((i for i in idx if get_string(partial[i])), None)
                start_ms = tick_clock[first_tick] if first_tick is not None else d['go_cue_redis_time'][0][x]
            else:
                out = get_string(d['ngram_decoder_final_output'][x][0]).strip()
                start_ms = d['go_cue_redis_time'][0][x]
            dur = (d['trial_end_redis_time'][0][x] - start_ms) / 1000.0
            chars_full = len(out)
            chars_nospace = len(out.replace(' ', ''))
            cpm_full = 60 * chars_full / dur if dur > 0 else 0
            we = calc_wer(cue.split(), out.split())
            tot_err += we
            tot_word += len(cue.split())
            rows.append((dur, chars_full, cpm_full, chars_nospace, we, cue[:30], out[:30]))
        print(f'== {label} {fname}: {n} trials ==')
        cpms = [r[2] for r in rows]
        print(f'  per-trial cpm: mean {np.mean(cpms):.1f} | median {np.median(cpms):.1f} | peak {np.max(cpms):.1f}')
        print(f'  fastest 3: {sorted(cpms)[-3:][::-1]}')
        print(f'  session WER(LM final vs cue): {tot_err}/{tot_word} = {tot_err/max(1,tot_word):.4f}')
        for r in rows[:5]:
            print(f'    dur {r[0]:.1f}s | {r[1]} chars | {r[2]:.1f} cpm | wer {r[4]} | true {r[5]!r} | out {r[6]!r}')
        print(flush=True)


if __name__ == '__main__':
    base = r'.\20260316_jude\data\20260316_jude\Data'
    # t18.2025.02.04 block 5 = 论文峰值会话的自由打字块
    analyze(base + r'\t18\t18.2025.02.04\Typing\RedisMat\*(5).mat', 't18.2025.02.04-b5')
    # T17 自由打字最多的会话 t17.2024.07.09（blocks 4,7,8,10,12,13）
    for b in [4, 7, 8, 10, 12, 13]:
        analyze(base + rf'\t17\t17.2024.07.09\Typing\RedisMat\*({b}).mat', f't17.2024.07.09-b{b}')
