# -*- coding: utf-8 -*-
"""T7: OOV audit — reference words vs the TLG 125k vocabulary.

For each dataset's final tsv: normalize refs like wer_from_tsv, then count
how many ref words are outside words.txt (the TLG word list, "WORD id" per
line). OOV ref words set a floor on achievable WER (each costs >=1 error).
Usage: vac_vocab_audit.py (paths hardcoded, run from repo root)
"""
import collections
import re

WORDS = (r"data\20230616_willett_BtoT24"
         r"\languageModel_5gram_work_words.txt")
SETS = {
    "willett": r"data\20230616_willett_BtoT24\run_60k\wl_final_full.tsv",
    "card":    r"data\20241009_card_BtoT25\run_120k\final_full.tsv",
    "jude":    r"data\20260428_jude\run_25k\off_final_full.tsv",
    "kunz_t12": r"data\20250821_kunz\vac\t12_bp1_a0.325_final.tsv",
    "kunz_t15": r"data\20250821_kunz\vac\t15_bp1_a0.5_final.tsv",
    "kunz_t16": r"data\20250821_kunz\vac\t16_bp1_a0.325_final.tsv",
}


def norm(s):
    return re.sub(r"[^A-Za-z'< ]", " ", s).upper().split()


def main():
    vocab = set()
    for line in open(WORDS, encoding="utf-8"):
        w = line.split()[0]
        if not w.startswith("#") and w != "<eps>":
            vocab.add(w.upper())
    print(f"vocab: {len(vocab)} entries")
    for name, path in SETS.items():
        n_words = n_oov = n_trials = t_oov = 0
        oov_counter = collections.Counter()
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[0] == "key":
                continue
            n_trials += 1
            ws = norm(p[1])
            oovs = [w for w in ws if w not in vocab]
            n_words += len(ws)
            n_oov += len(oovs)
            if oovs:
                t_oov += 1
            oov_counter.update(oovs)
        print(f"{name:9s}: {n_trials:5d} trials, {n_words:6d} ref words, "
              f"OOV {n_oov} ({100*n_oov/max(1,n_words):.2f}%), "
              f"trials w/ OOV {t_oov} ({100*t_oov/max(1,n_trials):.1f}%) "
              f"-> WER floor ~{100*n_oov/max(1,n_words):.2f}")
        for w, c in oov_counter.most_common(5):
            print(f"    OOVx{c}: {w}")


if __name__ == "__main__":
    main()
