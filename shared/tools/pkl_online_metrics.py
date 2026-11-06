# -*- coding: utf-8 -*-
"""Online decoding metrics from the paper's own records (t15_copyTask.pkl).

Computes, for the 1,718 online evaluation trials:
- greedy phoneme PER (acoustic-only, BEFORE language model; 'decoded_phonemes_raw')
- final PER / WER (AFTER n-gram LM + rescoring; 'decoded_sentence*')
Usage: python pkl_online_metrics.py
"""
import pickle, re
import editdistance

PKL = r".\20260713_Eddie_and_BrainGate_23_24_3_papers\datasets\BtoT25\t15_copyTask.pkl"

def remove_punctuation(s):
    s = re.sub(r"[^a-zA-Z\- \']", "", s)
    s = s.replace("- ", " ").lower().replace("--", "").replace(" '", "'").strip()
    return " ".join(w for w in s.split() if w)

def main():
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    n = len(d["cue_sentence"])
    for name, idx in [("all", range(n)),
                      ("50-word", [i for i in range(n) if d["vocab_size"][i] == 50]),
                      ("125k", [i for i in range(n) if d["vocab_size"][i] == 125000])]:
        idx = list(idx)
        e = sum(editdistance.eval(d["decoded_phonemes_raw"][i], d["cue_sentence_phonemes"][i]) for i in idx)
        l = sum(len(d["cue_sentence_phonemes"][i]) for i in idx)
        e2 = sum(editdistance.eval(d["decoded_sentence_phonemes"][i], d["cue_sentence_phonemes"][i]) for i in idx)
        we = sum(editdistance.eval(remove_punctuation(d["cue_sentence"][i]).split(),
                                   remove_punctuation(d["decoded_sentence"][i]).split()) for i in idx)
        wl = sum(len(remove_punctuation(d["cue_sentence"][i]).split()) for i in idx)
        print(f"[{name:8s}] n={len(idx):4d}  greedy-PER {100*e/l:6.2f}%  LM-PER {100*e2/l:6.2f}%  LM-WER {100*we/wl:6.2f}%")

if __name__ == "__main__":
    main()
