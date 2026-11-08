# `20230831_willett/code/` — Code Guide

Reproduces the **Willett et al., *Nature* 2023** speech neuroprosthesis (participant
T12): a 5-layer GRU emits CTC phoneme logits from neural features, then word-level
decoding runs either through a 1.46 GB KenLM 3-gram beam search (pure Python, this
folder) or through the C++ 5-gram TLG decoder in `shared/tools/`.
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
(Dryad competitionData .mat)
   └─ convert_btot24.py ──→ hdf5 in the nejm-brain-to-text layout
        └─ train_btot24.py ──→ GRU checkpoint (FP16, ~3.5 GB VRAM)
             ├─ eval_btot24.py ──→ greedy PER (test partition)
             └─ dump_btot24_logits.py ──→ per-trial CTC log-probs (hdf5)
                  ├─ decode_words_btot24_v2.py ──→ KenLM 3-gram word decode (WSL)
                  │       (supplies word_decode_joint3gram.csv)
                  └─ export_trials_npy.py ──→ per-trial .npy + meta.tsv
                        └─ shared/tools/tlg_viterbi.cc ──→ 5-gram WFST decode (WSL)
                             └─ shared/tools/opt_rescore.py ──→ OPT/Qwen rescoring
plot_willett2023_wer.py ──→ PER-vs-WER reproduction figure
```

## File-by-file

| File | Role |
|---|---|
| `convert_btot24.py` | Converts the released per-day `.mat` files into the nejm-brain-to-text hdf5 layout (spikePow+tx1 features, per-block z-scoring, g2p phoneme targets). |
| `train_btot24.py` | Trains the GRUDecoder on the converted data by programmatically adapting the official `rnn_args.yaml` (Windows-safe settings, scaled LR schedule). |
| `eval_btot24.py` | Greedy CTC evaluation of a checkpoint on the test partition (PER = Σeditdist / Σlen, official semantics). |
| `dump_btot24_logits.py` | Same model/iteration path as `eval_btot24.py`, but saves per-trial `log_softmax` outputs for word-level decoding. |
| `decode_words_btot24_v2.py` | **Canonical word decoder** (replaces the retired buggy v1, see `docs/CLEANUP_AUDIT.md`): joint lexicon-constrained CTC beam search over a pronunciation trie with KenLM 3-gram scoring; the 41× RAM-reduction path documented in the README. |
| `export_trials_npy.py` | Exports per-trial log-prob `.npy` + `meta.tsv` into `~/tlv` for the C++ `tlg_viterbi` 5-gram decoder (`shared/tools/tlg_viterbi.cc`). |
| `plot_willett2023_wer.py` | Draws the paper-vs-ours PER/WER figure from the CSVs produced above. |
