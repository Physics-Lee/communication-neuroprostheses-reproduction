# `20210512_willett/code/` — Code Guide

Reproduces the **Willett et al., *Nature* 2021** handwriting BCI pipeline (participant
T5): an RNN decodes attempted-handwriting neural features into character probabilities,
a Kaldi bigram FST turns them into text, and GPT-2 rescoring finishes the job.
Study-level context (memory adaptations, results): [`../README.md`](../README.md).

## Pipeline order

```text
(Dryad data)
   └─ step2_relabel.py ── HMM forced-alignment labels (re-run + audit vs official)
   └─ step3_make_synthetic.py ── synthetic data augmentation → .npy training folds
   └─ torch_port/train.py ── PyTorch RNN training (replaces official TF1 Step4)
        └─ torch_port/infer.py ── Step5 inference → logits + errCounts
             ├─ step5_eval_from_errcounts.py ── raw CER/WER + bootstrap CI
             ├─ step6_make_kaldimat.py ── logits → Kaldi probability matrices
             │    └─ step6_decode_ours.py ── bigram FST decode (WSL latgen_lazy2)
             │         └─ (m4_report.py aggregates everything)
             └─ fig_perday_cer_ours.py ── per-day official-vs-ours CER figure
Official-reference path (no training, uses shipped outputs):
   └─ step6_select_from_kaldioutput.py ── official n-best selection audit
   └─ step6_decode_batch.py ── same decoder over OFFICIAL RNN kaldiMats
   └─ step7_rescore_gpt2.py ── GPT-2 rescoring of official n-best lists
pipeline_watchdog_v4.py drives steps automatically (see below)
```

## File-by-file

| File | Role |
|---|---|
| `step2_relabel.py` | Re-runs the official HMM forced-alignment labeling (Step2) and audits agreement with the shipped official labels (letter starts, frame argmax, …). |
| `forcedAlignmentHMM.py` | numpy-2.x compatibility shim for the official module of the same name (spawned workers re-import it; `code/` precedes the official repo on `sys.path`). |
| `step3_make_synthetic.py` | Port of the official synthetic-data augmentation notebook; writes `.npy` folds (mmap-friendly) instead of TF1 `.tfrecord`s — the 56 GB → 26.9 GB RAM fix. |
| `step5_eval_from_errcounts.py` | Recomputes CER/WER (+ bootstrap 95% CI, fixed seed) from official/self `errCounts.mat` files; also a library (`summarize`) used by `m4_report.py`. |
| `step6_make_kaldimat.py` | Converts self-trained RNN outputs into Kaldi probability matrices via the official `rnnOutputToKaldiMatrices`. |
| `step6_decode_batch.py` | Batch bigram decode of the **official** RNN outputs through the WSL `latgen_lazy2` C++ decoder; also a library (`decode_one`, `wer`) imported by `step6_decode_ours.py`. |
| `step6_decode_ours.py` | Same decode over the **self-trained** model's matrices (M3/M4 milestone). |
| `step6_decode_bigram.py` | Pure-Python bigram decoder with lazy HCL×G composition (replicates `latgen-lazylm` semantics) — no WSL/Kaldi needed; the memory-wall mitigation documented in the study README. |
| `step6_select_from_kaldioutput.py` | Reproduces the official n-best selection (`argmin(ac + 2.0·lm)`) from shipped Step6 KaldiOutput files. |
| `step7_rescore_gpt2.py` | GPT-2-XL rescoring of the official n-best lists (HF transformers port of the official notebook). |
| `fig_perday_cer_ours.py` | Fig. 2c-style per-session raw CER bars, official checkpoint vs. self-trained, both held-out partitions. |
| `m4_report.py` | M4 milestone: aggregates all metrics into `docs/20261020_复现报告.md` and updates README/PROGRESS. Idempotent. |
| `pipeline_watchdog_v4.py` | The final one-shot orchestrator (guard Blocks training → infer → step5 → kaldimat → decode → figure → report). Machine-specific paths (conda env, GPU polling); kept as executable documentation of pipeline order. Superseded watchdog versions were removed (see `docs/CLEANUP_AUDIT.md`). |
| `torch_port/` | PyTorch port of the TF1 RNN (model / train / infer / augmentation / sanity check). See [`torch_port/README.md`](torch_port/README.md). |
| `latgen_lazy.cc.txt`, `latgen_nbest.cc.txt` | Reference C++ sources for the WSL-side lazy Kaldi decoders used by the `step6_decode_*` scripts. |
