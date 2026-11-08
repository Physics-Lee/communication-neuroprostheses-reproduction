# `shared` — Cross-Study C++ WFST Decoders, KenLM/LLM Rescorers & Finetune Utilities

Shared infrastructure used across the speech and handwriting reproduction pipelines:

- **`tools/tlg_viterbi.cc` & `tools/wenet_lattice.cc`**: Memory-mapped (`mmap` + `.idx`) C++ Kaldi OpenFST 5-gram WFST beam-search decoders that reduce host RAM peak from **300 GB** to **19.6 GB**.
- **`tools/opt_rescore.py` & `tools/qwen_rank_probe.py`**: Second-pass N-best lattice rescorers supporting Facebook OPT-6.7B (FP16, 13.0 GB VRAM) and 4-bit Qwen2.5-7B / Qwen3-8B on a single 16 GB GPU.
- **`tools/wer_from_tsv.py` & `tools/compare_rescorers.py`**: Case-insensitive, punctuation-normalized word error rate (WER) evaluators matching the official Brain-to-Text sidecar convention.
- **`finetune_chain/` & `vacation_tlg/`**: Cross-day adaptation and hyperparameter sweep scripts for acoustic scale, word insertion penalty (WIP), and LLM interpolation weights ($\alpha$, $\beta$).

## File-by-file

### `tools/`

| File | Role |
|---|---|
| `tlg_viterbi.cc` | Frame-synced CTC Viterbi over the composed `TLG.fst` (mmap + `.idx` state offsets): the 19.6 GB 5-gram WFST decode path. |
| `wenet_lattice.cc` | N-best lattice version of the same mmap decoder (feeds LLM rescoring). |
| `glog_shim.h` | Minimal glog header shim so the two `.cc` files build without a full glog install. |
| `opt_rescore.py` | Second-pass rescoring of decoder n-best TSVs with OPT-6.7B / Qwen (FP16/4-bit, single 16 GB GPU); interpolation weights $\alpha$, $\beta$. |
| `qwen_rank_probe.py` | Diagnosis: does the rescorer's raw sentence score rank the near-reference candidate highly within each n-best pool? |
| `wer_from_tsv.py` | WER for decoder TSV output (case/punctuation-normalized, official sidecar convention). |
| `compare_rescorers.py` | Pooled-WER comparison of 5-gram / OPT / Qwen rescoring across the four papers. |
| `parse_train_log.py` | Post-hoc parser of the nejm-brain-to-text trainer log → curves.csv + training_curves.png. |
| `pkl_online_metrics.py` | Online-decoding metrics computed from the paper's own `t15_copyTask.pkl` records (greedy PER before LM; final PER/WER after). |
| `extract_four_papers.py` | Fault-tolerant per-page PDF→text extraction of the four papers (anchor mining). |
| `*.sh` | WSL-side sweep drivers for the decoders/rescorers. |

### `plots/` (supplementary figures — distinct from the root benchmark figure)

| File | Figure |
|---|---|
| `plot_overview.py` | Six-panel BtoT24/BtoT25 run dashboard (training trajectory, per-day PER, corpus/timeline panels). |
| `plot_figs_separate.py` | The same six panels as separate figures. |
| `plot_wer_summary.py` | Four-paper WER reproduction summary (PER and WER, ours vs anchors). |
| `plot_wer_spectrum.py` | Same-pipeline greedy-PER difficulty spectrum across the five participants. |
| `plot_kunz_jude.py` | Kunz + Jude-2026b figures: PER spectrum extension + ours-vs-author-online comparison. |
| `plot_recording_days.py` | Recording-day calendar timelines of the four BrainGate speech datasets. |
| `plot_two_experiments.py` | 2026-09-29/30 experiments: official-decoder port WERs and rescoring-model swap. |
| `plot_vacation_tlg.py` | Vacation-week TLG results (oracle depth, rescorers, error anatomy, day adaptation). |

### `vacation_tlg/` (TLG-decoder experiment series; runners `vac_*.sh`)

| File | Role |
|---|---|
| `vac_day_adapt.py` | T6: label-free per-session logit mean-shift correction (day adaptation sweep over λ). |
| `vac_error_ops.py` | T7: word-level INS/DEL/SUB error anatomy of final TSVs. |
| `vac_oracle_depth.py` | Oracle WER vs n-best depth from an nbest TSV. |
| `vac_vocab_audit.py` | T7: OOV audit of reference words against the TLG 125k vocabulary. |
| `vac_patch_wip.py` | One-shot patcher that added the word-insertion-penalty knob to the WSL-side `wenet_lattice.cc` (kept for provenance of `wenet_lattice_wip`). |

### `finetune_chain/`

| File | Role |
|---|---|
| `train_finetune.py` | Rapid-calibration Step 3: fine-tune from the expanded Step-2 checkpoint on the target day's first N trials (driven by `run_finetune_chain.sh`). |
