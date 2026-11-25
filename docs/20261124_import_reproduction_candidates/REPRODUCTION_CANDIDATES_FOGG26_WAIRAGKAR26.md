# Reproduction Candidates: Fogg et al. 2026 & Wairagkar et al. 2026 (Brain2voice 2.0)

**Handoff document for AI coding agents.** Written 2026-11-18 after a full feasibility
study of both preprints against this machine's downloaded datasets and GPU
(1× RTX 5060 Ti 16 GB VRAM, 64 GB DDR5 host RAM). Everything below is
self-contained: paper specs, on-disk data audit, code-release status, scoped
reproduction plans, and open verification items. No re-fetching of the preprints
should be needed for planning (but DO re-check the "watch" items in §7).

Both papers already exist in this repo **as literature data points only**
(`figures/reproduction_benchmark.yaml` panel-a slots `fogg2026` / `wairagkar2026`;
`figures/code/render_figure2.py:74` `PANEL_A_IDS`; timeline metadata in the sibling repo
`D:\repositories\20261106_communication_neuroprostheses_timeline\studies.yaml`
lines 565–601). No study folders, code, or results exist for either yet.

---

## 1. TL;DR verdicts

| Study | Verdict | Data on disk? | New download? | Est. GPU cost (5060 Ti) | Main gap |
|---|---|---|---|---|---|
| **wairagkar26** (Brain2voice 2.0) | ✅ Reproduce near-fully | ✅ 100% (same Dryad T15 set as our `studies/20250612_wairagkar` study) | None | ~4–8 h per full training run (official: 1× RTX 5090 × 1.5 h) | Official code is a placeholder repo → reimplement from methods |
| **fogg26** (generalizable speech neuroprosthesis) | ✅ Reproduce as a **scoped 4-of-6-user offline** study | ◐ 3.5–4 of 6 participants (T12 ✅, T15-Small ✅, T16 ◐, T17 ✅; T21/T22/T15-Large ❌) | None (for the scoped plan) | SU models: hours. MU model: a few days at reduced budget (see §5.2) | 2 new participants + T15-Large (278k sentences) not public; closed-loop online sessions impossible by definition |

Suggested study folders (repo `yyyymmdd_author` convention, keyed to preprint
DOI dates): `20260630_wairagkar/` and `20260723_fogg/`. If peer-reviewed
versions appear with new dates, follow the publication-date convention instead.

**Order of attack: wairagkar26 first** (smaller, zero new data, direct sequel to
an existing study), fogg26 second.

---

## 2. Paper identification

### wairagkar26
- **Title:** *Brain2voice 2.0: High-performance voice synthesis brain-computer interface*
- **Authors:** Wairagkar, Srinivasan, Card, Singer-Clark, Hou, Iacobacci, Miller, Hochberg, Brandman, Stavisky (UC Davis + Brown)
- **Preprint:** bioRxiv, posted 2026-07-06, DOI `10.64898/2026.06.30.735633`, CC BY-NC-ND 4.0, 19 pp.
- **Task:** causal, real-time voice synthesis from intracortical signals; participant **T15** (ALS, 4 arrays, 256 electrodes, 512 neural features @ 10 ms bins).
- **Headline numbers:** human-listener WER **5.24%** (8× better than 2025's 43.75%), causal raw PER 7.03%, feature-domain r 0.77, mel r 0.93 ± 0.02, MCD 0.94 ± 0.22 dB. Prior SOTA (their Nature 2025): mel r 0.83 ± 0.04, MCD 2.89 ± 0.51 dB.

### fogg26
- **Title:** *A generalizable speech neuroprosthesis*
- **Authors:** Fogg, Card, Wairagkar, ... Jude, ... Pandarinath, Rubin, Brandman, Stavisky (28 authors; UC Davis / MGH / Emory-GaTech / Brown)
- **Preprint:** bioRxiv, posted 2026-07-27, DOI `10.64898/2026.07.23.739430`, "All rights reserved", 54 pp. (PMID 42619707)
- **Task:** multi-user (MU) transformer phoneme decoder trained jointly across **six** BrainGate2 participants **T12, T15, T16, T17, T21, T22**; cascaded phoneme→word LM pipeline; 311,602 sentences total.
- **Headline numbers:** MU beats SU for every user (mean 51.1% relative WER reduction offline; 55.9% closed-loop); T17 (LIS, lowest SNR) rescued from SU 65.6% → MU 23.1% WER; generalization to held-out users improves monotonically with pretraining-pool size (mean WER 17.6% @ n=1 → 9.5% @ n=4); few-shot adaptation reaches 10–11% WER from ≤200 sentences (~17–37 min of calibration data).

---

## 3. wairagkar26 — full reproduction brief

### 3.1 Data (already on disk — verify nothing)
Uses the **same** Dryad dataset as Wairagkar et al. 2025 (`doi:10.5061/dryad.2280gb64f`), which we have fully downloaded and extracted in the sibling repo:

```
D:\repositories\20261009_reproduce_20250612_wairagkar\data\
  raw\t15_neural_data.zip            26,158,469,306 B (24.4 GiB)
  raw\t15_control_experiment.zip        710,843,261 B
  <working>\Dryad_T15_all\{day}\       37 day dirs (du ~31 GB), 112 .mat + 9 .h5
```

Per-day contents consumed by our existing `studies/20250612_wairagkar/code/` scripts:
`{day}_neural_data.mat` (~0.9 GB/day), `{day}_target_speech_feats.mat` (65 MB),
`{day}_speech_indices.mat` (93 KB), `brain2voice_t15_dayXXX.h5` (14.8 MB official
Keras decoders). 8,489 trials / 53,278 words total. The dataset also defines the
**128 benchmark test trials + video set** used by both the 2025 paper and 2.0
(test split: 128 benchmark trials held out; 30 validation trials; rest training).

### 3.2 Model (reimplement in PyTorch — paper itself is PyTorch + torch.compile)
Multimodal **causal** Transformer encoder:
- Input: 512 neural features → linear projection → sinusoidal positional encoding.
- Backbone: **8 layers, hidden size 384, 6 heads**, causal-masked self-attention.
- Four parallel output heads (each = 2 linear layers, ReLU after the first):
  1. **Continuous acoustic head** — 20-d LPCNet features (18 spectral + pitch period + pitch strength); primary output, vocoded per 10 ms frame.
  2. **Tokenized acoustic head** — custom RVQ tokenizer over LPCNet features: **8 sequential K-means codebooks × 128 centroids**, fit on residuals (MiniBatch K-means, trained only on training-trial targets). Decoding = sum of 8 centroids. Predicted per codebook via parallel classification heads.
  3. **Phoneme head** — 41 classes (39 phonemes + silence + blank), trained with **CTC** (no time alignment needed; targets via g2p from trial text).
  4. **SSL head** — data2vec-style: mask random neural spans, predict the unmasked forward pass's hidden states; cosine-dissimilarity loss.
- **Multiscale discriminator**: continuous LPCNet features average-pooled at 1×/4×/16×/32× (10/40/160/320 ms); each scale an identical 4-layer 1D conv net (channels 20→64→128→256); frame-wise realness + intermediate feature maps.

### 3.3 Losses & training (all numbers from Methods §3.4 + Appendix A.2)
- `L_cont`: MSE with upweighting of hard high-frequency features; speech-vs-silence mask weighting ramped **1:1 → 5:0.2 over the first 30%** of training.
- `L_tokn`: per-codebook cross-entropy, harmonically weighted (later codebooks matter less).
- `L_phon`: CTC. `L_SSL`: cosine. Adversarial: softplus `L_D`, generator `L_G_adv`, feature-matching `L_fm` after a warm-up.
- Combination: `L_cont`/`L_tokn` merged via **learned uncertainty weighting** (Kendall log-variance weights). Fixed aux weights: **w3 = 0.1**, **w4 ramped 0.01 → 0.1 over 10,000 steps** (after adversarial warm-up), **w5 = 2.0**, **w6 = 0.3**.
- Optimizer: AdamW lr 1e-3, wd 0.01, fused; cosine decay + linear warmup; **25,000 batches**. Discriminator: separate AdamW lr 5e-5, updated **every alternate batch**, warm-up **5,000 steps**. Grad clip 1.0; **bfloat16** AMP; everything `torch.compile`d.
- Batches: **32 trials**, each a random contiguous **6.4 s window**, **downsampled 10 ms → 50 ms (5-bin) for training compute efficiency** (inference still 10 ms). Augmentation in the last 20% of training: Gaussian noise, baseline offsets, cumulative drift noise, 10% channel dropout.
- Official cost: **single RTX 5090, ≈1.5 h**. 5060 Ti estimate: 4–8 h per run (≈1/3–1/4 of a 5090's compute; model ~20M params; VRAM nowhere near 16 GB).
- Hyperparameters were "tuned manually at each development stage" — expect a small reproduction gap; the ablation table (A.3) is the fallback yardstick.

### 3.4 Targets (the one substantive methodological upgrade over 2025)
Improved targets: LPCNet features extracted from **own-voice TTS audio cloned from T15's pre-ALS voice samples**, then time-stretched **in the LPCNet feature domain via cubic interpolation** (not waveform stretching). Dataset-shipped targets remain the fallback; ablation Table A.3 quantifies each component's contribution (baseline = continuous head only + original targets; every row is a listener-WER evaluation — we proxy with ASR, see 3.6).

### 3.5 Inference (simulated real-time)
Sliding window of **80 bins (800 ms)**, stride 1 bin, keep only the last prediction → new output every 10 ms, fully causal. Official timings on RTX 5090: continuous 1.47 ± 0.11 ms/step; tokenized 2.11 ± 0.21 ms/step (incl. EMA logit smoothing, per-codebook argmax, RVQ reconstruction, causal smoothing); LPCNet vocoding 1.2 ms/frame. The 5060 Ti keeps all of this well under the 10 ms budget.

### 3.6 Evaluation protocol for us
- **Cannot** replicate the crowdsourced 7-listener median WER (5.24%). Proxy with **Whisper ASR transcription WER** — the paper itself reports Whisper numbers alongside listener numbers for every condition, so the proxy is calibrated. Pin the Whisper version.
- Objective metrics are fully reproducible: feature-domain Pearson r + MSE (20 normalized features), 40-Mel Pearson r + MCD (speech periods only, silences removed — match their masking or numbers inflate), causal raw phoneme PER (argmax, drop blanks, collapse repeats).
- Ablation targets: Table A.3 (heads/GAN/SSL/speech-weighting/target-quality ladder) and A.4 (electrode-count 64/128/192/256 and per-array v6v > M1 ≈ 55b > d6v).

### 3.7 Code-release status
`https://github.com/neuroprosthetics-lab/brain2voice2.0` → **live but placeholder**
(README.md + docs/ only, 13 KB, last push 2026-07-08; full code promised "with
the peer-reviewed published paper"). Demo audio/video:
`https://neuroprosthetics-lab.github.io/brain2voice2.0` (useful for informal
listening comparisons). **Plan for reimplementation; poll the repo** (§7).

### 3.8 Suggested tiering (matching `studies/20250612_wairagkar` conventions)
- **Tier 1 — reimplementation + official-style eval:** RVQ tokenizer training, full model, dataset-shipped targets; benchmark-set r/MCD/PER + Whisper WER. Compare against both 2.0 (r 0.77 / mel r 0.93 / MCD 0.94) and our 2025 mel-layer result (grand mean 0.649 vs paper 0.83).
- **Tier 2 — ablations:** head ladder (cont → +tokn → +phon → +GAN → +SSL → improved targets), electrode subsets, tokenizer sweep.
- **Tier 3 — improved targets:** own-voice TTS cloning (XTTS-class) + LPCNet-domain stretching; fall back gracefully if pre-ALS voice samples are inadequate.
- Reuse: our LPCNet build lives in **WSL native `~/`** (Esafenet CDG encrypts D:-drive WSL writes — see `studies/20250612_wairagkar/AGENTS.md`); lag-63 frame alignment fix and mel-eval utilities in `studies/20250612_wairagkar/code/` apply directly.

---

## 4. fogg26 — full reproduction brief

### 4.1 Architecture & training (Methods §4.4)
Transformer **decoder** (encoder-style blocks, no cross-attention):
- Per-participant input projection (SU models omit it): pretraining = **linear + tanh**; held-out adaptation = linear + tanh **or** 2-layer FFN (intermediate 512). Neural dims per participant range **128–768** → projected to hidden **512**.
- **Patch embedding:** concat 4 consecutive 20 ms steps → linear → stride 4 (i.e., 20 ms → 80 ms; phoneme prediction every 80 ms). Scaling analyses used patch size/stride 5.
- Blocks: self-attention + GLU-SiLU FFN (2× expansion), pre-RMSNorm, **RoPE**, residual scaling ×1/√(2L) on attn-out and FFN-down projections; init N(0, 1/√fan-in).
- Output: 41 classes (39 phonemes + silence + blank) via **CTC** + **Intermediate-CTC** aux loss (shared head; SU-8-block at blocks 2/4/6; 16-block at 4/8/12; 24-block at 8/12/16/20).
- Regularization: **stochastic depth** (linear ramp, pmax 0.2 SU / 0.3 MU, surviving blocks rescaled 1/(1−p)); dropout on sublayer outputs + attention scores.
- Augmentation (on-the-fly): time-warp (1–5 control points, stretch ~ N(1, 0.1) clamped [0.5, 2.0]); time-mask (span 20–40 bins, 40–60% of trial, learned mask vector); channel-mask (0–10% electrodes); white noise σ 0.5. Plus Gaussian smoothing σ 40 ms at train AND inference.
- Optimizers: **AdamW + Muon** (Muon on all 2D weights except the phoneme output projection; 500k-batch runs used AdamW 2e-4 / Muon 6e-4 max lr), cosine schedule + warmup, grad clip 10, bf16. Batch 64 (48 in scaling analyses). **Trials > 75 s excluded explicitly "to limit GPU VRAM"** — keep this rule.
- **Weight averaging** over the last 2/3 of validation checkpoints (greedy add if val PER doesn't worsen) instead of best-checkpoint selection.
- Depth/budget: SU = 8 blocks; MU = 12–24 blocks. Full 6-user MU = 24 blocks × **500,000 batches**. Sanctioned reductions: Fig. S7 (100k batches ≈ within ~1 WER point of 500k for SU), Table S6 (12 blocks / 300k batches for the dedup analysis), Fig. 4 (16 blocks; 100k/200k/400k/500k for pool sizes n=1/2/3/4, patch 5, batch 48).
- Official compute: **6 machines** (Ubuntu 22.04, 1–2 GPUs each: RTX 4090 / 5090 / 6000 Pro Blackwell; 128–512 GB RAM each). We have 1× 5060 Ti — hence the budget reductions above; expect a multi-day overnight run for one MU model at 12 blocks / 100–300k batches.

### 4.2 Language-model pipeline (the part that is a gift to us)
Two interchangeable phoneme→word pipelines, same underlying **5-gram LM trained on OpenWebText2, 125,000-word vocabulary from CMUdict** (same LM family we already hold on disk — see §6):
1. **WFST/Kaldi TLG** (`github.com/Neuroprosthetics-Lab/nejm-brain-to-text`, live): unpruned 5-gram ≈ **350 GB RAM**, OPT-6.7B fp16 rescoring ≈ 13 GB VRAM. **DO NOT USE on this machine** (violates the repo's RAM ceiling; our mmap decoders exist precisely for this).
2. **NEW KenLM + Flashlight-Text LexiconDecoder** (`github.com/Neuroprosthetics-Lab/phoneme-to-words-lm`, live with real code: `flashlight_text/`, `ngram/`, `phoneme_to_words_lm/`, `examples/`, `sweep/`, `env_setup.sh`): mmap trie **< 10 GB RAM**, < 10 s load, mean **0.316 s/trial** latency, 3.11× faster than WFST, accuracy within 0.27 WER points (8.81% vs 9.08% aggregate on 9,616 trials). n-best (≤100) rescored by **Qwen3.5-4B bf16 (~8–10 GB VRAM)**, optional LoRA finetune, hotword biasing, Optuna-TPE sweep support. Paper's own words: "making it suitable for deployment on **a single consumer GPU**."

Operational notes for us:
- This pipeline is Python-native and should run on Windows directly — a path around our WSL2/OpenFST constraint for first-pass decoding.
- Qwen3.5-4B needs downloading (~8 GB) — it is smaller than our OPT-6.7B/Qwen3-8B rescorers, so it fits the 16 GB card, but still run it **as a decoupled process from acoustic training** (repo Hard Prohibition #1).
- The 5-gram we hold (`studies/20230831_willett` sibling, `languageModel_5gram.tar.gz`, 38,245,502,441 B) is the Willett-2023 competition LM; verify whether the KenLM trie can be built from it directly or whether the OpenWebText2 5-gram must be rebuilt (§7 item 6).

### 4.3 Participant-by-participant data audit (as of 2026-11-18)

| Participant | Paper's data source | Splits used by paper | Our on-disk source | Status |
|---|---|---|---|---|
| **T12** | BrainToText **'24** Benchmark (= Willett 2023 competition data; only the two 6v arrays used, area-44 arrays dropped) | benchmark train+val merged → train; benchmark test | `D:\repositories\20261008_reproduce_20230831_willett\data\` (255.8 GB total; `competitionData/t12.2022.*.mat` + derived + tfRecords) | ✅ |
| **T15-Small** (~11k Copy-Task trials) | = public BrainToText **'25** Competition release | competition train+val → train; competition test | `D:\repositories\20261008_reproduce_20240815_card\data\` (35.45 GB; `hdf5_data_final/t15.2023.*/data_{train,val,test}.hdf5`, 45 sessions) | ✅ (verify trial-count match, §7 item 3) |
| **T15-Large** (~278,000 sentences; full multi-year incl. self-generated + pseudo-labeled trials) | Card 2026 *Nat. Med.* dataset | BrainToText '25 test set; train = everything else | — | ❌ not public |
| **T16** | Nason-Tomaszewski et al. 2026 preprint copy-task sessions (only the single 6v array used) | last Copy-Task block(s) of some sessions ≈ 10% held out | `D:\repositories\20261008_reproduce_20250821_kunz\data\` (30.71 GB; `sentenceData.zip` 3.93 GB) has **T16 attempted-speech sentence data**, but provenance differs from the Fogg sessions | ◐ needs verification (§7 item 1) |
| **T17** | Jude 2026 *Cell Reports* (only 6v + 55b arrays used) | ≈ 10% held out | `D:\repositories\20261008_reproduce_20260428_jude\data\` (95.92 GB; `Data/t17/t17.2024.*`, 12 sessions) | ✅ (we already extract 768-dim features — subset to the speech arrays, §7 item 7) |
| **T21** (new) | introduced in this preprint (6 arrays incl. 2 right-hemisphere) | session-final blocks | — | ❌ no release |
| **T22** (new) | introduced in this preprint (5 left arrays + 1 IFG; 6v+55b used) | session-final blocks | — | ❌ no release |

Paper results we can therefore target:
- ✅ **Fig. 3 analog** (SU vs MU per participant) on T12 / T15-Small / T16 / T17.
- ✅ **Fig. 4 analog** (pretraining-pool size 1→4 + frozen-decoder adaptation to the held-out user) — the paper's own analysis used pools of at most 4 users, exactly our regime.
- ✅ **Fig. 5 analog** (few-shot calibration curves: frozen MU + new user-projection vs from-scratch SU over 10–200 trials) with T21/T22 **replaced by a held-out public user**.
- ✅ KenLM-vs-WFST pipeline comparison (both repos public; WFST leg can reuse our archived mmap-decoder TSVs instead of re-running Kaldi).
- ✅ Offline continuous-finetuning simulation (Methods §4.5; code "included with the manuscript" — check the repo/zenodo attachment).
- ❌ Closed-loop online sessions (Fig. 2), T21/T22-specific results, T15-Large results (14.3% MU-vs-SU gap regime), left→right-hemisphere transfer (Table S11, needs T21 right-hemisphere arrays).

### 4.4 Suggested tiering
- **Tier 1 — infrastructure:** stand up `phoneme-to-words-lm` natively; build/verify the KenLM trie; download Qwen3.5-4B; per-participant dataset adapters harmonizing 20 ms threshold-crossing + SBP features, z-scoring, channel subsets, g2p labels; verify WER pipeline on our existing archived 20230831 logits.
- **Tier 2 — SU models:** 8-block SU per participant (hours each on the 5060 Ti); reproduce SU baselines (paper: T17 SU 65.6% WER, T15-Small SU ~30%-ish regime).
- **Tier 3 — MU model:** 12-block, 100–300k batches, 4 public users; SU-vs-MU deltas; frozen-decoder adaptation; few-shot curves; pool-size scaling n=1→4.
- **Tier 4 (stretch):** continuous-finetuning simulation; KenLM-vs-WFST comparison table.

---

## 5. GPU / disk feasibility snapshot (measured 2026-11-18)

- GPU: RTX 5060 Ti 16 GB (Blackwell, bf16 + torch.compile fine). Acoustic trainers for both papers are ≤ ~50M params; with 80 ms patching (fogg) and 50 ms training resolution (wairagkar) neither approaches the VRAM ceiling. Keep trials ≤ 75 s (fogg's own rule) and stream batches via memmap (Prohibition #1).
- Host RAM 64 GB: fine for both. fogg's KenLM trie < 10 GB; never load the 350 GB WFST grammar.
- Disk: **D: 217 GB free (89% used), C: 477 GB free.** No new downloads needed for either scoped plan except: Qwen3.5-4B (~8 GB), Whisper model (~1–3 GB), optional XTTS voice-clone model (~2–6 GB). If T21/T22/T15-Large ever land on Dryad they will likely be 100s of GB → C: only, after cleanup.
- 5060 Ti ≈ 1/3–1/4 of an RTX 5090's throughput → wairagkar26 full run ≈ 4–8 h; fogg26 SU models hours each; one 12-block/300k-batch MU run ≈ low single-digit days (run overnight, checkpoint per validation pass, resume-safe).

## 6. Reusable assets already in this repo

- `studies/20250612_wairagkar/code/`: Dryad T15 loaders, lag-63 alignment fix, mel/LPCNet feature evaluation, dataset census; LPCNet vocoder build (WSL native). The 2.0 study continues its gap story (our mel-layer grand mean 0.649 vs 2025 anchor 0.83; 2.0 reaches 0.93).
- `studies/20230831_willett/`: KenLM 3-gram pronunciation-trie decoder (1.46 GB), OPT/Qwen rescoring harness patterns, archived per-trial TSVs for LM-pipeline regression checks.
- `shared/tools/wer_from_tsv.py`, `figures/code/build_session_per_vs_wer.py`: WER tooling.
- fogg26's `phoneme-to-words-lm` (public) supersedes much of the WSL2 OpenFST path for the new study — treat it as the default first-pass decoder.
- Known ops quirks: TF 2.14 on Windows has no GPU path (2025 study ran CPU, 27 s/trial) — 2.0 is PyTorch so this constraint disappears; Esafenet CDG encrypts WSL writes on D: (build vocoders in WSL `~/`).

## 7. Open verification items (DO THESE FIRST)

1. **T16 data adequacy:** open `D:\repositories\20261008_reproduce_20250821_kunz\data\...\sentenceData.zip` and check whether T16 sentence trials are Switchboard-style copy-task with per-trial sentence labels, 20 ms binned features, and enough sessions/blocks for a ≈10% hold-out. Fogg's T16 sessions come from the (unreleased) Nason-Tomaszewski preprint — any provenance mismatch must be documented in the study README.
2. **Poll for code/data releases:** (a) `neuroprosthetics-lab/brain2voice2.0` (placeholder as of 2026-11-18); (b) the published versions of both preprints (Fogg = "All rights reserved", no data statement for T21/T22 yet; watch Dryad `10.5061/dryad.*` and the *Nat. Med.* / journal versions); (c) whether the fogg26 supplement's offline-finetuning codebase was attached to the preprint (Methods §4.5 says "included with this manuscript").
3. **T15-Small equivalence:** confirm our Card NEJM Dryad 45-session hdf5 ≈ the BrainToText '25 Competition release (~11k trials, same train/val/test splits). The fogg26 paper says they are identical.
4. **Feature-bin harmonization:** fogg26 assumes 20 ms bins of threshold crossings + SBP, z-scored over the preceding 5–20 trials. Verify each source dataset's actual bin width/feature convention (Willett T12 competition data may differ; Card/Wairagkar 10 ms vs 20 ms; Jude T17 768-dim = which arrays × which features). Write one adapter per participant with documented resampling.
5. **T17 channel subset:** Fogg uses only T17's 6v + 55b arrays; our `studies/20260428_jude` pipeline extracts 768-dim (all-array) features — subset accordingly and re-derive dims (expect 512 = 256 electrodes × 2 features).
6. **5-gram LM provenance:** determine whether the KenLM/Flashlight trie can be built from the Willett-2023 5-gram we hold or must be retrained on OpenWebText2 (125k CMUdict vocab). The `phoneme-to-words-lm` repo's `ngram/` + `env_setup.sh` should clarify the expected artifacts.
7. **Rescorer availability:** check Qwen3.5-4B weights are downloadable in our region/HF setup, and whether our `shared/tools/opt_rescore.py` harness extends to it (bf16, ~8 GB — must remain a separate process from any acoustic training).
8. **Whisper pinning:** pick and pin the Whisper checkpoint for the wairagkar26 ASR-WER proxy; note the paper's own Whisper numbers as the calibration anchor.
9. **Phoneme inventory alignment:** fogg26 uses g2p-en (g2pE), 39 phonemes + silence + blank; Willett 2023 uses its own phone set — build the mapping table during Tier-1 adapter work.

## 8. Repo-rule reminders when starting these studies

- New folders `20260630_wairagkar/` / `20260723_fogg/` with their own `AGENTS.md` + `code/README.md` (per root AGENTS.md conventions); never write into existing `results/` archives.
- Stream all training data with `numpy.memmap` / per-session iterators; never `np.concatenate` whole studies; never co-load an LLM rescorer and an acoustic trainer in one process.
- Register outcomes in `figures/reproduction_benchmark.yaml` (the YAML only — JSON auto-syncs via `figures/code/render_figure2.py`), then run the fast regression check:
  `python figures/code/build_session_per_vs_wer.py && python figures/code/build_session_cer_vs_wer.py && python shared/tools/wer_from_tsv.py studies/20230831_willett/results/20261124_import_btot24/wl_5g_opt_official.tsv && python figures/code/render_figure2.py && python figures/code/render_figure3.py`
- All code/comments/docstrings in English; NumPy-style docstrings.

---

*Sources: bioRxiv 10.64898/2026.06.30.735633 (full text retrieved 2026-11-18); bioRxiv 10.64898/2026.07.23.739430 (full text retrieved 2026-11-18); github.com/neuroprosthetics-lab/{brain2voice2.0,phoneme-to-words-lm} and github.com/Neuroprosthetics-Lab/nejm-brain-to-text (HTTP 200, contents inspected 2026-11-18); on-disk dataset audit of the eight sibling repos under `D:\repositories\` (2026-11-18).*
