# `studies/20260428_jude/code/` — Code Guide

Reproduces the **Jude et al., *Cell Reports* 2026** intended-speech study (participant
T17, long-standing anarthria/LIS) on the `nejm-brain-to-text` GRU stack: 12
conversational sessions with a 125,000-word vocabulary.
Study-level context: [`../README.md`](../README.md).

## Public commands

Run these commands from the repository root in the environment required by the study. Use `--help` on an entry point for its arguments.

| Public command | Role |
| --- | --- |
| [`prepare_data.py`](prepare_data.py) | Prepare the released intended-speech trials. |
| [`export_sessions.py`](export_sessions.py) | Export one recording day to the common session format. |
| [`train.py`](train.py) | Train the intended-speech model. |
| [`infer.py`](infer.py) | Export intended-speech logits. |
| [`evaluate.py`](evaluate.py) | Evaluate greedy phoneme errors for intended speech. |
| [`decode.py`](decode.py) | Run the indexed WFST decoder in WSL. |

Run the indexed WFST command in WSL with the compiled `shared/tools/wenet_lattice.cc` decoder. Supply the graph, words, exported trials, metadata, and the run-specific decoding options.

Use [`record_study_run.py`](../../../shared/tools/record_study_run.py) to name scored outputs from a completed command. See the [eight-study naming map](../../../docs/20261124_repository_migration/naming_map.md) for the result and data paths.

The export command writes `data/processed/t17/<release>_<date>.h5` for one session. `python shared/tools/validate_session.py <file> --source trials` re-reads every trial from the publisher's files. The format is defined in [`shared/session_format.py`](../../../shared/session_format.py).

## Pipeline order

```text
(Dryad RedisMat .mat blocks)
   └─ inspect_jude.py / scan_jude_blocks.py ── survey structure & block inventory
   └─ convert_jude.py ──→ hdf5 in the nejm-brain-to-text layout (768-dim features)
        └─ train_jude.py ──→ GRUDecoder (16 GB VRAM recipe)
             └─ eval_jude.py ──→ greedy PER by session/vocab/loop
             └─ dump_jude_logits.py ──→ per-trial logits
                  └─ shared decode chain (KenLM 3-gram / TLG 5-gram / OPT / Qwen)
```

## File-by-file

| File | Role |
|---|---|
| `export_sessions.py` | Writes one recording day to the common session format (`data/processed/t17/dryad_vq83bk481_<date>.h5`): each sentence cue of the sentence blocks `convert_jude.py` converts, with the released 10 ms features. `t17.2024.03.25` and `t17.2024.05.23` hold no such blocks and are listed in `EXCLUDED_SESSIONS`. |
| `build_split_manifest.py` | Writes `data/splits/dryad_vq83bk481__validation_blocks.csv` from the exported session files and checks the validation sentences against the archived evaluation list. |
| `inspect_jude.py` | Prints the variables/shapes/cue structure of one RedisMat `.mat` block. |
| `scan_jude_blocks.py` | Inventory scan over all `.mat` blocks: block numbers, trial counts, cue samples, feature shapes. |
| `convert_jude.py` | Per-block extraction following the official `mat_to_tfrecord.py` (trial windowing on 10 ms neural timestamps, 768-dim features with 256→384 channel padding). |
| `train_jude.py` | Trains the 5-layer GRU phoneme decoder on the converted conversational data. |
| `eval_jude.py` | Greedy CTC PER evaluation with breakdowns by session, vocabulary (50w/125k), and loop (open/closed). |
| `dump_jude_logits.py` | Dumps per-trial CTC log-probs for the run's val partition → hdf5 for the decode/rescore chain. |
