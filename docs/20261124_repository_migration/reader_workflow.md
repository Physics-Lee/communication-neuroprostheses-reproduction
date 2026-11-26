# Reproduce a study from a fresh clone

First time on this machine? [fresh_machine_setup.md](fresh_machine_setup.md) maps all eight studies' downloads and paths on one page, with the WSL toolchain and a suggested first-pass order.

Clone the repository and register its Python environment as a Notebook kernel.

```powershell
git clone https://github.com/Physics-Lee/communication-neuroprostheses-reproduction.git
Set-Location communication-neuroprostheses-reproduction
conda env create -f environment.yml
conda activate bci-repro
python -m ipykernel install --user --name bci-repro --display-name bci-repro
```

Open the study's Notebook with that kernel in VS Code or JupyterLab. Start with **Prepare the data** to get the publisher archives, extract them, and validate the first session HDF5. The terminal commands in **Reproduce and score** train or load the model and calculate the study's metrics. The Notebook code cells also recompute the archived metrics from the files shipped with the repository.

For the Willett 2023, Kunz 2025, and Jude 2026b conversions, install the English pronunciation resources once in `bci-repro`.

```powershell
python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

| Study Notebook | Fresh-run metrics | Linux or WSL step |
| --- | --- | --- |
| [Willett 2021](../../studies/20210512_willett/tutorial.ipynb) | CER and WER | Lazy bigram graph decoding |
| [Willett 2023](../../studies/20230831_willett/tutorial.ipynb) | PER and WER | Indexed speech 5-gram decoding |
| [Fan 2023](../../studies/20231201_fan/tutorial.ipynb) | Matched held-out WER | Python replay runs on Windows |
| [Card 2024](../../studies/20240815_card/tutorial.ipynb) | PER and WER | Indexed speech 5-gram decoding |
| [Wairagkar 2025](../../studies/20250612_wairagkar/tutorial.ipynb) | Mel correlation and spectral distortion | LPCNet audio synthesis |
| [Kunz 2025](../../studies/20250821_kunz/tutorial.ipynb) | PER and WER | Indexed speech 5-gram decoding |
| [Jude 2026a](../../studies/20260316_jude/tutorial.ipynb) | CER, WER, and online typing speed | Separate typing 5-gram runtime |
| [Jude 2026b](../../studies/20260428_jude/tutorial.ipynb) | PER and WER | Indexed speech 5-gram decoding |

## Where files go

Every Notebook starts under `studies/<study>/`. Use the same checkout from Windows and WSL. For example, WSL's `wslpath` prints the Linux path for a Windows checkout under `D:\repositories`.

```powershell
wsl.exe --exec wslpath -a -u (Get-Location).Path
```

| Location within a study | Contents |
| --- | --- |
| `data/raw/<release>/` | Publisher ZIP or TAR, saved with the `<release>__` filename prefix |
| `data/extracted/<release>/` | Publisher files with their archive directory structure |
| `data/processed/<participant>/<release>_<session>.h5` | Common session files. Use `export_sessions.py --list` to check coverage |
| `data/derived/<release>/<format>/<variant>/` | Model-specific training inputs, such as the native HDF5 or TFRecords |
| `data/splits/`, `data/checkpoints/`, `data/language_models/` | Split manifests, released checkpoints, and language models |
| `results/<run_id>/`, `log/<run_id>/` | Newly generated results and logs. Native checkpoints and intermediates go in `runtime/` |

A new run ID uses `<yyyymmdd>_<release>__<format>__<decoder>__<experiment>`. If you repeat a command, choose a different experiment label. When a command accepts an explicit runtime directory, pass `results/<run_id>/runtime`. For a scored output that needs publishing, use [`record_study_run.py`](../../shared/tools/record_study_run.py) with an existing trial CSV or TSV or summary JSON. The resulting run contains `run.yaml` and the files you supplied as `trials.csv` or `summary.json`.

The Notebook commands assume inputs are under `studies/`. For a dedicated data disk, copy the study IDs and their `data/` trees under a directory of your choice, set `BCI_DATA_ROOT` to that directory, and use the equivalent location in the manual extraction and validation commands. Set the variable in each shell that uses it, using the shell's path notation.

## Speech word error rate in WSL

The four speech studies below use the same memory-mapped C++ decoder. Keep the publisher graph in the Willett 2023 study and use it from all four studies. On the [Willett 2023 Dryad release](https://doi.org/10.5061/dryad.x69p8czpq), download `languageModel_5gram.tar.gz` and save it as `studies/20230831_willett/data/raw/dryad_x69p8czpq/dryad_x69p8czpq__languageModel_5gram.tar.gz`. Plan for about 83 GB for the archive and graph together, plus the index and decoding outputs.

From the repository root in WSL, extract the two members used for 1-best and N-best decoding.

```bash
base=studies/20230831_willett/data
lm="$base/language_models/wfst_5gram"
mkdir -p "$lm"
tar -xzf "$base/raw/dryad_x69p8czpq/dryad_x69p8czpq__languageModel_5gram.tar.gz" \
  -C "$lm" ./speech_5gram/lang_test/TLG.fst ./speech_5gram/lang_test/words.txt
```

**Decode from a Linux-filesystem copy of the graph.** The decoder reaches `TLG.fst` through random page reads of the memory map. On a Windows checkout reached through `/mnt/<drive>` (drvfs/9P), those reads are slow enough that a full decode stalls for hours. Copy the extracted graph to the WSL home filesystem once and decode against that copy; keep the study's copy for provenance.

```bash
mkdir -p "$HOME/tg5"
cp "$lm/speech_5gram/lang_test/TLG.fst" "$lm/speech_5gram/lang_test/words.txt" "$HOME/tg5/"
```

The first decode that passes `--build-index` writes `TLG.fst.idx` beside the graph it reads (about 22 minutes from the WSL home filesystem; do not attempt the build against `/mnt/<drive>`). Keep the `.idx` beside `TLG.fst` for later runs. The first trial batch also fills the WSL page cache with the graph states the beam search touches (up to about 20 GB); later decodes reuse that cache and run several times faster, so decode one study at a time rather than in parallel.

Build [`wenet_lattice.cc`](../../shared/tools/wenet_lattice.cc) once in WSL with a built [Kaldi](https://github.com/kaldi-asr/kaldi) tree and its matching OpenFST. Set `kaldi` to the location of your Kaldi tree. This command assumes the standard `src/lib` and `tools/openfst` locations from that build.

```bash
kaldi="$HOME/kaldi"
fst="$kaldi/tools/openfst"
mkdir -p "$HOME/bin"
g++ -O3 -std=c++17 -DHAVE_CXXABI_H -DKALDI_DOUBLEPRECISION=0 \
  -I"$kaldi/src" -I"$fst/include" shared/tools/wenet_lattice.cc \
  -L"$kaldi/src/lib" -L"$fst/lib" -Wl,-rpath,"$kaldi/src/lib:$fst/lib" \
  -lkaldi-decoder -lkaldi-lat -lkaldi-fstext -lkaldi-util -lkaldi-matrix \
  -lkaldi-base -lfst -ldl -pthread -o "$HOME/bin/wenet_lattice"
```

Export logits on Windows or with a Linux `bci-repro` environment. The table gives the study ID, the notebook's example training run suffix, and the file under its `runtime/` directory. Use the date you supplied when training.

| Study ID | Run suffix after the date and underscore | Logits file |
| --- | --- | --- |
| `20230831_willett` | `dryad_x69p8czpq__hdf5__ctc_greedy__reader` | `t12_logits.hdf5` |
| `20240815_card` | `dryad_dncjsxm85__hdf5__ctc_greedy__reader` | `t15_logits_fixed.hdf5` |
| `20250821_kunz` | `dryad_gf1vhhn1j__hdf5__ctc_greedy__t12_mixed_reader` | `kunz_logits.hdf5` |
| `20260428_jude` | `dryad_vq83bk481__hdf5__ctc_greedy__reader` | `jude_logits.hdf5` |

From the repository root in WSL, activate a Linux Python environment with `h5py`, `numpy`, `pyyaml`, and `editdistance`. With Conda installed inside WSL, you can create it with these commands.

```bash
conda create -n bci-wfst python=3.11 pip -y
conda activate bci-wfst
python -m pip install h5py numpy pyyaml editdistance
```

Set `study`, `run_suffix`, and `logits_name` from the same row of the table. This example uses Willett 2023. The first decode builds `TLG.fst.idx`. Keep it beside `TLG.fst` for later runs.

```bash
study=20230831_willett
run_suffix=dryad_x69p8czpq__hdf5__ctc_greedy__reader
logits_name=t12_logits.hdf5
run_date=$(date +%Y%m%d)
logits="studies/$study/results/${run_date}_$run_suffix/runtime/$logits_name"
graph="$HOME/tg5"
work=$(mktemp -d)

python shared/tools/export_wfst_trials.py "$logits" "$work/trials" "$work/meta.tsv"
python shared/tools/decode_wfst.py --binary "$HOME/bin/wenet_lattice" \
  --graph "$graph/TLG.fst" --words "$graph/words.txt" --build-index \
  --trials-dir "$work/trials" --meta "$work/meta.tsv" --output "$work/final.tsv" \
  --nbest-prefix "$work/nbest"
python shared/tools/wer_from_tsv.py --no-sidecar "$work/final.tsv"
python shared/tools/record_study_run.py "$study" --trials "$work/final.tsv" \
  --input-format npy --decoder wfst_5gram --label reader_wer
```

The scorer prints the summed word errors divided by the summed reference words. Keep `final.tsv` and `nbest.tsv` under the new run's `runtime/` if you want the original decoder outputs. For the benchmark variants that use a second-pass language model, run [`opt_rescore.py`](../../shared/tools/opt_rescore.py) on an N-best TSV in a separate process once acoustic inference has finished.

## Handwriting bigram

The Willett 2021 publisher archive contains the `BigramLM` graph. With OpenFST 1.8.4 already built, compile the repository's [lazy bigram decoder](../../studies/20210512_willett/code/latgen_lazy.cc.txt) inside WSL. Set `fst` to your OpenFST install prefix.

```bash
fst="$HOME/kaldi/tools/openfst-1.8.4"
mkdir -p "$HOME/bin"
g++ -O2 -std=c++17 -x c++ -I"$fst/include" \
  studies/20210512_willett/code/latgen_lazy.cc.txt -o "$HOME/bin/latgen_lazy2" \
  -L"$fst/lib" -lfst -Wl,-rpath,"$fst/lib"
```

Return to the Notebook's PowerShell commands to export matrices from the trained model and calculate WER and CER with this binary.

## Voice synthesis

Build the pretrained [Xiph LPCNet](https://github.com/xiph/LPCNet) vocoder in WSL following its build instructions, with the checkout in `~/lpcnet`. Verify that `~/lpcnet/lpcnet_demo` is executable. Return to PowerShell and run the Wairagkar Notebook's `evaluate.py` command. It uses WSL for the vocoder and the local Python environment for Mel-band correlations.

## Typing word decoding

In the [Jude typing Dryad release](https://doi.org/10.5061/dryad.cz8w9gjjk), download `LanguageModels.zip` and save it as `studies/20260316_jude/data/raw/dryad_cz8w9gjjk/dryad_cz8w9gjjk__LanguageModels.zip`. From the repository root in WSL, extract the pruned typing graph and its vocabulary.

```bash
study=studies/20260316_jude
lm="$study/data/language_models/wfst_5gram"
mkdir -p "$lm"
unzip -j -n "$study/data/raw/dryad_cz8w9gjjk/dryad_cz8w9gjjk__LanguageModels.zip" \
  LanguageModels/Typing5gramPunc/TLG.fst LanguageModels/Typing5gramPunc/words.txt -d "$lm"
```

In a separate WSL Python 3.10 environment (`py310lm`), run `python shared/tools/setup_upstream.py --study 20230831_willett` if you still need the pinned Willett 2023 `speechBCI` checkout. If Windows created it as a sparse checkout, restore the native source files in WSL with `git -C studies/20230831_willett/open_source_code_for_reference/speechBCI sparse-checkout disable`. Windows cannot materialize the `runtime/server/x86` Git symlinks (`bin`, `decoder`, `frontend`, `grpc`, `kaldi`, `patch`, `utils`, `websocket` all point into `runtime/core`), so relink them before building.

```bash
lm=studies/20230831_willett/open_source_code_for_reference/speechBCI/LanguageModelDecoder
cd "$lm/runtime/server/x86"
for name in bin decoder frontend grpc kaldi patch utils websocket; do
  rm -rf "$name"; ln -s "$lm/runtime/core/$name" "$name"
done
```

Build the `lm_decoder` extension using its `LanguageModelDecoder/README.md` and PyTorch 1.13.1 runtime (`python setup.py install` from `runtime/server/x86` with that interpreter; `CMAKE_POLICY_VERSION_MINIMUM=3.5` reconciles the bundled gflags with CMake 4, and its FetchContent downloads of boost/glog/gflags need a working WSL network). On GCC 15 the bundled openfst 1.6.5 templates trip the new `-Wtemplate-body` diagnostic as an error; add `add_compile_options(-Wno-template-body)` near the top of `runtime/server/x86/CMakeLists.txt` (and `-Wno-template-body -Wno-error` in that file's openfst `CPPFLAGS=` string) before building. This decoder's class order and graph differ from the four shared speech pipelines. After training, supply the directory with the `t17` and/or `t18` subdirectories to the typing decoder.

```bash
conda activate py310lm
run_date=$(date +%Y%m%d)
python studies/20260316_jude/code/decode.py --input-dir \
  "studies/20260316_jude/results/${run_date}_dryad_cz8w9gjjk__npz__ctc_greedy__multisession/runtime/runs_multisess"
```
