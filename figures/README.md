# Benchmark figures

The [published bundle](results/20261124_repository_migration/) contains the figures, the interactive edition, the source tables, and a build receipt with input checksums. The [figure notebook](tutorial.ipynb) reproduces the daily calculations and checks the online and offline trial coverage.

| Panel | Source and interpretation |
| --- | --- |
| 2a–b | Vocabulary estimates, released data counts, and their study annotations in the benchmark configuration. |
| 2c–d | [Selected trials from the authors' online outputs](docs/20261124_evaluation_protocol/figure2cd.md), grouped by recording day. The voice-synthesis listener result is a paper-reported median. |
| 2e–f | [Acoustic and full-pipeline comparison values](results/20261124_repository_migration/figure2ef_bars.csv). The [daily reproduction counts](results/20261124_repository_migration/figure2f_daily.csv) reproduce the equal-day WER means and sample SDs in 2f. Check selection and source columns before comparing with Figure 3. |
| 3a–b | [Matched offline evaluation points](results/20261124_repository_migration/figure3ab_session_points.csv). Each marker uses the same trials for greedy PER or CER and post-language-model WER. The table includes the available raw counts and the source of both axes. |
| 3c–d | [Host RAM and GPU VRAM values](results/20261124_repository_migration/figure3cd_memory.csv) from the benchmark annotations. Per-run monitor records are still unlinked for historical values without a verified log. |

Edit [`reproduction_benchmark.yaml`](reproduction_benchmark.yaml) for figure inputs. Study outputs and their original file names are described by the imported `run.yaml` files in `studies/`. The online daily counts are shipped with the published bundle. The [extraction instructions](docs/20261124_evaluation_protocol/figure2cd.md#reproduce-the-numbers) describe rebuilding them from the released recordings.

From the repository root with `bci-repro` active:

```bash
python figures/code/build_figures.py --output-dir figures/results/yyyymmdd_revised_manuscript
```

Use a fresh directory for a new release. To deliberately refresh the current published bundle after reviewing numeric changes:

```bash
python figures/code/build_figures.py --output-dir figures/results/20261124_repository_migration --publish
```
