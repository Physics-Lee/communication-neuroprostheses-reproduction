# Windows workspace cleanup review

2026-11-25, UTC+8

The 09:00 one-time task ran in Linux and could not see the Windows D drive. The checks below were performed on the Windows computer. The old archive was restored to its original path after verification. The nine reviewed directories were removed after approval.

## Files moved into the repository

121 files totaling 166.01 GiB moved on the same NTFS volume. File ID, byte size, link count, and modification time matched before and after the move. Git ignores the relocated files.

| Study | Files | Repository destination |
| --- | ---: | --- |
| Willett 2021 | 8 | `data/raw/dryad_wh70rxwmv/` and `data/checkpoints/dryad_wh70rxwmv/gpt2-xl/` |
| Willett 2023 | 8 | `data/raw/dryad_x69p8czpq/`, including the 35.62 GiB `languageModel_5gram.tar.gz` package |
| Card 2024 | 24 | `data/raw/dryad_dncjsxm85/` and missing run packages under `data/checkpoints/dryad_dncjsxm85/` |
| Wairagkar 2025 | 1 | The control dataset ZIP under `data/raw/dryad_2280gb64f/` |
| Kunz 2025 | 11 | `data/raw/dryad_gf1vhhn1j/` |
| Jude 2026a | 68 | The raw data and language-model archives under `data/raw/dryad_cz8w9gjjk/` and 64 checkpoint and logits files alongside their existing single-session and multisession reports |
| Jude 2026b | 1 | The original `Data.zip` under `data/raw/dryad_vq83bk481/` |

The Willett 2023 5-gram package is a retained source for restoring the graph. The extracted graph stays under `data/language_models/wfst_5gram/`. The 166.01 GiB is relocated space on the same disk.

## Verification with the old archive unavailable

The old archive was temporarily renamed for these checks, with both path override variables cleared. The eight study tutorials passed their path and command checks and ran all 24 Python cells. The pinned upstream checkout checks passed. Running the common validator on each tutorial's example read the trials again from the publisher files in the repository.

| Study | Example trials traced and identical |
| --- | ---: |
| Willett 2021 | 102 |
| Willett 2023 | 200 |
| Fan 2023 | 60 |
| Card 2024 | 288 |
| Wairagkar 2025 | 305 |
| Kunz 2025 | 100 |
| Jude 2026a | 80 |
| Jude 2026b | 200 |
| **Total** | **1,335** |

The Willett 2023 archived evaluation rescored to 1,277 word errors over 5,500 reference words. All 32 moved Jude checkpoint containers opened, and all 32 logits archives contained paired references and 31-class logits. A moved Jude multisession checkpoint loaded on CPU and its logits archive held 20 test entries. A copied Jude single-session checkpoint ran through the public inference entry point and regenerated the test output for the one trial in that input split. The Windows checks cover these paths, source files, tutorial evaluations, and representative inference.

## WSL

The current Windows session receives `Wsl/E_ACCESSDENIED` when listing or querying distributions. Direct access via the Ubuntu WSL share was also unavailable. The WSL filesystem has no verified file or space totals in this review.

The Willett 2023 exporter defaults to a `~/tlv` directory for decoder input arrays and trial metadata. Check that directory in the environment that ran the exporter. Windows files opened via `/mnt/d/` are the same files on D. Files copied to a native WSL home directory occupy space in the distribution's filesystem and need a separate read-only audit. Full WSL decoder runs were outside the verification above.

## Verified deletion candidates

The paths below are under `D:\workspace`. The Jude entries are alternate hard links to the same files now in the repository. They are included for directory cleanup, with zero estimated file data reclaimed.

| Candidate absolute path | Evidence | Estimated space reclaimed |
| --- | --- | ---: |
| `D:\workspace\communication_neuroprostheses_archive\20261009_reproduce_20260316_jude\data\unzipped\Data` | 462 hard links to repository inputs, including the typing intermediates now in `data/derived/` | 0 GiB |
| `D:\workspace\communication_neuroprostheses_archive\20261009_reproduce_20260316_jude\data\unzipped\LanguageModels\Typing5gramPunc` | 6 hard links to repository language-model files | 0 GiB |
| `D:\workspace\communication_neuroprostheses_archive\20261008_reproduce_20260428_jude\data\unzipped\Data` | 285 hard links to repository inputs | 0 GiB |
| `D:\workspace\communication_neuroprostheses_archive\20261008_reproduce_20230831_willett\data\unzipped\competitionData` | 63 files match the repository competition data by SHA-256 | 6.47 GiB when both Willett paths are removed |
| `D:\workspace\communication_neuroprostheses_archive\20261008_reproduce_20230831_willett\data\20230616_willett_BtoT24\competitionData` | A second old hard link for each of the 63 files in the preceding row | Included in the preceding row |
| `D:\workspace\communication_neuroprostheses_archive\20261008_reproduce_20250821_kunz\data\unzipped\sentenceData` | 16 files match the repository sentence data by SHA-256 | 3.68 GiB |
| `D:\workspace\communication_neuroprostheses_archive\20261009_reproduce_20250612_wairagkar\data\20250612_wairagkar\Dryad_T15_all` | All 125 file sizes and CRC-32 values match the retained publisher ZIP | 29.62 GiB when both Wairagkar paths are removed |
| `D:\workspace\communication_neuroprostheses_archive\20261009_reproduce_20250612_wairagkar\data\unzipped\Dryad_T15_all` | A second old hard link for each of the 125 files in the preceding row | Included in the preceding row |
| `D:\workspace\communication_neuroprostheses_archive\20261022_reproduce_20231201_fan\data\unzipped\CORP_data_release` | All 99 file sizes and CRC-32 values match the retained publisher ZIP | 13.61 GiB |
| **Total for the nine listed directories** | Each NTFS file counted once, with a 4 KiB allocation unit | **53.38 GiB** |

The estimate excludes directory and filesystem metadata. The rest of the old archive is retained, including unverified files and legacy build directories.

## Cleanup outcome

After the user approved the list, all nine directories were checked against the inventory and deleted. Windows reported **53.38 GiB** reclaimed, with D drive free space rising from **103.70 GiB** to **157.09 GiB**. All eight tutorial HDF5 examples passed the post-cleanup source-file check.
