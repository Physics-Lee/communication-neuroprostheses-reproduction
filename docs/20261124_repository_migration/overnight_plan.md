# 16-hour overnight plan / 16 小时夜间计划

2026-11-24

## English

**Finish the naming audit first.** By morning, a reader should be able to open any of the eight notebooks and see exactly what to download, where to put it, which command to run, and where the converted data will appear. Then validate the common `.h5` format on two contrasting studies before rolling it out.

| Time | Work | Acceptance check |
| --- | --- | --- |
| **Hours 0–4** | Reconcile the file tree, actual directories, script inputs, and tutorial instructions across all eight studies. Put data setup at the start of each notebook. Show the archive location, extraction destination, conversion command, and output path. Explain the two direct-ZIP cases. | Every notebook points to a real input path and an executable command. Generated session filenames include the dataset ID, for example `data/processed/t12/dryad_x69p8czpq_2022-05-19.h5`. |
| **Hours 4–6** | Define the first version of the common HDF5 format and implement a shared reader, writer, and validator. Use one file per session. Record trial IDs, source locations, neural features, timing, and task targets. Keep split assignments in separate manifests so different experiments can reference the same trials. | A single validation command checks the file structure and traces each trial to its source. |
| **Hours 6–9** | Export Willett 2023’s May 19 session. Read the 20 trials through the shared reader and run inference with the same checkpoint and recording-day index. | Reference sentences, labels, predictions, and error counts match trial by trial. The total remains **245 errors over 637 reference phonemes**. |
| **Hours 9–11** | Export one Wairagkar 2025 session using the same format. Check neural and audio features against their respective time axes. | Trial counts, frame timing, target features, and silence intervals agree with the source. |
| **Hours 11–14** | Apply the validated format to the other six studies, starting with one representative session each. Check Fan’s adaptation split, Card’s calibration split, Kunz’s condition and fold labels, and the two Jude studies’ task metadata. Start batch export with the sessions used in the main results. | Each study has a validation result. The export inventory shows which sessions passed, failed, or are still pending. |
| **Hours 14–16** | Rerun the eight zero-download notebooks and rebuild Figures 2 and 3. Write a short example showing how a new study can register its source, export one session, and verify it with the same command. | Published figure values match the archive, and the onboarding example exercises the shared validator. |

The **minimum morning handoff** is eight corrected notebook introductions, both pilot reports, and a per-study export inventory. The target is one validated `.h5` session for every study plus as much of the main evaluation data as the window allows. Process one session at a time and check disk space before batch export. Fix any pilot differences in trial identity, timing, or scores before expanding the conversion.

## 中文

**先完成命名验收。** 到第二天早上，读者打开任意一篇研究的教程，都能马上找到要下载的文件、放置目录、转换命令和输出位置。随后用两篇任务类型不同的研究验证统一的 `.h5` 格式，再推广到其余研究。

| 时间 | 工作 | 验收标准 |
| --- | --- | --- |
| **第 0–4 小时** | 逐篇核对文件树、磁盘上的实际目录、脚本输入路径和教程说明。把数据准备放在八份教程的开头，写清压缩包放哪、解压到哪、运行什么命令、输出到哪。说明两种脚本直接读取 ZIP 的情况。 | 每份教程都指向真实的输入路径和可运行的命令。生成的 session 文件名包含数据集 ID，例如 `data/processed/t12/dryad_x69p8czpq_2022-05-19.h5`。 |
| **第 4–6 小时** | 定义统一 HDF5 格式的第一版，实现共用的读取、写入和校验工具。每个 session 一个文件，记录试次 ID、原始文件位置、神经特征、时间信息和任务目标。划分单独存入清单，方便不同实验引用同一批试次。 | 同一个校验命令能检查文件结构，并把每条试次追溯到来源。 |
| **第 6–9 小时** | 导出 Willett 2023 的 5 月 19 日 session。通过共用工具读回 20 条试次，用同一个 checkpoint 和录制日索引重新推理。 | 参考句、标签、预测和错误数逐条与归档一致。总计仍为 **637 个参考音素中有 245 个错误**。 |
| **第 9–11 小时** | 用同一格式导出 Wairagkar 2025 的一个 session。沿各自的时间轴核对神经特征和音频特征。 | 试次数、帧时间、目标特征和静音区间均与原始数据一致。 |
| **第 11–14 小时** | 将通过试点的格式用到其余六篇，每篇先验收一个代表性 session。重点检查 Fan 的适应试次划分、Card 的校准划分、Kunz 的实验条件和折号，以及两篇 Jude 研究各自的任务字段。开始批量导出，优先覆盖主结果用到的 session。 | 每篇都有校验结果。导出清单逐个列明通过、失败和待处理的 session。 |
| **第 14–16 小时** | 重跑八份零下载教程，重建 Fig. 2 和 Fig. 3。写一个简短的新研究接入示例，演示登记来源、导出一个 session，以及如何用同一命令校验。 | 图表数值与归档一致。新研究接入示例实际调用共用校验工具。 |

**次日最低交付**是八份已修正的教程开头、两份试点报告和一份按研究列出的导出进度清单。目标是八篇各有至少一个通过校验的 `.h5` session，并在剩余时间内尽可能覆盖主评估数据。按 session 逐个处理，批量导出前核对磁盘空间。试点若发现试次、时间或评分发生变化，先解决差异再扩大转换范围。
