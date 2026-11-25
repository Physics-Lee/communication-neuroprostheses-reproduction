# D:\workspace 清理

**预计启动：2026 年 11 月 25 日 09:00（UTC+8）。** 等夜间任务结束后再移动文件。

1. 清点 `D:\workspace\communication_neuroprostheses_archive`，把复现仍需的文件移到仓库中对应研究目录。先核对 Jude 的模型权重和测试 logits，以及 Willett 2023 的 5-gram 语言模型压缩包是否仍有用途。
2. 暂时改名旧归档目录，用仓库里的路径验证八篇研究的数据读取和评估入口。验证后恢复旧归档目录名。
3. 列出确认可以删除的路径和预计可释放空间，确认后删除。清理范围限定在 `D:\workspace`。WinDirStat 的 `<Hardlinks>` 用于统计，同一文件在仓库里还有硬链接时，删旧路径不会释放该文件占用的空间。
