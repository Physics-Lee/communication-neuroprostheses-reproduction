# patch_table_s9g.py — 录入 S9g 三会话 post-LM 结果（2026-10-11）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()
old = '说明 | LM 把 2/38 错字全部纠正'
new = '''t18.2025.01.21（20 句） | 0.0820 | **0.0268** | **0.0826** | T18 全 block WER <7%（图 3c） | result/logs/20261011_0440_S9g_lm_all5.log |
| t18.2025.01.22（20 句） | 0.0943 | **0.0177** | **0.0569** | 同上 | 同上 |
| 说明 | 单句：LM 把 2/38 错字全纠正；批量（01.21/01.22 各 20 句）：raw→LM 增益显著（CER 降至 2-3%，WER 5.7-8.3%），与论文 WER<7% 口径基本吻合；02.04/02.05 补跑中'''
assert old in s, 'old string not found'
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
print('TABLE_OK')
