# patch_table_s9h.py — 录入 S9h（02.04/02.05）post-LM 结果 + T17 状态（2026-10-11）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()
old = '说明 | 单句：LM 把 2/38 错字全纠正'
new = '''t18.2025.02.04（20 句） | 0.0864 | **0.0444** | **0.1111** | 同上 | result/logs/20261011_0635_S9h_lm_0204_0205.log |
| t18.2025.02.05（20 句） | 0.1421 | **0.0581** | **0.1360** | 同上 | 同上 |
| 汇总 | 五会话 raw CER 5.3-14.2% → post-LM CER 0-5.8%（LM 平均把字符错误再砍 ~60%）；WER 0-13.6% vs 论文 <7%（两会话达标、两会话略超；差异来源：num_test 适配的随机划分 vs 官方按 block 评估、单会话训练 vs 官方跨 block、bf16/patch 序偏差） | | | | |
| 说明 | 单句：LM 把 2/38 错字全纠正'''
assert old in s, 'old string not found'
s = s.replace(old, new, 1)
open(p, 'w', encoding='utf-8').write(s)
print('TABLE_OK')
