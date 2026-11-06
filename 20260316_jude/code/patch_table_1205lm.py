# patch_table_1205lm.py — 录入 S9m（12.05 post-LM），T18 全 8 会话 LM 曲线闭环（2026-10-12 00:35）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()
old = '| 汇总（T18 七会话） |'
new = '''| t18.2024.12.05（20 句） | 0.1022 | **0.0392** | **0.0948** | 首日第二部分 | result/logs/20261011_2240_S9m_lm_1205.log |
| 汇总（T18 全 8 会话） |'''
assert old in s
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print('TABLE_OK')
