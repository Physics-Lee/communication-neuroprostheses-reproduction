# patch_table_r16.py — 录入 S9l（01.14/01.15 post-LM）（2026-10-11 20:35）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()
old = 't18.2025.01.15（20 句） | 0.1421'
# 上面这行不存在——01.15 行在 G1 节前；直接在 LM 表 02.05 行后追加
old2 = '| 汇总 | 五会话 raw CER'
new2 = '''| t18.2025.01.14（20 句） | 0.1420 | **0.0886** | **0.2160** | WER<7% 为后段会话口径 | result/logs/20261011_1835_S9l_lm_0114_0115.log |
| t18.2025.01.15（20 句） | 0.1088 | **0.0556** | **0.1729** | 同上 | 同上 |
| 汇总（T18 七会话） | 五会话 raw CER'''
assert old2 in s
open(p, 'w', encoding='utf-8').write(s.replace(old2, new2, 1))
print('TABLE_OK')
