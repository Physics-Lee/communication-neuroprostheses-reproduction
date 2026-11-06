# patch_table_t17.py — 录入 T17 两会话（2026-10-11）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()
old = '## G1 速度（110 cpm / 22 wpm）与 G4-G9：未开始（待 M3 多会话与 LM 解码）'
new = '''## G3 T17 句子解码 raw CER

| 会话 | 论文口径 | 复现值 | 口径说明 | 产物 |
|---|---|---|---|---|
| t17.2024.06.03（T17 首个校准日，120 训/20 测） | 首日 raw CER 较高（Fig 3b 首日点最差，WER 多数 <15% 为后续会话口径） | **best 0.2141**（4851 批早停/1.41h） | T17 首日；实测 384 通道=768 特征（README 称 512，不符） | result/runs/m3_t17_20240603_torch/final_report.json |
| t17.2024.06.04（65 训/20 测，含句复制块 6） | 同上 | 训练中 | blocks 1,2,3,5,6（官方值）；num_test 20 适配 | result/runs/m3_t17_20240604_torch/ |

## G1 速度（110 cpm / 22 wpm）与 G4-G9：未开始（G1 重算待做；G4-G9 留后续窗口）'''
assert old in s, 'old not found'
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print('TABLE_OK')
