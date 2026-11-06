# patch_table_r14.py — 录入 S9k（T17 LM）+ 01.14（2026-10-11 16:35）
p = r'.\20260316_jude\result\复现对照表.md'
s = open(p, encoding='utf-8').read()

old1 = '| t17.2024.06.04（65 训/20 测，含句复制块 6） | 同上 | 训练中 | blocks 1,2,3,5,6（官方值）；num_test 20 适配 | result/runs/m3_t17_20240604_torch/ |'
new1 = '''| t17.2024.06.04（65 训/20 测，含句复制块 6） | 同上 | **best 0.2670**（8351 批/0.79h） | blocks 1,2,3,5,6（官方值） | result/runs/m3_t17_20240604_torch/ |

### T17 post-LM（5-gram）

| 会话 | raw CER | post-LM CER | post-LM WER | 论文口径（WER 多数 <15%） | 产物 |
|---|---|---|---|---|---|
| t17.2024.06.03 | 0.2141 | **0.0571** | **0.1149** | ✔ 达标 | result/logs/20261011_1510_S9k_lm_t17.log |
| t17.2024.06.04 | 0.2670 | 0.1680 | 0.3810 | ✗（差会话） | 同上 |
| t17.2024.07.09 | 0.2129 | 0.0997 | 0.2705 | ✗（"多数<15%"需 8 会话全跑判定，当前 3 中 1） | 同上 |'''
assert old1 in s
s = s.replace(old1, new1, 1)

old2 = '## G1 速度重算'
new2 = '''| t18.2025.01.14（60 训/20 测） | <12% | **best 0.1420**（7001 批/1.05h） | 略超口径，如实记录 | result/runs/m3_t18_20250114_torch/final_report.json |
| t18.2025.01.15（78 训/20 测） | <12% | 训练中 | blocks 5-14 官方值；剔除 {5:[8],7:[9]} | result/runs/m3_t18_20250115_torch/ |

## G1 速度重算'''
assert old2 in s
open(p, 'w', encoding='utf-8').write(s.replace(old2, new2, 1))
print('TABLE_OK')
