# verify_and_patch_1205.py — W5 备料：核对全部 final_report.json + 录入 12.05（2026-10-11 22:35）
import json
from pathlib import Path

RUNS = Path(r'.\20260316_jude\result\runs')
print('=== 数字核对（best_cer / batches / hours）===')
rows = []
for d in sorted(RUNS.iterdir()):
    f = d / 'final_report.json'
    if f.exists():
        r = json.loads(f.read_text(encoding='utf-8'))
        rows.append((d.name, r['best_cer'], r['n_batches_trained'], r['hours']))
        print(f"{d.name}: {r['best_cer']:.4f} | {r['n_batches_trained']} 批 | {r['hours']:.2f}h")

p = Path(r'.\20260316_jude\result\复现对照表.md')
s = p.read_text(encoding='utf-8')
old = '| t18.2025.01.15（78 训/20 测） | <12% | 训练中 | blocks 5-14 官方值；剔除 {5:[8],7:[9]} | result/runs/m3_t18_20250115_torch/ |'
new = '''| t18.2025.01.15（78 训/20 测） | <12% | **best 0.1088**（8401 批/0.80h） | blocks 5-14 官方值；剔除 {5:[8],7:[9]} | result/runs/m3_t18_20250115_torch/final_report.json |
| t18.2024.12.05（首日第二部分，53 训/20 测） | <12%（首日后口径） | **best 0.1022**（3150 批早停/3.80h，句子超长 T≤8041） | blocks 1,2,3,4,6 官方值 | result/runs/m3_t18_20241205_torch/final_report.json |'''
assert old in s, 'old row not found'
p.write_text(s.replace(old, new, 1), encoding='utf-8')
print('TABLE_OK')
