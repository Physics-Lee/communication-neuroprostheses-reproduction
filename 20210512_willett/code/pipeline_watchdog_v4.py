# -*- coding: utf-8 -*-
"""训练流水线看门狗 v4（detached，Blocks 尾段自动收尾 + 自动生成复现报告）。

相对 v3：末尾新增 m4_report.py（汇总两划分全部指标 → docs/20261020_复现报告.md → README/PROGRESS 收口），
使用户要求的"明早 8:45 前完成"提前到当晚 ~24:00 落地。m4_report 幂等，重跑安全。

前置状态（2026-10-20 10:09 由 v2 交代）：HeldOutBlocks 训练已在独立进程运行
（result/train_HeldOutBlocks.log），预计 ~22:00 出 model_step100000.pt。

流程：
1. 守护 Blocks 训练：进程在→等；进程死且无 checkpoint 且日志停止刷新→独立拉起重启（从批 0）
2. checkpoint 出现 → 等 GPU 空闲 → infer.py（Blocks 推理）
3. step5_eval_from_errcounts.py 汇总 val CER/WER（对照官方 5.32/23.28，容差 ±2pp CER）
4. step6_make_kaldimat.py 生成 kaldiMat（仅 val 句）
5. step6_decode_ours.py bigram 解码（WSL latgen，CPU）
6. fig_perday_cer_ours.py 出官方 vs 自训逐日图
7. m4_report.py 生成复现报告并收口 README/PROGRESS
全部产物落盘后看门狗退出；明早 09:05 会话按门检查应为"已完成，直接结束"。

心跳约定：只写「看门狗」事件行，不算会话心跳。
"""
import os
import subprocess
import time

PY = r'/c/Users/user\.conda\envs\reproduce_20210512_willett\python.exe'
REPO = r'.\20210512_willett'
RUNS = os.path.join(REPO, 'data', '20210512_willett', 'runs')
BLOCKS_CKPT = os.path.join(RUNS, 'HeldOutBlocks_synth', 'model_step100000.pt')
PROGRESS = os.path.join(REPO, 'result', 'PROGRESS.md')
BLOCKS_LOG = os.path.join(REPO, 'result', 'train_HeldOutBlocks.log')
INFER_LOG = os.path.join(REPO, 'result', 'infer_HeldOutBlocks.log')


def beat(msg):
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M")} | 看门狗v3 | {msg}\n')
    print(msg, flush=True)


def log_fresh_seconds(max_age=1800):
    try:
        return time.time() - os.path.getmtime(BLOCKS_LOG) < max_age
    except OSError:
        return False


def train_alive():
    try:
        out = subprocess.check_output(
            ['powershell', '-NoProfile', '-Command',
             "(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train.py' }).Count"],
            stderr=subprocess.DEVNULL).decode().strip()
        return int(out or '0') > 0
    except Exception:
        return False


def gpu_busy():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            stderr=subprocess.DEVNULL).decode()
        return int(out.strip().split()[0]) > 2000
    except Exception:
        return True


def launch_detached_train():
    ps = ("Start-Process -WindowStyle Hidden -FilePath '%s' -ArgumentList "
          "'%s\\code\\torch_port\\train.py','--partition','HeldOutBlocks','--synth' "
          "-RedirectStandardOutput '%s' -RedirectStandardError '%s.err' -WorkingDirectory '%s'"
          % (PY, REPO, BLOCKS_LOG, BLOCKS_LOG, REPO))
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=False)
    time.sleep(30)


def run_step(cmd, log_path):
    with open(log_path, 'w', encoding='utf-8') as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO)
    return r.returncode


def main():
    beat('v4 启动（替代 v3）：守护 Blocks 训练直至 checkpoint → 全链 → 自动生成 M4 复现报告')
    guard_restarts = 0
    while not os.path.isfile(BLOCKS_CKPT):
        time.sleep(300)
        if os.path.isfile(BLOCKS_CKPT):
            break
        if not train_alive() and not log_fresh_seconds():
            guard_restarts += 1
            beat(f'Blocks 训练进程已死且无 checkpoint（第 {guard_restarts} 次）→ 独立拉起重启（从批 0）')
            launch_detached_train()
            if guard_restarts >= 3:
                beat('重启达 3 次上限，看门狗退出，等 09:05 会话人工介入')
                return
    beat('Blocks checkpoint 出现；等训练进程退出（GPU 释放）')
    while gpu_busy():
        time.sleep(120)
    beat('GPU 空闲：开始 infer.py（HeldOutBlocks 推理）')
    rc = run_step([PY, os.path.join(REPO, 'code', 'torch_port', 'infer.py'),
                   '--partition', 'HeldOutBlocks',
                   '--runDir', os.path.join(RUNS, 'HeldOutBlocks_synth')], INFER_LOG)
    beat(f'infer.py 退出码 {rc}（日志 result/infer_HeldOutBlocks.log）')
    if rc != 0:
        beat('推理失败，看门狗退出，等会话介入')
        return

    rc = run_step([PY, os.path.join(REPO, 'code', 'step5_eval_from_errcounts.py'),
                   os.path.join(RUNS, 'HeldOutBlocks_synth', 'step5_inference'), 'HeldOutBlocks'],
                  os.path.join(REPO, 'result', 'eval_blocks_watchdog.log'))
    beat(f'step5 汇总退出码 {rc}（日志 result/eval_blocks_watchdog.log）')

    rc = run_step([PY, os.path.join(REPO, 'code', 'step6_make_kaldimat.py'),
                   '--inferDir', os.path.join(RUNS, 'HeldOutBlocks_synth', 'step5_inference'),
                   '--cvPart', 'HeldOutBlocks'],
                  os.path.join(REPO, 'result', 'kaldimat_blocks_watchdog.log'))
    beat(f'kaldiMat 生成退出码 {rc}')

    rc = run_step([PY, os.path.join(REPO, 'code', 'step6_decode_ours.py'),
                   '--runName', 'HeldOutBlocks_synth', '--cvPart', 'HeldOutBlocks'],
                  os.path.join(REPO, 'result', 'm3_own_decode_HeldOutBlocks.log'))
    beat(f'Blocks bigram 解码退出码 {rc}（日志 result/m3_own_decode_HeldOutBlocks.log）')

    # per-day raw CER 图（官方 vs 自训，两划分各一栏；CPU，几分钟）
    rc = run_step([PY, os.path.join(REPO, 'code', 'fig_perday_cer_ours.py')],
                  os.path.join(REPO, 'result', 'fig_perday_watchdog.log'))
    beat(f'per-day 图退出码 {rc}（产出 result/figs/perday_cer_official_vs_ours.png）')

    # 自动生成复现报告并收口 README/PROGRESS（幂等；m4_report 缺 Blocks 产物会拒绝，--skipMissing 不加）
    rc = run_step([PY, os.path.join(REPO, 'code', 'm4_report.py')],
                  os.path.join(REPO, 'result', 'm4_report_watchdog.log'))
    beat(f'复现报告生成退出码 {rc}（日志 result/m4_report_watchdog.log）')
    beat('Blocks 全链 + M4 报告自动收尾完成。看门狗退出。')


if __name__ == '__main__':
    main()
