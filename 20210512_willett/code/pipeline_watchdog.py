# -*- coding: utf-8 -*-
"""训练流水线看门狗 v2（detached，守护模式）。

新增（相对 v1）：若 Trials 训练进程死亡且最终 checkpoint 未出现 → 以独立进程重新拉起
train.py（从头训——train.py 无断点续训，重启即从批 0，如实接受）。
心跳约定：只写带「看门狗」标记的事件行（不定期刷），不算会话心跳——
续跑会话的心跳门请只看含「心跳」二字的行。

流程：
1. 守护 Trials 训练：进程在→等；进程死且无 model_step100000.pt→独立拉起训练
2. checkpoint 出现 → 等 GPU 空闲 → infer.py 推理
3. GPU 空闲 → 启动 HeldOutBlocks 训练（独立进程）→ 看门狗退出
"""
import os
import subprocess
import time

PY = r'/c/Users/user\.conda\envs\reproduce_20210512_willett\python.exe'
REPO = r'.\20210512_willett'
RUNS = os.path.join(REPO, 'data', '20210512_willett', 'runs')
TRIALS_CKPT = os.path.join(RUNS, 'HeldOutTrials_synth', 'model_step100000.pt')
PROGRESS = os.path.join(REPO, 'result', 'PROGRESS.md')
TRAIN_LOG = os.path.join(REPO, 'result', 'train_HeldOutTrials.log')
BLOCKS_LOG = os.path.join(REPO, 'result', 'train_HeldOutBlocks.log')
INFER_LOG = os.path.join(REPO, 'result', 'infer_HeldOutTrials.log')


def beat(msg):
    with open(PROGRESS, 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%d %H:%M")} | 看门狗 | {msg}\n')
    print(msg, flush=True)


def log_fresh_seconds(max_age=1800):
    try:
        return time.time() - os.path.getmtime(TRAIN_LOG) < max_age
    except OSError:
        return False


def train_alive():
    """train.py 进程存在（按命令行匹配）。"""
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


def launch_detached(partition, log_path):
    ps = ("Start-Process -WindowStyle Hidden -FilePath '%s' -ArgumentList "
          "'%s\\code\\torch_port\\train.py','--partition','%s','--synth' "
          "-RedirectStandardOutput '%s' -RedirectStandardError '%s.err' -WorkingDirectory '%s'"
          % (PY, REPO, partition, log_path, log_path, REPO))
    subprocess.run(['powershell', '-NoProfile', '-Command', ps], check=False)
    time.sleep(30)


def main():
    beat('v2 启动（守护模式）：守护 Trials 训练直至 checkpoint')
    guard_restarts = 0
    while not os.path.isfile(TRIALS_CKPT):
        time.sleep(300)
        if os.path.isfile(TRIALS_CKPT):
            break
        if not train_alive() and not log_fresh_seconds():
            guard_restarts += 1
            beat(f'Trials 训练进程已死且无 checkpoint（第 {guard_restarts} 次）→ 独立拉起重启（从批 0）')
            launch_detached('HeldOutTrials', TRAIN_LOG)
            if guard_restarts >= 3:
                beat('重启达 3 次上限，看门狗退出，等 09:05 会话人工介入')
                return
    beat('Trials checkpoint 出现；等训练进程退出（GPU 释放）')
    while gpu_busy():
        time.sleep(120)
    beat('GPU 空闲：开始 infer.py（HeldOutTrials 推理）')
    with open(INFER_LOG, 'w', encoding='utf-8') as lf:
        r = subprocess.run([PY, os.path.join(REPO, 'code', 'torch_port', 'infer.py'),
                            '--partition', 'HeldOutTrials',
                            '--runDir', os.path.join(RUNS, 'HeldOutTrials_synth')],
                           stdout=lf, stderr=subprocess.STDOUT, cwd=REPO)
    beat(f'infer.py 退出码 {r.returncode}（日志 result/infer_HeldOutTrials.log）；等 GPU 后启动 Blocks')
    time.sleep(60)
    while gpu_busy():
        time.sleep(120)
    launch_detached('HeldOutBlocks', BLOCKS_LOG)
    beat('HeldOutBlocks 训练已独立启动（日志 result/train_HeldOutBlocks.log）。看门狗退出。')


if __name__ == '__main__':
    main()
