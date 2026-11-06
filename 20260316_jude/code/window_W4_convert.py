# window_W4_convert.py — 任意 T18 会话：官方 mat_to_tfrecord 转换 + 导出 npz（2026-10-10）
# 用法：reproduce_20260316_jude 环境 python window_W4_convert.py <session> <block_csv> <num_test> <rm_spec>
# rm_spec 格式："5:9,12,16;7:9" （block:trial列表；空串=无剔除）
import sys
from pathlib import Path

REF = Path(r'.\20260316_jude\open_source_code_for_reference\20260316_jude_typingBCI')
sys.path.insert(0, str(REF / 'utils'))

session = sys.argv[1]
blocks = [int(b) for b in sys.argv[2].split(',')]
num_test = int(sys.argv[3])
trials_to_remove = {}
if len(sys.argv) > 4 and sys.argv[4]:
    for part in sys.argv[4].split(';'):
        b, tls = part.split(':')
        trials_to_remove[int(b)] = [int(x) for x in tls.split(',')]

import mat_to_tfrecord  # noqa: E402

participant = 't17' if session.startswith('t17') else 't18'
session_path = str(Path(r'.\20260316_jude\data\20260316_jude\Data', participant, session, 'Typing'))
args = {
    'session_mat_path': session_path,
    'block_nums': blocks,
    'num_test_trials': num_test,
    'trials_to_remove': trials_to_remove,
    'channels_to_exclude': [],
    'channels_to_zero': [],
    'include_thresh_crossings': True,
    'include_spike_power': True,
    'spike_pow_max': 50000,
    'z_score_data': True,
    'global_std': True,
    'bin_compression_factor': 2,
    'save_path': str(Path(session_path, 'tfdata_20ms')),
}
mat_to_tfrecord.main(args)
print('CONVERT_DONE')
