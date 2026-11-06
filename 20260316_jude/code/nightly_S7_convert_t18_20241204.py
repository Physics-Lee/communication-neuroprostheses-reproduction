# S7 夜间流水线：M1 转换 t18.2024.12.04 blocks 5,6,7 -> tfrecord（2026-10-09）
# 严格复刻官方 notebook 25 "T18_train_model.ipynb" CELL 1 的调用方式
# （mat_to_tfrecord.main(args)，bin_compression_factor=2 -> tfdata_20ms；训练线用它，而非 *_analysis 变体）。
import sys
from pathlib import Path

REF = r'.\20260316_jude\open_source_code_for_reference\20260316_jude_typingBCI'
sys.path.insert(0, str(Path(REF, 'utils')))

import mat_to_tfrecord  # noqa: E402

session = 't18.2024.12.04'
participant = 't18'
session_path = str(Path(r'.\20260316_jude\data\20260316_jude\Data', participant, session, 'Typing'))
tfdata_path = str(Path(session_path, 'tfdata_20ms'))

# 官方 T18_SessionArgs.get_session_info('t18.2024.12.04')（notebook 25 版，原样照抄）
block_nums = [5, 6, 7]
num_test_trials = 1
trials_to_remove = {5: [9, 12, 16]}

args = {
    'session_mat_path': session_path,
    'block_nums': block_nums,
    'num_test_trials': num_test_trials,
    'trials_to_remove': trials_to_remove,
    'channels_to_exclude': [],   # 官方 Cell 1: list(range(0,0))
    'channels_to_zero': [],      # 官方 Cell 1: list(range(0,0))
    'include_thresh_crossings': True,
    'include_spike_power': True,
    'spike_pow_max': 50000,
    'z_score_data': True,
    'global_std': True,
    'bin_compression_factor': 2,
    'save_path': tfdata_path,
}

mat_to_tfrecord.main(args)
print('S7_DONE')
