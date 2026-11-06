# export_tfrecord_to_npz.py — 把官方 tfrecord（已 z-score）一次性导出为 npz，供 PyTorch 训练（2026-10-10）
# 用 reproduce_20260316_jude 环境（有 TF）运行。解析规格逐字复刻官方 brainToText_trainDecoder.py 的 datasetFeatures。
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

SESSION = sys.argv[1] if len(sys.argv) > 1 else 't18.2024.12.04'
N_FEATURES = 768  # 实测 t17.2024.06.03 与 t18 均为 384 通道×2=768（raw README 的“T17=512 features”与发布数据不符）
TFDATA = Path(r'.\20260316_jude\data\20260316_jude\Data', 't17' if SESSION.startswith('t17') else 't18', SESSION, 'Typing', 'tfdata_20ms')
OUT = TFDATA.parent / 'm1_torch_data.npz'
MAX_SEQ = 60

DATASET_FEATURES = {
    'inputFeatures': tf.io.FixedLenSequenceFeature([N_FEATURES], tf.float32, allow_missing=True),
    'newClassSignal': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'ceMask': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'seqClassIDs': tf.io.FixedLenFeature((MAX_SEQ,), tf.int64),
    'nTimeSteps': tf.io.FixedLenFeature((), tf.int64),
    'nSeqElements': tf.io.FixedLenFeature((), tf.int64),
    'transcription': tf.io.FixedLenFeature((MAX_SEQ,), tf.int64),
}


def parse(example_proto):
    return tf.io.parse_single_example(example_proto, DATASET_FEATURES)


def dump_split(split):
    files = sorted(str(x) for x in (TFDATA / split).glob('*.tfrecord'))
    ds = tf.data.TFRecordDataset(files).map(parse)
    feats, seq_ids, n_ts, n_el, trans = [], [], [], [], []
    for d in ds:
        t = int(d['nTimeSteps'].numpy())
        feats.append(d['inputFeatures'].numpy()[:t].astype(np.float32))  # 截掉 padded_batch 才有的尾部 pad（本路径无 pad，防御性截断）
        seq_ids.append(d['seqClassIDs'].numpy().astype(np.int64))
        n_ts.append(t)
        n_el.append(int(d['nSeqElements'].numpy()))
        trans.append(d['transcription'].numpy().astype(np.int64))
    print(split, ':', len(feats), 'trials; T range', min(x.shape[0] for x in feats), '-', max(x.shape[0] for x in feats),
          '; 标签长 range', min(n_el), '-', max(n_el))
    return feats, np.stack(seq_ids), np.array(n_ts, dtype=np.int64), np.array(n_el, dtype=np.int64), np.stack(trans)


def to_object_array(list_of_arrays):
    """安全构造 1-D object 数组（np.array([单个二维数组], dtype=object) 会被 numpy 升维，埋验证路径的雷）。"""
    arr = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        arr[i] = a
    return arr


if __name__ == '__main__':
    tr = dump_split('train')
    te = dump_split('test')
    np.savez_compressed(
        OUT,
        train_features=to_object_array(tr[0]), train_seqClassIDs=tr[1], train_nTimeSteps=tr[2],
        train_nSeqElements=tr[3], train_transcription=tr[4],
        test_features=to_object_array(te[0]), test_seqClassIDs=te[1], test_nTimeSteps=te[2],
        test_nSeqElements=te[3], test_transcription=te[4],
    )
    print('EXPORT_DONE ->', OUT)
