"""Run inference on the test split with best.pt and dump per-trial logits to disk.

Logits are stored in the model's native class order (blank=30 in the last
position). They feed the WSL-side lm_decoder 5-gram decoding, which moves the
blank column to index 0 (official rearrange_speech_logits) before feeding the
decoder.

Notes
-----
Written 2026-10-10. CLI: ``python dump_logits.py [ckpt_dir] [npz]`` (defaults:
run ``m1_t18_20241204_torch`` and its source npz).
"""
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import TypingDataset, gauss_smooth
from model import InputNet, TypingGRU

CKPT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r'.\20260316_jude\result\runs\m1_t18_20241204_torch')
NPZ = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r'.\20260316_jude\data\20260316_jude\Data\t18\t18.2024.12.04\Typing\m1_torch_data.npz')

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(CKPT_DIR / 'best.pt', map_location=dev, weights_only=True)
model = TypingGRU().to(dev)
model.load_state_dict(ck['model'])
inet = InputNet().to(dev)
inet.load_state_dict(ck['input_net'])
model.eval()
inet.eval()

ds = TypingDataset(NPZ, 'test')
all_logits, all_true, all_lens = [], [], []
with torch.no_grad():
    idx_all = list(range(len(ds)))
    for s in range(0, len(idx_all), 64):
        idx = idx_all[s:s + 64]
        x, il, _, tl = ds.make_batch(idx, augment=False)
        logits = model(inet(x.to(dev), training=False), training=False)
        for j, i in enumerate(idx):
            n = int(il[j])
            all_logits.append(logits[j, :n].float().cpu().numpy())
            end = int(np.argwhere(ds.transcriptions[i] == 0)[0, 0]) if (ds.transcriptions[i] == 0).any() else len(ds.transcriptions[i])
            all_true.append(''.join(chr(int(c)) for c in ds.transcriptions[i][:end]))

np.savez_compressed(CKPT_DIR / 'test_logits.npz',
                    logits=np.array(all_logits, dtype=object),
                    true_sentences=np.array(all_true, dtype=object))
print(f'DUMP_DONE: {len(all_logits)} trials -> {CKPT_DIR / "test_logits.npz"} | step={ck.get("step")} cer={ck.get("cer")}')
