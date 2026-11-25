"""Replay the Fan CORP held-out experiment and save its trial transcripts.

Notes
-----
The adaptation follows the repository's v7 experiment using recorded
pseudo-labels on the adaptation pool. Test trials are reserved before
adaptation. The character language model and CTC beam search are reused
from the v9 experiment and trained on seed-session text.
"""

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import editdistance
import numpy as np
import tensorflow as tf
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_data_path


FIGURE_CODE = Path(__file__).resolve().parents[3] / "figures" / "code"
FIELDS = (
    "session", "online_trial_index", "heldout_trial_index", "reference",
    "online_text", "greedy_text", "beam_text", "reference_words",
    "online_word_errors", "greedy_word_errors", "beam_word_errors",
)


def load_session(archive, session, normalize_key, display_text, words):
    """Index released trials by an unambiguous normalized reference.

    Parameters
    ----------
    archive : zipfile.ZipFile
        Original Dryad CORP release.
    session : str
        Evaluation session identifier.
    normalize_key, display_text, words : callable
        Cue-matching and transcription normalization functions.

    Returns
    -------
    trials : list of tuple
        Released cue keys, indices, references, and online hypotheses in
        display and training forms.
    unique : dict
        Online records indexed by unique, scoreable cue keys.
    """
    fields = ("sentences", "lm_decoded")
    member = f"CORP_data_release/online_evaluation_data/recalibration/mat/{session}.mat"
    data = loadmat(io.BytesIO(archive.read(member)), variable_names=fields, squeeze_me=True)
    trials = []
    for index, (reference, decoded) in enumerate(zip(*(data[k] for k in fields), strict=True)):
        reference = display_text(reference)
        trials.append((normalize_key(reference), index, reference, display_text(decoded),
                       normalize_key(decoded)))
    multiplicities = Counter(key for key, *_ in trials)
    unique = {key: (index, reference, decoded, training_text)
              for key, index, reference, decoded, training_text in trials
              if multiplicities[key] == 1 and words(reference)}
    return trials, unique


def training_step(model, layer, learning_rate, clip):
    """Build one graph-compiled CTC update.

    Parameters
    ----------
    model, layer : tf.keras.layers.Layer
        Shared acoustic decoder and current session input layer.
    learning_rate, clip : float
        Adam learning rate and global gradient norm threshold.

    Returns
    -------
    callable
        Train step accepting smoothed features and 1-based character IDs.
    """
    trainable = list(layer.trainable_variables) + list(model.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer.build(trainable)

    @tf.function(input_signature=(
        tf.TensorSpec([None, 192], tf.float32), tf.TensorSpec([None], tf.int32),
    ))
    def step(features, labels):
        """Apply the original v7 loss and gradient clipping."""
        length = tf.size(labels, out_type=tf.int64)
        indices = tf.stack((tf.zeros(length, dtype=tf.int64), tf.range(length)), axis=1)
        sparse = tf.SparseTensor(indices, labels - 1, tf.stack((tf.constant(1, tf.int64), length)))
        with tf.GradientTape() as tape:
            logits = model(layer(features[None], training=True), training=True)
            loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss_v2(
                sparse, logits, None, [tf.shape(logits)[1]], logits_time_major=False,
                blank_index=-1,
            ))
            tf.debugging.assert_all_finite(loss, "CTC loss became nonfinite")
        gradients, _ = tf.clip_by_global_norm(tape.gradient(loss, trainable), clip)
        optimizer.apply_gradients(zip(gradients, trainable, strict=True))
        return loss

    return step


def cuda_decoder(original, seed):
    """Convert the loaded Keras layers to cuDNN GRUs.

    Parameters
    ----------
    original : NeuralSequenceDecoder
        Restored TensorFlow decoder.
    seed : int
        Seed for GPU training.

    Returns
    -------
    predict, make_step : callable
        Inference with optional Keras parity check and per-session training
        step factory.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def linear(layer):
        weights, bias = layer.get_weights()
        result = nn.Linear(weights.shape[0], weights.shape[1], device="cuda")
        result.weight.data.copy_(torch.from_numpy(weights.T))
        result.bias.data.copy_(torch.from_numpy(bias))
        return result

    def gru(layer):
        kernel, recurrent, bias = layer.get_weights()
        width = recurrent.shape[0]
        order = np.r_[width:2 * width, :width, 2 * width:3 * width]
        result = nn.GRU(kernel.shape[0], width, batch_first=True, device="cuda")
        for parameter, values in (
            (result.weight_ih_l0, kernel.T), (result.weight_hh_l0, recurrent.T),
            (result.bias_ih_l0, bias[0]), (result.bias_hh_l0, bias[1]),
        ):
            parameter.data.copy_(torch.from_numpy(values[order]))
        return result

    class Decoder(nn.Module):
        """Implement the seed decoder's layer order and shared time dropout."""

        def __init__(self):
            super().__init__()
            self.inputs = nn.ModuleList(map(linear, original.inputLayers))
            self.recurrent = nn.ModuleList(map(gru, original.model.rnnLayers))
            self.output = linear(original.model.dense)
            self.initial = nn.Parameter(torch.as_tensor(
                original.model.initStates.numpy(), device="cuda",
            ).unsqueeze(0))

        def forward(self, data, layer, training):
            value = self.inputs[layer](data[None])[:, ::2]
            state = self.initial
            for recurrent in self.recurrent:
                if training:
                    value = value * F.dropout(torch.ones_like(value[:, :1]), p=0.4)
                value, _ = recurrent(value, state)
                state = None
            return self.output(value)[0]

    decoder = Decoder()

    def verify_and_predict(layer, smoothed, reference=None):
        """Decode one trial and optionally compare against Keras logits."""
        with torch.no_grad():
            logits = decoder(torch.as_tensor(smoothed, device="cuda"), layer, False).cpu().numpy()
        if reference is not None:
            np.testing.assert_allclose(logits, reference, atol=2e-4, rtol=2e-4)
        return logits

    def make_step(layer):
        """Build a bounded-memory GPU update for the current session."""
        parameters = [
            *decoder.inputs[layer].parameters(), *decoder.recurrent.parameters(),
            *decoder.output.parameters(), decoder.initial,
        ]
        optimizer = torch.optim.Adam(parameters, lr=5e-5, eps=1e-7)

        def step(features, labels):
            optimizer.zero_grad(set_to_none=True)
            logits = decoder(torch.as_tensor(features, device="cuda"), layer, True)
            logits = torch.cat((logits[:, -1:], logits[:, :-1]), dim=1)
            target = torch.as_tensor(labels, device="cuda", dtype=torch.int32)
            loss = F.ctc_loss(logits.log_softmax(1).unsqueeze(1), target,
                              [logits.shape[0]], [len(labels)], blank=0, reduction="sum")
            if not torch.isfinite(loss).item():
                raise ValueError("CTC loss became nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()

        return step

    return verify_and_predict, make_step


def reproduce(study, output, seed, sessions, steps, backend):
    """Train the held-out adaptation experiment and score both outputs.

    Parameters
    ----------
    study : pathlib.Path
        Original Fan reproduction checkout with source modules and release.
    output : pathlib.Path
        Fresh directory for the trial CSV, summary, and runtime files.
    seed, sessions, steps : int
        Random seed, number of ordered sessions, and updates per session.
    backend : {"tensorflow", "torch-cuda"}
        Native TensorFlow or converted CUDA decoder.
    """
    sys.path.insert(0, str(study / "code"))
    from tier3_corp_sim_official_labels import (
        CHAR_DEF, EVAL_SESSIONS, NeuralSequenceDecoder, forward_logits, gather_trials,
        gauss_smooth_np, greedy_decode, norm, text_to_ids,
    )
    from tier3_corp_sim_v6 import make_args
    from tier3_corp_sim_v9_lm_labels import ctc_beam_lm, train_lm_from_seed_sessions

    sys.path.insert(0, str(FIGURE_CODE))
    from extract_online_trials import display_text, words

    output.mkdir(parents=True, exist_ok=False)
    args = make_args()
    args.outputDir = str(output / "runtime")
    data = study_data_path("20231201_fan", "data/extracted/CORP_data_release", checkout=study)
    args.loadDir = str(data / "pretrained_seed_model")
    args.dataset.dataDir = [str(data / "seed_model_training_data/tfrecords")] * 11 + [
        str(data / "online_evaluation_data/recalibration/tfrecords"),
    ] * len(EVAL_SESSIONS)
    args.seed = seed
    model = NeuralSequenceDecoder(args=args)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    model.tfValDatasets = [dataset.with_options(options) for dataset in model.tfValDatasets]
    lm, seed_sentences = train_lm_from_seed_sessions(model)
    print(f"Restored seed decoder and fitted LM on {seed_sentences} seed sentences.", flush=True)
    seed_bank = []
    for session_index in range(0, 11, 2):
        features, labels, _ = gather_trials(model, session_index)
        for index in range(0, len(features), 4):
            seed_bank.append((gauss_smooth_np(features[index]).astype(np.float32),
                              np.asarray(labels[index][labels[index] > 0], dtype=np.int32)))
    probe = np.linspace(-0.5, 0.5, 23 * 192, dtype=np.float32).reshape(23, 192)
    probe_logits = forward_logits(model, 10, probe)
    if backend == "torch-cuda":
        cuda_predict, cuda_step = cuda_decoder(model, seed)
        cuda_predict(10, gauss_smooth_np(probe).astype(np.float32), probe_logits)
        print("Verified that the CUDA decoder matches the restored Keras logits.", flush=True)
    print(f"Seed replay pool has {len(seed_bank)} trials.", flush=True)
    raw = study_data_path("20231201_fan", "data/raw/CORP_data_release.zip", checkout=study)
    summary = {
        "seed": seed, "backend": backend, "steps_per_session": steps, "heldout_per_session": 6,
        "replay_fraction": 0.30, "learning_rate": 5e-5, "gradient_clip": 10.0,
        "beam": 12, "topk": 9, "alpha": 0.15, "lm_seed_sentences": seed_sentences,
        "adaptation_labels": "authors' online text on the non-holdout pool",
        "sessions": [],
    }
    with zipfile.ZipFile(raw) as archive, (output / "trial_transcripts.csv").open(
        "x", encoding="utf-8", newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for offset, session in enumerate(EVAL_SESSIONS[:sessions]):
            start = time.monotonic()
            layer_index = 11 + offset
            features, _, cues = gather_trials(model, layer_index)
            released, online_by_cue = load_session(archive, session, norm, display_text, words)
            cues = [norm(display_text(cue)) for cue in cues]
            if Counter(cues) != Counter(key for key, *_ in released):
                raise ValueError(f"Online and TFRecord cue inventories differ in {session}")
            hold = range(len(features) - 6, len(features))
            if any(cues[i] not in online_by_cue for i in hold):
                raise ValueError(f"Cannot identify every held-out online trial in {session}")
            pool = []
            for feat, cue in zip(features[:hold.start], cues[:hold.start], strict=True):
                if cue in online_by_cue:
                    labels = text_to_ids(online_by_cue[cue][3])
                    if labels:
                        pool.append((gauss_smooth_np(feat).astype(np.float32),
                                     np.asarray(labels, dtype=np.int32)))
            if not pool:
                raise ValueError(f"Empty CORP adaptation pool in {session}")
            print(f"{session}: {len(pool)} adaptation trials and {len(hold)} held out", flush=True)
            step = (cuda_step(layer_index) if backend == "torch-cuda" else
                    training_step(model.model, model.inputLayers[layer_index], 5e-5, 10.0))
            for i in range(steps):
                bank = seed_bank if np.random.rand() < 0.30 else pool
                step(*bank[np.random.randint(len(bank))])
                if (i + 1) % 100 == 0:
                    print(f"{session}: {i + 1}/{steps} updates in {time.monotonic() - start:.1f}s",
                          flush=True)
            scores = []
            for i in hold:
                online_index, ref, online, _ = online_by_cue[cues[i]]
                logits = (
                    cuda_predict(layer_index, gauss_smooth_np(features[i]).astype(np.float32))
                    if backend == "torch-cuda" else forward_logits(model, layer_index, features[i])
                )
                greedy = display_text("".join(CHAR_DEF[k - 1] for k in greedy_decode(logits)))
                beam = display_text("".join(
                    CHAR_DEF[k - 1] for k in ctc_beam_lm(logits, lm, alpha=0.15, beam=12, topk=9)
                ))
                ref_words = words(ref)
                errors = tuple(editdistance.eval(ref_words, words(h)) for h in (online, greedy, beam))
                writer.writerow(dict(zip(FIELDS, (
                    session, online_index, i, ref, online, greedy, beam, len(ref_words), *errors,
                ), strict=True)))
                scores.append((len(ref_words), *errors))
            denominators, online_errors, greedy_errors, beam_errors = map(sum, zip(*scores, strict=True))
            record = {"session": session, "matched_trials": len(scores),
                      "reference_words": denominators, "online_word_errors": online_errors,
                      "greedy_word_errors": greedy_errors, "beam_word_errors": beam_errors}
            summary["sessions"].append(record)
            stream.flush()
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print(record, flush=True)


def main():
    """Run the paired trial reproduction with a configurable seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="Original Fan reproduction checkout")
    parser.add_argument("--output", type=Path, required=True, help="Unused output directory")
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--sessions", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--backend", choices=("tensorflow", "torch-cuda"), default="tensorflow")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    reproduce(args.study.resolve(), args.output.resolve(), args.seed, args.sessions, args.steps,
              args.backend)


if __name__ == "__main__":
    main()
