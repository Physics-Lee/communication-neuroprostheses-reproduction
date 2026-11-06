# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-2 reproduction: run the OFFICIAL pretrained seed
model (Dryad pretrained_seed_model/ckpt-12500, TF checkpoint) on the
released tfrecords using the official NeuralDecoder package, entirely
offline (static decoder, no CORP recalibration).

Three arms:
  seed  : the 11 seed sessions' own test splits (checkpoint sanity check)
  norec : online_evaluation_data/no_recalibration test splits
  recal : online_evaluation_data/recalibration test splits

All eval arms map to input layer 10 (the last seed session t5.2022.09.01),
which is what the participant's decoder state would have been at the start
of the evaluation period without any recalibration.

Run with the TF-2.14 env:
  cd <repo>/open_source_code_for_reference/20231201_fan_CORP/NeuralDecoder
  python <repo>/code/tier2_seed_model_inference.py
Writes <repo>/result/tier2_seed_model_inference.json
"""
import json
import os
import sys

import numpy as np

NSD_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../open_source_code_for_reference/20231201_fan_CORP/NeuralDecoder"))
sys.path.insert(0, NSD_ROOT)
DATA = ("./20231201_fan/data/20231201_fan/"
        "CORP_data_release/")
OUT = ("./20231201_fan/result/"
       "tier2_seed_model_inference.json")

SEED_SESSIONS = [
    "t5.2022.05.18", "t5.2022.05.23", "t5.2022.05.25", "t5.2022.06.01",
    "t5.2022.06.03", "t5.2022.06.06", "t5.2022.06.08", "t5.2022.06.13",
    "t5.2022.06.15", "t5.2022.06.22", "t5.2022.09.01"]
EVAL_SESSIONS = [
    "t5.2022.09.29", "t5.2022.10.06", "t5.2022.10.18", "t5.2022.10.25",
    "t5.2022.10.27", "t5.2022.11.01", "t5.2022.11.03", "t5.2022.12.08",
    "t5.2022.12.15", "t5.2023.02.28"]

MODEL_YAML = {
    "nUnits": 512, "subsampleFactor": 1, "weightReg": 1e-5, "actReg": 0.0,
    "bidirectional": False, "dropout": 0.4, "trainable": True, "nLayers": 2,
    "stack_kwargs": {"kernel_size": 1, "strides": 2},
}


def make_args(data_dirs, sessions, layer_map, val_probs, test_on_train=False):
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "mode": "inference",
        "gpuNumber": "0",
        "outputDir": os.path.join(
            "./20231201_fan/tmp_run"),
        "loadDir": DATA + "pretrained_seed_model",
        "loadCheckpointIdx": 12500,
        "smoothInputs": 1,
        "smoothKernelSD": 2,
        "learnRateStart": 0.01, "learnRateEnd": 0.0,
        "learnRateDecaySteps": 100000, "learnRatePower": 1.0,
        "trainableInput": 1, "trainableBackend": 1,
        "seed": -1, "batchesPerSave": 0, "batchesPerVal": 50,
        "nBatchesToTrain": 100000, "batchSize": 48, "inputScale": 1.0,
        "inferenceOutputFileName": "", "inputLayerForInference": None,
        "gradClipValue": 10, "lossType": "ctc", "normLayer": False,
        "randomWalkAxis": -1, "warmUpSteps": 0,
        "model": MODEL_YAML,
        "dataset": {
            "name": "handwriting",
            "sessions": sessions,
            "dataDir": data_dirs,
            "datasetToLayerMap": layer_map,
            "datasetProbability": [1] * len(sessions),
            "datasetProbabilityVal": val_probs,
            # the eval arms ship EMPTY test tfrecords -- all evaluation
            # trials live in the train files, so inference must read them
            # via the package's own testOnTrain switch
            "testOnTrain": test_on_train,
            "syntheticDataDir": None, "syntheticMixingRate": 0.0,
            "whiteNoiseSD": 1.2, "constantOffsetSD": 0.6, "randomWalkSD": 0.02,
            "staticGainSD": 0.0, "nInputFeatures": 192, "timeWarpFactor": 0.0,
            "nClasses": 31, "maxSeqElements": 500, "bufferSize": 192,
        },
    })


def run_arm(arm, data_dirs, sessions, layer_map, val_probs,
            test_on_train=False):
    from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
    args = make_args(data_dirs, sessions, layer_map, val_probs,
                     test_on_train)
    nsd = NeuralSequenceDecoder(args=args)
    inf = nsd.inference()
    ed = np.asarray(inf["editDistances"])
    tl = np.asarray(inf["trueSeqLengths"])
    cer = float(np.sum(ed) / np.sum(tl))
    return {
        "arm": arm,
        "n_trials": int(len(ed)),
        "pooled_cer_pct": round(100 * cer, 2),
        "mean_trial_cer_pct": round(100 * float(np.mean(ed / tl)), 2),
    }


def main():
    os.makedirs("./20231201_fan/tmp_run",
                exist_ok=True)
    seed_dir = DATA + "seed_model_training_data/tfrecords"
    report = {}
    report["seed_selftest"] = run_arm(
        "seed", [seed_dir] * 11, SEED_SESSIONS, list(range(11)),
        [1] * 11)
    print("seed self-test:", report["seed_selftest"], flush=True)
    # The 11 seed sessions occupy input layers 0-10 (all restored from the
    # checkpoint); the 10 evaluation sessions ride input layer 10 (the last
    # seed session t5.2022.09.01 -- the decoder's state at the start of the
    # evaluation period, i.e. static, no recalibration) and are the only
    # datasets actually inferred (val_probs 0 for the seed sessions).
    for arm, sub in [("eval_no_recalibration", "no_recalibration"),
                     ("eval_recalibration_static", "recalibration")]:
        ev_dir = DATA + f"online_evaluation_data/{sub}/tfrecords"
        report[arm] = run_arm(
            arm, [seed_dir] * 11 + [ev_dir] * 10,
            SEED_SESSIONS + EVAL_SESSIONS,
            list(range(11)) + [10] * 10,
            [0] * 11 + [1] * 10, test_on_train=True)
        print(arm, report[arm], flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()
