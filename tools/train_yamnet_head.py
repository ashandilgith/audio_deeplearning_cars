"""Train the car fault classifier head on YAMNet embeddings and export it for the phone app.

Input : data/car_windows.npz (E: YAMNet embeddings from tools/build_car_dataset.py)
Eval  : 3 seeds of whole-video hold-out (~20 % of each class's windows, never the
        same recording on both sides). Reports balanced accuracy, top-3,
        fault-vs-normal, video accuracy and per-class recall.
Model : standardise -> Dropout .3 -> Dense 256 ReLU -> Dropout .4 -> Dense k softmax,
        class-weighted, 40 epochs (the same head the browser runs).
Export: final head fitted on ALL videos -> docs/models/fault_classifier.json
        (classes, labels, mean, std, weights, eval report); models/classifier_report.json
"""
import base64
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from exp_car_heads import metrics, mlp  # noqa: E402
from train_car_classifier import CLASSES, LABELS, split_videos  # noqa: E402

SEEDS = (7, 8, 9)


def fit_head(X, y, k, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((X.shape[1],)), L.Dropout(.3), L.Dense(256, activation="relu"), L.Dropout(.4),
                             L.Dense(k, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    cw = np.bincount(y, minlength=k); cw = {i: len(y) / (k * max(c, 1)) for i, c in enumerate(cw)}
    m.fit(X, y, epochs=40, batch_size=64, class_weight=cw, verbose=0)
    return m


def main():
    d = np.load(os.path.join(ROOT, "data", "car_windows.npz"))
    E, y, video = d["E"].astype(np.float32), d["y"], d["video"]
    classes = [c for c in CLASSES if c in set(y)]
    yi = np.array([classes.index(c) for c in y])
    runs, cms = [], []
    for seed in SEEDS:
        test = split_videos(y, video, 0.2, seed, 3)
        mu, sd = E[~test].mean(0), E[~test].std(0) + 1e-6
        p = mlp((E[~test] - mu) / sd, yi[~test], (E[test] - mu) / sd, len(classes), seed)
        runs.append(metrics(p, yi[test], video[test], classes))
        cms.append(confusion_matrix(yi[test], p.argmax(1), labels=list(range(len(classes)))))
        print("seed", seed, {k: round(float(v), 3) for k, v in runs[-1].items()}, flush=True)
    cm = sum(cms)
    recall = {c: round(float(cm[i, i] / max(cm[i].sum(), 1)), 3) for i, c in enumerate(classes)}
    summary = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}
    report = {"model": "YAMNet embeddings + MLP head", "seeds": list(SEEDS), "classes": classes,
              "balanced_accuracy": summary["bal"], "top3_accuracy": summary["top3"],
              "fault_vs_normal_accuracy": summary["fvn"], "video_accuracy": summary["video"],
              "chance_balanced": round(1 / len(classes), 3), "per_class_recall": recall,
              "confusion_summed_over_seeds": cm.tolist(), "windows": int(len(y)), "videos": int(len(set(video)))}
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_summed_over_seeds"}, indent=1))
    mu, sd = E.mean(0), E.std(0) + 1e-6
    m = fit_head((E - mu) / sd, yi, len(classes), 7)
    W = m.get_weights()
    web = {"v": 2, "type": "yamnet-mlp", "classes": classes, "labels": [LABELS[c] for c in classes],
           "mean": [float(v) for v in mu], "std": [float(v) for v in sd],
           "report": {k: report[k] for k in ("balanced_accuracy", "top3_accuracy", "fault_vs_normal_accuracy",
                                             "video_accuracy", "chance_balanced", "per_class_recall", "videos")},
           "weights": [{"shape": list(w.shape), "data": base64.b64encode(w.astype(np.float32).tobytes()).decode()} for w in W]}
    os.makedirs(os.path.join(ROOT, "docs", "models"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "models", "fault_classifier.json"), "w") as f:
        json.dump(web, f)
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    with open(os.path.join(ROOT, "models", "classifier_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    m.save(os.path.join(ROOT, "models", "fault_head.keras"))
    print("exported docs/models/fault_classifier.json")


if __name__ == "__main__":
    main()
