"""Train the car fault classifier on YouTube windows and export it for the phone app.

Input : data/car_windows.npz (tools/build_car_dataset.py), browser-identical
        log-mel windows (30 frames x 64 bands).
Split : whole videos held out (never the same recording on both sides), per
        class ~20 % of windows, chosen with a fixed seed.
Model : small CNN (3 conv blocks, global average pooling, softmax), class
        weights, SpecAugment-style time/frequency masking, early stopping on
        a validation split carved from the training videos.
Output: models/fault_cnn_web.json  (baseline; the app ships the YAMNet head from train_yamnet_head.py;
        loaded by docs/index.html with TensorFlow.js)
        models/fault_classifier.keras, models/classifier_report.json

Usage: python tools/train_car_classifier.py [--seed 7] [--epochs 60] [--min-videos 3]
"""
import argparse
import base64
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA, MODELS, WEB = (os.path.join(ROOT, d) for d in ("data", "models", os.path.join("docs", "models")))

CLASSES = ["normal", "belt_squeal", "engine_knock", "valve_tick", "rough_idle", "exhaust_leak",
           "startup_rattle", "accessory_whine", "trouble_starting"]
LABELS = {
    "normal": "Healthy engine", "belt_squeal": "Belt squeal", "engine_knock": "Engine knock (rod/bearing)",
    "valve_tick": "Valve or lifter tick", "rough_idle": "Misfire / rough idle", "exhaust_leak": "Exhaust leak",
    "startup_rattle": "Start-up rattle (timing chain)", "accessory_whine": "Whine (alternator, pump, pulley)",
    "trouble_starting": "Starting problem (starter/clicking)",
}


def split_videos(y, video, frac, seed, min_videos):
    rng = random.Random(seed)
    test_vids = set()
    for c in sorted(set(y)):
        vids = Counter(v for v, l in zip(video, y) if l == c)
        if len(vids) < min_videos:
            continue
        total, best = sum(vids.values()), None
        names = sorted(vids)
        for _ in range(2000):
            rng.shuffle(names)
            pick, n = [], 0
            for v in names:
                if len(pick) >= len(names) - 2:
                    break
                if n < frac * total:
                    pick.append(v); n += vids[v]
            err = abs(n - frac * total)
            if best is None or err < best[0]:
                best = (err, list(pick))
        test_vids.update(best[1])
    return np.array([v in test_vids for v in video])


def build(n_classes, frames=30, mels=64):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input((frames, mels, 1))
    h = inp
    for f in (16, 32, 64):
        h = L.Conv2D(f, 3, padding="same", activation="relu")(h)
        h = L.Conv2D(f, 3, padding="same", activation="relu")(h)
        h = L.MaxPooling2D(2)(h)
    h = L.GlobalAveragePooling2D()(h)
    h = L.Dropout(0.4)(h)
    out = L.Dense(n_classes, activation="softmax")(h)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def augment(X, rng):
    X = X.copy()
    for i in range(len(X)):
        if rng.random() < .8:
            t = rng.integers(0, 6); s = rng.integers(0, X.shape[1] - t)
            X[i, s:s + t] = 0
        if rng.random() < .8:
            f = rng.integers(0, 10); s = rng.integers(0, X.shape[2] - f)
            X[i, :, s:s + f] = 0
        X[i] *= rng.uniform(.85, 1.15)           # mic sensitivity / level spread
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--min-videos", type=int, default=3)
    ap.add_argument("--final", action="store_true", help="after evaluating, retrain on all videos for export")
    a = ap.parse_args()
    import tensorflow as tf
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    tf.keras.utils.set_random_seed(a.seed)
    d = np.load(os.path.join(DATA, "car_windows.npz"))
    X, y, video = d["X"].astype(np.float32), d["y"], d["video"]
    keep = np.isin(y, CLASSES)
    X, y, video = X[keep], y[keep], video[keep]
    classes = [c for c in CLASSES if c in set(y)]
    yi = np.array([classes.index(c) for c in y])
    test = split_videos(y, video, 0.2, a.seed, a.min_videos)
    tr_vids = sorted(set(video[~test]))
    rng = np.random.default_rng(a.seed)
    val_vids = set(rng.choice(tr_vids, size=max(1, len(tr_vids) // 8), replace=False))
    val = np.array([v in val_vids for v in video]) & ~test
    tr = ~test & ~val
    mean, std = X[tr].mean(axis=(0, 1)), X[tr].std(axis=(0, 1)) + 1e-3
    norm = lambda Z: ((Z - mean) / std)[..., None]
    cw = Counter(yi[tr])
    weights = {k: len(yi[tr]) / (len(cw) * v) for k, v in cw.items()}

    def fit(mask, epochs, val_data=None):
        m = build(len(classes))
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)] if val_data else []
        for ep in range(epochs):
            Xa = norm(augment(X[mask], rng))
            h = m.fit(Xa, yi[mask], epochs=1, batch_size=64, class_weight=weights, verbose=0,
                      validation_data=val_data, callbacks=[])
            if val_data:
                vl = h.history["val_loss"][-1]
                if ep == 0 or vl < best[0] - 1e-4:
                    best = (vl, m.get_weights(), ep); wait = 0
                else:
                    wait += 1
                    if wait >= 8:
                        break
        if val_data:
            m.set_weights(best[1])
            print(f"  stopped after {ep + 1} epochs, best epoch {best[2] + 1}")
        return m

    print(f"windows: train {tr.sum()}  val {val.sum()}  test {test.sum()}   classes {classes}")
    m = fit(tr, a.epochs, (norm(X[val]), yi[val]))
    p = m.predict(norm(X[test]), verbose=0)
    pred = p.argmax(1)
    bal = balanced_accuracy_score(yi[test], pred)
    acc = float((pred == yi[test]).mean())
    # video level: average probabilities over each held-out video
    vp = defaultdict(list)
    for v, pp, t in zip(video[test], p, yi[test]):
        vp[v].append((pp, t))
    vids_ok = [int(np.mean([q for q, _ in r], 0).argmax() == r[0][1]) for r in vp.values()]
    top3 = float(np.mean([t in np.argsort(-pp)[:3] for pp, t in zip(p, yi[test])]))
    nvf_true = yi[test] != classes.index("normal")
    nvf_pred = pred != classes.index("normal")
    cm = confusion_matrix(yi[test], pred, labels=list(range(len(classes))))
    per = {c: {"recall": round(float(cm[i, i] / max(cm[i].sum(), 1)), 3), "test_windows": int(cm[i].sum()),
               "test_videos": int(len(set(video[test & (yi == i)])))} for i, c in enumerate(classes)}
    report = {"seed": a.seed, "classes": classes, "test_windows": int(test.sum()),
              "test_videos": len(vp), "window_accuracy": round(acc, 3), "balanced_accuracy": round(float(bal), 3),
              "top3_accuracy": round(top3, 3), "video_accuracy": round(float(np.mean(vids_ok)), 3),
              "fault_vs_normal_accuracy": round(float((nvf_true == nvf_pred).mean()), 3),
              "chance_balanced": round(1 / len(classes), 3), "per_class": per, "confusion": cm.tolist()}
    print(json.dumps({k: v for k, v in report.items() if k not in ("confusion", "per_class")}, indent=1))
    for c, r in per.items():
        print(f"  {c:16s} recall {r['recall']:.2f}  ({r['test_windows']} windows, {r['test_videos']} videos)")

    if a.final:
        print("retraining on all videos for export")
        allm = ~val
        m = fit(allm, a.epochs, (norm(X[val]), yi[val]))
        report["exported"] = "all videos (val videos for early stopping)"
    else:
        report["exported"] = "train split only"
    os.makedirs(MODELS, exist_ok=True); os.makedirs(WEB, exist_ok=True)
    m.save(os.path.join(MODELS, "fault_classifier.keras"))
    with open(os.path.join(MODELS, "classifier_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    web = {"v": 1, "classes": classes, "labels": [LABELS[c] for c in classes],
           "mean": [float(v) for v in mean], "std": [float(v) for v in std],
           "arch": "conv16x2-pool-conv32x2-pool-conv64x2-pool-gap-dense",
           "report": {k: report[k] for k in ("balanced_accuracy", "top3_accuracy", "video_accuracy", "test_videos", "chance_balanced")},
           "weights": [{"shape": list(w.shape), "data": base64.b64encode(w.astype(np.float32).tobytes()).decode()}
                       for w in m.get_weights()]}
    with open(os.path.join(MODELS, "fault_cnn_web.json"), "w") as f:  # baseline only; the app ships the YAMNet head
        json.dump(web, f)
    print("exported", os.path.join(MODELS, "fault_cnn_web.json"))


if __name__ == "__main__":
    main()
