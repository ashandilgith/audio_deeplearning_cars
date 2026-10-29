"""EXP-C1: YouTube car-fault classification, same held-out-video splits, three seeds.
Compares heads on YAMNet embeddings (logistic regression, small MLP) with the
from-scratch CNN on browser log-mel (tools/train_car_classifier.py, run separately).
Prints balanced accuracy, top-3, fault-vs-normal and video accuracy per seed."""
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from train_car_classifier import CLASSES, split_videos  # noqa: E402


def metrics(p, yt, video, classes):
    pred = p.argmax(1)
    vp = defaultdict(list)
    for v, pp, t in zip(video, p, yt):
        vp[v].append((pp, t))
    vid = np.mean([np.mean([q for q, _ in r], 0).argmax() == r[0][1] for r in vp.values()])
    n = classes.index("normal")
    return dict(bal=balanced_accuracy_score(yt, pred), top3=np.mean([t in np.argsort(-q)[:3] for q, t in zip(p, yt)]),
                fvn=np.mean((yt != n) == (pred != n)), video=vid)


def mlp(Xtr, ytr, Xte, k, seed):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    m = tf.keras.Sequential([L.Input((Xtr.shape[1],)), L.Dropout(.3), L.Dense(256, activation="relu"), L.Dropout(.4),
                             L.Dense(k, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    cw = np.bincount(ytr, minlength=k); cw = {i: len(ytr) / (k * max(c, 1)) for i, c in enumerate(cw)}
    m.fit(Xtr, ytr, epochs=40, batch_size=64, class_weight=cw, verbose=0)
    return m.predict(Xte, verbose=0)


def main():
    d = np.load(os.path.join(ROOT, "data", "car_windows.npz"))
    E, y, video = d["E"].astype(np.float32), d["y"], d["video"]
    classes = [c for c in CLASSES if c in set(y)]
    yi = np.array([classes.index(c) for c in y])
    res = defaultdict(list)
    for seed in (7, 8, 9):
        test = split_videos(y, video, 0.2, seed, 3)
        sc = StandardScaler().fit(E[~test])
        Xtr, Xte = sc.transform(E[~test]), sc.transform(E[test])
        lr = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit(Xtr, yi[~test])
        res["yamnet+logreg"].append(metrics(lr.predict_proba(Xte), yi[test], video[test], classes))
        res["yamnet+mlp"].append(metrics(mlp(Xtr, yi[~test], Xte, len(classes), seed), yi[test], video[test], classes))
        print("seed", seed, {k: {m: round(v[-1][m], 3) for m in v[-1]} for k, v in res.items()}, flush=True)
    for k, v in res.items():
        print(k, {m: f"{np.mean([r[m] for r in v]):.3f}±{np.std([r[m] for r in v]):.3f}" for m in v[0]})


if __name__ == "__main__":
    main()
