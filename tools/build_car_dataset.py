"""Turn downloaded YouTube audio into labelled 1-second feature windows.

Reads data/car_sources.csv (video_id,label,title,duration_s,channel,query,
include,start_s,end_s,notes) and data/raw_audio/<video_id>.*; for each
included video: decodes to 16 kHz mono, cuts 1 s windows (0.5 s hop for
short videos, 1 s otherwise), drops silence and talking (syllable-rate
modulation heuristic) and stores the browser-identical log-mel features.

Output: data/car_windows.npz  (X: (n, 30, 64) float16 browser log-mel, E: (n, 1024) YAMNet
                               embeddings, y: labels, video, start)
        data/car_windows_index.csv

Label noise warning: a "rod knock" video also contains talking, idling
between demos and a healthy engine for comparison. The heuristics remove
talking and silence only; a listening review is the real fix.
"""
import csv
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.web_features import SR, WIN, decode16k, mel_window, speech_mod  # noqa: E402
from modules.yamnet_embed import YamnetEmbedder  # noqa: E402

DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw_audio")
MAX_S = 240          # use at most the first 4 minutes of a video (demos are usually early)


def main():
    src = [r for r in csv.DictReader(open(os.path.join(DATA, "car_sources.csv"))) if r.get("include", "1") == "1"]
    files = {}
    for f in os.listdir(RAW):
        stem, ext = os.path.splitext(f)
        if ext not in (".part", ".ytdl", ".txt"):
            files[stem] = os.path.join(RAW, f)
    X, E, y, vids, starts, idx = [], [], [], [], [], []
    emb = YamnetEmbedder()
    for r in src:
        vid = r["video_id"]
        if vid not in files:
            continue
        try:
            a = decode16k(files[vid])
        except subprocess.CalledProcessError:
            print("decode failed", vid)
            continue
        s0 = float(r.get("start_s") or 0)
        s1 = float(r["end_s"]) if r.get("end_s") else min(len(a) / SR, s0 + MAX_S)
        a = a[int(s0 * SR): int(s1 * SR)]
        hop = WIN // 2 if len(a) < 60 * SR else WIN
        wins = [(t, a[t:t + WIN]) for t in range(0, len(a) - WIN + 1, hop)]
        if not wins:
            continue
        rms = np.array([20 * np.log10(np.sqrt(np.mean(w ** 2)) + 1e-9) for _, w in wins])
        ref = np.percentile(rms, 90)
        kept = 0
        kept_w = []
        for (t, w), lv in zip(wins, rms):
            if lv < ref - 25:
                continue
            sm = speech_mod(w)
            if sm > 0.25:
                continue
            X.append(mel_window(w).astype(np.float16))
            kept_w.append(w)
            y.append(r["label"]); vids.append(vid); starts.append(s0 + t / SR)
            idx.append({"video_id": vid, "label": r["label"], "start_s": round(s0 + t / SR, 2),
                        "rms_db": round(float(lv), 1), "speech_mod": round(sm, 3)})
            kept += 1
        for i in range(0, len(kept_w), 64):
            E.extend(emb.batch(kept_w[i:i + 64]).astype(np.float16))
        print(f"{vid} {r['label']:16s} {len(wins):4d} windows, {kept:4d} kept  {r['title'][:50]}")
    np.savez_compressed(os.path.join(DATA, "car_windows.npz"), X=np.stack(X), E=np.stack(E), y=np.array(y),
                        video=np.array(vids), start=np.array(starts, dtype=np.float32))
    with open(os.path.join(DATA, "car_windows_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(idx[0]))
        w.writeheader(); w.writerows(idx)
    from collections import Counter
    c, v = Counter(y), {}
    for vid, lab in zip(vids, y):
        v.setdefault(lab, set()).add(vid)
    print("\nlabel             windows  videos")
    for k in sorted(c):
        print(f"{k:16s} {c[k]:8d} {len(v[k]):7d}")


if __name__ == "__main__":
    main()
