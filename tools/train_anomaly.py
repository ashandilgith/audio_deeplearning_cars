"""Fit a healthy baseline (both anomaly detectors) for one car.

Record the car healthy, parked, phone in the same place every time (see
README: warm idle, ~2,000 rpm held, idle with AC on; ideally a cold start and
a second session on another day), put the files in one folder and optionally
describe each file in a CSV (rpm and coolant temperature from an OBD-II app):

    file,rpm,temp_c,run_time_s,voltage
    idle_warm_1.wav,760,90,1200,14.2
    rev2000_1.wav,2010,91,1260,14.3

Any column may be blank / missing: rpm is then estimated from the sound,
other variables are left out of the model.

Usage:
  python tools/train_anomaly.py --engine-id my_corolla --audio-dir rec/corolla \
      --manifest rec/corolla/baseline.csv --cylinders 4 --idle-rpm 750 --max-rpm 6500
Writes models/anomaly/<engine-id>/ (profile.json, autoencoder.*, idnn.*, regression.json, report.json).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.anomaly.autoencoder import ConditionalAutoencoder  # noqa: E402
from modules.anomaly.context import EngineProfile  # noqa: E402
from modules.anomaly.regression import PolynomialNormalModel  # noqa: E402
from modules.anomaly.service import prepare_windows  # noqa: E402
from modules.audio_features import SAMPLE_RATE  # noqa: E402

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".webm", ".aac", ".mp4")


def read_audio(path):
    """Any audio file -> (mono float32, sr). soundfile first, ffmpeg for the rest."""
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=False)
        return a, sr
    except Exception:
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
                              "-f", "f32le", "-"], check=True, capture_output=True).stdout
        return np.frombuffer(out, dtype=np.float32), SAMPLE_RATE


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def fit_baseline(samples, groups, profile: EngineProfile, out_dir, detectors=("ae", "idnn", "regression"),
                 epochs=60, seed=0, verbose=0):
    """samples: list of WindowSample; groups: recording id per window."""
    os.makedirs(out_dir, exist_ok=True)
    profile.save(os.path.join(out_dir, "profile.json"))
    ctxs = [s.ctx for s in samples]
    states = [c.state for c in ctxs]
    report = {"windows": len(samples), "recordings": len(set(groups)),
              "states": {s: states.count(s) for s in set(states)},
              "rpm_sources": {k: [c.rpm_source for c in ctxs].count(k) for k in {c.rpm_source for c in ctxs}},
              "detectors": {}}
    for mode in [d for d in detectors if d in ("ae", "idnn")]:
        t0 = time.time()
        ae = ConditionalAutoencoder(mode=mode).fit([s.mel for s in samples],
                                                   np.stack([c.vector(profile) for c in ctxs]), states,
                                                   groups, epochs=epochs, seed=seed, verbose=verbose)
        ae.save(out_dir)
        report["detectors"][ae.name] = {"seconds": round(time.time() - t0, 1),
                                        "epochs_run": len(ae.history["loss"]),
                                        "val_loss": round(min(ae.history["val_loss"]), 4),
                                        "thresholds": ae.thresholds["mse"]}
    if "regression" in detectors:
        t0 = time.time()
        reg = PolynomialNormalModel(max_rpm=profile.max_rpm).fit([s.desc for s in samples], ctxs, groups)
        reg.save(os.path.join(out_dir, "regression.json"))
        report["detectors"]["regression"] = {
            "seconds": round(time.time() - t0, 1), "context_vars": reg.vars,
            "degrees": {k: t["degree"] for k, t in reg.targets.items()}, "thresholds": reg.thresholds}
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-id", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--cylinders", type=int, default=4)
    ap.add_argument("--strokes", type=int, default=4, choices=(2, 4))
    ap.add_argument("--idle-rpm", type=float, default=750)
    ap.add_argument("--max-rpm", type=float, default=6500)
    ap.add_argument("--make", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--year", type=int)
    ap.add_argument("--fuel", default="petrol", choices=("petrol", "diesel", "hybrid"))
    ap.add_argument("--detectors", default="ae,idnn,regression")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--include-level", action="store_true",
                    help="model absolute loudness (only with a fixed phone mount and volume)")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "models", "anomaly"))
    a = ap.parse_args()

    profile = EngineProfile(a.engine_id, a.make, a.model, a.year, a.fuel, a.cylinders, a.strokes, a.idle_rpm, a.max_rpm)
    meta = {}
    if a.manifest:
        for r in csv.DictReader(open(a.manifest)):
            meta[r["file"]] = r
    files = sorted(f for f in os.listdir(a.audio_dir) if f.lower().endswith(AUDIO_EXT))
    samples, groups = [], []
    for fn in files:
        m = meta.get(fn, {})
        audio, sr = read_audio(os.path.join(a.audio_dir, fn))
        ws = prepare_windows(audio, sr, profile, rpm=_num(m.get("rpm")), temp_c=_num(m.get("temp_c")),
                             run_time_s=_num(m.get("run_time_s")), voltage=_num(m.get("voltage")),
                             include_level=a.include_level)
        samples += ws
        groups += [fn] * len(ws)
        print(f"{fn}: {len(ws)} windows")
    if len(samples) < 60:
        sys.exit(f"Only {len(samples)} s of audio; record at least a few minutes of normal running.")
    rep = fit_baseline(samples, groups, profile, os.path.join(a.out_root, a.engine_id),
                       a.detectors.split(","), a.epochs, verbose=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
