"""Run both anomaly detectors on a recording and turn scores into a verdict.

Baselines live in models/anomaly/<engine_id>/:
    profile.json            EngineProfile (cylinders, strokes, idle/max rpm)
    autoencoder.*           detector A, mode "ae"     (optional)
    idnn.*                  detector A, mode "idnn"   (optional)
    regression.json         detector B                (optional)
An engine without its own baseline falls back to models/anomaly/generic/.

Verdict per detector (window = 1 s):
    ANOMALY  >= 3 of any 5 consecutive windows above the alarm threshold
             (clips shorter than 5 windows: >= 60 % of windows)
    WATCH    median window score above the warning threshold, or any alarm window
    NORMAL   otherwise
The overall verdict is the most severe across detectors; every detector's
scores are always returned so the two approaches can be compared in the field.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from modules.anomaly.context import EngineProfile, OperatingContext, resolve_context
from modules.anomaly.features import anomaly_mel, descriptors, estimate_firing_hz, windows

DEFAULT_ROOT = "models/anomaly"
SEVERITY = {"NORMAL": 0, "WATCH": 1, "ANOMALY": 2}


@dataclass
class WindowSample:
    mel: np.ndarray
    desc: dict
    ctx: OperatingContext


def prepare_windows(audio: np.ndarray, sr: int, profile: EngineProfile, rpm=None, temp_c=None,
                    run_time_s=None, voltage=None, rpm_source="sensor", hop_s=1.0,
                    include_level=False, min_rpm_conf=0.5) -> List[WindowSample]:
    """Raw audio + telemetry -> per-window features and contexts.
    Used identically by training (tools/train_anomaly.py) and scoring."""
    out = []
    for i, w in enumerate(windows(audio, sr, hop_s)):
        audio_rpm, fhz = None, None
        if rpm is not None and rpm > 0:
            fhz = profile.rpm_to_firing_hz(rpm)
        else:
            hz, conf = estimate_firing_hz(w, profile)
            if hz is not None and conf >= min_rpm_conf:
                audio_rpm, fhz = profile.firing_hz_to_rpm(hz), hz
        rt = None if run_time_s is None else run_time_s + i * hop_s
        ctx = resolve_context(profile, rpm, temp_c, rt, voltage, rpm_source, audio_rpm)
        d = descriptors(w, profile, firing_hz=fhz, include_level=include_level)
        if temp_c is not None:
            d["temp_c"] = float(temp_c)
        if voltage is not None:
            d["voltage"] = float(voltage)
        out.append(WindowSample(anomaly_mel(w), d, ctx))
    return out


def window_levels(scores, states, thresholds: Dict[str, dict]) -> np.ndarray:
    lv = np.zeros(len(scores), int)
    for i, (s, st) in enumerate(zip(scores, states)):
        t = thresholds.get(st, thresholds["all"])
        lv[i] = 2 if s > t["alarm"] else (1 if s > t["warn"] else 0)
    return lv


def verdict_from_levels(levels, scores, states, thresholds) -> str:
    levels = np.asarray(levels)
    n = len(levels)
    if n == 0:
        return "NORMAL"
    alarm = levels == 2
    if n < 5:
        if alarm.sum() >= math.ceil(0.6 * n):
            return "ANOMALY"
    elif max(alarm[i:i + 5].sum() for i in range(n - 4)) >= 3:
        return "ANOMALY"
    main_state = max(set(states), key=list(states).count)
    warn = thresholds.get(main_state, thresholds["all"])["warn"]
    if alarm.any() or np.median(scores) > warn:
        return "WATCH"
    return "NORMAL"


class AnomalyService:
    def __init__(self, root: str = DEFAULT_ROOT, ae_score: str = "mse"):
        self.root = root
        self.ae_score = ae_score
        self._cache: Dict[str, dict] = {}

    def baseline_dir(self, engine_id: str) -> Optional[str]:
        for eid in (engine_id, "generic"):
            d = os.path.join(self.root, eid or "generic")
            if os.path.exists(os.path.join(d, "profile.json")):
                return d
        return None

    def _load(self, directory: str) -> dict:
        if directory in self._cache:
            return self._cache[directory]
        from modules.anomaly.autoencoder import ConditionalAutoencoder
        from modules.anomaly.regression import PolynomialNormalModel
        b = {"profile": EngineProfile.load(os.path.join(directory, "profile.json")), "detectors": {}}
        for mode in ("ae", "idnn"):
            if ConditionalAutoencoder.exists(directory, mode):
                m = ConditionalAutoencoder.load(directory, mode)
                b["detectors"][m.name] = m
        rp = os.path.join(directory, "regression.json")
        if os.path.exists(rp):
            b["detectors"]["regression"] = PolynomialNormalModel.load(rp)
        self._cache[directory] = b
        return b

    def reload(self):
        self._cache.clear()

    def analyze(self, audio: np.ndarray, sr: int, engine_id: str = "generic", rpm=None, temp_c=None,
                run_time_s=None, voltage=None, rpm_source="sensor") -> dict:
        directory = self.baseline_dir(engine_id)
        if directory is None:
            return {"verdict": "NO_BASELINE", "engine_id": engine_id,
                    "message": "No normal baseline recorded for this engine yet."}
        b = self._load(directory)
        profile: EngineProfile = b["profile"]
        include_level = False
        reg = b["detectors"].get("regression")
        if reg is not None and "level_db" in reg.targets:
            include_level = True
        samples = prepare_windows(audio, sr, profile, rpm, temp_c, run_time_s, voltage, rpm_source,
                                  include_level=include_level)
        if not samples:
            return {"verdict": "NO_AUDIO", "engine_id": engine_id}
        ctxs = [s.ctx for s in samples]
        states = [c.state for c in ctxs]
        results = {}
        for name, det in b["detectors"].items():
            if name == "regression":
                r = det.score([s.desc for s in samples], ctxs)
                scores, thr = r["window_scores"], r["thresholds"]
                extra = {"top_features": r["top_features"], "context_vars": r["context_vars"]}
            else:
                sc = det.score([s.mel for s in samples], np.stack([c.vector(profile) for c in ctxs]))
                scores, thr = sc[self.ae_score], det.thresholds[self.ae_score]
                extra = {"score_type": self.ae_score}
            levels = window_levels(scores, states, thr)
            results[name] = {
                "verdict": verdict_from_levels(levels, scores, states, thr),
                "score_median": round(float(np.median(scores)), 4),
                "score_max": round(float(np.max(scores)), 4),
                "warn": round(thr.get(states[0], thr["all"])["warn"], 4),
                "alarm": round(thr.get(states[0], thr["all"])["alarm"], 4),
                "alarm_windows": int((levels == 2).sum()),
                "window_scores": [round(float(x), 4) for x in scores],
                **extra,
            }
        overall = max((r["verdict"] for r in results.values()), key=SEVERITY.get, default="NO_BASELINE")
        rpms = [c.rpm for c in ctxs if c.rpm]
        return {
            "verdict": overall,
            "engine_id": engine_id,
            "baseline": os.path.basename(directory),
            "windows": len(samples),
            "context": {**ctxs[0].to_dict(),
                        "rpm": round(float(np.median(rpms)), 0) if rpms else None,
                        "state": max(set(states), key=states.count)},
            "detectors": results,
        }
