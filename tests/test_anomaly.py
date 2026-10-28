"""Anomaly detection tests on synthetic engine sounds (modules/anomaly/synth.py).

Run: python -m pytest tests/test_anomaly.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.anomaly.context import EngineProfile, resolve_context  # noqa: E402
from modules.anomaly.features import estimate_rpm, windows  # noqa: E402
from modules.anomaly.service import AnomalyService, prepare_windows, verdict_from_levels  # noqa: E402
from modules.anomaly.synth import engine_sound  # noqa: E402

SR = 22050
PROFILE = EngineProfile(engine_id="synth", cylinders=4, strokes=4, idle_rpm=750, max_rpm=6500)
RPMS = (750, 1000, 1500, 2000, 2600, 3400, 4200)


def test_states():
    assert PROFILE.state_for_rpm(750) == "idle"
    assert PROFILE.state_for_rpm(2000) == "slow"
    assert PROFILE.state_for_rpm(4000) == "fast"
    assert PROFILE.state_for_rpm(None) == "unknown"
    c = resolve_context(PROFILE, rpm=None, audio_rpm=4000)
    assert c.rpm_source == "audio" and c.state == "fast"
    assert len(c.vector(PROFILE)) == 10


@pytest.mark.parametrize("rpm", [750, 2000, 4500])
def test_rpm_from_audio(rpm):
    est, n = estimate_rpm(windows(engine_sound(rpm, PROFILE, 3, seed=rpm), SR), PROFILE)
    assert n > 0 and abs(est - rpm) / rpm < 0.03


def test_persistence_rule():
    thr = {"all": {"warn": 1.0, "alarm": 2.0}}
    st = ["idle"] * 6
    assert verdict_from_levels([2, 0, 2, 0, 2, 0], [3, 0, 3, 0, 3, 0], st, thr) == "ANOMALY"
    assert verdict_from_levels([2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], st, thr) == "WATCH"
    assert verdict_from_levels([0] * 6, [0.1] * 6, st, thr) == "NORMAL"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from tools.train_anomaly import fit_baseline
    samples, groups = [], []
    for i, rpm in enumerate(RPMS):
        for rep in range(3):
            temp = 82 + 3 * rep + rpm / 1000           # thermostat-held coolant
            a = engine_sound(rpm, PROFILE, seconds=6, seed=100 * i + rep, gain=0.7 + 0.15 * rep)
            ws = prepare_windows(a, SR, PROFILE, rpm=rpm, temp_c=temp, run_time_s=300 * rep)
            samples += ws
            groups += [f"{rpm}_{rep}"] * len(ws)
    root = tmp_path_factory.mktemp("anomaly")
    fit_baseline(samples, groups, PROFILE, str(root / "synth"), ("ae", "regression"), epochs=25)
    return AnomalyService(str(root))


def _score(svc, audio, rpm, temp):
    return svc.analyze(audio, SR, "synth", rpm=rpm, temp_c=temp, run_time_s=600)


def test_normal_vs_faults(trained):
    rpm, temp = 2000, 88 + 2000 / 1000
    normal = _score(trained, engine_sound(rpm, PROFILE, 6, seed=999), rpm, temp)
    faults = {
        "squeal": _score(trained, engine_sound(rpm, PROFILE, 6, seed=998, squeal_hz=6500), rpm, temp),
        "dead": _score(trained, engine_sound(rpm, PROFILE, 6, seed=997, dead_cylinder=True), rpm, temp),
        "misfire": _score(trained, engine_sound(rpm, PROFILE, 6, seed=996, misfire_p=0.25), rpm, temp),
    }
    for det in ("autoencoder", "regression"):
        n = normal["detectors"][det]["score_median"]
        for name, r in faults.items():
            assert r["detectors"][det]["score_median"] > n, (det, name)
    assert normal["verdict"] in ("NORMAL", "WATCH")
    assert faults["dead"]["detectors"]["regression"]["verdict"] == "ANOMALY"
    assert faults["squeal"]["detectors"]["autoencoder"]["verdict"] == "ANOMALY"


def test_overheat_is_flagged_by_regression(trained):
    rpm = 1500
    r = _score(trained, engine_sound(rpm, PROFILE, 6, seed=995), rpm, temp=112.0)
    top = [f["feature"] for f in r["detectors"]["regression"]["top_features"]]
    assert top[0] == "temp_c"
    assert r["detectors"]["regression"]["verdict"] == "ANOMALY"


def test_unknown_engine_without_generic_has_no_baseline(tmp_path):
    svc = AnomalyService(str(tmp_path))
    assert svc.analyze(np.zeros(SR * 2, np.float32), SR, "nope")["verdict"] == "NO_BASELINE"
