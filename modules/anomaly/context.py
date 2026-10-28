"""Operating context shared by both anomaly detectors.

A car engine sounds different at idle vs 3,000 rpm, cold vs warm, with the
air-conditioning on or off. A detector that
ignores this flags every throttle change as an anomaly (the DCASE "domain
shift" problem), so every reading carries an OperatingContext:

    rpm         tach / user / estimated from the audio firing frequency
    state       idle | slow | fast | unknown  (coarse band from rpm + profile)
    temp_c      coolant temperature (OBD-II or a probe), optional
    run_time_s  seconds since engine start, optional
    voltage     12 V system voltage while running (charging), optional

Missing values are allowed everywhere; models fall back gracefully.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

STATES = ("idle", "slow", "fast", "unknown")

# Context vector layout fed to the autoencoder (also mirrored in docs/index.html).
CONTEXT_NAMES = ["is_idle", "is_slow", "is_fast", "is_unknown",
                 "rpm_norm", "rpm_known", "temp_norm", "temp_known",
                 "runtime_norm", "runtime_known"]
CONTEXT_DIM = len(CONTEXT_NAMES)


@dataclass
class EngineProfile:
    """What we need to know about an engine to interpret its sound."""
    engine_id: str = "generic"
    make: str = ""
    model: str = ""
    year: Optional[int] = None
    fuel: str = "petrol"        # petrol | diesel | hybrid
    cylinders: int = 4
    strokes: int = 4            # car engines are 4-stroke; 2 kept for completeness
    idle_rpm: float = 750.0
    max_rpm: float = 6500.0

    @property
    def firings_per_rev(self) -> float:
        # 4-stroke: every cylinder fires every 2nd revolution (4 cyl -> 2 firings/rev).
        return self.cylinders * (1.0 if self.strokes == 2 else 0.5)

    def rpm_to_firing_hz(self, rpm: float) -> float:
        return rpm / 60.0 * self.firings_per_rev

    def firing_hz_to_rpm(self, hz: float) -> float:
        return hz * 60.0 / self.firings_per_rev

    def state_for_rpm(self, rpm: Optional[float]) -> str:
        if rpm is None or not np.isfinite(rpm) or rpm <= 0:
            return "unknown"
        if rpm <= 1.5 * self.idle_rpm:
            return "idle"
        if rpm <= 0.5 * self.max_rpm:
            return "slow"
        return "fast"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EngineProfile":
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class OperatingContext:
    rpm: Optional[float] = None
    rpm_source: str = "none"          # sensor | user | audio | none
    state: str = "unknown"
    temp_c: Optional[float] = None
    run_time_s: Optional[float] = None
    voltage: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def vector(self, profile: EngineProfile) -> np.ndarray:
        """Fixed-length numeric encoding (CONTEXT_NAMES) for the autoencoder."""
        onehot = [1.0 if self.state == s else 0.0 for s in STATES]
        rpm_known = _known(self.rpm)
        temp_known = _known(self.temp_c)
        rt_known = _known(self.run_time_s)
        return np.array(onehot + [
            (self.rpm / profile.max_rpm) if rpm_known else 0.0, float(rpm_known),
            (self.temp_c / 100.0) if temp_known else 0.0, float(temp_known),
            runtime_norm(self.run_time_s) if rt_known else 0.0, float(rt_known),
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def runtime_norm(run_time_s: float) -> float:
    """log-compress run time: 0 s -> 0, 1 h -> 1 (warm-up happens in the first ~15 min)."""
    return math.log1p(max(run_time_s, 0.0) / 60.0) / math.log1p(60.0)


def _known(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def resolve_context(profile: EngineProfile, rpm=None, temp_c=None, run_time_s=None,
                    voltage=None, rpm_source="sensor", audio_rpm=None) -> OperatingContext:
    """Build a context, using the audio rpm estimate only when no measured rpm exists."""
    if _known(rpm) and rpm > 0:
        r, src = float(rpm), rpm_source
    elif _known(audio_rpm):
        r, src = float(audio_rpm), "audio"
    else:
        r, src = None, "none"
    return OperatingContext(rpm=r, rpm_source=src, state=profile.state_for_rpm(r),
                            temp_c=float(temp_c) if _known(temp_c) else None,
                            run_time_s=float(run_time_s) if _known(run_time_s) else None,
                            voltage=float(voltage) if _known(voltage) else None)
