"""Coolant-temperature sanity check for a warm car engine.

Car engines are thermostat-controlled, so a healthy warm engine sits in a
fairly narrow coolant band regardless of rpm (typically ~85-105 C; check the
car's manual, some run hotter). The anomaly detector's regression model learns
each car's own coolant behaviour from its baseline; this module is the simple
rule-based backstop for when no baseline exists yet.
"""
from typing import Any, Dict, Optional


class CoolantMonitor:
    def __init__(self, normal_low_c: float = 80.0, normal_high_c: float = 105.0,
                 overheat_c: float = 112.0, warmup_s: float = 900.0):
        self.normal_low_c = normal_low_c
        self.normal_high_c = normal_high_c
        self.overheat_c = overheat_c
        self.warmup_s = warmup_s

    def evaluate(self, coolant_c: float, run_time_s: Optional[float] = None) -> Dict[str, Any]:
        if coolant_c >= self.overheat_c:
            status = "OVERHEATING"
        elif coolant_c > self.normal_high_c:
            status = "RUNNING_HOT"
        elif coolant_c < self.normal_low_c and (run_time_s is None or run_time_s >= self.warmup_s):
            # still cold after a full warm-up: classic stuck-open thermostat
            status = "NOT_WARMING_UP" if run_time_s is not None else "COLD_OR_NOT_WARMED_UP"
        else:
            status = "NORMAL"
        return {"status": status, "coolant_c": round(coolant_c, 1),
                "normal_band_c": [self.normal_low_c, self.normal_high_c]}
