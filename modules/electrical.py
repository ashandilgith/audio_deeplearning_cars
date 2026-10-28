from typing import Dict, Any

class ElectricalBusMonitor:
    """Evaluates a car's 12 V system: battery at rest, cranking sag, and charging voltage.
    Some cars with smart alternators charge as low as ~12.5 V at times; treat
    an undercharging warning as "check", not proof."""
    
    THRESHOLDS = {
        "dead": 11.8,
        "weak": 12.2,
        "sag_limit": 9.6,
        "charging_low": 13.2,
        "charging_high": 15.1
    }

    @classmethod
    def evaluate(cls, resting_v: float, cranking_v: float, running_v: float) -> Dict[str, Any]:
        faults = []
        status = "NORMAL"

        if resting_v < cls.THRESHOLDS["dead"]:
            faults.append(f"CRITICAL: Battery depleted ({resting_v}V).")
            status = "ELECTRICAL_FAULT"
        
        if cranking_v and cranking_v < cls.THRESHOLDS["sag_limit"]:
            faults.append(f"CRITICAL: Excessive starter voltage sag ({cranking_v}V).")
            status = "ELECTRICAL_FAULT"
            
        if running_v:
            if running_v < cls.THRESHOLDS["charging_low"]:
                faults.append(f"WARNING: Undercharging ({running_v}V). Stator/Regulator fault.")
                status = "ELECTRICAL_WARNING"
            elif running_v > cls.THRESHOLDS["charging_high"]:
                faults.append(f"CRITICAL: Overcharging ({running_v}V). Regulator blown.")
                status = "ELECTRICAL_FAULT"

        return {"status": status, "faults": faults}