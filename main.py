"""Car engine sound check: FastAPI backend.

Serves the phone web app (docs/index.html) and an API that scores a
recording against a car's healthy baseline. The web app also works fully on
the phone without this server; the server is for fleets, mechanics and
experiments that want baselines trained with the Python tools.
"""
import io
import json
import os

import soundfile as sf
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse

from modules.anomaly.service import AnomalyService
from modules.electrical import ElectricalBusMonitor
from modules.thermal import CoolantMonitor

app = FastAPI(title="Car Engine Sound Check")
ROOT = os.path.dirname(os.path.abspath(__file__))
anomaly_service = AnomalyService(os.environ.get("CAR_ANOMALY_DIR", os.path.join(ROOT, "models", "anomaly")))
coolant = CoolantMonitor()


def _anomaly(audio, sr, **kw):
    try:
        return anomaly_service.analyze(audio, sr, **kw)
    except Exception as e:  # keep the API answering even if a baseline is broken
        return {"verdict": "ERROR", "error": str(e)}


@app.get("/")
def web_app():
    """The phone web app."""
    return FileResponse(os.path.join(ROOT, "docs", "index.html"))


@app.post("/api/v1/check")
async def check(
    audio_file: UploadFile,
    car_id: str = Form("generic"),
    rpm: float = Form(None),           # OBD-II or tach; estimated from the sound if missing
    coolant_c: float = Form(None),     # OBD-II coolant temperature
    run_time_s: float = Form(None),    # seconds since the engine started
    running_v: float = Form(None),     # system voltage with the engine running
    resting_v: float = Form(None),     # battery voltage with the engine off
    cranking_v: float = Form(None),    # lowest voltage while cranking
):
    """Score a recording against the car's healthy baseline (both detectors),
    plus rule-based coolant and 12 V checks when those values are supplied."""
    audio, sr = sf.read(io.BytesIO(await audio_file.read()))
    out = {"anomaly": _anomaly(audio, sr, engine_id=car_id, rpm=rpm, temp_c=coolant_c,
                               run_time_s=run_time_s, voltage=running_v)}
    if coolant_c is not None:
        out["coolant"] = coolant.evaluate(coolant_c, run_time_s)
    if resting_v is not None or running_v is not None:
        out["electrical"] = ElectricalBusMonitor.evaluate(resting_v or 12.6, cranking_v, running_v)
    return out


@app.get("/api/v1/baselines")
def list_baselines():
    """Cars with a recorded healthy baseline and how they were trained."""
    root, out = anomaly_service.root, {}
    if os.path.isdir(root):
        for cid in sorted(os.listdir(root)):
            rp = os.path.join(root, cid, "report.json")
            if os.path.exists(rp):
                with open(rp) as f:
                    out[cid] = json.load(f)
    return out


@app.post("/api/v1/baselines/reload")
def reload_baselines():
    """Pick up baselines retrained with tools/train_anomaly.py without restarting."""
    anomaly_service.reload()
    return {"status": "reloaded"}
