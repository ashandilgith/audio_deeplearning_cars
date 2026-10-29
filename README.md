# Engine Ear: car engine sound check

A phone web app that learns what **your** car engine sounds like when it's healthy, then tells you when it starts to sound different, and what changed.

- No fault recordings are needed. It learns the healthy sound, so it works from day one.
- Everything runs on the phone. Nothing is uploaded.
- This is a side project spun off from the Piranaware boat-engine work. It uses the same anomaly-detection design, adapted to cars.

> A change in sound is a reason to have the car looked at, not a diagnosis. It doesn't replace a mechanic or the car's warning lights.

## Use it

1. Open the app: `docs/index.html`. Host it with GitHub Pages (below) or any static host, and open it on your phone.
2. **Add your car.** Enter the cylinders, fuel, and usual warm idle rpm, and choose where the phone will sit:
   - a dashboard holder (recommended)
   - the same spot under the hood
3. **Record the healthy sound.** Make 3 takes of 15 s for each state:
   - **Warm idle**
   - **About 2,000 rpm** (held steady in Park or Neutral)
   - **Idle with AC on** (optional)

   That's about 3 minutes. The phone then learns the sound, which takes under a minute.
4. **Check** any time, with the same phone in the same spot. You get *Sounds normal*, *Slightly different* or *Sounds different*, with plain-language reasons. For example:
   - "More high-pitched sound: a belt squeal…"
   - "Idle speed is higher than usual (850 vs 760 rpm)"
5. **On another day, while the car is fine,** tap *Add a healthy recording* once. Alarm levels set from two different days give far fewer false alarms.

**What's that noise?** works without a baseline. Record the noise, and a model trained on YouTube examples of 8 common faults says what it sounds most like. Faults covered: belt squeal, engine knock, valve tick, misfire/rough idle, exhaust leak, start-up rattle, alternator/pump/pulley whine and starting problems. When a check of a car with a baseline comes out different, the same model adds a "sounds most like" hint. The right fault is in its top 3 about 60% of the time on videos it never heard (`docs/EXPERIMENTS.md`), so treat it as a hint for the mechanic.

**Try the demo** first. It uses a synthetic engine, so you can see how a belt squeal, a misfire or a weak cylinder shows up.

**Safety:** parked on level ground, handbrake on, in Park or Neutral, outdoors or with the garage open. Keep away from the fan and belts.

## How it works

Every recording is cut into 1-second windows. Two detectors score each window against this car's own healthy recordings, for the same engine state:

| | A. Sound pattern (autoencoder) | B. Feature check |
|---|---|---|
| What it learns | A small neural network (TensorFlow.js, trained on the phone) learns to rebuild 5 consecutive log-mel frames (64 bands, 16 kHz) of the healthy sound. It is told the engine state | The healthy spread of 18 named features per state. Bands: 8 band energies, brightness, high-pitch share, hiss. Rhythm: firing-rhythm strength, uneven firing across cylinders, crest/knock, idle wander. Plus rpm estimated from the firing frequency |
| Anomaly score | How badly it rebuilds the new sound | Average squared z-score across the features |
| Strength | Catches new sounds no feature was designed for | Explains itself in plain language. Tracks idle-speed changes |

**Alarm levels.**
- The last take of each state is held back from training and used to set the levels: warning at about 2.5 robust standard deviations, alarm at about 4.
- *Add a healthy recording* and *Mark as normal* add more healthy scores to that pool.
- A state is *Sounds different* when 3 of any 5 consecutive seconds are above the alarm level. A single blip only counts as *Slightly different*.
- Recordings that are silent or clipping, or 2,000 rpm takes whose revs don't match the baseline, are rejected rather than scored.

The design and its limits come from the boat project's experiments (EXP-005 to EXP-007). Those tested the same detectors on real engine audio with fault-like distortions mixed in:
- **Added sounds** (squeal, knock, hiss) are caught well.
- **Rhythm faults** (misfire, weak cylinder) are harder.
- **False alarms** mostly come from changing the recording setup. Same phone and same position matter more than the model.

## Repository layout

```
docs/index.html          the phone web app (single file; GitHub Pages serves /docs)
main.py                  optional FastAPI server: serves the app + /api/v1/check scoring API
modules/anomaly/         Python version of both detectors (training, scoring, verdicts)
  context.py             engine profile (car defaults: 4-cyl 4-stroke, idle 750, max 6500), states
  features.py            log-mel frames + named features (incl. uneven firing for N cylinders)
  autoencoder.py         detector A (DCASE-style conditional autoencoder / IDNN, TFLite export)
  regression.py          detector B (polynomial normal-behaviour model with rpm/coolant/run time)
  service.py             per-car baselines, persistence rule, verdict
  synth.py, distort.py   synthetic engine sounds and fault-like distortions for testing
modules/electrical.py    12 V battery / cranking / charging rules
modules/thermal.py       coolant-temperature backstop (overheating, not warming up)
tools/train_anomaly.py   fit a car baseline from a folder of recordings (+ optional OBD-II CSV)
tests/test_anomaly.py    Python tests on synthetic engines
tests/e2e_web.mjs        browser test of the web app using its demo engine
tests/test_web_parity.py, tests/test_yamnet_parity.py   browser vs Python feature checks
modules/web_features.py  numpy copy of the browser's log-mel front end
modules/yamnet_embed.py  YAMNet embeddings with the browser's exact maths
docs/models/             YAMNet weights (float16, Apache-2.0, Google) + fault head, loaded by the app
models/yamnet/           official YAMNet float TFLite (source of the web weights) + class map
tools/select_car_videos.py, tools/build_car_dataset.py, tools/train_yamnet_head.py
                         YouTube fault dataset -> classifier (tools/windows/*.bat run yt-dlp on a PC)
docs/EXPERIMENTS.md      classifier results and limits
```

The Python side is for fleets, mechanics and experiments. It can use extra inputs from an OBD-II app: rpm, coolant temperature and voltage. The web app needs none of it.

## Run

**Web app on GitHub Pages:** Settings → Pages → *Deploy from a branch* → `main` / `/docs`. A private repository needs a paid GitHub plan for Pages. The microphone needs HTTPS, which Pages provides.

**Python server:**

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000          # app at http://localhost:8000, API at /api/v1/check
python tools/train_anomaly.py --engine-id my_car --audio-dir recordings/my_car --cylinders 4 --idle-rpm 750
```

**Tests:**

```bash
python -m pytest tests/test_anomaly.py -q
npm i playwright && node tests/e2e_web.mjs     # demo baseline + healthy/squeal/misfire/weak checks
```

## Rebuild the fault classifier

1. On a Windows PC with internet, run `tools/windows/search_car_videos.bat`. Then run `python tools/select_car_videos.py` to pick the videos.
2. Run `tools/windows/download_car_audio.bat` on the PC, then copy its `raw_audio` folder to `data/raw_audio/`.
3. Run `python tools/build_car_dataset.py`, then `python tools/train_yamnet_head.py`. This writes `docs/models/fault_classifier.json`.

## Limits

- **Not validated on real faulty cars yet.** The demo and tests use synthetic sounds. Real checks across several cars, healthy and faulty, are the next step.
- **The fault classifier learned from YouTube.** Its labels are noisy, and 23% balanced accuracy over 9 classes is a hint, not a diagnosis. See `docs/EXPERIMENTS.md`.
- **Hybrids** may switch the engine off at idle. Hold it running (for example with the AC on) or skip idle checks.
- **Phone microphones differ.** Keep one baseline per phone and per position. Re-record after a repair that changes the sound.
- **Storage is the phone browser's.** Baselines live in this browser's storage. Clearing site data deletes them, so use *Save a backup file*.
