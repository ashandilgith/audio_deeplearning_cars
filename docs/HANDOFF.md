# Hand-over: Engine Ear (cars side project)

Paste the prompt below into a new chat to continue this project without the boat context.

---

**Starter prompt**

> Continue my car side project **Engine Ear**. The repo is `ashandilgith/audio_deeplearning_cars` (branch `main`); clone it. Local copy on my PC: `C:\Users\PC1\Desktop\Car Deep Learning Fault Prediction`. Read `docs/ENGINE_EAR_REPORT.md` (overview, diagram, data, performance, compatibility, pricing, way forward) and `docs/EXPERIMENTS.md` first.
>
> **State as of 29 Oct 2026**
> - The phone web app is `docs/index.html`, one file using TensorFlow.js. It has a per-car healthy baseline, an autoencoder plus a feature check, "What's that noise?", and a "sounds most like" hint from a YAMNet + MLP head trained on 224 YouTube videos.
> - Fault-head accuracy on held-out videos: 22.6% balanced over 9 classes (chance 11%), top-3 60%, healthy vs fault 73%.
> - The raw YouTube audio (232 MB) is on my PC in `Desktop\Piranaware3\cars_download\raw_audio`. It is not in git.
> - The published claude.ai artifact (private) is "Engine Ear"; republish it from `docs/index.html` with the model files.
>
> **Next steps, in order**
> 1. A listening-review page to clean the YouTube labels, then retrain.
> 2. Enable GitHub Pages for live microphone use.
> 3. A shop pilot for real before/after-repair recordings.
> 4. Optional OBD-II (ELM327) input.
>
> Keep this separate from the Piranaware boat project.

---

**Useful facts for the next session**
- **Downloads:** the container can't reach YouTube. Run `tools/windows/*.bat` on the Windows PC (yt-dlp), then stage files back.
- **Model files:** the YAMNet weights in `docs/models/yamnet.bin` were exported from the official TFLite model by `tools/export_yamnet_web.py`. The TFLite stores int8 weights, which are dequantised. Training embeddings must use `modules/yamnet_embed.py` (browser-identical), not the TFLite interpreter.
- **Artifact hosting** serves only text types. Publish `yamnet.bin` as `models/yamnet.b64.txt` (base64); the app falls back to it automatically.
- **Tests:**
  - `pytest tests/test_anomaly.py`
  - `tests/test_web_parity.py`
  - `tests/test_yamnet_parity.py` (pass `--tfjs` with a local `tf.min.js` when offline)
  - `node tests/e2e_web.mjs`
