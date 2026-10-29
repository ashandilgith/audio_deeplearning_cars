# Experiments

## C1: fault classification from YouTube audio (29 Oct 2026)

**Data**
- **Sources:** 240 YouTube videos found with yt-dlp searches (`tools/windows/`, `data/car_sources.csv`). 228 downloaded, 224 had usable audio.
- **Classes (9):**
  - normal (48 videos)
  - belt squeal
  - engine knock
  - valve/lifter tick
  - rough idle/misfire
  - exhaust leak
  - start-up rattle
  - accessory whine
  - trouble starting (12 usable videos; the others were mostly talking)

  The fault classes have 23–24 videos each unless stated.
- **Windows:** 1 s windows, first 4 minutes of each video, with silence and talking removed (`tools/build_car_dataset.py`). 5,934 windows in total.
- **Labels are the video's topic, not a listening check.** Fault videos also contain healthy idling and tool noise, so the labels are noisy.

**Protocol**
- Whole videos are held out: about 20% of each class's windows, never the same recording on both sides.
- Seeds 7, 8 and 9.
- Chance level for balanced accuracy is 0.111.

| Model | Balanced acc. | Top-3 | Fault vs normal | Video acc. |
|---|---|---|---|---|
| Small CNN from scratch on browser log-mel (`train_car_classifier.py`, seed 7) | 0.152 | 0.536 | 0.595 | 0.25 |
| YAMNet embeddings + logistic regression (`exp_car_heads.py`) | 0.220 ± 0.003 | 0.568 | 0.719 | 0.32 |
| **YAMNet embeddings + MLP (shipped, `train_yamnet_head.py`)** | **0.226** | **0.598** | **0.725** | 0.26 |

Per-class recall of the shipped head, summed over the three seeds:

| Class | Recall |
|---|---|
| exhaust leak | 0.37 |
| normal | 0.32 |
| accessory whine | 0.29 |
| belt squeal | 0.27 |
| trouble starting | 0.23 |
| rough idle | 0.18 |
| engine knock | 0.15 |
| valve tick | 0.12 |
| start-up rattle | 0.11 |

**What it means**
- Pretrained YAMNet (Google, AudioSet) clearly beats training from scratch on this data.
- It is still a weak classifier. The right fault is in its top 3 about 60% of the time, so the app shows it only as a "sounds most like" hint when the car's own baseline has flagged a change. In "What's that noise?" it is shown with that same caveat.
- **Biggest next gain: clean labels.** A listening review that keeps only the seconds where the fault is actually audible, like the boat project's vetting page, would help more than further model changes.
- **Also worth doing:**
  - Recordings of real cars with a known fault, taken by a mechanic.
  - More trouble-starting and start-up rattle videos.

**Browser parity**
- YAMNet runs in the phone app with tf.js. Its weights come from the official float TFLite model, dequantised to float16 (`tools/export_yamnet_web.py`, 7 MB).
- Training embeddings use exactly the same maths (`modules/yamnet_embed.py`).
- `tests/test_yamnet_parity.py` reports a cosine similarity of 1.0000 between browser and Python embeddings.
