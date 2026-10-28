"""Unsupervised car-engine anomaly detection (trained on normal recordings only).

Two detectors run side by side so they can be compared in the field:
  A. autoencoder.ConditionalAutoencoder  - log-mel autoencoder / IDNN (DCASE recipe)
  B. regression.PolynomialNormalModel    - per-feature polynomial normal-behaviour model
service.AnomalyService loads per-engine baselines and turns scores into a verdict.
See docs/ANOMALY_DETECTION_PLAN.md.
"""
