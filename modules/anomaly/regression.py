"""Detector B: polynomial normal-behaviour model.

For every named target feature y (sound descriptors, block temperature, bus
voltage) we learn how it depends on the operating point of a HEALTHY engine:

    y ~ ridge( polynomial(rpm, temp, run time; degree d) + state offsets )

d in {1, 2, 3} and the ridge strength are picked per feature by grouped
K-fold cross-validation. The residual spread is estimated per state from the
OUT-OF-FOLD residuals (in-sample residuals are optimistically small and
would cause false alarms in the field). At run time:

    z_k   = (y_k - y_hat_k) / spread_k(state)
    score = mean_k z_k^2           (diagonal Mahalanobis distance)

and the largest |z| features explain the verdict. Inference needs only
numpy and a JSON of coefficients, so it can run in a phone browser or on a microcontroller.
This generalises modules/thermal.py (a degree-1 coolant-temp-vs-rpm model).
"""
from __future__ import annotations

import itertools
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from modules.anomaly.context import STATES, OperatingContext, runtime_norm

CONT_VARS = ("rpm", "temp", "runtime")          # continuous context variables
DEGREES = (1, 2, 3)
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
# Non-audio targets and the context variables they must NOT be regressed on.
SENSOR_TARGETS = {"temp_c": ("temp",), "voltage": ()}
# Smallest residual spread we believe, per target: sensor resolution plus day-to-day
# variation (water temperature, battery state). Stops a perfectly repeatable
# training session from making a 1 degree difference look like 8 sigma.
MIN_SPREAD = {"temp_c": 1.5, "voltage": 0.1}
DEFAULT_MIN_SPREAD = 0.05


def _cont(ctx: OperatingContext, max_rpm: float) -> Dict[str, Optional[float]]:
    return {
        "rpm": ctx.rpm / max_rpm if ctx.rpm else None,
        "temp": ctx.temp_c / 100.0 if ctx.temp_c is not None else None,
        "runtime": runtime_norm(ctx.run_time_s) if ctx.run_time_s is not None else None,
    }


def _monomials(n_vars: int, degree: int):
    out = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_vars), d):
            e = [0] * n_vars
            for v in combo:
                e[v] += 1
            out.append(tuple(e))
    return out


def _design(C: np.ndarray, S: np.ndarray, monos, states_used) -> np.ndarray:
    """C: (n, v) continuous context, S: (n,) state names -> design matrix (no intercept)."""
    cols = [np.prod(C ** np.array(m)[None, :], axis=1) for m in monos]
    for s in states_used[1:]:                      # first used state is the reference level
        cols.append((S == s).astype(float))
    return np.stack(cols, axis=1) if cols else np.zeros((len(C), 0))


def _ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Xs.T @ Xs + alpha * len(y) * np.eye(Xs.shape[1]), Xs.T @ (y - ym))
    coef = w / sd
    return coef, float(ym - mu @ coef)


def _robust_sd(r):
    r = np.asarray(r)
    if len(r) < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9)


def _folds(groups, k, seed=0):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups]), k


class _SubsetModel:
    """Normal-behaviour regressions using a fixed set of context variables."""

    def __init__(self, max_rpm: float, folds: int, vars: Sequence[str]):
        self.max_rpm = max_rpm
        self.folds = folds
        self.vars: List[str] = list(vars)
        self.targets: Dict[str, dict] = {}
        self.states_used: List[str] = []
        self.medians: Dict[str, float] = {}
        self.thresholds: Dict[str, dict] = {}

    # ------------------------------------------------------------------ data
    def _matrix(self, contexts: Sequence[OperatingContext]):
        rows = []
        for c in contexts:
            cv = _cont(c, self.max_rpm)
            rows.append([self.medians[v] if cv[v] is None else cv[v] for v in self.vars])
        C = np.array(rows, dtype=float).reshape(len(contexts), len(self.vars))
        S = np.array([c.state for c in contexts])
        return C, S

    # ------------------------------------------------------------------ fit
    def fit(self, feats: List[dict], contexts: List[OperatingContext], groups: Sequence):
        n = len(feats)
        groups = np.asarray(groups)
        conts = [_cont(c, self.max_rpm) for c in contexts]
        self.medians = {v: float(np.median([c[v] for c in conts if c[v] is not None])) for v in self.vars}
        self.states_used = [s for s in STATES if any(c.state == s for c in contexts)]
        C, S = self._matrix(contexts)
        fold, k = _folds(groups, self.folds)

        names = sorted({k for f in feats for k in f})
        for name in names:
            y = np.array([f.get(name, np.nan) for f in feats], dtype=float)
            ok = np.isfinite(y)
            if ok.sum() < 20:
                continue
            excl = SENSOR_TARGETS.get(name, ())
            vidx = [i for i, v in enumerate(self.vars) if v not in excl]
            best = None
            for deg in DEGREES:
                monos = _monomials(len(vidx), deg)
                if len(monos) + len(self.states_used) > ok.sum() / 5:
                    continue
                X = _design(C[:, vidx], S, monos, self.states_used)
                for alpha in ALPHAS:
                    oof = np.full(n, np.nan)
                    for j in range(k):
                        tr, te = ok & (fold != j), ok & (fold == j)
                        if tr.sum() < 10 or te.sum() == 0:
                            continue
                        coef, b = _ridge_fit(X[tr], y[tr], alpha)
                        oof[te] = X[te] @ coef + b
                    m = ok & np.isfinite(oof)
                    if m.sum() < 10:
                        continue
                    err = float(np.mean((y[m] - oof[m]) ** 2))
                    if best is None or err < best[0] - 1e-12:
                        best = (err, deg, alpha, oof)
            if best is None:  # no valid CV: constant model
                X = np.zeros((n, 0)); deg, alpha = 0, 1.0
                oof = np.full(n, np.nanmean(y[ok]))
            else:
                _, deg, alpha, oof = best
            monos = _monomials(len(vidx), deg) if deg else []
            X = _design(C[:, vidx], S, monos, self.states_used) if deg else np.zeros((n, 0))
            if X.shape[1]:
                coef, b = _ridge_fit(X[ok], y[ok], alpha)
            else:
                coef, b = np.zeros(0), float(np.mean(y[ok]))
            res = (y - oof)[ok & np.isfinite(oof)]
            st = S[ok & np.isfinite(oof)]
            floor = MIN_SPREAD.get(name, DEFAULT_MIN_SPREAD)
            g_sd = max(_robust_sd(res), floor)
            spread = {"all": g_sd}
            for s in self.states_used:
                r = res[st == s]
                spread[s] = max(_robust_sd(r), 0.5 * g_sd, floor) if len(r) >= 15 else g_sd
            self.targets[name] = dict(vars=[self.vars[i] for i in vidx], degree=deg, alpha=alpha,
                                      monomials=[list(m) for m in monos], coef=coef.tolist(),
                                      intercept=b, spread=spread)
        # thresholds from out-of-fold scores of the training windows (honest normal scores)
        oof_scores = self._oof_scores(feats, contexts, fold, k)
        self.thresholds = _percentile_thresholds(oof_scores, S, n_features=len(self.targets))
        return self

    def _oof_scores(self, feats, contexts, fold, k):
        """Re-score each fold with models refit without it (same chosen degree/alpha)."""
        C, S = self._matrix(contexts)
        z2 = np.zeros(len(feats)); cnt = np.zeros(len(feats))
        for name, t in self.targets.items():
            y = np.array([f.get(name, np.nan) for f in feats], dtype=float)
            ok = np.isfinite(y)
            vidx = [self.vars.index(v) for v in t["vars"]]
            monos = [tuple(m) for m in t["monomials"]]
            X = _design(C[:, vidx], S, monos, self.states_used) if t["degree"] else np.zeros((len(y), 0))
            pred = np.full(len(y), np.nan)
            for j in range(k):
                tr, te = ok & (fold != j), ok & (fold == j)
                if tr.sum() < 10 or te.sum() == 0:
                    continue
                if X.shape[1]:
                    coef, b = _ridge_fit(X[tr], y[tr], t["alpha"])
                    pred[te] = X[te] @ coef + b
                else:
                    pred[te] = y[tr].mean()
            sd = np.array([t["spread"].get(s, t["spread"]["all"]) for s in S])
            m = ok & np.isfinite(pred)
            z2[m] += ((y[m] - pred[m]) / sd[m]) ** 2
            cnt[m] += 1
        return z2 / np.maximum(cnt, 1)

    # ------------------------------------------------------------------ score
    def predict(self, contexts: List[OperatingContext]) -> Dict[str, np.ndarray]:
        C, S = self._matrix(contexts)
        out = {}
        for name, t in self.targets.items():
            vidx = [self.vars.index(v) for v in t["vars"]]
            if t["degree"]:
                X = _design(C[:, vidx], S, [tuple(m) for m in t["monomials"]], self.states_used)
                out[name] = X @ np.array(t["coef"]) + t["intercept"]
            else:
                out[name] = np.full(len(contexts), t["intercept"])
        return out

    def score(self, feats: List[dict], contexts: List[OperatingContext]) -> dict:
        """Per-window scores plus the features that deviate most."""
        # states never seen in training fall back to the reference level (and 'all' spread)
        pred = self.predict(contexts)
        n = len(feats)
        z = {}
        for name, t in self.targets.items():
            y = np.array([f.get(name, np.nan) for f in feats], dtype=float)
            sd = np.array([t["spread"].get(c.state, t["spread"]["all"]) for c in contexts])
            z[name] = (y - pred[name]) / sd
        Z = np.array([z[k] for k in z]) if z else np.zeros((0, n))
        valid = np.isfinite(Z)
        scores = (np.where(valid, Z, 0.0) ** 2).sum(0) / np.maximum(valid.sum(0), 1)
        mean_z = {k: float(np.nanmean(v)) for k, v in z.items() if np.isfinite(v).any()}
        top = sorted(mean_z.items(), key=lambda kv: -abs(kv[1]))[:3]
        return {"window_scores": scores, "z": z, "pred": pred,
                "top_features": [{"feature": k, "z": round(v, 2),
                                  "expected": round(float(np.nanmean(pred[k])), 2)} for k, v in top]}


class PolynomialNormalModel:
    """Detector B. One _SubsetModel per combination of available context
    variables, so a reading without (say) a temperature uses a model that was
    trained without temperature instead of guessing a value for it."""
    name = "regression"

    def __init__(self, max_rpm: float = 5500.0, folds: int = 5):
        self.max_rpm = max_rpm
        self.folds = folds
        self.vars: List[str] = []
        self.models: Dict[str, _SubsetModel] = {}

    @staticmethod
    def _key(vars_):
        return "+".join(vars_) or "none"

    def fit(self, feats: List[dict], contexts: List[OperatingContext], groups: Sequence = None,
            min_known: float = 0.8):
        """feats: per-window dicts of descriptor values (+ optional temp_c / voltage).
        groups: recording id per window, so CV never splits one recording."""
        n = len(feats)
        groups = np.asarray(groups if groups is not None else np.arange(n))
        conts = [_cont(c, self.max_rpm) for c in contexts]
        # a context variable is usable if known for most windows and actually varies
        self.vars = [v for v in CONT_VARS
                     if sum(c[v] is not None for c in conts) >= min_known * n
                     and np.std([c[v] for c in conts if c[v] is not None]) > 1e-6]
        for r in range(len(self.vars), -1, -1):
            for sub in itertools.combinations(self.vars, r):
                rows = [i for i, c in enumerate(conts) if all(c[v] is not None for v in sub)]
                if len(rows) < 30:
                    continue
                m = _SubsetModel(self.max_rpm, self.folds, sub)
                self.models[self._key(sub)] = m.fit([feats[i] for i in rows], [contexts[i] for i in rows],
                                                    groups[rows])
        return self

    @property
    def full(self) -> _SubsetModel:
        """The model using the most context variables that could be fitted."""
        return max(self.models.values(), key=lambda m: len(m.vars))

    @property
    def targets(self):
        return self.full.targets

    @property
    def thresholds(self):
        return self.full.thresholds

    def _model_for(self, ctx: OperatingContext) -> _SubsetModel:
        cv = _cont(ctx, self.max_rpm)
        avail = [v for v in self.vars if cv[v] is not None]
        for r in range(len(avail), -1, -1):
            for sub in itertools.combinations(avail, r):
                if self._key(sub) in self.models:
                    return self.models[self._key(sub)]
        return self.full

    def score(self, feats: List[dict], contexts: List[OperatingContext]) -> dict:
        """Per-window scores, the thresholds that apply, and the most deviant features."""
        n = len(feats)
        scores = np.zeros(n)
        keys = [self._key(self._model_for(c).vars) for c in contexts]
        main = max(set(keys), key=keys.count)
        z_all, pred_all = {}, {}
        for key in set(keys):
            idx = [i for i in range(n) if keys[i] == key]
            r = self.models[key].score([feats[i] for i in idx], [contexts[i] for i in idx])
            scores[idx] = r["window_scores"]
            if key == main:
                z_all, pred_all = r["z"], r["pred"]
        mean_z = {k: float(np.nanmean(v)) for k, v in z_all.items() if np.isfinite(v).any()}
        top = sorted(mean_z.items(), key=lambda kv: -abs(kv[1]))[:3]
        return {"window_scores": scores, "thresholds": self.models[main].thresholds, "context_vars": main,
                "top_features": [{"feature": k, "z": round(v, 2),
                                  "expected": round(float(np.nanmean(pred_all[k])), 2)} for k, v in top]}

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(max_rpm=self.max_rpm, vars=self.vars, models={
                k: dict(vars=m.vars, medians=m.medians, states_used=m.states_used,
                        targets=m.targets, thresholds=m.thresholds) for k, m in self.models.items()}),
                f, indent=1)

    @classmethod
    def load(cls, path: str) -> "PolynomialNormalModel":
        with open(path) as f:
            d = json.load(f)
        m = cls(max_rpm=d["max_rpm"])
        m.vars = d["vars"]
        for k, sd in d["models"].items():
            sm = _SubsetModel(m.max_rpm, m.folds, sd["vars"])
            sm.medians, sm.states_used = sd["medians"], sd["states_used"]
            sm.targets, sm.thresholds = sd["targets"], sd["thresholds"]
            m.models[k] = sm
        return m


def _percentile_thresholds(scores, states, warn_q=95.0, alarm_q=99.5, n_features=10) -> Dict[str, dict]:
    """Warning / alarm thresholds per state (and overall).

    Empirical percentiles of honest (out-of-fold) normal scores, but never below
    what pure Gaussian noise would give: if every z were N(0,1), the score
    (mean of k squared z values) follows chi2(k)/k. Small baselines otherwise
    produce thresholds that are too tight.
    """
    from scipy.stats import chi2
    k = max(n_features, 1)
    floor = {"warn": float(chi2.ppf(warn_q / 100, k) / k), "alarm": float(chi2.ppf(0.999, k) / k)}
    scores = np.asarray(scores)

    def th(v):
        return {"warn": max(float(np.percentile(v, warn_q)), floor["warn"]),
                "alarm": max(float(np.percentile(v, alarm_q)), floor["alarm"])}

    out = {"all": th(scores)}
    for s in set(states):
        v = scores[np.asarray(states) == s]
        if len(v) >= 30:
            out[s] = th(v)
    return out
