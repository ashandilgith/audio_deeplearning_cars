"""Check the Python front end (modules/web_features.py) matches the browser's melWindow.

Runs the web app headless (Playwright), feeds both the same 1 s signal and
compares the log-mel output. Run: python tests/test_web_parity.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.web_features import mel_window  # noqa: E402


def main():
    from playwright.sync_api import sync_playwright
    rng = np.random.default_rng(0)
    t = np.arange(16000) / 16000
    x = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * rng.standard_normal(16000)
         + 0.2 * np.sin(2 * np.pi * 3700 * t) * (np.sin(2 * np.pi * 25 * t) > 0)).astype(np.float32)
    py = mel_window(x)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.route("https://**/*", lambda r: r.fulfill(body="", content_type="text/plain"))
        pg.goto("file://" + os.path.join(ROOT, "docs", "index.html"))
        js = np.array(pg.evaluate("x => Array.from(window.EngineEar.melWindow(Float32Array.from(x)))",
                                  x.tolist()), dtype=np.float32).reshape(py.shape)
        b.close()
    err = float(np.abs(py - js).max())
    print(f"max abs difference (dB): {err:.5f}")
    assert err < 0.01, err


if __name__ == "__main__":
    main()
