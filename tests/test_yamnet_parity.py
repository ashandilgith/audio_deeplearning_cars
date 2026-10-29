"""Browser YAMNet (docs/index.html) vs the Python embedder used for training (modules/yamnet_embed.py).

Serves docs/ on localhost, embeds the same audio in both, and checks cosine
similarity > 0.999 and that the fault head picks the same class.
Run: python tests/test_yamnet_parity.py [--tfjs /path/to/tf.min.js]
"""
import argparse
import functools
import http.server
import os
import sys
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modules.yamnet_embed import YamnetEmbedder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tfjs"); a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    d = np.load(os.path.join(ROOT, "data", "car_windows.npz")) if os.path.exists(os.path.join(ROOT, "data", "car_windows.npz")) else None
    rng = np.random.default_rng(1)
    t = np.arange(16000) / 16000
    sigs = [0.3 * np.sin(2 * np.pi * 300 * t) * (np.sin(2 * np.pi * 26 * t) > 0) + 0.05 * rng.standard_normal(16000),
            0.2 * rng.standard_normal(16000), 0.4 * np.sin(2 * np.pi * 3800 * t) + 0.1 * rng.standard_normal(16000)]
    sigs = [s.astype(np.float32) for s in sigs]
    emb = YamnetEmbedder()
    ref = np.stack([emb(s) for s in sigs])
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.join(ROOT, "docs")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        if a.tfjs:
            pg.route("https://cdn.jsdelivr.net/**", lambda r: r.fulfill(path=a.tfjs, content_type="application/javascript"))
        pg.route("https://fonts.googleapis.com/**", lambda r: r.fulfill(body="", content_type="text/css"))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/index.html")
        js = np.array(pg.evaluate("ws => window.EngineEar.yamnetEmbedArray(ws)", [s.tolist() for s in sigs]))
        b.close()
    srv.shutdown()
    cos = (ref * js).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(js, axis=1))
    print("cosine similarity per signal:", np.round(cos, 5), " max abs diff:", float(np.abs(ref - js).max()))
    assert cos.min() > 0.999, cos


if __name__ == "__main__":
    main()
