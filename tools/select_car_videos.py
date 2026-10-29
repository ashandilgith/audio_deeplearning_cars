"""Pick YouTube videos per fault class from yt-dlp search results.

Input : data/car_search_results*.txt (from cars_download/search_car_videos*.bat)
Output: data/car_sources.csv and data/car_urls.txt (for the download script)

Rules: 8 s - 10 min long; title must name the sound and not be a comparison
("X vs Y"), a compilation, music/ASMR, a game, or a non-car engine; a video
found under two different classes is dropped (ambiguous); at most PER_CLASS
per class (twice that for "normal", whose clips are short), preferring titles that promise the sound itself ("sound",
"noise", "what does ... sound like") and shorter videos.
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 22

KEYWORDS = {
    "normal": r"idle|idling|running|start|engine sound",
    "belt_squeal": r"belt|squeal|squeak",
    "engine_knock": r"knock",
    "valve_tick": r"tick|tapping|lifter|valve",
    "rough_idle": r"misfir|rough|shak|stumbl|sputter",
    "exhaust_leak": r"exhaust|manifold",
    "startup_rattle": r"rattl|timing chain|tensioner",
    "accessory_whine": r"whin|alternator|power steering|pulley|bearing",
    "trouble_starting": r"start|crank|click|starter",
}
BAD = re.compile(r"\bvs\b|versus|compilation|top \d|\btop10\b|asmr|sleep|hours?\b|relax|white noise|music|song|remix|"
                 r"forza|gta|beamng|simulator|game|toy|\brc\b|motorcycle|motorbike|scooter|bike|lawn|mower|boat|"
                 r"outboard|generator|chainsaw|tractor|jet ski|atv|dirt|minecraft|roblox|animation|#shorts.*#shorts",
                 re.I)
GOOD = re.compile(r"sound|noise|sounds like|what does|listen", re.I)


def main():
    hits, label = {}, None
    import glob
    lines = [l for p in sorted(glob.glob(os.path.join(DATA, "car_search_results*.txt")))
             for l in open(p, encoding="utf-8", errors="replace")]
    for line in lines:
        line = line.rstrip("\r\n")
        if line.startswith("###"):
            label, query = line[4:].split("|", 1)
            continue
        parts = line.split("\t")
        if len(parts) < 4 or not label:
            continue
        vid, dur, chan, title = parts[0], parts[1], parts[2], "\t".join(parts[3:])
        try:
            dur = float(dur)
        except ValueError:
            continue
        hits.setdefault(vid, {"labels": set(), "title": title, "dur": dur, "chan": chan, "queries": set()})
        hits[vid]["labels"].add(label)
        hits[vid]["queries"].add(query)
    rows = []
    for lab, kw in KEYWORDS.items():
        cands = []
        for vid, h in hits.items():
            if h["labels"] != {lab}:
                continue
            t = h["title"]
            if not (8 <= h["dur"] <= 600) or BAD.search(t) or not re.search(kw, t, re.I):
                continue
            score = (2 if GOOD.search(t) else 0) + (1 if h["dur"] <= 180 else 0) + len(h["queries"]) * .5
            cands.append((-score, h["dur"], vid))
        cands.sort()
        chans = {}
        for _, _, vid in cands:
            h = hits[vid]
            if chans.get(h["chan"], 0) >= 3:      # no more than 3 videos per channel per class
                continue
            chans[h["chan"]] = chans.get(h["chan"], 0) + 1
            rows.append({"video_id": vid, "label": lab, "title": h["title"], "duration_s": int(h["dur"]),
                         "channel": h["chan"], "query": "; ".join(sorted(h["queries"])), "include": 1,
                         "start_s": "", "end_s": "", "notes": ""})
            if sum(r["label"] == lab for r in rows) >= (2 * PER_CLASS if lab == "normal" else PER_CLASS):
                break
    with open(os.path.join(DATA, "car_sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(DATA, "car_urls.txt"), "w", newline="\r\n") as f:
        f.write("\n".join(f"https://www.youtube.com/watch?v={r['video_id']}" for r in rows) + "\n")
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    mins = {k: round(sum(r["duration_s"] for r in rows if r["label"] == k) / 60) for k in c}
    for k in KEYWORDS:
        print(f"{k:16s} {c.get(k, 0):3d} videos  {mins.get(k, 0):4d} min")
    print("total", len(rows))


if __name__ == "__main__":
    main()
