"""Render docs/ENGINE_EAR_REPORT.md (with its mermaid diagram) to docs/ENGINE_EAR_REPORT.pdf.
Needs: markdown, playwright (chromium), mmdc (mermaid-cli). Usage: python tools/build_report.py"""
import os
import re
import subprocess
import tempfile

import markdown
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "ENGINE_EAR_REPORT.md")
OUT = os.path.join(ROOT, "docs", "ENGINE_EAR_REPORT.pdf")
CSS = """
@page{size:A4;margin:14mm 13mm 16mm 13mm}
body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:9.3pt;line-height:1.45;color:#1b2024}
h1{font-size:21pt;color:#b8560f;margin:0 0 4pt}
h2{font-size:13.5pt;color:#b8560f;border-bottom:1.5px solid #ecd8c6;padding-bottom:2pt;margin:16pt 0 6pt;page-break-after:avoid}
h3{font-size:11pt;margin:10pt 0 4pt;page-break-after:avoid}
p{margin:4pt 0} em{color:#5e676d}
table{border-collapse:collapse;width:100%;font-size:8.3pt;margin:4pt 0 8pt}
tr{page-break-inside:avoid}
th{background:#f6e6d8;text-align:left}
th,td{border:1px solid #d9ddda;padding:3pt 5pt;vertical-align:top}
code{font-family:'DejaVu Sans Mono',monospace;font-size:8pt;background:#f3f4f2;padding:0 2pt}
ol,ul{margin:3pt 0;padding-left:16pt} li{margin:2pt 0}
.diagram{text-align:center;margin:6pt 0;page-break-inside:avoid}
.diagram img{max-width:100%;max-height:200mm}
a{color:#b8560f;text-decoration:none}
"""


def main():
    md = open(SRC, encoding="utf-8").read()
    tmp = tempfile.mkdtemp()
    imgs = []

    def render(m):
        i = len(imgs)
        src, svg = os.path.join(tmp, f"d{i}.mmd"), os.path.join(tmp, f"d{i}.svg")
        open(src, "w").write(m.group(1))
        subprocess.run(["mmdc", "-i", src, "-o", svg, "-b", "white", "-w", "1600"], check=True, capture_output=True)
        imgs.append(svg)
        return f'<div class="diagram"><img src="file://{svg}"></div>'

    md = re.sub(r"```mermaid\n(.*?)```", render, md, flags=re.S)
    body = markdown.markdown(md, extensions=["tables"])
    html = os.path.join(tmp, "report.html")
    open(html, "w", encoding="utf-8").write(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + html)
        pg.wait_for_timeout(500)
        pg.pdf(path=OUT, format="A4", print_background=True, display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:7pt;color:#777;width:100%;text-align:center'>Engine Ear · project report · <span class='pageNumber'></span>/<span class='totalPages'></span></div>",
               margin={"top": "14mm", "bottom": "16mm", "left": "13mm", "right": "13mm"})
        b.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
