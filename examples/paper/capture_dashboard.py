"""Capture the dashboard figure of the PPEECS supplement (Fig. S2).

Start the dashboard on localhost first, then run this script from the repository root:

    streamlit run examples/dashboard/app.py --server.headless true --server.address 127.0.0.1 --server.port 8599
    python examples/paper/capture_dashboard.py

It drives headless Chromium over the DevTools protocol (needs `chromium` and the
`websockets` package), captures (a) the overview of the ACS tab and (b) the drift type
step at three times the CSS resolution, removes the explanatory paragraphs that the
caption repeats and writes paper/ppeecs/figures/FigS2.png.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

URL = "http://127.0.0.1:8599/"
OUT = Path("paper/ppeecs/figures/FigS2.png")
WIDTH, HEIGHT, SCALE = 1100, 16000, 3        # CSS pixels; the tall viewport avoids inner scrolling
FIG_WIDTH_MM = 174

HIDE = ("(()=>{const s=document.createElement('style'); s.textContent='.modebar-container,.modebar{display:none"
        " !important} header[data-testid=stHeader],[data-testid=stToolbar]{display:none !important}';"
        " document.head.appendChild(s);})()")
# page regions [x, y, w, h]: from the title to the next section, and the drift type step
OVERVIEW = ("(()=>{const h=[...document.querySelectorAll('h1')].find(e=>e.textContent.includes('Data Drift Autopsy'));"
            " const n=[...document.querySelectorAll('h1,h2,h3')].find(e=>e.textContent.includes('1. Initial setup'));"
            " const a=h.getBoundingClientRect(), b=n.getBoundingClientRect(); return [0, a.top-16, %d, b.top-a.top+4];})()"
            % WIDTH)
DRIFT_TYPE = ("(()=>{const h=[...document.querySelectorAll('h1,h2,h3,h4')].find(e=>e.textContent.includes('3b. Drift type'));"
              " const r=h.getBoundingClientRect(); const bx=[...document.querySelectorAll('div')].filter(d=>"
              "d.textContent.trim().startsWith('What the data shows') && d.getBoundingClientRect().top>r.top)"
              ".sort((x,y)=>x.getBoundingClientRect().top-y.getBoundingClientRect().top)[0].getBoundingClientRect();"
              " return [0, r.top-8, %d, bx.bottom-r.top+16];})()" % WIDTH)


async def capture(regions):
    prof = tempfile.mkdtemp()
    proc = subprocess.Popen(["chromium", "--headless=new", "--remote-debugging-port=9333", f"--user-data-dir={prof}",
                             "--hide-scrollbars", f"--window-size={WIDTH},{HEIGHT}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import websockets
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(ws_url, max_size=2 ** 28) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            await call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=SCALE,
                       mobile=False)
            await call("Page.navigate", url=URL)
            await asyncio.sleep(20)                 # let every chart render
            await call("Runtime.evaluate", expression=HIDE)
            await asyncio.sleep(1)
            images = []
            for js in regions:
                x, y, w, h = (await call("Runtime.evaluate", expression=js, returnByValue=True))["result"]["value"]
                shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                  clip={"x": x, "y": y, "width": w, "height": h, "scale": 1})
                images.append(Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB"))
            return images
    finally:
        proc.terminate()


def bands(im: Image.Image):
    """Vertical extents of the blocks of non-white rows (text lines, charts)."""
    rows = (np.array(im.convert("L")) < 235).any(axis=1)
    out, start = [], None
    for i, v in enumerate(rows):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 3:
                out.append((start, i))
            start = None
    return out


def vstack(parts, gap=0):
    out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts) + gap * (len(parts) - 1)), "white")
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + gap
    return out


def main() -> None:
    overview, drift = asyncio.run(capture([OVERVIEW, DRIFT_TYPE]))
    # overview: keep title, tabs and selectors, then the headline numbers and the step strip;
    # drop the dataset description (bands 6 to 8) and the focus window selector at the end.
    # Bands: 0 title, 1 subtitle, 2 tabs, 3 tab underline, 4 selector labels, 5 radio buttons,
    # 6-8 description, 9 metric labels, 10 metric values, 11 step strip, 12+ focus selector.
    b = bands(overview)
    if len(b) < 12:
        raise RuntimeError(f"unexpected overview layout ({len(b)} blocks)")
    pad = 3 * SCALE
    a = vstack([overview.crop((0, b[0][0] - 12 * pad, overview.width, b[5][1] + 6 * pad)),
                overview.crop((0, b[9][0] - 6 * pad, overview.width, b[11][1] + 4 * pad))])
    # drift type: keep the heading, then the matrix and its note; drop the method paragraph
    b = bands(drift)
    d = vstack([drift.crop((0, 0, drift.width, b[0][1] + 4 * pad)), drift.crop((0, b[4][0] - 8 * pad, drift.width, drift.height))])

    def hbox(im):
        box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
        return box[0], box[2]

    x0 = min(hbox(a)[0], hbox(d)[0]) - 20
    x1 = max(hbox(a)[1], hbox(d)[1]) + 20
    a, d = a.crop((x0, 0, x1, a.height)), d.crop((x0, 0, x1, d.height))
    lab, gap = 120, 70
    out = Image.new("RGB", (a.width + lab, a.height + d.height + gap), "white")
    out.paste(a, (lab, 0))
    out.paste(d, (lab, a.height + gap))
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype("LiberationSerif-Regular.ttf", 84)
    draw.line([(lab, a.height + gap // 2), (out.width, a.height + gap // 2)], fill=(205, 205, 205), width=4)
    draw.text((0, 40), "(a)", font=font, fill="black")
    draw.text((0, a.height + gap + 40), "(b)", font=font, fill="black")
    dpi = out.width / (FIG_WIDTH_MM / 25.4)
    out.save(OUT, dpi=(dpi, dpi))
    print(f"wrote {OUT}: {out.size[0]} x {out.size[1]} px, {dpi:.0f} dpi at {FIG_WIDTH_MM} mm")


if __name__ == "__main__":
    main()
