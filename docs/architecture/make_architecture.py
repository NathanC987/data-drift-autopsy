"""Draws the ResilientAI architecture diagram as SVG and renders a high-resolution PNG and a vector PDF.

Usage: python docs/architecture/make_architecture.py
Needs rsvg-convert and the Noto Sans font.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent
W, H = 1600, 870
FONT = "Noto Sans"
INK = "#0f172a"
MUTED = "#475569"
LINE = "#94a3b8"

STAGES = {
    "A": ("#2563eb", "#eef4ff"),
    "B": ("#7c3aed", "#f4efff"),
    "C": ("#d97706", "#fff6e8"),
    "D": ("#e11d48", "#fff0f3"),
    "EF": ("#059669", "#ebfaf3"),
    "RUN": ("#475569", "#f1f5f9"),
    "OUT": ("#0e7490", "#ecfafd"),
}

parts: list[str] = []


def text(x, y, s, size=12, weight="normal", color=INK, anchor="start", style="normal"):
    parts.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
        f'font-style="{style}" fill="{color}" text-anchor="{anchor}">{escape(s)}</text>'
    )


def panel(x, y, w, h, key, letter, name, question):
    color, tint = STAGES[key]
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{tint}" '
        f'stroke="{color}" stroke-opacity="0.35" stroke-width="1.5"/>'
    )
    if letter:
        bw = 26 if len(letter) == 1 else 40
        parts.append(f'<rect x="{x + 16}" y="{y + 16}" width="{bw}" height="26" rx="13" fill="{color}"/>')
        text(x + 16 + bw / 2, y + 34, letter, 13, "bold", "#ffffff", "middle")
        tx = x + 16 + bw + 10
    else:
        tx = x + 18
    text(tx, y + 34, name, 16.5, "bold", color)
    text(x + 18, y + 60, question, 12.5, color=MUTED, style="italic")


def card(x, y, w, title, lines, key, h=None):
    color, _ = STAGES[key]
    h = h or 24 + 18 + 17 * len(lines)
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#ffffff" '
        f'stroke="#e2e8f0" stroke-width="1" filter="url(#shadow)"/>'
    )
    parts.append(f'<rect x="{x}" y="{y + 10}" width="4" height="{h - 20}" rx="2" fill="{color}"/>')
    text(x + 16, y + 26, title, 14.5, "bold")
    for i, line in enumerate(lines):
        text(x + 16, y + 46 + 17 * i, line, 12, color=MUTED)
    return h


def chip(x, y, w, label, color):
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="13" fill="{color}" fill-opacity="0.1" '
        f'stroke="{color}" stroke-width="1.2"/>'
    )
    text(x + w / 2, y + 17.5, label, 11.5, "bold", color, "middle")


markers: dict[str, str] = {}


def marker_for(color):
    return markers.setdefault(color, f"arrow{len(markers)}")


def path(d, color=LINE, width=1.8, dash=None, arrow=True):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    marker = f' marker-end="url(#{marker_for(color)})"' if arrow else ""
    parts.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{extra}{marker}/>'
    )


def chevron(x, y):
    """Large arrow between two stage panels, centered at (x, y)."""
    parts.append(
        f'<path d="M{x - 12},{y - 16} L{x + 6},{y - 16} L{x + 18},{y} L{x + 6},{y + 16} '
        f'L{x - 12},{y + 16} L{x},{y} Z" fill="#cbd5e1"/>'
    )


# Title
text(50, 58, "ResilientAI: Drift Detection and Diagnosis System", 28, "bold")
text(
    50, 86,
    "Label-free drift detection, accuracy estimation and drift-type diagnosis with a stated false-alarm rate",
    15, color=MUTED,
)

# Stage panels
PY, PH = 112, 500
cols = {
    "A": (50, 230),
    "B": (323, 213),
    "C": (580, 270),
    "D": (895, 380),
    "EF": (1320, 230),
}
panel(*(cols["A"][0], PY, cols["A"][1], PH), "A", "A", "Inputs", "What is being monitored?")
panel(*(cols["B"][0], PY, cols["B"][1], PH), "B", "B", "Preparation", "Bring data to a common form")
panel(*(cols["C"][0], PY, cols["C"][1], PH), "C", "C", "Detection & estimation", "Did it change? How much accuracy is lost?")
panel(*(cols["D"][0], PY, cols["D"][1], PH), "D", "D", "Diagnosis and root cause", "What type of drift is it and why?")
panel(*(cols["EF"][0], PY, cols["EF"][1], PH), "EF", "E·F", "Acting on findings", "What should the team do?")

for left, right in (("A", "B"), ("B", "C"), ("C", "D"), ("D", "EF")):
    lx = cols[left][0] + cols[left][1]
    chevron((lx + cols[right][0]) / 2 - 3, PY + 250)


def stack(key, items, top=PY + 84, gap=None):
    x, w = cols[key]
    heights = [24 + 18 + 17 * len(lines) for _, lines in items]
    if gap is None:
        gap = (PY + PH - 18 - top - sum(heights)) / (len(items) - 1)
    y, pos = top, []
    for (title, lines), h in zip(items, heights):
        card(x + 14, y, w - 28, title, lines, key)
        pos.append((y, h))
        y += h + gap
    return pos


stack("A", [
    ("Deployed model", ["Logistic regression or", "gradient boosting"]),
    ("Labeled reference data", ["Data the model was", "built from"]),
    ("Production windows", ["Unlabeled: years, states", "or sensor batches"]),
    ("Audit labels (optional)", ["600 late labels", "per window"]),
])

stack("B", [
    ("Image embeddings", ["Frozen ResNet-18,", "512 features per image"]),
    ("Feature typing", ["Numeric vs nominal codes;", "codes are never numbers"]),
    ("Stratified split", ["60% train · 20% calibration", "20% test (control window)"]),
], gap=None)

stack("C", [
    ("Shift detectors", ["KS · PSI · MMD · confidence χ²", "Baselines: BBSD, domain classifier"]),
    ("Accuracy estimate", ["CBPE from model confidence", "Compared with AC, DoC, ATC"]),
    ("Feature localization", ["Per-feature tests with Holm,", "effect sizes, unseen categories"]),
])

stack("EF", [
    ("Reliability (E)", ["Risk of each prediction:", "confidence, instability,", "out-of-distribution score"]),
    ("Retrain triage (F)", ["Will a cheap retrain help?", "Importance-weighted", "validation, Holm"]),
    ("Fix by drift type", ["Covariate → reweight", "Prior → recalibrate", "Concept → fresh labels", "Noise → check labeling"]),
])

# Diagnosis panel internals
dx, dw = cols["D"]
red = STAGES["D"][0]
cw = 218
s1y = PY + 84
h1 = card(dx + 14, s1y, cw, "Stage 1 · label-free", ["Weighted Holm over feature,", "prediction and MMD tests", "BBSE + label-shift test"], "D")
chips_x = dx + 14 + cw + 20
chw = 112
chip(chips_x, s1y + 14, chw, "Covariate shift", red)
chip(chips_x, s1y + 50, chw, "Prior shift", red)
path(f"M{dx + 14 + cw},{s1y + 27} L{chips_x - 4},{s1y + 27}", red, 1.6)
path(f"M{dx + 14 + cw},{s1y + 63} L{chips_x - 4},{s1y + 63}", red, 1.6)

s2y = s1y + h1 + 46
path(f"M{dx + 14 + cw / 2},{s1y + h1} L{dx + 14 + cw / 2},{s2y - 4}", red, 1.6)
text(dx + 14 + cw / 2 + 10, s1y + h1 + 27, "nothing rejected", 11.5, color=red, style="italic")
h2 = card(dx + 14, s2y, cw, "Stage 2 · audit", ["Did accuracy drop?", "Does the drop depend on a", "feature beyond confidence?"], "D")
for i, label in enumerate(("Concept drift", "Label noise", "No drift")):
    cy = s2y + 6 + 32 * i
    chip(chips_x, cy, chw, label, red)
    path(f"M{dx + 14 + cw},{s2y + h2 / 2} C{dx + cw + 30},{s2y + h2 / 2} {dx + cw + 24},{cy + 13} {chips_x - 4},{cy + 13}", red, 1.4)

ny = s2y + h2 + 30
parts.append(f'<rect x="{dx + 14}" y="{ny - 18}" width="{dw - 28}" height="28" rx="8" fill="#ffffff" stroke="{red}" stroke-opacity="0.4" stroke-dasharray="4 3"/>')
text(dx + dw / 2, ny + 1, "α = 0.05 per stage → false verdicts ≤ 2α (measured 6.5%)", 11, "bold", red, "middle")

card(dx + 14, PY + PH - 18 - 76, dw - 28, "Root cause", ["SHAP attribution change with bootstrap intervals (tabular)", "Class activation maps for images"], "D")

# Audit labels feed stage 2 (dashed route below the panels)
ax0 = cols["A"][0] + cols["A"][1] - 14
ay = PY + PH - 18 - 59 / 2
gx1 = (cols["A"][0] + cols["A"][1] + cols["B"][0]) / 2 + 6
gx2 = (cols["C"][0] + cols["C"][1] + cols["D"][0]) / 2 + 4
by = PY + PH + 14
s2mid = s2y + h2 / 2
path(
    f"M{ax0},{ay} L{gx1 - 8},{ay} Q{gx1},{ay} {gx1},{ay + 8} L{gx1},{by - 8} Q{gx1},{by} {gx1 + 8},{by} "
    f"L{gx2 - 8},{by} Q{gx2},{by} {gx2},{by - 8} L{gx2},{s2mid + 8} Q{gx2},{s2mid} {gx2 + 8},{s2mid} L{dx + 10},{s2mid}",
    STAGES["A"][0], 1.6, "5 4",
)
text((gx1 + gx2) / 2, by + 15, "audit labels go only to Stage 2", 11.5, color=STAGES["A"][0], anchor="middle", style="italic")

# Lower panels: study runners and outputs
LY, LH = 668, 170
panel(50, LY, 560, LH, "RUN", "", "Study runners", "")
panel(650, LY, 900, LH, "OUT", "", "Outputs", "")

rc = card(68, LY + 54, 255, "Real-data studies", ["ACS Income and Employment,", "UCI gas sensor, CLEAR-10"], "RUN")
card(337, LY + 54, 255, "Injection benchmark", ["5 states × 10 seeds, known type", "Designs A (matched), B (held out)"], "RUN")
# runners -> inputs
path(f"M165,{LY} L165,{PY + PH + 4}", LINE, 1.8)
text(176, PY + PH + 28, "windows", 11.5, color=MUTED, style="italic")

oc = STAGES["OUT"][0]
jx, jy, jw = 668, LY + 54, 230
card(jx, jy, jw, "JSON report per study", ["Detections, estimates, verdicts,", "risks and timings per window"], "OUT")
card(950, LY + 44, 270, "Streamlit dashboard", ["Step-by-step story per dataset"], "OUT", h=56)
card(950, LY + 106, 270, "Results build", ["Every number is a named key"], "OUT", h=52)
card(1270, LY + 106, 262, "Paper and black book", ["No hand-typed numbers"], "OUT", h=52)
path(f"M{jx + jw},{jy + 28} L946,{jy + 28}", LINE, 1.8)
path(f"M{jx + jw},{jy + 28} L916,{jy + 28} Q924,{jy + 28} 924,{jy + 36} L924,{LY + 124} Q924,{LY + 132} 932,{LY + 132} L946,{LY + 132}", LINE, 1.8)
path(f"M1220,{LY + 132} L1266,{LY + 132}", LINE, 1.8)

# Analysis panels -> JSON report (bus just above the outputs panel)
bus = LY - 14
for key in ("C", "D", "EF"):
    x, w = cols[key]
    path(f"M{x + w / 2},{PY + PH} L{x + w / 2},{bus}", LINE, 1.8, arrow=False)
path(f"M{cols['C'][0] + cols['C'][1] / 2},{bus} L{cols['EF'][0] + cols['EF'][1] / 2},{bus}", LINE, 1.8, arrow=False)
path(f"M{jx + jw / 2 + 30},{bus} L{jx + jw / 2 + 30},{jy - 4}", LINE, 1.8)

marker_defs = "\n".join(
    f'  <marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,1 L9,5 L0,9 Z" fill="{color}"/></marker>'
    for color, mid in markers.items()
)
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
{marker_defs}
  <filter id="shadow" x="-10%" y="-20%" width="120%" height="150%">
    <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.08"/>
  </filter>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{chr(10).join(parts)}
</svg>
"""
(OUT / "architecture.svg").write_text(svg)
subprocess.run(
    ["rsvg-convert", "-z", "3", "-b", "white", "-o", str(OUT / "architecture.png"), str(OUT / "architecture.svg")],
    check=True,
)
subprocess.run(
    ["rsvg-convert", "-f", "pdf", "-o", str(OUT / "architecture.pdf"), str(OUT / "architecture.svg")],
    check=True,
)
print("wrote architecture.svg, architecture.png and architecture.pdf in", OUT)
