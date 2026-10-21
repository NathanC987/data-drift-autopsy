"""Build the Periodica Polytechnica EECS version of the paper as Word and PDF.

The journal works from Word files made with its template, so this script writes
paper/ppeecs/build/PPEECS_paper.docx (and the supplement) on top of
paper/ppeecs/template.docx, keeping the template's styles and header. The text
lives in paper/ppeecs/{paper,supplement}.md, a small Markdown dialect:

    # Heading {#sec:x}     ## Subheading      # Acknowledgement {-}
    *italic*  **bold**  $inline math$  [@key; @key]  @fig:x @tab:x @eq:x @sec:x
    $$ display math $$, {#eq:x}           (punctuation after the closing $$)
    ![caption](Fig01.png){#fig:x width=83}
    Table: caption {#tab:x}               followed by <<gen:tab_name>> or pipe rows
    1. item                               (numbered list)
    \\noindent text                       (paragraph without first-line indent)
    \\newpage                             (page break, used only in the supplement)

Numbers are the same \\res{key} values as the AGH paper (paper/generated/results.tex),
generated tables are parsed from paper/generated/tab_*.tex and math is converted to
Word equations by pandoc (cached in paper/ppeecs/math_cache.json). The PDF is made
with LibreOffice. Run from the repository root:

    python examples/paper/build_ppeecs.py              # paper and supplement
    python examples/paper/build_ppeecs.py --figures    # redraw the PPEECS figures first

PANDOC and TECTONIC may point to the binaries if they are not on PATH.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
PP = PAPER / "ppeecs"
FIGS = PP / "figures"
BUILD = PP / "build"
TEMPLATE = PP / "template.docx"
MATH_CACHE = PP / "math_cache.json"

# page geometry of the template (twips): A4, two columns of 83.5 mm
COL_W = 4734
PAGE_W = 9865
ABSTRACT_MAX_WORDS = 250

SECT_ONE = ('<w:sectPr><w:headerReference w:type="default" r:id="{hdr}"/>'
            '<w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
            '<w:pgMar w:top="1956" w:right="907" w:bottom="1418" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>'
            '<w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>')
SECT_COLS = ('<w:sectPr><w:headerReference w:type="default" r:id="{hdr}"/><w:type w:val="continuous"/>'
             '<w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
             '<w:pgMar w:top="1956" w:right="907" w:bottom="1418" w:left="1134" w:header="709" w:footer="709" w:gutter="0"/>'
             '<w:cols w:num="{n}" w:space="397"/><w:docGrid w:linePitch="360"/></w:sectPr>')

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_LINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
LINK_COLOR = "275B9B"   # colour of links in the journal's sample file


# --------------------------------------------------------------------------- #
# small helpers

def esc(s: str) -> str:
    return html.escape(s, quote=False)


def run(text: str, rpr: str = "") -> str:
    if not text:
        return ""
    props = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{props}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(style: str, body: str, ppr: str = "") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr}</w:pPr>{body}</w:p>'


DOLLAR = "\ue000"       # an escaped \$ survives math tokenizing as this character


def tex_value(s: str) -> str:
    """A generated LaTeX fragment (result value or table cell) in the Markdown dialect."""
    s = s.replace("\\$", DOLLAR).replace("{,}", ",").replace("\\%", "%").replace("\\&", "&")
    s = re.sub(r"\\nminus\s?", "\u2212", s)
    s = s.replace("$^{\\downarrow}$", "\u2193").replace("$^{\\uparrow}$", "\u2191")
    s = s.replace("$\\Delta$", "\u0394").replace("$\\to$", "\u2192").replace("\\checkmark", "\u2713")
    s = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", s)
    s = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", s)
    s = s.replace("\\ ", " ").replace("~", "\u00a0")
    return s


def plain_text(s: str) -> str:
    """Typographic text: dashes, apostrophes, non-breaking spaces."""
    s = s.replace("---", "\u2014").replace("--", "\u2013").replace("~", "\u00a0")
    s = re.sub(r"(?<=\w)'", "\u2019", s)
    return s.replace(DOLLAR, "$")


def png_size(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


# --------------------------------------------------------------------------- #
# results and generated tables

def load_results() -> Dict[str, str]:
    vals = {}
    for line in (PAPER / "generated" / "results.tex").read_text().splitlines():
        m = re.match(r"\\expandafter\\def\\csname res@(.+?)\\endcsname\{(.*)\}$", line)
        if m:
            vals[m.group(1)] = tex_value(m.group(2))
    return vals


@dataclass
class Cell:
    text: str
    span: int = 1


@dataclass
class Row:
    cells: List[Cell]
    rule_above: Optional[str] = None          # "heavy" or "light"
    cmid_below: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class Table:
    aligns: List[str]
    rows: List[Row]
    header_rows: int = 1


def split_cells(line: str) -> List[str]:
    out, depth, cur = [], 0, ""
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line):
            cur += line[i:i + 2]
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    out.append(cur)
    return [c.strip() for c in out]


def parse_colspec(spec: str) -> List[str]:
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r">\{[^}]*\}", "", spec)
    aligns = []
    i = 0
    while i < len(spec):
        ch = spec[i]
        if ch in "lrc":
            aligns.append(ch)
        elif ch in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            aligns.append("l")
            i = spec.index("}", i)
        i += 1
    return aligns


def parse_tabular(name: str) -> Table:
    """A generated booktabs tabular as rows of cells with rules."""
    src = (PAPER / "generated" / f"{name}.tex").read_text()
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%"))
    m = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*)\\end\{tabular\}", src, re.S)
    aligns = parse_colspec(m.group(1))
    body = m.group(2)
    rows: List[Row] = []
    pending: Optional[str] = None
    header_rows = 0
    for chunk in re.split(r"\\\\", body):
        chunk = chunk.strip()
        while True:
            rule = re.match(r"\\(toprule|midrule|bottomrule)\s*", chunk)
            cmid = re.match(r"\\cmidrule(?:\([a-z]*\))?\{(\d+)-(\d+)\}\s*", chunk)
            if rule:
                kind = rule.group(1)
                if kind == "midrule" and header_rows == 0:
                    header_rows = len(rows)
                if kind != "bottomrule":
                    pending = "heavy" if kind == "toprule" else "light"
                chunk = chunk[rule.end():]
            elif cmid:
                rows[-1].cmid_below.append((int(cmid.group(1)) - 1, int(cmid.group(2)) - 1))
                chunk = chunk[cmid.end():]
            else:
                break
        if not chunk:
            continue
        cells = []
        for c in split_cells(chunk):
            mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", c, re.S)
            if mc:
                cells.append(Cell(tex_value(mc.group(2)), int(mc.group(1))))
            else:
                cells.append(Cell(tex_value(c)))
        rows.append(Row(cells, pending))
        pending = None
    return Table(aligns, rows, max(header_rows, 1))


def parse_pipe_table(lines: List[str]) -> Table:
    rows: List[Row] = []
    aligns: List[str] = []
    header_rows = 0
    pending: Optional[str] = "heavy"
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            if not aligns:
                aligns = ["c" if c.startswith(":") and c.endswith(":") else "r" if c.endswith(":") else "l"
                          for c in cells]
                header_rows = len(rows)
            pending = "light"
            continue
        rows.append(Row([Cell(c) for c in cells], pending))
        pending = None
    return Table(aligns or ["l"] * len(rows[0].cells), rows, max(header_rows, 1))


def stack_models(t: Table) -> Table:
    """Methods table, one column wide: the two model families as stacked row groups."""
    groups = [c.text for c in t.rows[0].cells if c.span > 1]
    head = t.rows[1]
    n = (len(head.cells) - 1) // 2
    names = {"Ours (hierarchical tests)": "Ours", "Learned meta-classifier": "Meta-classifier",
             "Error-rate monitoring": "Error-rate monitoring", "Ours without MMD block": "Ours, no MMD",
             "Ours, audit null without confidence": "Ours, no confidence",
             "Ours, audit null with class cells": "Ours, class cells"}
    rows = [Row([Cell("")] + [Cell(c.text) for c in head.cells[1:1 + n]], "heavy")]
    for g, group in enumerate(groups):
        rows.append(Row([Cell(group, 1 + n)], "light"))
        for r in t.rows[2:]:
            first = r.cells[0].text
            cells = [Cell(names.get(first, first))] + [Cell(c.text) for c in r.cells[1 + g * n:1 + (g + 1) * n]]
            rows.append(Row(cells, "light" if r.rule_above == "light" and first.startswith("Ours ") else None))
    return Table(["l"] + ["r"] * n, rows, 1)


def stack_models_main(t: Table) -> Table:
    """Methods table without the ablation rows, which the supplement reports in full."""
    out = stack_models(t)
    out.rows = [r for r in out.rows if not r.cells[0].text.startswith("Ours,")]
    return out


def income_narrow(t: Table) -> Table:
    """ACS Income table, one column wide: one-letter detector headers (defined in the caption)."""
    short = {"KS": "K", "MMD": "M", "Conf.": "C", "BBSD": "B", "DC": "D", "Est. err.": "Est. err."}
    for r in t.rows:
        for c in r.cells:
            c.text = short.get(c.text, c.text).replace("CA 2014 (control)", "Control")
            if re.fullmatch(r"[A-Z]{2} \d{4}", c.text):
                c.text = c.text.replace(" ", "\u00a0")       # keep "CA 2015" on one line
            if c.text == "Alarms (LR)":
                c.text = "Alarms"
    # one Type column: the verdict of both models, or both verdicts where they differ
    for i, r in enumerate(t.rows):
        if i == 0:
            r.cells[-1] = Cell("")
            r.cmid_below = [cm for cm in r.cmid_below if cm[1] < len(t.aligns) - 2]
            continue
        lr, gb = r.cells[-2].text, r.cells[-1].text
        r.cells[-2:] = [Cell("Type" if i == 1 else (lr if lr == gb else f"{lr}/{gb}"))]
    t.aligns = t.aligns[:-1]
    return t


def estimators_short(t: Table) -> Table:
    """Estimator table, one column wide: short column headers."""
    short = {"No drift": "None", "Covariate": "Cov.", "Concept, noise": "Conc., noise", "ACS Inc.": "Inc.",
             "ACS Emp.": "Emp.", "Benchmark, design A": "Benchmark (design A)"}
    for r in t.rows:
        for c in r.cells:
            c.text = short.get(c.text, c.text)
    return t


TRANSFORMS = {"stack_models": stack_models, "stack_models_main": stack_models_main, "income_narrow": income_narrow,
              "estimators_short": estimators_short}

# RELP labels from the 2018 PUMS data dictionary
RELP_LABELS = {
    0: "Reference person", 1: "Husband/wife", 2: "Biological son or daughter", 3: "Adopted son or daughter",
    4: "Stepson or stepdaughter", 5: "Brother or sister", 6: "Father or mother", 7: "Grandchild",
    8: "Parent-in-law", 9: "Son-in-law or daughter-in-law", 10: "Other relative", 11: "Roomer or boarder",
    12: "Housemate or roommate", 13: "Unmarried partner", 14: "Foster child", 15: "Other nonrelative",
    16: "Institutionalized group quarters population", 17: "Noninstitutionalized group quarters population"}


def relp_map() -> Table:
    """The RELSHIPP to RELP mapping, read from the code that applies it."""
    from drift_autopsy.experiments.acs import RELSHIPP_TO_RELP
    rows = [Row([Cell("RELP"), Cell("Label (2018)"), Cell("RELSHIPP codes (2019 on)")], "heavy")]
    for code, label in RELP_LABELS.items():
        new = sorted(k for k, v in RELSHIPP_TO_RELP.items() if v == code)
        rows.append(Row([Cell(str(code)), Cell(label), Cell(", ".join(map(str, new)) or "none")],
                        "light" if code == 0 else None))
    return Table(["r", "l", "l"], rows, 1)


GENERATORS = {"relp_map": relp_map}


# --------------------------------------------------------------------------- #
# bibliography in the journal's style

ACCENTS = {'"': "\u0308", "'": "\u0301", "`": "\u0300", "~": "\u0303", "^": "\u0302", "c": "\u0327", "v": "\u030c"}


def tex_unicode(s: str) -> str:
    s = re.sub(r"\{\\l\}|\\l(?![a-zA-Z])", "\u0142", s)
    s = re.sub(r"\\i(?![a-zA-Z])", "i", s)

    def accent(m: "re.Match[str]") -> str:
        return unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)])

    s = re.sub(r"\{?\\([\"'`~^])\{?([A-Za-z])\}?\}?", accent, s)
    s = re.sub(r"\{?\\([cv])\{([A-Za-z])\}\}?", accent, s)
    s = s.replace("$\\chi^2$", "\u03c7\u00b2").replace("\\&", "&").replace("--", "\u2013")
    return s.replace("{", "").replace("}", "").strip()


def parse_bib(path: Path) -> Dict[str, Dict[str, str]]:
    text = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("%"))
    out: Dict[str, Dict[str, str]] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        depth, j = 0, text.index("{", m.start())
        for j in range(j, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
        body, fields, i = text[m.end():j], {"_type": m.group(1).lower()}, 0
        while True:
            fm = re.compile(r"\s*(\w+)\s*=\s*").match(body, i)
            if not fm:
                break
            i = fm.end()
            if body[i] == "{":
                depth, k = 0, i
                for k in range(i, len(body)):
                    depth += {"{": 1, "}": -1}.get(body[k], 0)
                    if depth == 0:
                        break
                value, i = body[i + 1:k], k + 1
            else:
                k = body.find(",", i)
                k = len(body) if k < 0 else k
                value, i = body[i:k], k
            fields[fm.group(1).lower()] = value.strip()
            i = body.find(",", i) + 1 if "," in body[i:] else len(body)
        out[m.group(2)] = fields
    return out


def initial(part: str) -> str:
    if "-" in part:
        return "-".join(initial(p) for p in part.split("-") if p)
    return part if (part.endswith(".") and len(part) <= 2) else part[0] + "."


def fmt_name(name: str) -> str:
    name = name.strip()
    if name.startswith("{") and name.endswith("}"):
        return tex_unicode(name[1:-1])
    if "," in name:
        last, first = (p.strip() for p in name.split(",", 1))
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    first = tex_unicode(first)
    return f"{tex_unicode(last)}, {' '.join(initial(p) for p in first.split())}".strip().rstrip(",")


def fmt_reference(e: Dict[str, str]) -> List[Tuple[str, Optional[str]]]:
    """One reference as (text, link) segments in the format of the journal's sample file."""
    names = [fmt_name(a) for a in re.split(r"\s+and\s+", e["author"])]
    authors = ", ".join(names)
    title = tex_unicode(e["title"])
    year = e.get("year", "")
    pages = tex_unicode(e.get("pages", ""))
    typ = e["_type"]
    if typ == "article":
        vol = tex_unicode(e.get("volume", "")) + (f"({e['number']})" if e.get("number") else "")
        loc = f"pp. {pages}" if pages else e.get("eid", "")
        text = f'{authors} "{title}", {tex_unicode(e["journal"])}, {vol}, {loc}, {year}.'
    elif typ == "inproceedings":
        where = f", {tex_unicode(e['address'])}" if e.get("address") else ""
        text = f'{authors} "{title}", In: {tex_unicode(e["booktitle"])}{where}, {year}'
        text += f", pp. {pages}." if pages else "."
    elif typ == "book":
        where = f", {tex_unicode(e['address'])}" if e.get("address") else ""
        text = f'{authors} "{title}", {tex_unicode(e["publisher"])}{where}, {year}.'
        if e.get("isbn"):
            text += f" ISBN {e['isbn']}"
    elif typ == "techreport":
        where = f", {tex_unicode(e['address'])}" if e.get("address") else ""
        text = f'{authors} "{title}", {tex_unicode(e["institution"])}{where}, Rep. {e["number"]}, {year}.'
    elif typ == "online":
        text = f'{authors} "{title}",'
    else:
        raise ValueError(f"unsupported entry type {typ}")
    segs: List[Tuple[str, Optional[str]]] = [(text + " ", None)]
    if e.get("doi"):
        url = f"https://doi.org/{e['doi']}"
        segs.append((url, url))
    elif e.get("url"):
        segs += [("[online] Available at: ", None), (e["url"], e["url"]), (f" [Accessed: {e['urldate']}]", None)]
    return segs


# --------------------------------------------------------------------------- #
# math: LaTeX to Word equations through pandoc, cached

class MathEngine:
    def __init__(self) -> None:
        self.cache: Dict[str, str] = json.loads(MATH_CACHE.read_text()) if MATH_CACHE.exists() else {}
        self.missing: List[str] = []

    def get(self, latex: str, display: bool = False) -> str:
        key = ("D:" if display else "I:") + latex.strip()
        if key in self.cache:
            return self.cache[key]
        if key not in self.missing:
            self.missing.append(key)
        return run("[math]")

    def convert_missing(self) -> None:
        if not self.missing:
            return
        exe = os.environ.get("PANDOC") or shutil.which("pandoc")
        if not exe:
            raise RuntimeError("pandoc is needed to convert new math (set PANDOC)")
        with tempfile.TemporaryDirectory() as tmp:
            md = Path(tmp) / "m.md"
            md.write_text("\n\n".join(("$$" + k[2:] + "$$") if k.startswith("D:") else ("$" + k[2:] + "$")
                                      for k in self.missing) + "\n")
            out = Path(tmp) / "m.docx"
            subprocess.run([exe, "-f", "markdown", "-t", "docx", "-o", str(out), str(md)], check=True)
            xml = zipfile.ZipFile(out).read("word/document.xml").decode()
        paras = re.findall(r"<w:p>.*?</w:p>|<w:p .*?</w:p>", xml, re.S)
        maths = [re.search(r"<m:oMath>.*?</m:oMath>", p, re.S) for p in paras]
        maths = [m.group(0) for m in maths if m]
        if len(maths) != len(self.missing):
            raise RuntimeError(f"pandoc returned {len(maths)} equations for {len(self.missing)} snippets")
        for k, m in zip(self.missing, maths):
            self.cache[k] = m
        self.missing = []
        MATH_CACHE.write_text(json.dumps(self.cache, indent=0, sort_keys=True, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# source parsing

@dataclass
class Block:
    kind: str                     # heading, para, eq, fig, table, list, abstract
    text: str = ""
    level: int = 0
    attrs: Dict[str, str] = field(default_factory=dict)
    lines: List[str] = field(default_factory=list)
    number: str = ""


def parse_attrs(s: str) -> Dict[str, str]:
    out = {}
    for tok in (s or "").split():
        if tok.startswith("#"):
            out["id"] = tok[1:]
        elif tok == "-":
            out["unnumbered"] = "1"
        elif "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def parse_source(path: Path) -> Tuple[Dict[str, List[str]], List[Block]]:
    text = "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("%"))
    front: Dict[str, List[str]] = {}
    if text.startswith("---"):
        head, text = text[3:].split("\n---", 1)
        for line in head.strip().splitlines():
            k, v = line.split(":", 1)
            front.setdefault(k.strip(), []).append(v.strip())
    blocks: List[Block] = []
    for chunk in re.split(r"\n\s*\n", text.strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk == "\\newpage":
            blocks.append(Block("pagebreak"))
        elif chunk.startswith("::: abstract"):
            blocks.append(Block("abstract", " ".join(chunk.splitlines()[1:-1])))
        elif chunk.startswith("#"):
            m = re.match(r"(#{1,3})\s+(.*?)\s*(?:\{([^}]*)\})?$", chunk)
            blocks.append(Block("heading", m.group(2), len(m.group(1)), parse_attrs(m.group(3))))
        elif chunk.startswith("$$"):
            m = re.match(r"\$\$(.*?)\$\$\s*([.,;]?)\s*(?:\{([^}]*)\})?$", chunk, re.S)
            b = Block("eq", " ".join(m.group(1).split()), attrs=parse_attrs(m.group(3)))
            b.attrs["punct"] = m.group(2)
            blocks.append(b)
        elif chunk.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*?)\)\{([^}]*)\}$", chunk, re.S)
            b = Block("fig", " ".join(m.group(1).split()), attrs=parse_attrs(m.group(3)))
            b.attrs["file"] = m.group(2)
            blocks.append(b)
        elif chunk.startswith("Table:"):
            lines = chunk.splitlines()
            m = re.match(r"Table:\s*(.*?)\s*\{([^}]*)\}$", lines[0])
            blocks.append(Block("table", m.group(1), attrs=parse_attrs(m.group(2)), lines=lines[1:]))
        elif re.match(r"\d+\.\s", chunk):
            items = re.split(r"\n(?=\d+\.\s)", chunk)
            blocks.append(Block("list", lines=[" ".join(re.sub(r"^\d+\.\s+", "", i).split()) for i in items]))
        else:
            blocks.append(Block("para", " ".join(chunk.split())))
    return front, blocks


# --------------------------------------------------------------------------- #
# document writer

INLINE = re.compile(
    r"(?P<math>\$[^$]+\$)|(?P<bold>\*\*.+?\*\*)|(?P<ital>\*[^*\s][^*]*?\*)|(?P<cite>\[@[^\]]+\])"
    r"|(?P<ref>@(?:m:|s:)?(?:sec|fig|tab|eq):[A-Za-z0-9_-]+)")


class Doc:
    def __init__(self, src: Path, results: Dict[str, str], math: MathEngine, bib: Dict[str, Dict[str, str]],
                 prefix: str = "", columns: int = 2, main_labels: Optional[Dict[str, str]] = None,
                 supp_labels: Optional[Dict[str, str]] = None) -> None:
        self.front, self.blocks = parse_source(src)
        self.src = src
        self.res = results
        self.math = math
        self.bib = bib
        self.prefix = prefix
        self.columns = columns
        self.main_labels = main_labels or {}
        self.supp_labels = supp_labels or {}
        self.labels: Dict[str, str] = {}
        self.order: Dict[str, int] = {}        # block index of each labelled block
        self.first_ref: Dict[str, int] = {}    # block index of the first reference to a label
        self.cites: List[str] = []
        self.rels: List[Tuple[str, str, str, bool]] = []   # id, type, target, external
        self.media: List[Tuple[str, Path]] = []
        self.captions: List[Tuple[str, str]] = []          # ("Fig. 1", caption text)
        self.used_res: set = set()
        self.cur = 0
        self.number_blocks()

    # ---- numbering -------------------------------------------------------- #
    def number_blocks(self) -> None:
        sec = [0, 0, 0]
        counts = {"fig": 0, "tab": 0, "eq": 0}
        for i, b in enumerate(self.blocks):
            if b.kind == "heading" and "unnumbered" not in b.attrs:
                sec[b.level - 1] += 1
                for j in range(b.level, 3):
                    sec[j] = 0
                b.number = self.prefix + ".".join(str(n) for n in sec[:b.level])
            elif b.kind in ("fig", "table", "eq"):
                kind = "tab" if b.kind == "table" else b.kind
                counts[kind] += 1
                b.number = f"{self.prefix}{counts[kind]}"
            if "id" in b.attrs:
                if b.attrs["id"] in self.labels:
                    raise ValueError(f"duplicate label {b.attrs['id']}")
                self.labels[b.attrs["id"]] = b.number
                self.order[b.attrs["id"]] = i

    # ---- inline text ------------------------------------------------------ #
    def substitute(self, text: str) -> str:
        def res(m: "re.Match[str]") -> str:
            k = m.group(1)
            if k not in self.res:
                raise KeyError(f"{self.src.name}: undefined result key {k}")
            self.used_res.add(k)
            return self.res[k]
        text = text.replace("\\$", DOLLAR)
        text = re.sub(r"\\res\{([^}]+)\}", res, text)
        return re.sub(r'"([^"]*)"', "\u201c\\1\u201d", text)

    def cite(self, group: str) -> str:
        keys = [k.strip().lstrip("@") for k in group[1:-1].split(";")]
        nums = []
        for k in keys:
            if k not in self.bib:
                raise KeyError(f"{self.src.name}: unknown citation {k}")
            if k not in self.cites:
                self.cites.append(k)
            nums.append(self.cites.index(k) + 1)
        nums = sorted(set(nums))
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}\u2013{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(parts) + "]"

    def ref(self, token: str) -> str:
        label = token[1:]
        if label.startswith("s:"):
            if label[2:] not in self.supp_labels:
                raise KeyError(f"{self.src.name}: unknown supplement label {label}")
            return self.supp_labels[label[2:]]
        if label.startswith("m:"):
            if label[2:] not in self.main_labels:
                raise KeyError(f"{self.src.name}: unknown main-paper label {label}")
            return self.main_labels[label[2:]]
        if label not in self.labels:
            raise KeyError(f"{self.src.name}: unknown label {label}")
        self.first_ref.setdefault(label, self.cur)
        return self.labels[label]

    def inline(self, text: str, rpr: str = "", subst: bool = True) -> str:
        if subst:
            text = self.substitute(text)
        out, pos = [], 0
        for m in INLINE.finditer(text):
            out.append(run(plain_text(text[pos:m.start()]), rpr))
            tok = m.group(0)
            if m.group("math"):
                out.append(self.math.get(tok[1:-1].replace(DOLLAR, "\\$")))
            elif m.group("bold"):
                out.append(self.inline(tok[2:-2], rpr + "<w:b/>", False))
            elif m.group("ital"):
                out.append(self.inline(tok[1:-1], rpr + "<w:i/>", False))
            elif m.group("cite"):
                out.append(run(self.cite(tok), rpr))
            else:
                out.append(run(self.ref(tok), rpr))
            pos = m.end()
        out.append(run(plain_text(text[pos:]), rpr))
        return "".join(out)

    def link(self, url: str, text: str) -> str:
        rid = f"rIdL{len(self.rels) + 1}"
        self.rels.append((rid, REL_LINK, url, True))
        return (f'<w:hyperlink r:id="{rid}" w:history="1">'
                f'<w:r><w:rPr><w:color w:val="{LINK_COLOR}"/></w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'
                "</w:hyperlink>")

    # ---- blocks ----------------------------------------------------------- #
    def front_matter(self) -> str:
        f = self.front
        title = re.sub(r'"([^"]*)"', "\u201c\\1\u201d", f["title"][0])
        out = [para("PPTitle", run(title, "<w:b/><w:bCs/>"), "<w:rPr><w:b/><w:bCs/></w:rPr>")]
        authors = []
        for i, a in enumerate(f.get("author", [])):
            name, mark = (p.strip() for p in a.split(";"))
            authors.append(run(("" if i == 0 else ", ") + name) + run(mark, '<w:vertAlign w:val="superscript"/>'))
        out.append(para("PPAuthors", "".join(authors)))
        for a in f.get("affiliation", []):
            mark, text = (p.strip() for p in a.split(";", 1))
            sup = '<w:vertAlign w:val="superscript"/>'
            out.append(para("PPAffiliation", run(mark, sup) + f"<w:r><w:rPr>{sup}</w:rPr><w:tab/></w:r>" + run(text)))
        if f.get("corresponding"):
            mail = f["corresponding"][0]
            sup = '<w:b/><w:vertAlign w:val="superscript"/>'
            out.append(para("PPAffiliation", run("*", sup) + f"<w:r><w:rPr>{sup}</w:rPr><w:tab/></w:r>"
                            + run("Corresponding author, e-mail: ") + self.link("mailto:" + mail, mail),
                            '<w:spacing w:after="80"/>'))
        for note in f.get("note", []):
            out.append(para("PPAffiliation", self.inline(note), '<w:spacing w:after="80"/>'))
        return "".join(out)

    def abstract(self, b: Block) -> str:
        words = len(re.findall(r"\S+", re.sub(r"\\res\{[^}]+\}", "0", b.text)))
        if words > ABSTRACT_MAX_WORDS:
            raise ValueError(f"abstract has {words} words (limit {ABSTRACT_MAX_WORDS})")
        if "[@" in b.text:
            raise ValueError("the abstract must not contain citations")
        self.abstract_words = words
        kw = self.front["keywords"][0]
        n_kw = len([k for k in kw.split(",") if k.strip()])
        if not 3 <= n_kw <= 5 or kw != kw.lower():
            raise ValueError("give 3 to 5 lowercase keywords")
        s = "PPAbstractandKeywords"
        return (para(s, run("Abstract", "<w:b/>")) + para(s, self.inline(b.text))
                + para(s, run("Keywords", "<w:b/>"), "<w:rPr><w:b/></w:rPr>")
                + para(s, run(kw), '<w:spacing w:after="360"/>'))

    def heading(self, b: Block) -> str:
        label = (b.number + " " if b.number else "") + b.text
        brk = "<w:pageBreakBefore/>" if getattr(self, "page_break_next", False) else ""
        self.page_break_next = False
        return para("PPHeading", self.inline(label), "<w:keepNext/>" + brk)

    def paragraph(self, b: Block, first: bool) -> str:
        text = b.text
        noindent = first or text.startswith("\\noindent")
        text = text.replace("\\noindent", "").strip()
        return para("PPBodyMainText", self.inline(text), '<w:ind w:firstLine="0"/>' if noindent else "")

    def numbered_list(self, b: Block) -> str:
        return "".join(para("PPListingNumbered", self.inline(item), '<w:jc w:val="both"/>') for item in b.lines)

    def equation(self, b: Block) -> str:
        width = COL_W if self.columns == 2 else PAGE_W
        tabs = f'<w:tabs><w:tab w:val="center" w:pos="{width // 2}"/><w:tab w:val="right" w:pos="{width}"/></w:tabs>'
        body = ("<w:r><w:tab/></w:r>" + self.math.get(b.text, display=True) + run(b.attrs.get("punct", ""))
                + "<w:r><w:tab/></w:r>" + run(f"({b.number})"))
        # automatic line height: the template's exact 14 pt would clip limits and fractions
        return para("PPEquation", body, tabs + '<w:spacing w:before="80" w:after="80" w:line="240" w:lineRule="auto"/>')

    def figure(self, b: Block) -> str:
        path = FIGS / b.attrs["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        w_px, h_px = png_size(path)
        width_mm = float(b.attrs.get("width", 83))
        cx = int(width_mm * 36000)
        cy = int(cx * h_px / w_px)
        n = len(self.media) + 1
        rid = f"rIdI{n}"
        self.media.append((f"media/image{n}.png", path))
        self.rels.append((rid, REL_IMAGE, f"media/image{n}.png", False))
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="Figure {b.number}" descr="{esc(b.attrs["file"])}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{NS_A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="{NS_A}"><a:graphicData uri="{NS_PIC}"><pic:pic xmlns:pic="{NS_PIC}">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{esc(b.attrs["file"])}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            "</wp:inline></w:drawing></w:r>")
        cap = self.inline(b.text)
        self.captions.append((f"Fig. {b.number}", b.text))
        return (para("PPFigureTableCaption", drawing,
                     '<w:keepNext/><w:spacing w:before="100" w:after="40" w:line="240" w:lineRule="auto"/>')
                + para("PPFigureTableCaption", run(f"Fig. {b.number}", "<w:b/>") + run(" ") + cap,
                       '<w:keepLines/><w:spacing w:after="120"/>'))

    def table(self, b: Block) -> str:
        src = b.lines
        gen = re.match(r"<<gen:(\w+)(?:\s+(\w+))?>>", src[0].strip()) if src else None
        code = re.match(r"<<code:(\w+)>>", src[0].strip()) if src else None
        if code:
            t = GENERATORS[code.group(1)]()
        elif gen:
            t = parse_tabular(gen.group(1))
            if gen.group(2):
                t = TRANSFORMS[gen.group(2)](t)
        else:
            t = parse_pipe_table(src)
        cap = self.inline(b.text)
        self.captions.append((f"Table {b.number}", b.text))
        out = para("PPFigureTableCaption", run(f"Table {b.number}", "<w:b/>") + run(" ") + cap,
                   '<w:keepNext/><w:keepLines/><w:spacing w:before="160" w:after="60"/>')
        avail = int(b.attrs.get("width", 0)) * 5669 // 100 if b.attrs.get("width") else (
            COL_W if self.columns == 2 and b.attrs.get("span") != "2" else PAGE_W)
        out += self.table_xml(t, avail)
        return out + '<w:p><w:pPr><w:spacing w:after="0" w:line="160" w:lineRule="exact"/></w:pPr></w:p>'

    def table_xml(self, t: Table, avail: int) -> str:
        ncol = len(t.aligns)
        rendered = [[self.inline(c.text) for c in r.cells] for r in t.rows]
        widths = column_widths(t, avail)
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        rows_xml = []
        last = len(t.rows) - 1
        for ri, (r, cells) in enumerate(zip(t.rows, rendered)):
            col = 0
            tcs = []
            for c, xml in zip(r.cells, cells):
                borders = []
                if ri == 0 or r.rule_above:
                    sz = 8 if (ri == 0 or r.rule_above == "heavy") else 4
                    borders.append(f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="auto"/>')
                bottom = None
                if ri == last:
                    bottom = 8
                elif any(a <= col and col + c.span - 1 <= b_ for a, b_ in r.cmid_below):
                    bottom = 4
                if bottom:
                    borders.append(f'<w:bottom w:val="single" w:sz="{bottom}" w:space="0" w:color="auto"/>')
                align = "center" if c.span > 1 else {"l": "left", "r": "right", "c": "center"}[t.aligns[col]]
                if ri < t.header_rows and c.span == 1 and col > 0:
                    align = "center" if t.aligns[col] == "c" else align
                w = sum(widths[col:col + c.span])
                span = f'<w:gridSpan w:val="{c.span}"/>' if c.span > 1 else ""
                keep = "<w:keepNext/>" if ri < last else ""
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{span}'
                           f'<w:tcBorders>{"".join(borders)}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                           + para("PPTableBody", xml, f'{keep}<w:spacing w:before="10" w:after="10"/><w:jc w:val="{align}"/>')
                           + "</w:tc>")
                col += c.span
            if col != ncol:
                raise ValueError(f"table row {ri} has {col} columns, expected {ncol}")
            trpr = "<w:cantSplit/>" + ("<w:tblHeader/>" if ri < t.header_rows else "")
            rows_xml.append(f"<w:tr><w:trPr>{trpr}</w:trPr>{''.join(tcs)}</w:tr>")
        return ('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
                '<w:tblCellMar><w:left w:w="45" w:type="dxa"/><w:right w:w="45" w:type="dxa"/></w:tblCellMar>'
                '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>'
                f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(rows_xml)}</w:tbl>')

    def references(self) -> str:
        out = [para("PPHeading", run("References"), "<w:keepNext/>")]
        for k in self.cites:
            body = "".join(self.link(url, text) if url else run(text) for text, url in fmt_reference(self.bib[k]))
            out.append(para("PPReferences", body))
        return "".join(out)

    # ---- assembly --------------------------------------------------------- #
    def body(self, hdr: str) -> str:
        self.cites, self.rels, self.media, self.captions, self.first_ref = [], [], [], [], {}
        out = [self.front_matter()]
        prev = "heading"
        two = SECT_COLS.format(hdr=hdr, n=self.columns)
        started = False
        for i, b in enumerate(self.blocks):
            self.cur = i
            if b.kind == "abstract":
                out.append(self.abstract(b))
                continue
            if not started:
                out.append(f'<w:p><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="exact"/>{SECT_ONE.format(hdr=hdr)}</w:pPr></w:p>')
                started = True
            if b.kind == "heading":
                if b.text == "References" and "unnumbered" in b.attrs:
                    out.append(self.references())
                else:
                    out.append(self.heading(b))
            elif b.kind == "para":
                out.append(self.paragraph(b, prev == "heading"))
            elif b.kind == "list":
                out.append(self.numbered_list(b))
            elif b.kind == "pagebreak":
                # set on the next heading, so a full page never leaves an empty one behind
                self.page_break_next = True
                continue
            elif b.kind == "eq":
                out.append(self.equation(b))
            elif b.kind == "fig":
                out.append(self.figure(b))
            elif b.kind == "table":
                if b.attrs.get("span") == "2" and self.columns == 2:
                    # a table across both columns sits in its own one-column section
                    brk = '<w:p><w:pPr><w:spacing w:after="0" w:line="20" w:lineRule="exact"/>{}</w:pPr></w:p>'
                    out.append(brk.format(SECT_COLS.format(hdr=hdr, n=2)))
                    out.append(self.table(b))
                    out.append(brk.format(SECT_COLS.format(hdr=hdr, n=1)))
                else:
                    out.append(self.table(b))
            prev = b.kind
        if not any(b.kind == "heading" and b.text == "References" for b in self.blocks):
            out.append(self.references())
        return "".join(out) + two

    def check_order(self) -> List[str]:
        problems = []
        for label, idx in self.order.items():
            if not label.startswith(("fig:", "tab:")):
                continue
            if label not in self.first_ref:
                problems.append(f"{label} is never referenced")
            elif self.first_ref[label] > idx:
                problems.append(f"{label} is first referenced after it appears")
        for label in self.labels:
            if label.startswith("eq:") and label not in self.first_ref:
                problems.append(f"{label} is never referenced")
        return problems

    def write(self, out: Path, title: str) -> None:
        zin = zipfile.ZipFile(TEMPLATE)
        parts = {n: zin.read(n) for n in zin.namelist()}
        doc = parts["word/document.xml"].decode()
        rels = parts["word/_rels/document.xml.rels"].decode()
        hdr = re.search(r'Id="(\w+)" Type="[^"]*/header"', rels).group(1)
        self.math.missing = []
        self.body(hdr)                          # first pass collects the math
        self.math.convert_missing()
        body = self.body(hdr)
        if self.math.missing:
            raise RuntimeError("math left unconverted")
        root = re.search(r"<w:document[^>]*>", doc).group(0)
        parts["word/document.xml"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                                      f"{root}<w:body>{body}</w:body></w:document>").encode()
        rels = re.sub(r'<Relationship Id="\w+" Type="[^"]*/hyperlink"[^>]*/>', "", rels)
        extra = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{esc(tg)}"' + (' TargetMode="External"' if ext else "")
                        + "/>" for i, t, tg, ext in self.rels)
        parts["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", extra + "</Relationships>").encode()
        ct = parts["[Content_Types].xml"].decode()
        if 'Extension="png"' not in ct:
            ct = ct.replace("<Default Extension=\"xml\"", '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"')
        parts["[Content_Types].xml"] = ct.encode()
        for name, path in self.media:
            parts["word/" + name] = path.read_bytes()
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        authors = "; ".join(a.split(";")[0].strip() for a in self.front.get("author", []))
        parts["docProps/core.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{esc(title)}</dc:title><dc:creator>{esc(authors)}</dc:creator>"
            f"<cp:lastModifiedBy>{esc(authors.split(';')[0])}</cp:lastModifiedBy><cp:revision>1</cp:revision>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>').encode()
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in zin.namelist() + [n for n in parts if n not in zin.namelist()]:
                z.writestr(name, parts[name])
        print(f"wrote {out.relative_to(ROOT)}: {len(self.cites)} references, {len(self.media)} figures, "
              f"{len(self.used_res)} result keys")


def text_width(text: str) -> float:
    """Width of 8 pt Times text in twips (Liberation Serif has the same metrics)."""
    font = _font()
    plain = re.sub(r"\[@[^\]]+\]", "[00]", text)
    plain = re.sub(r"[*$]", "", plain)
    if font is None:
        return len(plain) * 75.0
    return font.getlength(plain) / 10 * 20


_FONT = []


def _font():
    if not _FONT:
        try:
            from PIL import ImageFont
            _FONT.append(ImageFont.truetype("LiberationSerif-Regular.ttf", 80))
        except Exception:
            try:
                from PIL import ImageFont
                _FONT.append(ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 80))
            except Exception:
                _FONT.append(None)
    return _FONT[0]


def column_widths(t: Table, avail: int) -> List[int]:
    """Natural column widths, shrinking the widest text columns (which then wrap) to fit."""
    ncol = len(t.aligns)
    nat = [300.0] * ncol
    for r in t.rows:
        col = 0
        for c in r.cells:
            if c.span == 1:
                nat[col] = max(nat[col], text_width(c.text) + 2 * 45 + 30)
            col += c.span
    for r in t.rows:                       # spanning cells must fit their columns
        col = 0
        for c in r.cells:
            if c.span > 1:
                need = text_width(c.text) + 120
                have = sum(nat[col:col + c.span])
                if need > have:
                    for j in range(col, col + c.span):
                        nat[j] += (need - have) / c.span
            col += c.span
    minw = [300.0] * ncol                  # a column never gets narrower than its longest word
    for r in t.rows:
        col = 0
        for c in r.cells:
            if c.span == 1:
                words = re.split(r"[ \t]+|(?<=-)", re.sub(r"\[@[^\]]+\]", "[00]", c.text))
                minw[col] = max([minw[col]] + [text_width(w_) + 2 * 45 + 30 for w_ in words if w_])
            col += c.span
    if sum(nat) <= avail:
        return [int(w) for w in nat]
    lo, hi = 0.0, max(nat)                 # cap the widest columns at one width so the total fits
    for _ in range(50):
        cap = (lo + hi) / 2
        fits = sum(max(m, min(w, cap)) for w, m in zip(nat, minw)) <= avail
        lo, hi = (cap, hi) if fits else (lo, cap)
    widths = [max(m, min(w, lo)) for w, m in zip(nat, minw)]
    return [int(w) for w in widths]


# --------------------------------------------------------------------------- #
# PDF and checks

def to_pdf(docx: Path) -> Path:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run(["soffice", f"-env:UserInstallation=file://{profile}", "--headless", "--convert-to", "pdf",
                        "--outdir", str(docx.parent), str(docx)], check=True, capture_output=True)
    return docx.with_suffix(".pdf")


def pdf_pages(pdf: Path) -> List[str]:
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.check_output(["pdfinfo", str(pdf)], text=True)).group(1))
    return [" ".join(subprocess.check_output(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"],
                                             text=True).split()) for p in range(1, n + 1)]


def float_check(pdf: Path, captions: List[Tuple[str, str]]) -> int:
    """Every caption must be on the page of the first mention of its float."""
    pages = pdf_pages(pdf)
    bad = 0
    for tag, cap in captions:
        start = " ".join(re.sub(r"\[@[^\]]+\]|\$[^$]*\$|\\res\{[^}]+\}|[*]", " ", cap).split()[:4])
        key = f"{tag} {start}"
        cap_page = next((i for i, t in enumerate(pages) if key in t), None)
        pat = re.compile(re.escape(tag) + r"(?![0-9])")
        mention = None
        for i, t in enumerate(pages):
            if pat.search(t.replace(key, "")):
                mention = i
                break
        ok = cap_page is not None and mention == cap_page
        bad += not ok
        print(f"  {tag:9s} first mention p{mention + 1 if mention is not None else '?'}, "
              f"caption p{cap_page + 1 if cap_page is not None else '?'}" + ("" if ok else "  <-- MISMATCH"))
    return bad


BRITISH = r"\b(labell\w*|modell\w*|standardis\w*|normalis\w*|behaviour\w*|favour\w*|colour\w*|analys(?:e|ed|ing)\b|" \
          r"optimis\w*|centre\w*|catalogue\w*|randomis\w*|summaris\w*|characteris(?:e|ed|es|ing|ation)\b|generalis\w*|organis\w*)"


def lint(src: Path) -> List[str]:
    issues = []
    _, blocks = parse_source(src)
    for b in blocks:
        texts = [b.text] + (b.lines if b.kind == "list" else [])
        for t in texts:
            prose = re.sub(r"\$[^$]*\$|\[@[^\]]+\]|\\res\{[^}]+\}", "X", t)
            for pat, msg in ((r",\s+and\b", "comma before 'and'"), ("\u2014|---", "em dash"),
                             (r"\b(Moreover|Furthermore|Additionally|In conclusion)\b", "formulaic transition"),
                             (BRITISH, "British spelling"), (r"\bFigure\s", "write Fig."),
                             (r"\bequation\s*\(", "write Eq. (n)"), (r"\b[Rr]ef(erence)?\.?\s*\[", "Ref. [n]"),
                             (r"\bnot only\b", "'not only'")):
                for m in re.finditer(pat, prose):
                    issues.append(f"{src.name}: {msg}: ...{prose[max(0, m.start() - 40):m.end() + 20]}...")
    return issues


# --------------------------------------------------------------------------- #
# figures at the journal's sizes

def figures() -> None:
    sys.path.insert(0, str(ROOT / "examples" / "paper"))
    os.chdir(ROOT)
    import numpy as np
    import build_artifacts as ba

    FIGS.mkdir(parents=True, exist_ok=True)
    plt = ba.mpl()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans"})
    W = 83.5 / 25.4

    def save(fig, name: str) -> None:
        fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(FIGS / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # confusion matrices of our diagnosis, LR, 2 x 2
    labels = ["none", "covariate", "prior", "concept", "label_noise"]
    short = {"none": "None", "covariate": "Cov.", "prior": "Prior", "concept": "Conc.", "label_noise": "Noise"}
    fig, axes = plt.subplots(2, 2, figsize=(W, 2.75))
    for ax, (code, design) in zip(axes.flat, [("inc", "A"), ("inc", "B"), ("emp", "A"), ("emp", "B")]):
        ti = ba.bench(code, design, "lr")["summary"]["type_identification"]
        rows_ = [t for t in labels if sum(ti["confusion_matrix"][t].values())]
        M = np.array([[ti["confusion_matrix"][a][b] for b in labels] for a in rows_], dtype=float)
        frac = M / M.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows_)):
            for j in range(len(labels)):
                if M[i, j]:
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=6.5,
                            color="white" if frac[i, j] > 0.6 else ba.INK)
        bottom = code == "emp"
        ax.set_xticks(range(len(labels)), [short[t] for t in labels] if bottom else [], rotation=45, ha="right")
        ax.set_yticks(range(len(rows_)), [short[t] for t in rows_])
        ax.set_title(f"{ba.STUDY_TITLE[code].replace('ACS ', '')}, {design}", loc="left")
        ax.grid(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("injected")
    fig.supxlabel("diagnosed", fontsize=8, y=0.01)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    save(fig, "Fig03")

    # measured and estimated accuracy over the temporal windows, 2 x 2
    fig, axes = plt.subplots(2, 2, figsize=(W - 0.13, 3.0))
    for ax, code in zip(axes.flat, ("inc", "emp", "gas", "clr")):
        lr, gb = ba.study(code, "lr"), ba.study(code, "hgb")
        names = [k for k, w in lr["windows"].items() if w["analysis_type"] in ("control", "temporal")]
        x = np.arange(len(names))
        years = [lr["windows"][k].get("year") for k in names]
        breaks = [i for i in range(1, len(names)) if years[i] and years[i - 1] and years[i] - years[i - 1] > 1]

        def series(d, fn):
            y = [float(fn(d["windows"][k])) for k in names]
            xs = list(x.astype(float))
            for b in reversed(breaks):
                xs.insert(b, np.nan)
                y.insert(b, np.nan)
            return xs, y

        for d, color, lab in ((lr, ba.PALETTE[0], "LR measured"), (gb, ba.PALETTE[2], "GB measured")):
            ax.plot(*series(d, lambda w: w["actual_accuracy"]), "-o", color=color, ms=2.5, lw=1.1, label=lab)
        ax.plot(*series(lr, ba.cbpe_estimate), "--s", color=ba.PALETTE[1], ms=2.5, lw=1.0, mfc="white",
                label="LR estimate (CBPE)")
        ticks = []
        for k in names:
            lab = ba.wlabel(code, k).replace("control", "C")
            ticks.append(lab.replace("CA 20", "") if code in ("inc", "emp") else lab.replace("b", ""))
        ax.set_xticks(x, ticks, fontsize=6.5 if code in ("inc", "emp") else 7)
        ax.set_title(ba.STUDY_TITLE[code], loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, columnspacing=0.7, handletextpad=0.4)
    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.6, rect=(0, 0, 1, 0.93))
    save(fig, "Fig04")

    # audit budget (supplement)
    ab = ba.bench("inc", "A", "lr")["summary"]["audit_budget"]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    sizes = sorted(int(n) for n in next(iter(ab.values())))
    for c, t in zip(ba.PALETTE, ("none", "concept", "label_noise")):
        ax.plot(sizes, [ab[t][str(n)]["rate"] for n in sizes], "-o", color=c, ms=3.5, lw=1.3, label=ba.TYPE_LABEL[t])
        ax.fill_between(sizes, [ab[t][str(n)]["ci"][0] for n in sizes], [ab[t][str(n)]["ci"][1] for n in sizes],
                        color=c, alpha=0.12, lw=0)
    ax.set_xticks(sizes)
    ax.set_xlabel("labels in the delayed audit")
    ax.set_ylabel("correct diagnosis rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    save(fig, "FigS1")

    # TikZ drawings: vector PDF and a 600 dpi PNG
    tectonic = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not tectonic:
        print("TECTONIC not set: TikZ figures not rebuilt")
        return
    for src, name in (("pipeline", "Fig01"), ("decision", "Fig02")):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([tectonic, "--outdir", tmp, str(PP / "tikz" / f"{src}.tex")], check=True, capture_output=True)
            shutil.copy(Path(tmp) / f"{src}.pdf", FIGS / f"{name}.pdf")
        subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", str(FIGS / f"{name}.pdf"), str(FIGS / name)],
                       check=True)
    print(f"wrote figures to {FIGS.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #

def main() -> None:
    if "--figures" in sys.argv:
        figures()
    results = load_results()
    math = MathEngine()
    bib = parse_bib(PP / "references.bib")
    BUILD.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []

    supp_labels = Doc(PP / "supplement.md", results, math, bib, prefix="S", columns=1).labels
    main_doc = Doc(PP / "paper.md", results, math, bib, supp_labels=supp_labels)
    title = main_doc.front["title"][0]
    paper_docx = BUILD / "PPEECS_paper.docx"
    main_doc.write(paper_docx, title)
    failures += main_doc.check_order()
    unused = sorted(set(bib) - set(main_doc.cites))

    supp = Doc(PP / "supplement.md", results, math, bib, prefix="S", columns=1, main_labels=main_doc.labels)
    supp_docx = BUILD / "PPEECS_supplement.docx"
    supp.write(supp_docx, "Supplementary Material: " + title)
    failures += supp.check_order()
    unused = [k for k in unused if k not in supp.cites]
    if unused:
        failures.append(f"bib entries cited nowhere: {unused}")

    issues = lint(PP / "paper.md") + lint(PP / "supplement.md")
    for i in issues:
        print("LINT", i)

    if "--no-pdf" not in sys.argv:
        for d, docx in ((main_doc, paper_docx), (supp, supp_docx)):
            pdf = to_pdf(docx)
            n = len(pdf_pages(pdf))
            print(f"{pdf.relative_to(ROOT)}: {n} pages")
            bad = float_check(pdf, d.captions)
            if bad:
                failures.append(f"{pdf.name}: {bad} floats not on the page of their first mention")
    print(f"abstract: {main_doc.abstract_words} words")
    if failures:
        for f_ in failures:
            print("FAIL", f_)
        sys.exit(1)


if __name__ == "__main__":
    main()
