"""Build the ICMMDU 2026 conference version of the paper (Springer proceedings Word template).

The conference asks for a Word file on its template (icmmdu-conf-paper/splnproc2311.docm,
the Springer LNCS layout) of at least 10 pages and sets no upper limit, so this version
is the full paper: paper/icmmdu/paper.md is the PPEECS text with its supplement merged
in, plus a real-drift summary and the running times. The script reads it with the
parser of build_ppeecs.py (same results, references, figures and generated tables) and
writes paper/icmmdu/build/ICMMDU_paper.docx, keeping the template's styles, numbering
and running heads (the macros of the .docm are left out), and a PDF made with LibreOffice.

Run from the repository root:

    python examples/paper/build_icmmdu.py              # Word and PDF
    python examples/paper/build_icmmdu.py --figures    # redraw the timeline figure first
"""

from __future__ import annotations

import datetime as dt
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ppeecs import (FIGS, GENERATORS, TRANSFORMS, PAPER, PP, ROOT, Block, Doc, MathEngine, Table,  # noqa: E402
                          column_widths, esc, lint, load_results, para, parse_bib, pdf_pages, png_size, run,
                          tex_unicode, to_pdf)

TEMPLATE = ROOT / "icmmdu-conf-paper" / "splnproc2311.docm"
SRC = PAPER / "icmmdu" / "paper.md"
OUT = PAPER / "icmmdu" / "build"
ICM_FIGS = PAPER / "icmmdu" / "figures"      # figures made for this version
MIN_PAGES = 10
TEXT_W = 6918                     # twips: 122 mm text width of the template
ABSTRACT_MAX_WORDS = 250
MAIN_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
RUNIN = {"Acknowledgments", "Disclosure of Interests"}


# --------------------------------------------------------------------------- #
# tables computed from the results

RESULTS = load_results()
STAGES = ("ks", "psi", "mmd", "conf", "bbsd", "dc", "diagnosis", "reliability")    # the stages of Section 3


def window_times() -> Dict[str, Dict[str, float]]:
    """Seconds per window, summed over the stages the paper describes, keyed by study and window."""
    out: Dict[str, Dict[str, float]] = {}
    for key, val in RESULTS.items():
        m = re.fullmatch(r"((?:inc|emp|gas|clr)(?:-hgb)?)-(\w+)-time-(\w+)", key)
        if m and m.group(3) in STAGES:
            out.setdefault(m.group(1), {}).setdefault(m.group(2), 0.0)
            out[m.group(1)][m.group(2)] += float(val)
    return out


def income_wide(t: Table) -> Table:
    """ACS Income table at the full text width: a short control label, the two top shifts kept on one line."""
    for r in t.rows:
        for c in r.cells:
            c.text = c.text.replace("CA 2014 (control)", "Control")
            if re.fullmatch(r"[A-Z0-9]+, [A-Z0-9]+", c.text):
                c.text = c.text.replace(", ", ",\u00a0")
    return t


def datasets_wide(t: Table) -> Table:
    """Datasets table: each group of windows kept on one line, so the census rows take two lines, not four."""
    for r in t.rows:
        for c in r.cells:
            if c.text.startswith("CA 2015"):
                c.text = c.text.replace(" and ", ", ")
                c.text = " ".join(g.replace(" ", "\u00a0") for g in re.split(r"(?<=;) ", c.text))
    return t


TRANSFORMS.update({"income_wide": income_wide, "datasets_wide": datasets_wide})
RESULTS["icm-time-max"] = f"{max(max(v.values()) for v in window_times().values()):.0f}"


# --------------------------------------------------------------------------- #
# references in the style of the template (Springer LNCS)

def lncs_name(name: str) -> str:
    name = name.strip()
    if name.startswith("{") and name.endswith("}"):
        return tex_unicode(name[1:-1])
    if "," in name:
        last, first = (p.strip() for p in name.split(",", 1))
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    first = tex_unicode(first)

    def initial(part: str) -> str:
        if "-" in part:
            return "-".join(initial(p) for p in part.split("-") if p)
        return part if part.endswith(".") and len(part) <= 2 else part[0] + "."
    return f"{tex_unicode(last)}, {''.join(initial(p) for p in first.split())}".rstrip(", ")


def lncs_reference(e: Dict[str, str]) -> List[Tuple[str, Optional[str]]]:
    """Author, F., Author, S.: Title. Journal 2(5), 99–110 (2016), as in the template's examples.

    A paper in a journal or proceedings is found by its venue, so it carries its DOI but no URL;
    only web documents keep a URL and the date it was accessed."""
    authors = ", ".join(lncs_name(a) for a in re.split(r"\s+and\s+", e["author"]))
    title = tex_unicode(e["title"]).rstrip(".")
    year = e.get("year", "")
    pages = tex_unicode(e.get("pages", ""))
    where = f", {tex_unicode(e['address'])}" if e.get("address") else ""
    typ = e["_type"]
    if typ == "article":
        vol = tex_unicode(e.get("volume", "")) + (f"({e['number']})" if e.get("number") else "")
        loc = pages or e.get("eid", "")
        text = f"{authors}: {title}. {tex_unicode(e['journal'])} {vol}, {loc} ({year})"
    elif typ == "inproceedings":
        pp = f", pp. {pages}" if pages else ""
        text = f"{authors}: {title}. In: {tex_unicode(e['booktitle'])}{pp} ({year})"
    elif typ == "book":
        text = f"{authors}: {title}. {tex_unicode(e['publisher'])}{where} ({year})"
    elif typ == "techreport":
        text = f"{authors}: {title}. Tech. Rep. {e['number']}, {tex_unicode(e['institution'])}{where} ({year})"
    elif typ == "online":
        text = f"{authors}: {title} ({year})"
    else:
        raise ValueError(f"unsupported entry type {typ}")
    if e.get("doi"):
        url = f"https://doi.org/{e['doi']}"
        return [(text + ". ", None), (url, url)]
    if e.get("url") and typ in ("online", "techreport"):
        return [(text + ", ", None), (e["url"], e["url"]), (f", last accessed {e['urldate']}", None)]
    return [(text + ".", None)]


SMALL_WORDS = {"a", "an", "the", "and", "or", "but", "nor", "of", "in", "on", "at", "to", "for", "by", "with",
               "from", "against", "across", "per", "vs."}


def title_case(text: str) -> str:
    """Headings in title case, as the template's examples are; acronyms and math stay as they are."""
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        if "$" in w or "@" in w or any(ch.isupper() for ch in w[1:]) or (i and w.lower() in SMALL_WORDS):
            out.append(w)
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in w.split("-")))
    return " ".join(out)


# --------------------------------------------------------------------------- #
# document writer on the template's styles

class LncsDoc(Doc):
    template = TEMPLATE
    fig_dir = ICM_FIGS          # figures made for this version, looked up before the PPEECS ones
    kw_sep = ", "

    def front_matter(self) -> str:
        f = self.front
        title = re.sub(r'"([^"]*)"', "\u201c\\1\u201d", f["title"][0])
        names = [a.split(";")[0].strip() for a in f["author"]]
        line = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
        mail = f["corresponding"][0]
        address = (run(f["affiliation"][0]) + "<w:r><w:br/></w:r>"
                   + f'<w:r><w:rPr><w:rStyle w:val="e-mail"/></w:rPr><w:t>{esc(mail)}</w:t></w:r>')
        return para("papertitle", run(title)) + para("author", run(line)) + para("address", address)

    def abstract(self, b: Block) -> str:
        words = len(re.findall(r"\S+", re.sub(r"\\res\{[^}]+\}", "0", b.text)))
        if words > ABSTRACT_MAX_WORDS:
            raise ValueError(f"abstract has {words} words (limit {ABSTRACT_MAX_WORDS})")
        self.abstract_words = words
        kw = [k.strip() for k in self.front["keywords"][0].split(",")]
        return (para("abstract", run("Abstract. ", "<w:b/><w:bCs/>") + self.inline(b.text),
                     '<w:spacing w:after="0"/><w:ind w:firstLine="0"/>')
                + para("keywords", run("Keywords:", "<w:b/><w:bCs/>") + run(" " + self.kw_sep.join(kw) + ".")))

    def heading(self, b: Block) -> str:
        # the template numbers its headings itself (1, 1.1), in step with Doc.number_blocks
        style = "heading1" if b.level == 1 else "heading2"
        if "unnumbered" in b.attrs:
            return para(style, self.inline(title_case(b.text)),
                        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="0" w:firstLine="0"/>')
        return para(style, self.inline(title_case(b.text)))

    def paragraph(self, b: Block, first: bool) -> str:
        text = b.text
        noindent = first or text.startswith("\\noindent")
        text = text.replace("\\noindent", "").strip()
        if self.runin:
            body = run(self.runin + ".", '<w:rStyle w:val="heading30"/>') + run(" ") + self.inline(text)
            self.runin = ""
            return para("acknowlegments", body)
        return para("p1a" if noindent else "Normal", self.inline(text))

    def numbered_list(self, b: Block) -> str:
        return "".join(para("numitem", self.inline(item)) for item in b.lines)

    def equation(self, b: Block) -> str:
        body = ("<w:r><w:tab/></w:r>" + self.math.get(b.text, display=True) + run(b.attrs.get("punct", ""))
                + "<w:r><w:tab/></w:r>" + run(f"({b.number})"))
        tabs = f'<w:tabs><w:tab w:val="center" w:pos="{TEXT_W // 2}"/><w:tab w:val="right" w:pos="{TEXT_W}"/></w:tabs>'
        return para("equation", body, tabs)

    def caption(self, tag: str, b: Block, style: str) -> str:
        text = b.text.rstrip(".") + "."
        jc = '<w:jc w:val="both"/>' if len(text) > 110 else ""
        return para(style, run(tag + ".", "<w:b/>") + run(" ") + self.inline(text), jc)

    def figure(self, b: Block) -> str:
        path = self.fig_dir / b.attrs["file"]
        if not path.exists():
            path = FIGS / b.attrs["file"]
        w_px, h_px = png_size(path)
        cx = int(float(b.attrs.get("width", 100)) * 36000)
        cy = int(cx * h_px / w_px)
        n = len(self.media) + 1
        rid = f"rIdI{n}"
        self.media.append((f"media/image{n}.png", path))
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                          f"media/image{n}.png", False))
        a = "http://schemas.openxmlformats.org/drawingml/2006/main"
        pic = "http://schemas.openxmlformats.org/drawingml/2006/picture"
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="Figure {b.number}" descr="{esc(b.attrs["file"])}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{a}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="{a}"><a:graphicData uri="{pic}"><pic:pic xmlns:pic="{pic}">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{esc(b.attrs["file"])}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
            "</wp:inline></w:drawing></w:r>")
        self.captions.append((f"Fig. {b.number}", b.text))
        return para("image", drawing, "<w:keepNext/>") + self.caption(f"Fig. {b.number}", b, "figurecaption")

    def table(self, b: Block) -> str:
        src = b.lines
        gen = re.match(r"<<gen:(\w+)(?:\s+(\w+))?>>", src[0].strip()) if src else None
        code = re.match(r"<<code:(\w+)>>", src[0].strip()) if src else None
        from build_ppeecs import parse_pipe_table, parse_tabular
        if code:
            t = GENERATORS[code.group(1)]()
        elif gen:
            t = parse_tabular(gen.group(1))
            if gen.group(2):
                t = TRANSFORMS[gen.group(2)](t)
        else:
            t = parse_pipe_table(src)
        if not self.appendix:          # an appendix table is introduced in the appendix itself
            self.captions.append((f"Table {b.number}", b.text))
        size = int(b.attrs.get("size", 9))
        out = self.caption(f"Table {b.number}", b, "tablecaption")
        out += self.table_xml(t, TEXT_W - 60, size)
        return out + '<w:p><w:pPr><w:spacing w:after="0" w:line="120" w:lineRule="exact"/></w:pPr></w:p>'

    def table_xml(self, t: Table, avail: int, size: int = 9) -> str:
        ncol = len(t.aligns)
        rpr = f'<w:sz w:val="{2 * size}"/><w:szCs w:val="{2 * size}"/>'
        rendered = [[self.inline(c.text, rpr) for c in r.cells] for r in t.rows]
        # column_widths measures 8 pt text: scale the room to the font size used here
        widths = [int(w * size / 8) for w in column_widths(t, int(avail * 8 / size))]
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        rows_xml, last = [], len(t.rows) - 1
        for ri, (r, cells) in enumerate(zip(t.rows, rendered)):
            col, tcs = 0, []
            for c, xml in zip(r.cells, cells):
                borders = []
                if ri == 0 or r.rule_above:
                    sz = 12 if (ri == 0 or r.rule_above == "heavy") else 6
                    borders.append(f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="000000"/>')
                bottom = 12 if ri == last else (
                    6 if any(a <= col and col + c.span - 1 <= b_ for a, b_ in r.cmid_below) else None)
                if bottom:
                    borders.append(f'<w:bottom w:val="single" w:sz="{bottom}" w:space="0" w:color="000000"/>')
                align = "center" if c.span > 1 else {"l": "left", "r": "right", "c": "center"}[t.aligns[col]]
                w = sum(widths[col:col + c.span])
                span = f'<w:gridSpan w:val="{c.span}"/>' if c.span > 1 else ""
                keep = "<w:keepNext/>" if ri < last else ""
                tcs.append(f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{span}'
                           f'<w:tcBorders>{"".join(borders)}</w:tcBorders><w:vAlign w:val="center"/></w:tcPr>'
                           + para("Normal", xml, f'{keep}<w:spacing w:before="4" w:after="4" w:line="{25 * size}" '
                                  f'w:lineRule="atLeast"/><w:ind w:firstLine="0"/><w:jc w:val="{align}"/>')
                           + "</w:tc>")
                col += c.span
            if col != ncol:
                raise ValueError(f"table row {ri} has {col} columns, expected {ncol}")
            trpr = "<w:cantSplit/>" + ("<w:tblHeader/>" if ri < t.header_rows else "")
            rows_xml.append(f"<w:tr><w:trPr>{trpr}<w:jc w:val=\"center\"/></w:trPr>{''.join(tcs)}</w:tr>")
        return ('<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
                '<w:tblCellMar><w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar>'
                '<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" '
                f'w:noHBand="0" w:noVBand="0"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{"".join(rows_xml)}</w:tbl>')

    def link(self, url: str, text: str) -> str:
        rid = f"rIdL{len(self.rels) + 1}"
        self.rels.append((rid, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", url, True))
        return f'<w:hyperlink r:id="{rid}" w:history="1">{run(text)}</w:hyperlink>'

    def references(self) -> str:
        out = [para("heading1", run("References"),
                    '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:ind w:left="0" w:firstLine="0"/>')]
        for k in self.cites:
            body = "".join(self.link(url, text) if url else run(text) for text, url in lncs_reference(self.bib[k]))
            out.append(para("referenceitem", body, '<w:jc w:val="left"/>'))   # long URLs: no stretched lines
        return "".join(out)

    def body(self, sect: str) -> str:
        self.cites, self.rels, self.media, self.captions, self.first_ref = [], [], [], [], {}
        self.runin = ""
        self.appendix = False
        out = [self.front_matter()]
        prev = "heading"
        for i, b in enumerate(self.blocks):
            self.cur = i
            if b.kind == "abstract":
                out.append(self.abstract(b))
            elif b.kind == "heading" and "unnumbered" in b.attrs:
                if b.text == "References":
                    out.append(self.references())
                elif b.text in RUNIN:
                    self.runin = b.text
                else:
                    out.append(self.heading(b))
                    self.appendix = True
            elif b.kind == "heading":
                out.append(self.heading(b))
            elif b.kind == "para":
                out.append(self.paragraph(b, prev in ("heading", "fig", "table")))
            elif b.kind == "list":
                out.append(self.numbered_list(b))
            elif b.kind == "eq":
                out.append(self.equation(b))
            elif b.kind == "fig":
                out.append(self.figure(b))
            elif b.kind == "table":
                out.append(self.table(b))
            else:
                raise ValueError(f"block {b.kind} is not supported in this template")
            prev = b.kind
        return "".join(out) + sect

    def write(self, out: Path, title: str) -> None:
        zin = zipfile.ZipFile(self.template)
        parts = {n: zin.read(n) for n in zin.namelist()}
        # a plain .docx: drop the macros, the ribbon and the template's sample chart
        for n in list(parts):
            if n.startswith(("customUI/", "word/charts/")) or "vba" in n.lower():
                del parts[n]
        doc = parts["word/document.xml"].decode()
        sect = re.search(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S).group(0)
        self.math.missing = []
        self.body(sect)                         # first pass collects the math
        self.math.convert_missing()
        body = self.body(sect)
        if self.math.missing:
            raise RuntimeError("math left unconverted")
        root = re.search(r"<w:document[^>]*>", doc).group(0)
        parts["word/document.xml"] = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                                      f"{root}<w:body>{body}</w:body></w:document>").encode()
        rels = parts["word/_rels/document.xml.rels"].decode()
        rels = re.sub(r'<Relationship Id="\w+" Type="[^"]*/(hyperlink|chart|vbaProject)"[^>]*/>', "", rels)
        extra = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{esc(tg)}"' + (' TargetMode="External"' if ext else "")
                        + "/>" for i, t, tg, ext in self.rels)
        parts["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", extra + "</Relationships>").encode()
        top = parts["_rels/.rels"].decode()
        parts["_rels/.rels"] = re.sub(r'<Relationship Id="\w+" Type="[^"]*/ui/extensibility"[^>]*/>', "", top).encode()
        ct = parts["[Content_Types].xml"].decode()
        ct = re.sub(r'<Default Extension="(bin|ICO)"[^>]*/>', "", ct)
        ct = ct.replace('ContentType="image/.png"', 'ContentType="image/png"')
        ct = re.sub(r'<Override PartName="/word/(vbaData|charts/chart1)\.xml"[^>]*/>', "", ct)
        ct = re.sub(r'(PartName="/word/document.xml" ContentType=")[^"]*"', rf'\g<1>{MAIN_CT}"', ct)
        parts["[Content_Types].xml"] = ct.encode()
        notes = parts["word/footnotes.xml"].decode()          # the template's sample footnote
        parts["word/footnotes.xml"] = re.sub(r'<w:footnote w:id="1">.*?</w:footnote>', "", notes, flags=re.S).encode()
        settings = parts["word/settings.xml"].decode()       # the template is protected against editing
        settings = re.sub(r"<w:documentProtection[^>]*/>|<w:attachedTemplate[^>]*/>", "", settings)
        parts["word/settings.xml"] = settings.encode()
        # running heads: authors on even pages, short title on odd pages
        names = [a.split(";")[0].strip() for a in self.front["author"]]
        first = names[0].split()
        lead = f"{first[0][0]}. {' '.join(first[1:])}" + (" et al." if len(names) > 2 else "")
        lead = self.front.get("running-authors", [lead])[0]
        for name, old, new in (("word/header1.xml", "F. Author and S. Author", lead),
                               ("word/header2.xml", "Contribution Title (shortened if too long)",
                                self.front.get("running", [title])[0])):
            xml = parts[name].decode()
            xml = re.sub(r"(<w:t[^>]*>)[^<]*Author[^<]*(</w:t>)" if "Author" in old else
                         r"(<w:t[^>]*>)Contribution Title[^<]*(</w:t>)", lambda m: m.group(1) + esc(new) + m.group(2), xml)
            if esc(new) not in xml:
                raise RuntimeError(f"running head not set in {name}")
            parts[name] = xml.encode()
        for name, path in self.media:
            parts["word/" + name] = path.read_bytes()
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts["docProps/core.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{esc(title)}</dc:title><dc:creator>{esc('; '.join(names))}</dc:creator>"
            f"<cp:lastModifiedBy>{esc(names[0])}</cp:lastModifiedBy><cp:revision>1</cp:revision>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
            f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>').encode()
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for name in [n for n in zin.namelist() if n in parts] + [n for n in parts if n not in zin.namelist()]:
                z.writestr(name, parts[name])
        print(f"wrote {out.relative_to(ROOT)}: {len(self.cites)} references, {len(self.media)} figures, "
              f"{len(self.used_res)} result keys")


# --------------------------------------------------------------------------- #
# figures redrawn for the 122 mm text width

def figures() -> None:
    import os

    import numpy as np
    os.chdir(ROOT)
    import build_artifacts as ba

    ICM_FIGS.mkdir(parents=True, exist_ok=True)
    plt = ba.mpl()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans"})
    W = 88 / 25.4                 # printed at 88 mm, so the lettering stays at 7 to 8 pt

    # measured and estimated accuracy over the temporal windows, 2 x 2
    fig, axes = plt.subplots(2, 2, figsize=(W, 2.68))
    for ax, code in zip(axes.flat, ("inc", "emp", "gas", "clr")):
        lr, gb = ba.study(code, "lr"), ba.study(code, "hgb")
        names = [k for k, w in lr["windows"].items() if w["analysis_type"] in ("control", "temporal")]
        x = np.arange(len(names))
        years = [lr["windows"][k].get("year") for k in names]
        breaks = [i for i in range(1, len(names)) if years[i] and years[i - 1] and years[i] - years[i - 1] > 1]

        def series(d, fn):
            y = [float(fn(d["windows"][k])) for k in names]
            xs = list(x.astype(float))
            for br in reversed(breaks):            # a gap where a year is missing (no 2020 release)
                xs.insert(br, np.nan)
                y.insert(br, np.nan)
            return xs, y

        for d, color, lab in ((lr, ba.PALETTE[0], "LR measured"), (gb, ba.PALETTE[2], "GB measured")):
            ax.plot(*series(d, lambda w: w["actual_accuracy"]), "-o", color=color, ms=3, lw=1.2, label=lab)
        ax.plot(*series(lr, ba.cbpe_estimate), "--s", color=ba.PALETTE[1], ms=3, lw=1.0, mfc="white",
                label="LR estimate (CBPE)")
        ticks = [ba.wlabel(code, k).replace("control", "C") for k in names]
        ticks = [t.replace("CA 20", "") if code in ("inc", "emp") else t.replace("b", "") for t in ticks]
        ax.set_xticks(x, ticks)
        ax.set_title(ba.STUDY_TITLE[code], loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=1.0, rect=(0, 0, 1, 0.94))
    fig.savefig(ICM_FIGS / "Fig05.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(ICM_FIGS / "Fig05.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote figures to {ICM_FIGS.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# checks

def float_check(pdf: Path, captions: List[Tuple[str, str]],
                heads: str = r"\d+ R\. Guhagarkar et al\.|.*? Analysis \d+") -> int:
    """Every float follows its first mention: below it on the same page or, when it does not fit in the
    space left there, at the top of the next page (nothing but the running head above it)."""
    pages = pdf_pages(pdf)
    bad = 0
    for tag, cap in captions:
        start = " ".join(re.sub(r"\[@[^\]]+\]|\$[^$]*\$|\\res\{[^}]+\}|[*]", " ", cap).split()[:4])
        key = f"{tag}. {start}"
        cap_page = next((i for i, t in enumerate(pages) if key in t), None)
        pat = re.compile(re.escape(tag) + r"(?![0-9])")
        mention = next((i for i, t in enumerate(pages) if pat.search(t.replace(key, ""))), None)
        where = "?"
        if cap_page is not None and mention == cap_page:
            text = pages[cap_page]
            ok = pat.search(text.replace(key, "\0" * len(key))).start() < text.index(key)
            where = "same page"
        elif cap_page is not None and mention is not None and cap_page == mention + 1:
            above = pages[cap_page][:pages[cap_page].index(key)]
            above = re.sub(rf"^({heads})\s*", "", above)       # the running head
            ok = len(above) < 15
            where = "top of next page"
        else:
            ok = False
        bad += not ok
        print(f"  {tag:9s} mention p{mention + 1 if mention is not None else '?'}, "
              f"caption p{cap_page + 1 if cap_page is not None else '?'}, {where}" + ("" if ok else "  <-- MISPLACED"))
    return bad


def main() -> None:
    if "--figures" in sys.argv:
        figures()
    math = MathEngine()
    bib = parse_bib(PP / "references.bib")
    doc = LncsDoc(SRC, RESULTS, math, bib)
    docx = OUT / "ICMMDU_paper.docx"
    doc.write(docx, doc.front["title"][0])
    failures = doc.check_order()
    for i in lint(SRC):
        print("LINT", i)
    if "--no-pdf" not in sys.argv:
        pdf = to_pdf(docx)
        n = len(pdf_pages(pdf))
        print(f"{pdf.relative_to(ROOT)}: {n} pages")
        if float_check(pdf, doc.captions):
            failures.append(f"{pdf.name}: floats not on the page of their first mention")
        if n < MIN_PAGES:
            failures.append(f"{pdf.name} has {n} pages (minimum {MIN_PAGES})")
    print(f"abstract: {doc.abstract_words} words")
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
