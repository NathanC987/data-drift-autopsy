"""Build the DACS 2026 conference version of the paper (Springer Nature LaTeX template).

The conference allows 12 pages, so paper/dacs/paper.md is the PPEECS text
(paper/ppeecs/paper.md) with whole parts removed and nothing else reworded; there is
no supplement. The script reads it with the parser of build_ppeecs.py (same results,
references, figures, generated tables and table transforms) and writes LaTeX on the
sn-jnl class in dacs-conf-paper/ (project root). The output goes to paper/dacs/build/, a folder
that compiles on its own (Overleaf or any TeX installation):

    DACS_paper.tex / .pdf, references.bib, sn-jnl.cls, sn-mathphys-num.bst, figures/

Run from the repository root (TECTONIC may point to the binary if it is not on PATH):

    python examples/paper/build_dacs.py              # LaTeX and PDF
    python examples/paper/build_dacs.py --figures    # redraw the result figures first
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ppeecs import (DOLLAR, FIGS, GENERATORS, INLINE, PAPER, PP, ROOT, TRANSFORMS, Block,  # noqa: E402
                          Doc, float_check, load_results, parse_pipe_table, parse_source, parse_tabular,
                          pdf_pages)

TEMPLATE_DIR = ROOT / "dacs-conf-paper"
SRC = PAPER / "dacs" / "paper.md"
DACS_FIGS = PAPER / "dacs" / "figures"     # result figures redrawn for the 131 mm text width
OUT = PAPER / "dacs" / "build"
PAGE_LIMIT = 12

PREAMBLE = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{mathrsfs}%
\usepackage[title]{appendix}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{manyfoot}%
\usepackage{booktabs}%
% figures and tables are set in the text flow, right after the paragraph that introduces them
\makeatletter\providecommand\captionof[1]{\def\@captype{#1}\caption}\makeatother

\graphicspath{{figures/}}
\raggedbottom

% breakurl works only with pdflatex; other engines set reference URLs with \url
\usepackage{iftex}
\ifPDFTeX\else\AtBeginDocument{\let\burl\url}\fi
"""

# characters of the sources that pdflatex does not set from UTF-8 on its own
TEXT_CHARS = {"\u2013": "--", "\u2212": "\\textminus{}", "\u2193": "$\\downarrow$", "\u2191": "$\\uparrow$",
              "\u2192": "$\\to$", "\u0394": "$\\Delta$", "\u2713": "\\checkmark{}", "\u00a0": "~", DOLLAR: "\\$"}
MATH_CHARS = {"\u2212": "-", "\u2013": "-", "\u00a0": "~", DOLLAR: "\\$"}
SPECIAL = {"%": "\\%", "&": "\\&", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}", "^": "\\^{}"}


def tex_text(s: str) -> str:
    """Plain text of the Markdown dialect as LaTeX."""
    s = s.replace("\\noindent", "\x00")
    s = re.sub(r"\\(?=[^\x00])", r"\\textbackslash{}", s)
    s = "".join(SPECIAL.get(c, c) for c in s)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    s = "".join(TEXT_CHARS.get(c, c) for c in s)
    return s.replace("\x00", "\\noindent")


class TexDoc:
    """One source file of the dialect as an sn-jnl document."""

    def __init__(self, src: Path, results: Dict[str, str], prefix: str = "",
                 other_labels: Optional[Dict[str, str]] = None) -> None:
        self.src = src
        self.res = results
        self.prefix = prefix
        self.front, self.blocks = parse_source(src)
        self.numbers = Doc(src, results, None, {}, prefix=prefix).labels    # same numbering as the Word build
        self.other = other_labels or {}
        self.cites: List[str] = []
        self.captions: List[tuple] = []
        self.used_res: set = set()

    # ---- inline ----------------------------------------------------------- #
    def substitute(self, text: str) -> str:
        def res(m: "re.Match[str]") -> str:
            if m.group(1) not in self.res:
                raise KeyError(f"{self.src.name}: undefined result key {m.group(1)}")
            self.used_res.add(m.group(1))
            return self.res[m.group(1)]
        return re.sub(r"\\res\{([^}]+)\}", res, text.replace("\\$", DOLLAR))

    def inline(self, text: str, subst: bool = True) -> str:
        if subst:
            text = self.substitute(text)
        out, pos = [], 0
        for m in INLINE.finditer(text):
            out.append(tex_text(text[pos:m.start()]))
            tok = m.group(0)
            if m.group("math"):
                out.append("".join(MATH_CHARS.get(c, c) for c in tok))
            elif m.group("bold"):
                out.append("\\textbf{" + self.inline(tok[2:-2], False) + "}")
            elif m.group("ital"):
                out.append("\\emph{" + self.inline(tok[1:-1], False) + "}")
            elif m.group("cite"):
                keys = [k.strip().lstrip("@") for k in tok[1:-1].split(";")]
                for k in keys:
                    if k not in self.cites:
                        self.cites.append(k)
                if out and out[-1].endswith(" "):
                    out[-1] = out[-1][:-1] + "~"
                out.append("\\cite{" + ",".join(keys) + "}")
            else:
                out.append(self.ref(tok[1:]))
            pos = m.end()
        out.append(tex_text(text[pos:]))
        s = "".join(out)
        # keep "Fig. 1", "Table 2", "Section 3" and "Eq. (1)" together
        return re.sub(r"\b(Fig\.|Table|Tables|Section|Sections|Eq\.) (?=\\ref|\(\\ref|S?\d)", r"\1~", s)

    def ref(self, label: str) -> str:
        if label.startswith(("s:", "m:")):
            if label[2:] not in self.other:
                raise KeyError(f"{self.src.name}: unknown cross-document label {label}")
            return self.other[label[2:]]
        if label not in self.numbers:
            raise KeyError(f"{self.src.name}: unknown label {label}")
        return f"\\ref{{{label}}}"

    # ---- blocks ----------------------------------------------------------- #
    def front_matter(self) -> str:
        f = self.front
        title = tex_text(f["title"][0])
        out = [f"\\title[{tex_text(short_title(f['title'][0]))}]{{{title}}}", ""]
        mail = f.get("corresponding", [""])[0]
        for a in f["author"]:
            name, mark = (p.strip() for p in a.split(";"))
            first, last = name.rsplit(" ", 1)
            star = "*" if mark.endswith("*") else ""
            out.append(f"\\author{star}[{mark.rstrip('*')}]{{\\fnm{{{first}}} \\sur{{{last}}}}}"
                       + (f"\\email{{{mail}}}" if star else ""))
        out.append("")
        for a in f["affiliation"]:
            mark, text = (p.strip() for p in a.split(";", 1))
            parts = [p.strip() for p in text.split(",")]
            div, org, rest = parts[0], parts[1], parts[2:]
            city, post = re.match(r"(.*?)\s+(\d+)$", rest[-2]).groups()
            street = ", ".join(rest[:-2])
            out.append(f"\\affil*[{mark}]{{\\orgdiv{{{tex_text(div)}}}, \\orgname{{{tex_text(org)}}}, "
                       f"\\orgaddress{{\\street{{{tex_text(street)}}}, \\city{{{city}}}, \\postcode{{{post}}}, "
                       f"\\country{{{rest[-1]}}}}}}}")
        out.append("")
        abstract = next((b for b in self.blocks if b.kind == "abstract"), None)
        if abstract:
            out.append(f"\\abstract{{{self.inline(abstract.text)}}}")
            out.append("")
            out.append(f"\\keywords{{{f['keywords'][0]}}}")
        else:
            out.append("\\abstract{}")
        out += ["", "\\maketitle", ""]
        for note in f.get("note", []):
            out += ["\\noindent\\emph{" + self.inline(note) + "}", ""]
        return "\n".join(out)

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
        self.captions.append((f"Table {self.numbers[b.attrs['id']]}", b.text))
        lines = ["\\par\\vspace{0.7\\intextsep}\\noindent\\begin{minipage}{\\textwidth}\\centering\\tablebodyfont",
                 f"\\captionof{{table}}{{{self.inline(b.text)}}}\\label{{{b.attrs['id']}}}%",
                 "\\begin{tabular}{@{}" + "".join(t.aligns) + "@{}}", "\\toprule"]
        last = len(t.rows) - 1
        for ri, r in enumerate(t.rows):
            if ri > 0 and r.rule_above:
                lines.append("\\midrule")
            cells, col = [], 0
            for c in r.cells:
                body = self.inline(c.text)
                if c.span > 1:
                    body = f"\\multicolumn{{{c.span}}}{{c}}{{{body}}}"
                elif ri < t.header_rows and col > 0 and t.aligns[col] != "l":
                    body = f"\\multicolumn{{1}}{{c}}{{{body}}}"
                cells.append(body)
                col += c.span
            if col != len(t.aligns):
                raise ValueError(f"{b.attrs['id']}: row {ri} has {col} columns, expected {len(t.aligns)}")
            lines.append(" & ".join(cells) + " \\\\")
            for a, z in r.cmid_below:
                lines.append(f"\\cmidrule(lr){{{a + 1}-{z + 1}}}")
            if ri == last:
                lines.append("\\botrule")
        lines += ["\\end{tabular}", "\\end{minipage}\\par\\vspace{0.7\\intextsep}"]
        return "\n".join(lines)

    def figure(self, b: Block) -> str:
        name = Path(b.attrs["file"]).stem
        folder = DACS_FIGS if (DACS_FIGS / f"{name}.pdf").exists() else FIGS
        ext = ".pdf" if (folder / f"{name}.pdf").exists() else Path(b.attrs["file"]).suffix
        self.figures.append((folder, name + ext))
        width = float(b.attrs.get("width", 83.5))
        self.captions.append((f"Fig. {self.numbers[b.attrs['id']]}", b.text))
        return "\n".join([
            "\\par\\vspace{0.7\\intextsep}\\noindent\\begin{minipage}{\\textwidth}\\centering",
            f"\\includegraphics[width={width:g}mm,height=0.6\\textheight,keepaspectratio]{{{name}{ext}}}",
            f"\\captionof{{figure}}{{{self.inline(b.text)}}}\\label{{{b.attrs['id']}}}",
            "\\end{minipage}\\par\\vspace{0.7\\intextsep}"])

    def body(self) -> str:
        self.figures: List[tuple] = []
        out = []
        for b in self.blocks:
            if b.kind == "abstract":
                continue
            if b.kind == "heading":
                if b.text == "References" and "unnumbered" in b.attrs:
                    continue
                cmd = ["section", "subsection", "subsubsection"][b.level - 1]
                star = "*" if "unnumbered" in b.attrs else ""
                label = f"\\label{{{b.attrs['id']}}}" if "id" in b.attrs else ""
                out.append(f"\\{cmd}{star}{{{self.inline(b.text)}}}{label}")
            elif b.kind == "para":
                out.append(self.inline(b.text))
            elif b.kind == "list":
                out.append("\\begin{enumerate}\n" + "\n".join("\\item " + self.inline(i) for i in b.lines)
                           + "\n\\end{enumerate}")
            elif b.kind == "eq":
                eq = "".join(MATH_CHARS.get(c, c) for c in self.substitute(b.text))
                out.append(f"\\begin{{equation}}\n{eq}{b.attrs.get('punct', '')}\\label{{{b.attrs['id']}}}\n"
                           "\\end{equation}")
            elif b.kind == "fig":
                out.append(self.figure(b))
            elif b.kind == "table":
                out.append(self.table(b))
            elif b.kind == "pagebreak":
                out.append("\\clearpage")
        return "\n\n".join(out)

    def write(self, out: Path, bib: str) -> None:
        body = self.body()
        numbering = ""
        if self.prefix:
            numbering = "".join(f"\\renewcommand{{\\the{c}}}{{{self.prefix}\\arabic{{{c}}}}}\n"
                                for c in ("section", "figure", "table", "equation"))
        names = ", ".join(a.split(";")[0].strip() for a in self.front["author"])
        meta = f"\\hypersetup{{pdftitle={{{tex_text(self.front['title'][0])}}}, pdfauthor={{{names}}}}}\n"
        tex = (PREAMBLE + meta + "\n\\begin{document}\n\n" + self.front_matter() + "\n" + numbering + "\n" + body
               + f"\n\n\\bibliography{{{bib}}}\n\n\\end{{document}}\n")
        out.write_text(tex)
        print(f"wrote {out.relative_to(ROOT)}: {len(self.cites)} references, {len(self.figures)} figures, "
              f"{len(self.used_res)} result keys")


def short_title(title: str) -> str:
    """Running head: the title, or its part before a colon when it is long."""
    return title if len(title) <= 70 else title.split(":")[0]


def protect(title: str) -> str:
    """Brace acronyms and proper names (CLEAR, ImageNet, Folktables) so the style's sentence case keeps them."""
    def keep(w: str) -> bool:
        parts = [re.sub(r"[^A-Za-z]", "", p) for p in w.split("-")]
        return any(any(c.isupper() for c in p[1:]) for p in parts) or any(p in PROPER for p in parts)
    return re.sub(r"[^\s{}]+", lambda m: "{" + m.group(0) + "}" if keep(m.group(0)) else m.group(0), title)


PROPER = {"Bayesian", "Census", "Gaussian", "Holm", "Kolmogorov", "Smirnov", "Race", "Question", "User", "Note"}


def dacs_bib(src: Path, dst: Path) -> None:
    """The PPEECS references for the sn-mathphys-num style.

    Web pages become @misc with their access date. Journal and proceedings entries drop
    the full-text URL that PPEECS asks for, which Springer's reference style does not need.
    """
    entries = re.split(r"(?m)^(?=@)", src.read_text())
    out = []
    for e in entries:
        if re.match(r"@(article|inproceedings|book)\s*\{", e):
            e = re.sub(r"(?m)^\s*(url|urldate)\s*=.*\n", "", e)
            e = re.sub(r",(\s*\n\})", r"\1", e)
        e = re.sub(r"(?m)^(\s*title\s*=\s*)\{(.*)\},?$", lambda m: f"{m.group(1)}{{{protect(m.group(2))}}},", e)
        out.append(re.sub(r"^@online\s*\{", "@misc{", e))
    dst.write_text("".join(out))


def compile_tex(tex: Path) -> Path:
    exe = os.environ.get("TECTONIC") or shutil.which("tectonic")
    if not exe:
        raise RuntimeError("tectonic is needed for the PDF (set TECTONIC)")
    r = subprocess.run([exe, "--keep-logs", tex.name], cwd=tex.parent, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"tectonic failed on {tex.name}")
    log = tex.with_suffix(".log").read_text(errors="replace")
    for pat in (r"LaTeX Warning: (?:Reference|Citation) .*", r"Overfull \\hbox \((\d+\.\d+)pt too wide\).*"):
        for m in re.finditer(pat, log):
            if m.group(0).startswith("Overfull") and float(m.group(1)) < 1:
                continue
            print("  TEX", m.group(0))
    return tex.with_suffix(".pdf")


def figures() -> None:
    """Confusion matrices and accuracy timelines as one row of four panels across the text width."""
    os.chdir(ROOT)
    import numpy as np
    import build_artifacts as ba

    DACS_FIGS.mkdir(parents=True, exist_ok=True)
    plt = ba.mpl()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans"})
    W = 131 / 25.4

    def save(fig, name: str) -> None:
        fig.savefig(DACS_FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    labels = ["none", "covariate", "prior", "concept", "label_noise"]
    short = {"none": "None", "covariate": "Cov.", "prior": "Prior", "concept": "Conc.", "label_noise": "Noise"}
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.75))
    for ax, (code, design) in zip(axes, [("inc", "A"), ("inc", "B"), ("emp", "A"), ("emp", "B")]):
        ti = ba.bench(code, design, "lr")["summary"]["type_identification"]
        rows_ = [t for t in labels if sum(ti["confusion_matrix"][t].values())]
        M = np.array([[ti["confusion_matrix"][a][b] for b in labels] for a in rows_], dtype=float)
        frac = M / M.sum(axis=1, keepdims=True)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(rows_)):
            for j in range(len(labels)):
                if M[i, j]:
                    ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=6,
                            color="white" if frac[i, j] > 0.6 else ba.INK)
        ax.set_xticks(range(len(labels)), [short[t] for t in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(rows_)), [short[t] for t in rows_])
        ax.set_title(f"{ba.STUDY_TITLE[code]}, {design}", loc="left")
        ax.grid(False)
    axes[0].set_ylabel("injected")
    fig.supxlabel("diagnosed", fontsize=8, y=0.0)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "Fig03")

    fig, axes = plt.subplots(1, 4, figsize=(W, 1.8))
    for ax, code in zip(axes, ("inc", "emp", "gas", "clr")):
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
            ax.plot(*series(d, lambda w: w["actual_accuracy"]), "-o", color=color, ms=2.2, lw=1.0, label=lab)
        ax.plot(*series(lr, ba.cbpe_estimate), "--s", color=ba.PALETTE[1], ms=2.2, lw=0.9, mfc="white",
                label="LR estimate (CBPE)")
        ticks = []
        for k in names:
            lab = ba.wlabel(code, k).replace("control", "C")
            ticks.append(lab.replace("CA 20", "") if code in ("inc", "emp") else lab.replace("b", ""))
        ax.set_xticks(x, ticks, fontsize=6, rotation=90)
        ax.set_title(ba.STUDY_TITLE[code], loc="left")
    axes[0].set_ylabel("accuracy")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.88))
    save(fig, "Fig04")
    print(f"wrote figures to {DACS_FIGS.relative_to(ROOT)}")


def order_check(pdf: Path, captions: List[tuple]) -> int:
    """Each float must follow the text that first mentions it, on the same page."""
    pages = pdf_pages(pdf)
    bad = 0
    for tag, cap in captions:
        start = " ".join(re.sub(r"\[@[^\]]+\]|\$[^$]*\$|\\res\{[^}]+\}|[*]", " ", cap).split()[:4])
        key = f"{tag} {start}"
        page = next((i for i, t in enumerate(pages) if key in t), None)
        if page is None:
            bad += 1
            continue
        text = pages[page]
        m = re.search(re.escape(tag) + r"(?![0-9])", text.replace(key, "\0" * len(key)))
        if not m or m.start() > text.index(key):
            bad += 1
            print(f"  {tag:9s} appears before its first mention")
    return bad


def main() -> None:
    if "--figures" in sys.argv:
        figures()
    results = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in ("sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy(TEMPLATE_DIR / name, OUT / name)
    dacs_bib(PP / "references.bib", OUT / "references.bib")

    paper = TexDoc(SRC, results)
    paper.write(OUT / "DACS_paper.tex", "references")
    names = [name for _, name in paper.figures]
    for folder, name in paper.figures:
        shutil.copy(folder / name, OUT / "figures" / name)
    for stale in list((OUT / "figures").iterdir()) + list(OUT.glob("DACS_supplement.*")):
        if stale.name not in names:
            stale.unlink()

    failures = []
    if "--no-pdf" not in sys.argv:
        pdf = compile_tex(OUT / "DACS_paper.tex")
        n = len(pdf_pages(pdf))
        print(f"{pdf.relative_to(ROOT)}: {n} pages")
        if float_check(pdf, paper.captions):
            failures.append(f"{pdf.name}: floats not on the page of their first mention")
        if order_check(pdf, paper.captions):
            failures.append(f"{pdf.name}: floats placed above the text that introduces them")
        if n > PAGE_LIMIT:
            failures.append(f"{pdf.name} has {n} pages (limit {PAGE_LIMIT})")
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
