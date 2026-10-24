"""Build the DACS 2026 conference version of the paper (Springer Nature LaTeX template).

The conference allows 12 pages, so paper/dacs/paper.md is the PPEECS text
(paper/ppeecs/paper.md) with whole parts removed and nothing else reworded; there is
no supplement. The script reads it with the parser of build_ppeecs.py (same results,
references, figures, generated tables and table transforms) and writes LaTeX on the
sn-jnl class in paper/dacs-conf-paper/. The output goes to paper/dacs/build/, a folder
that compiles on its own (Overleaf or any TeX installation):

    DACS_paper.tex / .pdf, references.bib, sn-jnl.cls, sn-mathphys-num.bst, figures/

Run from the repository root (TECTONIC may point to the binary if it is not on PATH):

    python examples/paper/build_dacs.py
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

TEMPLATE_DIR = PAPER / "dacs-conf-paper"
SRC = PAPER / "dacs" / "paper.md"
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
        lines = ["\\begin{table}[!htb]", f"\\caption{{{self.inline(b.text)}}}\\label{{{b.attrs['id']}}}%",
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
        lines += ["\\end{tabular}", "\\end{table}"]
        return "\n".join(lines)

    def figure(self, b: Block) -> str:
        name = Path(b.attrs["file"]).stem
        ext = ".pdf" if (FIGS / f"{name}.pdf").exists() else Path(b.attrs["file"]).suffix
        self.figures.append(name + ext)
        width = float(b.attrs.get("width", 83.5))
        self.captions.append((f"Fig. {self.numbers[b.attrs['id']]}", b.text))
        return "\n".join([
            "\\begin{figure}[!htb]", "\\centering",
            f"\\includegraphics[width={width:g}mm,height=0.6\\textheight,keepaspectratio]{{{name}{ext}}}",
            f"\\caption{{{self.inline(b.text)}}}\\label{{{b.attrs['id']}}}", "\\end{figure}"])

    def body(self) -> str:
        self.figures: List[str] = []
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
        tex = (PREAMBLE + "\n\\begin{document}\n\n" + self.front_matter() + "\n" + numbering + "\n" + body
               + f"\n\n\\bibliography{{{bib}}}\n\n\\end{{document}}\n")
        out.write_text(tex)
        print(f"wrote {out.relative_to(ROOT)}: {len(self.cites)} references, {len(self.figures)} figures, "
              f"{len(self.used_res)} result keys")


def short_title(title: str) -> str:
    """Running head: the title up to its comma-free core."""
    return "Drift Detection, Diagnosis and Reliability Analysis" if title.startswith("Drift Detection") else title


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


def main() -> None:
    results = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in ("sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy(TEMPLATE_DIR / name, OUT / name)
    dacs_bib(PP / "references.bib", OUT / "references.bib")

    paper = TexDoc(SRC, results)
    paper.write(OUT / "DACS_paper.tex", "references")
    for name in paper.figures:
        shutil.copy(FIGS / name, OUT / "figures" / name)
    for stale in list((OUT / "figures").iterdir()) + list(OUT.glob("DACS_supplement.*")):
        if stale.name not in paper.figures:
            stale.unlink()

    failures = []
    if "--no-pdf" not in sys.argv:
        pdf = compile_tex(OUT / "DACS_paper.tex")
        n = len(pdf_pages(pdf))
        print(f"{pdf.relative_to(ROOT)}: {n} pages")
        if float_check(pdf, paper.captions):
            failures.append(f"{pdf.name}: floats not on the page of their first mention")
        if n > PAGE_LIMIT:
            failures.append(f"{pdf.name} has {n} pages (limit {PAGE_LIMIT})")
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
