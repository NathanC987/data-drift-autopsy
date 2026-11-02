"""Build the Bulletin of the National Research Centre version of the paper (Springer Nature LaTeX).

The journal (Springer Open) asks for a research article with a structured abstract
(Background, Results, Conclusions), the main text as Background, Methods, Results,
Discussion and Conclusions, a list of abbreviations, its Declarations and references in
the Springer Basic (author-year) style, set double-spaced with line and page numbers.
paper/bnrc/paper.md is the PPEECS text in that structure: headings are renamed or moved
one level down, the abstract carries the three labels, the Discussion and conclusions
section is split in two and the supplement is cited as Additional file 1; nothing else
is reworded. paper/bnrc/supplement.md is the PPEECS supplement, published as
Additional file 1.

The script reuses the LaTeX writer of build_dacs.py and writes to paper/bnrc/build/, a
folder that compiles on its own (pdflatex + bibtex, Overleaf or tectonic):

    BNRC_manuscript.tex / .pdf          the manuscript
    BNRC_additional_file_1.tex / .pdf   Additional file 1
    references.bib, sn-jnl.cls, sn-basic.bst, figures/

Run from the repository root (TECTONIC may point to the binary if it is not on PATH):

    python examples/paper/build_bnrc.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dacs as dacs  # noqa: E402
from build_dacs import TexDoc, compile_tex, dacs_bib, order_check  # noqa: E402
from build_ppeecs import FIGS, PAPER, PP, ROOT, Block, Doc, float_check, load_results, pdf_pages  # noqa: E402

TEMPLATE_DIR = ROOT / "sn-journal-paper"
SRC = PAPER / "bnrc" / "paper.md"
SUPP = PAPER / "bnrc" / "supplement.md"
OUT = PAPER / "bnrc" / "build"
MAIN, ADD = "BNRC_manuscript", "BNRC_additional_file_1"

# referee: double line spacing; lineno: line numbers in the margin; sn-basic: Springer Basic (author-year)
MANUSCRIPT_CLASS = r"\documentclass[pdflatex,referee,lineno,sn-basic]{sn-jnl}"
SUPPLEMENT_CLASS = r"\documentclass[pdflatex,sn-basic]{sn-jnl}"
TITLE_WORDS, ABSTRACT_WORDS = 15, 350


def split_caption(text: str) -> tuple:
    """Caption title (first sentence) and legend (the rest), as the journal sets them apart."""
    m = re.search(r"(?<!\bet al)(?<!\be\.g)(?<!\bi\.e)\.\s+(?=[A-Z*])", text)
    return (text[:m.start()], text[m.end():]) if m else (text, "")


class BnrcDoc(TexDoc):
    def __init__(self, *args, document_class: str = MANUSCRIPT_CLASS, **kw) -> None:
        super().__init__(*args, **kw)
        self.document_class = document_class

    def inline(self, text: str, subst: bool = True) -> str:
        s = super().inline(text, subst)
        # author-year citations: "Roschewitz et al. (2025)" after a named author, else "(Lipton et al. 2018)"
        s = re.sub(r"et al\.~\\cite\{", r"et al.~\\citeyearpar{", s)
        return re.sub(r"\\cite\{", r"\\citep{", s)

    def front_matter(self) -> str:
        s = super().front_matter()
        # structured abstract: each labelled part starts its own paragraph
        for head in ("Results:", "Conclusions:"):
            s = s.replace(f" \\textbf{{{head}}}", f"\\par \\textbf{{{head}}}")
        return s.replace("\\abstract{}", "")

    def table(self, b: Block) -> str:
        title, legend = split_caption(b.text)
        tex = super().table(Block(kind=b.kind, text=title, level=b.level, attrs=dict(b.attrs), lines=b.lines))
        self.captions[-1] = (self.captions[-1][0], b.text)
        if legend:
            tex = tex.replace("\\end{tabular}\n\\end{minipage}",
                              "\\end{tabular}\\par\\vspace{3pt}\n{\\footnotesize\\noindent " + self.inline(legend)
                              + "\\par}\n\\end{minipage}")
        return tex

    def figure(self, b: Block) -> str:
        name = Path(b.attrs["file"]).stem
        ext = ".pdf" if (FIGS / f"{name}.pdf").exists() else Path(b.attrs["file"]).suffix
        self.figures.append((FIGS, name + ext))
        width = float(b.attrs.get("width", 83.5))
        size = "\\textwidth" if width > 100 else f"{width:g}mm"
        self.captions.append((f"Fig. {self.numbers[b.attrs['id']]}", b.text))
        return "\n".join([
            "\\par\\vspace{0.7\\intextsep}\\noindent\\begin{minipage}{\\textwidth}\\centering",
            f"\\includegraphics[width={size},height=0.6\\textheight,keepaspectratio]{{{name}{ext}}}",
            f"\\captionof{{figure}}{{{self.inline(b.text)}}}\\label{{{b.attrs['id']}}}",
            "\\end{minipage}\\par\\vspace{0.7\\intextsep}"])

    def write(self, out: Path, bib: str) -> None:
        saved = dacs.PREAMBLE
        dacs.PREAMBLE = saved.replace(r"\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}", self.document_class)
        try:
            super().write(out, bib)
        finally:
            dacs.PREAMBLE = saved
        tex = out.read_text()
        # Declarations in the journal's form: unnumbered back-matter headings
        tex = re.sub(r"\\subsection\*\{([^}]*)\}", r"\\bmhead{\1}", tex)
        # text the authors still have to supply is printed in red
        tex = re.sub(r"\[\[(.*?)\]\]", r"\\textcolor{red}{\1}", tex)
        out.write_text(tex)


def journal_bib(src: Path, dst: Path) -> None:
    """The PPEECS references for sn-basic, with web pages carrying 'Accessed <date>' as the journal asks."""
    dacs_bib(src, dst)
    text = dst.read_text()
    text = re.sub(r"(?m)^(\s*)urldate(\s*)=\s*\{([^}]*)\}", r"\1note\2= {Accessed \3}", text)
    dst.write_text(text)


def checks(doc: BnrcDoc) -> List[str]:
    """Rules of the journal that the source must meet."""
    fails = []
    abstract = next(b for b in doc.blocks if b.kind == "abstract")
    plain = re.sub(r"\\res\{[^}]+\}", "0", abstract.text)
    n = len(plain.split())
    print(f"abstract: {n} words (limit {ABSTRACT_WORDS})")
    if n > ABSTRACT_WORDS:
        fails.append(f"abstract has {n} words")
    if "[@" in abstract.text:
        fails.append("abstract cites references")
    for head in ("**Background:**", "**Results:**", "**Conclusions:**"):
        if head not in abstract.text:
            fails.append(f"abstract lacks {head}")
    keywords = [k for k in doc.front["keywords"][0].split(",") if k.strip()]
    if not 3 <= len(keywords) <= 10:
        fails.append(f"{len(keywords)} keywords (3 to 10 allowed)")
    for b in doc.blocks:
        if b.kind in ("fig", "table"):
            title = split_caption(b.text)[0]
            words = len(re.sub(r"\$[^$]*\$", "x", title).split())
            if words > TITLE_WORDS:
                fails.append(f"{b.attrs['id']}: caption title has {words} words (limit {TITLE_WORDS})")
        if b.kind == "pagebreak":
            fails.append("the manuscript must not contain page breaks")
    return fails


def main() -> None:
    results = load_results()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    shutil.copy(TEMPLATE_DIR / "sn-jnl.cls", OUT / "sn-jnl.cls")
    shutil.copy(TEMPLATE_DIR / "bst" / "sn-basic.bst", OUT / "sn-basic.bst")
    journal_bib(PP / "references.bib", OUT / "references.bib")

    main_numbers = Doc(SRC, results, None, {}).labels
    supp_numbers = Doc(SUPP, results, None, {}, prefix="S").labels
    paper = BnrcDoc(SRC, results, other_labels=supp_numbers)
    supp = BnrcDoc(SUPP, results, prefix="S", other_labels=main_numbers, document_class=SUPPLEMENT_CLASS)
    failures = checks(paper)
    paper.write(OUT / f"{MAIN}.tex", "references")
    supp.write(OUT / f"{ADD}.tex", "references")

    names = set()
    for folder, name in paper.figures + supp.figures:
        shutil.copy(folder / name, OUT / "figures" / name)
        names.add(name)
    for stale in (OUT / "figures").iterdir():
        if stale.name not in names:
            stale.unlink()
    todo = re.findall(r"\[\[(.*?)\]\]", SRC.read_text())
    for t in todo:
        print("  TO FILL", t)

    if "--no-pdf" not in sys.argv:
        for doc, stem in ((paper, MAIN), (supp, ADD)):
            pdf = compile_tex(OUT / f"{stem}.tex")
            print(f"{pdf.relative_to(ROOT)}: {len(pdf_pages(pdf))} pages")
            if order_check(pdf, doc.captions):
                failures.append(f"{pdf.name}: a float is placed above the text that introduces it")
            float_check(pdf, doc.captions)      # reported only; the journal does not require it
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
