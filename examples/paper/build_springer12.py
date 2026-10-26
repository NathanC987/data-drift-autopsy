"""Build the 12-page Springer proceedings versions of the paper (E2A 2026 and DSISys 2026).

Both conferences ask for the Springer proceedings (LNCS) format and at most 12 pages,
DSISys exactly 12. The text is the 12-page DACS cut (paper/dacs/paper.md), unchanged,
in paper/e2a/paper.md and paper/dsisys/paper.md. Both are written on the Springer LNCS
Word template (icmmdu-conf-paper/splnproc2311.docm) with the writer of build_icmmdu.py;
E2A also accepts LaTeX, DSISys only Word. DSISys reviews double-blind, so its submission
file carries no author names, affiliation or e-mail, neither in the text, nor in the
running heads, nor in the file properties; --named builds the camera-ready version.

Run from the repository root:

    python examples/paper/build_springer12.py e2a              # Word and PDF
    python examples/paper/build_springer12.py dsisys           # anonymous submission
    python examples/paper/build_springer12.py dsisys --named   # with the authors
    python examples/paper/build_springer12.py e2a --figures    # redraw Fig. 3 and Fig. 4 first
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dacs import figures  # noqa: E402
from build_icmmdu import RESULTS, LncsDoc, float_check, para, run  # noqa: E402
from build_ppeecs import PAPER, PP, ROOT, MathEngine, lint, parse_bib, pdf_pages, to_pdf  # noqa: E402

VENUES = {
    "e2a": {"name": "E2A", "pages": (1, 12), "anonymous": False},
    "dsisys": {"name": "DSISys", "pages": (12, 12), "anonymous": True},
}
ANONYMOUS = "Anonymous Authors"


class Springer12Doc(LncsDoc):
    kw_sep = " · "               # the middots of the Springer guidelines

    def __init__(self, *args, anonymous: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.anonymous = anonymous
        if anonymous:
            self.front["author"] = [ANONYMOUS]
            self.front["running-authors"] = [ANONYMOUS]

    def front_matter(self) -> str:
        if not self.anonymous:
            return super().front_matter()
        return (para("papertitle", run(self.front["title"][0])) + para("author", run(ANONYMOUS))
                + para("address", run("Affiliation withheld for double-blind review")))


def main() -> None:
    venue = sys.argv[1] if len(sys.argv) > 1 else ""
    if venue not in VENUES:
        sys.exit(f"usage: build_springer12.py {{{'|'.join(VENUES)}}} [--named] [--figures] [--no-pdf]")
    v = VENUES[venue]
    folder = PAPER / venue
    if "--figures" in sys.argv:
        # Fig. 3 is the one-column 2 x 2 PPEECS drawing: one row of four is too narrow at 122 mm
        figures(folder / "figures", 122, png=True, names=("Fig04",))
    anonymous = v["anonymous"] and "--named" not in sys.argv
    src = folder / "paper.md"
    doc = Springer12Doc(src, RESULTS, MathEngine(), parse_bib(PP / "references.bib"), anonymous=anonymous)
    doc.fig_dir = folder / "figures"
    docx = folder / "build" / (f"{v['name']}_paper" + ("_anonymous" if anonymous else "") + ".docx")
    doc.write(docx, doc.front["title"][0])
    failures = doc.check_order()
    for i in lint(src):
        print("LINT", i)
    if "--no-pdf" not in sys.argv:
        pdf = to_pdf(docx)
        text = " ".join(pdf_pages(pdf))
        n = len(pdf_pages(pdf))
        print(f"{pdf.relative_to(ROOT)}: {n} pages")
        if float_check(pdf, doc.captions, heads=r"\d+ (?:R\. Guhagarkar et al\.|Anonymous Authors)|.*? Models \d+"):
            failures.append(f"{pdf.name}: floats not on the page of their first mention")
        low, high = v["pages"]
        if not low <= n <= high:
            failures.append(f"{pdf.name} has {n} pages (allowed {low} to {high})")
        if anonymous:
            with zipfile.ZipFile(docx) as z:
                parts = " ".join(z.read(n).decode(errors="ignore") for n in z.namelist() if n.endswith((".xml", ".rels")))
            leaks = [w for w in ("Guhagarkar", "Cherian", "Mirchandani", "Patre", "Alok", "Vivekanand", "ves.ac.in")
                     if w in text or w in parts]
            if leaks:
                failures.append(f"{docx.name} names the authors: {', '.join(leaks)}")
    print(f"abstract: {doc.abstract_words} words")
    for f_ in failures:
        print("FAIL", f_)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
