"""Turn the two delivery specs into real PDFs, so ingestion is tested on PDF input too.

Channel and platform specs usually arrive as PDFs. The metadata goes into a sidecar
<name>.meta.yaml, because a PDF has no reliable place for our fields.

    python scripts/make_pdfs.py      (needs: pip install fpdf2)
"""

import re
from pathlib import Path

import yaml
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "corpus" / "pdf_sources"
OUT = ROOT / "corpus" / "docs"


def render(md_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()
    pdf.set_title(meta["title"])
    body = body.translate(str.maketrans({"—": "-", "–": "-", "’": "'", "“": "\"", "”": "\""}))
    for line in body.strip().splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 9, line[2:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif line.startswith("> "):
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")  # keep "> " so the chunker skips the notice
            pdf.ln(3)
        elif m := re.match(r"##\s+(.*)", line):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, m.group(1), new_x="LMARGIN", new_y="NEXT")
        elif line.strip():
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")
    name = md_path.stem
    pdf.output(str(OUT / f"{name}.pdf"))
    (OUT / f"{name}.meta.yaml").write_text(yaml.safe_dump(meta, allow_unicode=True, sort_keys=False),
                                           encoding="utf-8")
    print(f"wrote {name}.pdf")


if __name__ == "__main__":
    for md in sorted(SRC.glob("*.md")):
        render(md)
