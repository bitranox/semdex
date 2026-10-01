"""Generate the extractor benchmark fixtures.

Self-authored, tiny, license-clean sample documents in every format the
extractor adapters must handle (markdown, text, HTML, PDF, DOCX, PPTX, XLSX,
PNG). Every fixture carries the same known content set so a test can assert
uniformly across formats; ``expected.toml`` records which phrases apply to
each fixture. Regenerate with::

    python tests/fixtures/extract/_generate.py

Requires generate-time-only libraries (not project runtime deps): reportlab,
python-docx, python-pptx, openpyxl, Pillow. Install them into the active
environment before running this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import rtoml
from docx import Document
from openpyxl import Workbook
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# The known content set every fixture embeds, so extraction can be checked
# uniformly across formats regardless of which library produced the file.
HEADING: Final[str] = "Semdex Extraction Benchmark"
PARAGRAPH: Final[str] = "The quick brown fox jumps over the lazy dog near the riverbank."
CODE_LINE: Final[str] = "print('hello semdex')"
TABLE_CELL: Final[str] = "Rotek-42"

FIXTURES_DIR: Final[Path] = Path(__file__).resolve().parent


def _write_markdown(path: Path) -> None:
    content = (
        f"# {HEADING}\n"
        "\n"
        f"{PARAGRAPH}\n"
        "\n"
        "```python\n"
        f"{CODE_LINE}\n"
        "```\n"
        "\n"
        "| Key | Value |\n"
        "| --- | --- |\n"
        f"| id | {TABLE_CELL} |\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_text(path: Path) -> None:
    content = f"{HEADING}\n\n{PARAGRAPH}\n"
    path.write_text(content, encoding="utf-8")


def _write_html(path: Path) -> None:
    content = (
        "<!doctype html>\n"
        "<html><head><title>Semdex fixture</title></head><body>\n"
        f"<h1>{HEADING}</h1>\n"
        f"<p>{PARAGRAPH}</p>\n"
        f"<code>{CODE_LINE}</code>\n"
        "<table><tr><td>id</td>"
        f"<td>{TABLE_CELL}</td></tr></table>\n"
        "</body></html>\n"
    )
    path.write_text(content, encoding="utf-8")


def _write_pdf(path: Path) -> None:
    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    _, height = LETTER
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, height - 72, HEADING)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, height - 100, PARAGRAPH)
    pdf.drawString(72, height - 120, f"id: {TABLE_CELL}")
    pdf.save()


def _write_docx(path: Path) -> None:
    document = Document()
    document.add_heading(HEADING, level=1)
    document.add_paragraph(PARAGRAPH)
    table = document.add_table(rows=1, cols=1)
    table.rows[0].cells[0].text = TABLE_CELL
    document.save(str(path))


def _write_pptx(path: Path) -> None:
    presentation = Presentation()
    slide_layout = presentation.slide_layouts[1]  # title + content
    slide = presentation.slides.add_slide(slide_layout)
    slide.shapes.title.text = HEADING
    textbox = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(1))
    textbox.text_frame.text = PARAGRAPH
    presentation.save(str(path))


def _write_xlsx(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet["A1"] = HEADING
    sheet["A2"] = PARAGRAPH
    sheet["A3"] = TABLE_CELL
    workbook.save(str(path))


def _write_scan_png(path: Path) -> None:
    width, height = 900, 300
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    try:
        font_heading = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        # No system TTF available; the default bitmap font still OCRs fine
        # at this size, it just is not as crisp.
        font_heading = ImageFont.load_default()
        font_body = ImageFont.load_default()
    draw.text((20, 30), HEADING, fill="black", font=font_heading)
    draw.text((20, 120), PARAGRAPH, fill="black", font=font_body)
    image.save(path)


def _write_expected_toml(path: Path) -> None:
    fox = "quick brown fox"
    lines = [
        '[fixtures."sample.md"]',
        f'phrases = ["{HEADING}", "{fox}", "{CODE_LINE}", "{TABLE_CELL}"]',
        "",
        '[fixtures."sample.txt"]',
        f'phrases = ["{HEADING}", "{fox}"]',
        "",
        '[fixtures."sample.html"]',
        f'phrases = ["{HEADING}", "{fox}", "{CODE_LINE}", "{TABLE_CELL}"]',
        "",
        '[fixtures."sample.pdf"]',
        f'phrases = ["{HEADING}", "{fox}", "{TABLE_CELL}"]',
        "",
        '[fixtures."sample.docx"]',
        f'phrases = ["{HEADING}", "{fox}", "{TABLE_CELL}"]',
        "",
        '[fixtures."sample.pptx"]',
        f'phrases = ["{HEADING}", "{fox}"]',
        "",
        '[fixtures."sample.xlsx"]',
        f'phrases = ["{HEADING}", "{fox}", "{TABLE_CELL}"]',
        "",
        '[fixtures."scan.png"]',
        f'phrases = ["{HEADING}", "{fox}"]',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def generate() -> None:
    _write_markdown(FIXTURES_DIR / "sample.md")
    _write_text(FIXTURES_DIR / "sample.txt")
    _write_html(FIXTURES_DIR / "sample.html")
    _write_pdf(FIXTURES_DIR / "sample.pdf")
    _write_docx(FIXTURES_DIR / "sample.docx")
    _write_pptx(FIXTURES_DIR / "sample.pptx")
    _write_xlsx(FIXTURES_DIR / "sample.xlsx")
    _write_scan_png(FIXTURES_DIR / "scan.png")
    _write_expected_toml(FIXTURES_DIR / "expected.toml")


if __name__ == "__main__":
    generate()
    rtoml.load(FIXTURES_DIR / "expected.toml")  # fail fast if the written TOML does not parse
    print(f"Generated fixtures in {FIXTURES_DIR}")
