# Extractor benchmark fixtures

Tiny, self-authored, license-clean sample documents used to measure how well
each extractor adapter recovers text from a given file format. Every
fixture embeds the same known content set (a heading, a paragraph, and
where the format supports it, a code line and a table cell) so extraction
can be checked uniformly across formats. `expected.toml` lists, per fixture
file, the phrases that a correct extraction must contain.

All content here (`Semdex Extraction Benchmark`, the pangram paragraph, the
code line, the table cell) was authored for this repository; the fixtures
carry no third-party text, images, or templates, so they are MIT-clean like
the rest of the project.

## Files

| File            | Format                                            |
|-----------------|---------------------------------------------------|
| `sample.md`     | Markdown (heading, paragraph, fenced code, table) |
| `sample.txt`    | Plain text (heading, paragraph)                   |
| `sample.html`   | HTML (`<h1>`, `<p>`, `<code>`, `<table>`)         |
| `sample.pdf`    | Text-layer PDF (reportlab)                        |
| `sample.docx`   | Word (Heading 1, paragraph, 1-cell table)         |
| `sample.pptx`   | PowerPoint (title slide + text box)               |
| `sample.xlsx`   | Excel (cells A1/A2/A3)                            |
| `scan.png`      | Rendered text image, for OCR extractors           |
| `expected.toml` | Required phrases per fixture, for assertions      |

## Regenerating

The fixtures are produced by `_generate.py`, which needs generate-time-only
libraries that are not part of the project's runtime dependencies:

```bash
uv pip install reportlab python-docx python-pptx openpyxl Pillow
python tests/fixtures/extract/_generate.py
```

Re-run it whenever the known content set changes; it overwrites all
fixtures and `expected.toml` in place.
