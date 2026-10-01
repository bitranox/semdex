"""The markdown twin of the MLDR English slice is the source text with its headings marked.

The rule is a heuristic; these tests pin what it must and must not do on synthetic text, and that
stripping the marks gives the source back byte for byte, so the paired comparison isolates markup.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_mldr_markdown_slice.py"

pytestmark = pytest.mark.os_agnostic

_PROSE = (
    "The compound was first synthesised in the laboratory in the early part "
    "of the twentieth century and later produced at industrial scale by several "
    "manufacturers across three continents."
)
assert len(_PROSE) >= 120 and _PROSE.endswith(".")


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("build_mldr_markdown_slice", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_mldr_markdown_slice"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> Any:
    return _load()


def test_a_short_line_before_a_prose_paragraph_is_a_heading(builder: Any) -> None:
    marked, count = builder.mark_headings(f"History\n{_PROSE}\n")
    assert marked == f"## History\n{_PROSE}\n"
    assert count == 1


def test_a_heading_followed_by_a_blank_line_then_prose_is_still_a_heading(builder: Any) -> None:
    marked, count = builder.mark_headings(f"Early life\n\n{_PROSE}\n")
    assert marked.startswith("## Early life\n\n")
    assert count == 1


@pytest.mark.parametrize(
    "line",
    [
        "See also",  # followed by short lines in the fixture below, never prose
        "Population: 1234",  # a run of three digits
        "x = y + z",  # formula characters
        "Ends with a full stop.",
        "A rather long line that has many more than eight spaces in it and so is a sentence not a heading",
        "",
        " ",  # a single space: whitespace-only, not empty
        "\t ",  # tab and space: whitespace-only, not empty
    ],
)
def test_lines_that_are_not_headings_are_left_alone(builder: Any, line: str) -> None:
    following = "Short item\nAnother\n" if line == "See also" else _PROSE
    text = f"{line}\n{following}\n"
    marked, count = builder.mark_headings(text)
    assert marked == text
    assert count == 0


def test_a_list_item_before_prose_is_not_a_heading(builder: Any) -> None:
    text = f"- an item\n{_PROSE}\n"
    assert builder.mark_headings(text) == (text, 0)


def test_stripping_the_marks_gives_the_source_back_byte_for_byte(builder: Any) -> None:
    source = f"Robert Bilott investigation\n{_PROSE}\nSynthesis\n{_PROSE}\nSee also\nShort\n"
    marked, count = builder.mark_headings(source)
    assert count == 2
    assert builder.strip_marks(marked) == source


def test_a_source_line_that_already_starts_with_the_mark_is_never_marked(builder: Any) -> None:
    # Otherwise strip_marks would remove a mark the source carried and break the identity.
    text = f"## Already marked\n{_PROSE}\n"
    assert builder.mark_headings(text) == (text, 0)


def _write_source(root: Path, docs: dict[str, str]) -> Path:
    src = root / "mldr_en_8k_slice"
    src.mkdir(parents=True)
    src.joinpath("corpus.jsonl").write_bytes(
        b"".join(json.dumps({"_id": d, "title": "", "text": t}).encode() + b"\n" for d, t in sorted(docs.items()))
    )
    src.joinpath("queries.json").write_text(json.dumps({"q1": "what is it"}))
    src.joinpath("qrels.json").write_text(json.dumps({"q1": {"doc-1": 1}}))
    src.joinpath("doc_ids.txt").write_text("\n".join(sorted(docs)) + "\n")
    src.joinpath("meta.json").write_text(json.dumps({"corpus": "mldr_en_8k_slice", "total": len(docs)}))
    return src


def test_build_writes_a_twin_slice_with_the_same_ids_queries_and_qrels(builder: Any, tmp_path: Path) -> None:
    docs = {"doc-1": f"History\n{_PROSE}\n", "doc-2": f"{_PROSE}\n"}
    src = _write_source(tmp_path, docs)
    out = tmp_path / "mldr_en_8k_md_slice"
    meta = builder.build(src, out, sample_size=5, seed=1)
    assert out.joinpath("doc_ids.txt").read_text() == src.joinpath("doc_ids.txt").read_text()
    assert out.joinpath("queries.json").read_bytes() == src.joinpath("queries.json").read_bytes()
    assert out.joinpath("qrels.json").read_bytes() == src.joinpath("qrels.json").read_bytes()
    rows = [json.loads(line) for line in out.joinpath("corpus.jsonl").read_text().splitlines()]
    assert [r["_id"] for r in rows] == ["doc-1", "doc-2"]
    assert rows[0]["text"] == f"## History\n{_PROSE}\n"
    assert rows[1]["text"] == docs["doc-2"]
    assert {builder.strip_marks(r["text"]) for r in rows} == set(docs.values())
    assert meta["corpus"] == "mldr_en_8k_md_slice"
    assert meta["source_corpus"] == "mldr_en_8k_slice"
    assert meta["marked_lines"] == 1 and meta["marked_docs"] == 1 and meta["total"] == 2
    assert meta["rule"]["max_chars"] == builder.MAX_HEADING_CHARS
    assert len(meta["source_corpus_sha256"]) == 64
    written = json.loads(out.joinpath("meta.json").read_text())
    assert written == meta
    sample = out.joinpath("heading-sample.txt").read_text().splitlines()
    assert sample and sample[0].startswith("doc-1\t## History\t"), "one marked line per row: doc id, heading, next line"


def test_build_is_idempotent_and_never_rewrites_an_existing_slice(builder: Any, tmp_path: Path) -> None:
    src = _write_source(tmp_path, {"doc-1": f"History\n{_PROSE}\n"})
    out = tmp_path / "mldr_en_8k_md_slice"
    builder.build(src, out, sample_size=5, seed=1)
    before = out.joinpath("corpus.jsonl").read_bytes()
    out.joinpath("corpus.jsonl").write_bytes(b"tampered\n")
    builder.build(src, out, sample_size=5, seed=1)
    assert out.joinpath("corpus.jsonl").read_bytes() == b"tampered\n", "an existing slice is left alone"
    assert before != b"tampered\n"
