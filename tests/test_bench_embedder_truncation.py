"""An embedder's own input cap can clip a chunk with nothing anywhere saying so.

A chunk set is capped by the chunker, in the chunker's tokenizer. The embedder then applies its
own cap, in its own tokenizer, at embed time - silently. The overlap ladder walks a chunk's length
up on purpose, so far enough up it meets that second cap, and from there the sweep is varying
overlap AND how much got clipped.

The audit therefore may not count tokens itself. It must drive the embedder's real tokenize path,
because a cut can happen BEFORE tokenization: ``model2vec`` slices the raw string to
``max_length * median_token_length`` CHARACTERS first, which no token count can see. The first
test below is that case, and it is the one a hand-rolled token counter gets wrong.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs] - no stubs; typed at the facade below
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs] - see above
import pytest

# Typed facade over the two unstubbed pyarrow calls this test needs. Declaring the signatures
# under TYPE_CHECKING types every call site below without suppressing anything: the checker reads
# these declarations, the runtime binds the real functions. Drop it when pyarrow ships stubs.
if TYPE_CHECKING:

    def _arrow_table(columns: dict[str, list[str]]) -> Any: ...

    def _write_parquet(table: Any, where: Path) -> None: ...

else:
    _arrow_table = pa.table
    _write_parquet = pq.write_table

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "audit_embedder_truncation.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("audit_embedder_truncation", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_embedder_truncation"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> Any:
    return _load()


class WordTokenizer:
    """Tokenizes on whitespace, applying an id-slice cap like a plain embedder."""

    def tokenize(self, texts: Sequence[str], max_length: int | None = None) -> list[list[int]]:
        out: list[list[int]] = []
        for text in texts:
            ids = list(range(len(text.split())))
            out.append(ids[:max_length] if max_length is not None else ids)
        return out


class CharCutTokenizer(WordTokenizer):
    """Cuts the raw string by CHARACTERS before tokenizing, as model2vec does.

    With a generous token cap the id slice never bites, so a probe that merely counted tokens
    would report every chunk as fitting while the embedder had already dropped text.
    """

    def __init__(self, char_budget: int) -> None:
        self._char_budget = char_budget

    def tokenize(self, texts: Sequence[str], max_length: int | None = None) -> list[list[int]]:
        if max_length is None:
            return super().tokenize(texts, None)
        cut = [t[: self._char_budget] for t in texts]
        return super().tokenize(cut, max_length)


def _probe(audit: Any, tokenizer: Any, *, cap: int, char_precut: int | None = None, label: str = "fake:probe") -> Any:
    # The label must be the one the registry was keyed under: results are keyed by probe.label,
    # so a hardcoded label makes every by-label assertion pass without asserting anything.
    return audit.Probe(label=label, tokenizes=tokenizer, token_cap=cap, char_precut=char_precut)


def test_a_character_cut_before_tokenizing_is_still_caught(audit: Any) -> None:
    # 40 words, one char each plus spaces = 79 chars; the char budget drops the tail.
    text = " ".join("w" for _ in range(40))
    probe = _probe(audit, CharCutTokenizer(char_budget=20), cap=1000, char_precut=20)
    counts = audit.count_batch([text], probe)
    # The token cap alone would never fire: 40 tokens against a cap of 1000.
    assert counts.max_tokens_full == 40
    assert counts.truncated == 1
    assert counts.tokens_kept < counts.tokens_full


def test_nothing_truncated_when_every_chunk_fits(audit: Any) -> None:
    probe = _probe(audit, WordTokenizer(), cap=10)
    counts = audit.count_batch(["a b c", "d e"], probe)
    assert counts.truncated == 0
    assert counts.tokens_kept == counts.tokens_full == 5
    assert counts.chunks == 2


def test_the_token_cut_is_counted(audit: Any) -> None:
    probe = _probe(audit, WordTokenizer(), cap=2)
    counts = audit.count_batch(["a b c d", "e f"], probe)
    assert counts.truncated == 1
    assert counts.tokens_full == 6
    assert counts.tokens_kept == 4  # 2 kept of 4, plus 2 of 2
    assert counts.max_tokens_full == 4


def test_an_empty_batch_counts_nothing(audit: Any) -> None:
    counts = audit.count_batch([], _probe(audit, WordTokenizer(), cap=5))
    assert counts == audit.Counts()


class RowDroppingTokenizer(WordTokenizer):
    """Returns fewer rows than it was given - a probe that would silently under-report."""

    def tokenize(self, texts: Sequence[str], max_length: int | None = None) -> list[list[int]]:
        return super().tokenize(texts, max_length)[:-1]


def test_a_probe_that_drops_rows_is_an_error_not_a_silent_pass(audit: Any) -> None:
    probe = _probe(audit, RowDroppingTokenizer(), cap=5)
    with pytest.raises(RuntimeError, match="rows for"):
        audit.count_batch(["a b", "c d"], probe)


def test_counts_merge_keeps_the_maxima(audit: Any) -> None:
    a = audit.Counts(chunks=2, truncated=1, tokens_full=10, tokens_kept=6, max_tokens_full=7, max_chars=30)
    b = audit.Counts(chunks=3, truncated=0, tokens_full=9, tokens_kept=9, max_tokens_full=4, max_chars=99)
    merged = a.merge(b)
    assert merged.chunks == 5
    assert merged.truncated == 1
    assert merged.tokens_full == 19
    assert merged.tokens_kept == 15
    assert merged.max_tokens_full == 7  # the max, not the last or the sum
    assert merged.max_chars == 99


def test_summarise_reports_headroom_and_lost_mass(audit: Any) -> None:
    counts = audit.Counts(chunks=4, truncated=1, tokens_full=200, tokens_kept=150, max_tokens_full=90, max_chars=300)
    rec = audit.summarise(counts, _probe(audit, WordTokenizer(), cap=100, char_precut=1000))
    assert rec["truncated_pct"] == 25.0
    assert rec["token_mass_lost_pct"] == 25.0
    assert rec["headroom_tokens"] == 10  # cap 100 minus the longest chunk at 90
    assert rec["headroom_chars"] == 700


def test_summarise_on_an_empty_set_does_not_divide_by_zero(audit: Any) -> None:
    rec = audit.summarise(audit.Counts(), _probe(audit, WordTokenizer(), cap=100))
    assert rec["truncated_pct"] == 0.0
    assert rec["token_mass_lost_pct"] == 0.0
    assert rec["headroom_chars"] is None


class PaddingTokenizer:
    """A `tokenizers.Tokenizer` stand-in that PADS every batch to its longest member.

    This is the real bge-base configuration. While padding is on, every encoding in a batch has
    the same id count, so a capped-vs-uncapped length comparison agrees by construction.
    """

    def __init__(self) -> None:
        self.padding = True
        self.no_padding_calls = 0
        self._truncate_to: int | None = None

    def no_padding(self) -> None:
        self.padding = False
        self.no_padding_calls += 1

    def no_truncation(self) -> None:
        self._truncate_to = None

    def enable_truncation(self, max_length: int) -> None:
        self._truncate_to = max_length

    def encode_batch(self, texts: list[str]) -> list[Any]:
        ids = [list(range(len(t.split()))) for t in texts]
        if self._truncate_to is not None:
            ids = [i[: self._truncate_to] for i in ids]
        if self.padding:
            width = max((len(i) for i in ids), default=0)
            ids = [i + [0] * (width - len(i)) for i in ids]
        return [SimpleNamespace(ids=i) for i in ids]


def test_the_driver_turns_padding_off_so_lengths_measure_content(audit: Any) -> None:
    tokenizer = PaddingTokenizer()
    driver = audit.TruncatingTokenizer(tokenizer)
    assert tokenizer.no_padding_calls == 1, "padding must be disabled once, at construction"
    out = driver.tokenize(["a b c d", "e"], max_length=None)
    assert [len(x) for x in out] == [4, 1], "padded lengths would both read 4 and hide every difference"


def test_a_padding_tokenizer_cannot_hide_truncation_from_the_audit(audit: Any) -> None:
    probe = _probe(audit, audit.TruncatingTokenizer(PaddingTokenizer()), cap=2)
    counts = audit.count_batch(["a b c d e", "f"], probe)
    assert counts.truncated == 1
    assert counts.tokens_full == 6
    assert counts.tokens_kept == 3


def test_padding_left_on_makes_the_count_describe_the_batch_and_not_the_rows(audit: Any) -> None:
    """The control for the two tests above, and the reason padding is disabled at construction.

    Padding pads to the BATCH maximum, and that maximum is not the same in the capped and the
    uncapped call - so the per-row comparison stops describing rows. Here exactly ONE of the two
    rows is over the cap and it reports BOTH. The opposite reading appears when no row exceeds the
    cap: the two calls then pad to the same width and it reports none, which is what the first
    bge-base measurement produced at every rung.
    """

    class NeverDisablesPadding(PaddingTokenizer):
        def no_padding(self) -> None:
            self.no_padding_calls += 1  # counted, but padding deliberately stays on

    padded = _probe(audit, audit.TruncatingTokenizer(NeverDisablesPadding()), cap=2)
    assert audit.count_batch(["a b c d e", "f"], padded).truncated == 2

    # The same input through the shipped driver, which turns padding off, counts the one real row.
    honest = _probe(audit, audit.TruncatingTokenizer(PaddingTokenizer()), cap=2)
    assert audit.count_batch(["a b c d e", "f"], honest).truncated == 1


def _write_chunks(path: Path, texts: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_parquet(_arrow_table({"text": texts}), path / "chunks.parquet")


def test_sample_spreads_across_the_file_rather_than_taking_the_head(audit: Any, tmp_path: Path) -> None:
    # The head of a real chunk set is one document's chunks, so a head sample is not a sample.
    texts = ["head"] * 500 + ["tail"] * 500
    _write_chunks(tmp_path / "set", texts)
    sampled = [t for batch in audit._batches(tmp_path / "set" / "chunks.parquet", 10) for t in batch]
    assert len(sampled) == 10
    assert "tail" in sampled, "a sample that never reaches the tail cannot see where truncation starts"


def test_sample_of_zero_reads_every_chunk(audit: Any, tmp_path: Path) -> None:
    _write_chunks(tmp_path / "set", [f"w{i}" for i in range(37)])
    sampled = [t for batch in audit._batches(tmp_path / "set" / "chunks.parquet", 0) for t in batch]
    assert len(sampled) == 37


def test_a_row_records_whether_the_embedder_actually_embedded_that_set(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An embedder can be measured against a chunk set it never embedded; the row must say so
    # rather than read as a measurement of a cell that exists.
    _write_chunks(tmp_path / "chunks" / "corpus__profile", ["a b"])
    audit.cell_dir(tmp_path, "corpus__profile", "fake:covered").mkdir(parents=True)
    _write_chunks(tmp_path / "chunks" / "corpus__unembedded", ["c d"])
    out = tmp_path / "report.json"

    def _factory(label: str) -> Any:
        return _probe(audit, WordTokenizer(), cap=100, label=label)

    monkeypatch.setitem(audit._PROBES, "fake:covered", _factory)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:covered")
    monkeypatch.setenv("OUT", str(out))
    assert audit.main() == 0
    import json

    rows = {r["chunk_set"]: r for r in json.loads(out.read_text())["rows"]}
    assert rows["corpus__profile"]["cell_exists"] is True
    assert rows["corpus__unembedded"]["cell_exists"] is False


def test_the_cell_path_writes_an_embedder_label_the_way_the_cache_does(audit: Any, tmp_path: Path) -> None:
    path = audit.cell_dir(tmp_path, "corpus__recursive-t256-o0-gpt2", "model2vec:potion-base-8M")
    assert path.name == "corpus__recursive-t256-o0-gpt2__model2vec-potion-base-8M"


def test_an_unknown_embedder_is_reported_uncovered_never_untruncated(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_chunks(tmp_path / "chunks" / "corpus__profile", ["a b c d e f"])
    out = tmp_path / "report.json"

    def _factory(label: str) -> Any:
        return _probe(audit, WordTokenizer(), cap=100, label=label)

    monkeypatch.setitem(audit._PROBES, "fake:covered", _factory)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:covered,fake:unknown")
    monkeypatch.setenv("OUT", str(out))
    assert audit.main() == 0
    import json

    payload = json.loads(out.read_text())
    assert payload["uncovered"] == ["fake:unknown"]
    measured = {r["embedder"] for r in payload["rows"]}
    # Require the covered label to be present, or the next assertion passes for the wrong reason.
    assert "fake:covered" in measured
    # The unknown embedder must not appear anywhere as a measured, untruncated result.
    assert "fake:unknown" not in measured
    assert "uncovered" in capsys.readouterr().err


def test_no_covered_embedder_is_an_error_not_a_clean_report(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "report.json"
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:unknown")
    monkeypatch.setenv("OUT", str(out))
    assert audit.main() == 2
    assert not out.exists(), "a run that measured nothing must not leave a report behind"


def test_a_finding_exits_1_so_it_can_gate(audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_chunks(tmp_path / "chunks" / "corpus__profile", ["a b c d e f g h"])
    out = tmp_path / "report.json"

    def _factory(label: str) -> Any:
        return _probe(audit, WordTokenizer(), cap=3, label=label)

    monkeypatch.setitem(audit._PROBES, "fake:tight", _factory)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:tight")
    monkeypatch.setenv("OUT", str(out))
    assert audit.main() == 1, "truncation is a finding, and a finding must be distinguishable from success"


def test_only_the_named_sets_are_audited(audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_chunks(tmp_path / "chunks" / "corpus__wanted", ["a b"])
    _write_chunks(tmp_path / "chunks" / "corpus__other", ["c d"])
    out = tmp_path / "report.json"

    def _factory(label: str) -> Any:
        return _probe(audit, WordTokenizer(), cap=100, label=label)

    monkeypatch.setitem(audit._PROBES, "fake:covered", _factory)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:covered")
    monkeypatch.setenv("AUDIT_SETS", "corpus__wanted")
    monkeypatch.setenv("OUT", str(out))
    assert audit.main() == 0
    import json

    assert [r["chunk_set"] for r in json.loads(out.read_text())["rows"]] == ["corpus__wanted"]


def _run_main(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, sets: str, cap: int, sample: int = 0
) -> Any:
    """Drive main() once against tmp_path's cache with one fake probe, returning the report."""
    import json

    def _factory(label: str) -> Any:
        return _probe(audit, WordTokenizer(), cap=cap, label=label)

    monkeypatch.setitem(audit._PROBES, "fake:merge", _factory)
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUDIT_EMBEDDINGS", "fake:merge")
    monkeypatch.setenv("AUDIT_SETS", sets)
    monkeypatch.setenv("AUDIT_SAMPLE", str(sample))
    monkeypatch.setenv("OUT", str(tmp_path / "report.json"))
    audit.main()
    return json.loads((tmp_path / "report.json").read_text())


def test_a_rerun_on_other_sets_keeps_the_rows_it_did_not_measure(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A census is built up over several runs (the ladder one day, the rest another); a run that
    # overwrote the report with only what it scanned would silently drop every earlier row.
    _write_chunks(tmp_path / "chunks" / "corpus__first", ["a b"])
    _write_chunks(tmp_path / "chunks" / "corpus__second", ["c d"])
    _run_main(audit, tmp_path, monkeypatch, sets="corpus__first", cap=100)
    report = _run_main(audit, tmp_path, monkeypatch, sets="corpus__second", cap=100)
    assert sorted(r["chunk_set"] for r in report["rows"]) == ["corpus__first", "corpus__second"]


def test_a_rerun_replaces_the_row_it_re_measures(audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_chunks(tmp_path / "chunks" / "corpus__set", ["a b c d e f"])
    first = _run_main(audit, tmp_path, monkeypatch, sets="corpus__set", cap=100)
    assert first["rows"][0]["truncated"] == 0
    report = _run_main(audit, tmp_path, monkeypatch, sets="corpus__set", cap=3)
    assert len(report["rows"]) == 1, "the same (set, embedder) must not appear twice"
    assert report["rows"][0]["truncated"] == 1


def test_every_row_records_the_sample_it_was_measured_on(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A merged report mixes full and sampled rows, so the sample size is a property of the row.
    _write_chunks(tmp_path / "chunks" / "corpus__set", ["a b"] * 10)
    report = _run_main(audit, tmp_path, monkeypatch, sets="corpus__set", cap=100, sample=4)
    assert report["rows"][0]["sample"] == 4


def test_a_sampled_rerun_never_replaces_a_full_measurement(
    audit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The ladder was measured on every chunk; a later screening census at a sample must not
    # downgrade those rows to an estimate.
    _write_chunks(tmp_path / "chunks" / "corpus__set", ["a b c d e f"] * 10)
    _run_main(audit, tmp_path, monkeypatch, sets="corpus__set", cap=3, sample=0)
    report = _run_main(audit, tmp_path, monkeypatch, sets="corpus__set", cap=3, sample=2)
    assert len(report["rows"]) == 1
    assert report["rows"][0]["sample"] == 0
    assert report["rows"][0]["chunks"] == 10


def test_a_legacy_report_with_one_top_level_sample_is_read_as_per_row_samples(audit: Any, tmp_path: Path) -> None:
    # The first reports carried one top-level sample for every row; a merge must read those rows
    # as measured on that sample, never as unmeasured.
    import json

    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"sample": 0, "uncovered": [], "rows": [{"chunk_set": "s", "embedder": "e", "chunks": 3}]})
    )
    rows = audit._read_report(legacy)["rows"]
    assert rows[0]["sample"] == 0
