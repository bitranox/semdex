#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy/pyarrow (no strict stubs); strict mode would only add
# reportUnknown* noise. Same stance as bench_dim_crossover.py and score_chunk_sweep.py.
"""Audit what each sweep axis ACTUALLY changed in the cached chunk sets.

A profile string names the knobs a chunk set was built with (strategy, max_tokens, overlap,
tokenizer, recipe, semantic breakpoint model). It does not prove any of them took effect: an
adapter can accept a knob and ignore it, and two profiles that differ only in a name can hold
byte-identical chunk boundaries. Reading the profile string and believing it is how a sweep comes
to report an effect for a knob that was never applied.

So this reads the chunk sets themselves. For every ``<corpus>__<profile>`` set it streams
``chunks.parquet`` once and records the row count, the document count, the token-length
distribution, the character total, how many chunks exceed the profile's TRUE cap
(``max_tokens + overlap``, since overlap is prepended context and deliberately pushes a chunk
over), and a sha256 over the chunk boundary sequence. Two profiles with the same boundary hash
are the same configuration wearing two names.

Characters matter as much as tokens here, because ``token_count`` is not comparable across
strategies: chonkie 1.7.0's ``SemanticChunker`` and ``LateChunker`` accept no tokenizer argument
at all and count in their embedding model's tokenizer, while ``RecursiveChunker`` counts in the
configured one. That alone shows up as a token difference of up to 22 percent over byte-identical
text, so the character total is what decides whether a profile really covers the corpus.

It then groups the sets so that exactly one axis varies at a time and reports what that axis did:
whether the boundaries moved, whether the token mass moved, and whether the nominal cap was
honoured. Each axis comes out labelled ``real``, ``content-only``, ``weak``, ``inert``, or
``single-level``, with the measurement that decided it attached.

Bounded memory: one column batch at a time, a bincount histogram for the length distribution
(never a list of per-chunk lengths), and documents counted as 8-byte digests rather than as full
uri strings. Peak stays a few hundred MB on the largest set.

Env:
  CACHE_ROOT   cache dir (default /embeddings)
  CORPORA      optional comma-separated filter (default: every corpus in the cache)
  OUT          results json (default tests/benchmarks/raw/chunk-dimension-audit.json)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
_BATCH = 16384
_HIST_CAP = 1 << 17  # token lengths above this are clipped into the top bin; nothing real is close
# Generous per-boundary allowance for a delimiter or whitespace a split may drop, used to bound
# the expected character difference between two profiles of the same corpus (see _integrity).
_BOUNDARY_CHARS = 4

# <strategy>-t<max_tokens>-o<overlap>-<tokenizer>[-r<recipe>][-bp<semantic_model>]
# Built by preembed_vectors._profile_tag; ":" in a model name is written as "-".
_PROFILE_RE = re.compile(r"^(?P<strategy>[a-z_]+)-t(?P<max_tokens>\d+)-o(?P<overlap>\d+)-(?P<tail>.+)$")
# chonkie recipes the chunker has written into a profile tag. A recipe segment is split off only
# when it names one of these, so a tokenizer whose own name contains "-r" is never cut in two.
_KNOWN_RECIPES = ("markdown",)


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def parse_profile(profile: str) -> dict[str, Any]:
    """Split a profile tag into its axes.

    The recipe segment is ambiguous to parse (a tokenizer name may itself contain "-r"), so only a
    trailing ``-r<recipe>`` naming one of ``_KNOWN_RECIPES`` is split off; any other ``-r`` segment
    is FLAGGED rather than guessed at: a wrong split would silently mislabel an axis, which is the
    exact failure this whole script exists to catch.
    """
    m = _PROFILE_RE.match(profile)
    if not m:
        return {"strategy": None, "parse_error": f"profile does not match the grammar: {profile}"}
    tail = m.group("tail")
    breakpoint_model = None
    if "-bp" in tail:
        tail, breakpoint_model = tail.rsplit("-bp", 1)
    recipe = next((r for r in _KNOWN_RECIPES if tail.endswith(f"-r{r}")), "")
    if recipe:
        tail = tail[: -len(f"-r{recipe}")]
    axes: dict[str, Any] = {
        "strategy": m.group("strategy"),
        "max_tokens": int(m.group("max_tokens")),
        "overlap_tokens": int(m.group("overlap")),
        "tokenizer": tail,
        "recipe": recipe,
        "breakpoint_model": breakpoint_model,
    }
    if "-r" in tail:
        axes["recipe_unparsed"] = tail  # never guess the split; see the docstring
    return axes


class _ChunkSetScan:
    """Streaming accumulator over one chunks.parquet."""

    def __init__(self) -> None:
        self.rows = 0
        self.tokens = 0
        self.chars = 0
        self.hist = np.zeros(_HIST_CAP, dtype=np.int64)
        self.sha = hashlib.sha256()
        # Two independent document counts, kept so they can DISAGREE. Counting the ordinal reset
        # alone looked right and silently returned 1 document for every repaired set, because the
        # repair renumbers ordinals across the whole set rather than per document. The distinct
        # count is authoritative; the transition count exists to prove rows are still grouped by
        # document, which several later comparisons assume.
        self._seen: set[bytes] = set()
        self._transitions = 0
        self._prev: str | None = None

    def feed(self, uris: list[str], ordinals: list[int], counts: list[int], texts: list[str]) -> None:
        self.rows += len(counts)
        arr = np.asarray(counts, dtype=np.int64)
        self.tokens += int(arr.sum())
        # Characters are the strategy-INDEPENDENT measure of how much text a profile covers.
        # token_count is not: chonkie's SemanticChunker/LateChunker take no tokenizer argument
        # (verified against chonkie 1.7.0), so they count in their embedding model's tokenizer
        # while recursive counts in the configured one. Comparing token totals across strategies
        # therefore shows a difference of up to 22 percent where the text is byte-identical.
        self.chars += sum(len(t) for t in texts)
        self.hist += np.bincount(np.clip(arr, 0, _HIST_CAP - 1), minlength=_HIST_CAP)
        for uri, ordinal, count in zip(uris, ordinals, counts, strict=True):
            self.sha.update(f"{uri}|{ordinal}|{count}\n".encode())
            # An 8-byte digest per DOCUMENT (not per chunk): bounded by the corpus document
            # count, which is the quantity being measured, and about 12x smaller than the uris.
            self._seen.add(hashlib.blake2b(uri.encode(), digest_size=8).digest())
            if uri != self._prev:
                self._transitions += 1
                self._prev = uri

    @property
    def docs(self) -> int:
        return len(self._seen)

    @property
    def grouped_by_doc(self) -> bool:
        return self._transitions == len(self._seen)

    def percentile(self, q: float) -> int:
        if self.rows == 0:
            return 0
        target = q * self.rows
        return int(np.searchsorted(np.cumsum(self.hist), target))

    def max_token(self) -> int:
        nz = np.nonzero(self.hist)[0]
        return int(nz[-1]) if nz.size else 0

    def over(self, cap: int) -> int:
        return int(self.hist[cap + 1 :].sum()) if cap > 0 else 0


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def true_cap(axes: dict[str, Any]) -> int:
    """The largest chunk a profile may legitimately hold, in the chunker's tokens.

    ``recursive`` appends its overlap on top of a chunk already cut to the cap, so its ceiling is
    the cap plus the overlap. ``markdown`` and ``fast`` re-cut the text into windows of the cap
    and honour overlap inside it, and the chonkie strategies ignore overlap, so for every other
    strategy the ceiling is the cap itself. Adding the overlap for those made "cap held" a check
    that could not fail on them.
    """
    nominal = axes.get("max_tokens")
    if not nominal:
        return 0
    overlap = int(axes.get("overlap_tokens") or 0) if axes.get("strategy") == "recursive" else 0
    return int(nominal) + overlap


def scan_chunk_set(directory: Path) -> dict[str, Any] | None:
    """Stream one chunk set and return its measured row. None when it holds no parquet."""
    parquet = directory / "chunks.parquet"
    if not parquet.exists():
        return None
    corpus, profile = directory.name.split("__", 1)
    axes = parse_profile(profile)
    scan = _ChunkSetScan()
    columns = ["source_uri", "ordinal", "token_count", "text"]
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=_BATCH, columns=columns):
        data = batch.to_pydict()
        scan.feed(data["source_uri"], data["ordinal"], data["token_count"], data["text"])
    return _row(directory, corpus, profile, axes, scan)


def _row(directory: Path, corpus: str, profile: str, axes: dict[str, Any], scan: _ChunkSetScan) -> dict[str, Any]:
    meta = _read_json(directory / "meta.json") or {}
    queries = _read_json(directory / "queries.json") or {}
    qrels = _read_json(directory / "qrels.json") or {}
    # The TRUE cap: overlap is prepended context, so the chunker deliberately overshoots
    # max_tokens by exactly the overlap. Measuring against max_tokens alone reports every
    # overlap chunk as oversized and hides the real violations.
    nominal = int(axes.get("max_tokens") or 0)
    true_cap_tokens = true_cap(axes)
    return {
        "corpus": corpus,
        "profile": profile,
        "axes": axes,
        "rows": scan.rows,
        "docs": scan.docs,
        "rows_grouped_by_doc": scan.grouped_by_doc,
        "chunks_per_doc": round(scan.rows / scan.docs, 3) if scan.docs else None,
        "tokens_total": scan.tokens,
        "chars_total": scan.chars,
        "chars_per_token": round(scan.chars / scan.tokens, 3) if scan.tokens else None,
        "token_mean": round(scan.tokens / scan.rows, 2) if scan.rows else 0.0,
        "token_p50": scan.percentile(0.50),
        "token_p90": scan.percentile(0.90),
        "token_p95": scan.percentile(0.95),
        "token_max": scan.max_token(),
        "nominal_cap": nominal,
        "true_cap": true_cap_tokens,
        "over_true_cap": scan.over(true_cap_tokens),
        "cap_enforced": scan.max_token() <= true_cap_tokens if true_cap_tokens else None,
        "boundary_sha256": scan.sha.hexdigest(),
        "meta_count": meta.get("count"),
        "meta_count_stale": (meta.get("count") is not None and meta.get("count") != scan.rows),
        "n_queries": len(queries),
        "n_qrels": len(qrels),
        "chunks_pre_repair": bool(list(directory.glob("*.pre-repair"))),
    }


def vector_repair_state(cache: Path) -> dict[str, dict[str, int]]:
    """Per chunk set, how many of its embedded cells still carry a .pre-repair sibling."""
    state: dict[str, dict[str, int]] = defaultdict(lambda: {"cells": 0, "pre_repair": 0})
    for cell in sorted((cache / "vectors").glob("*__*__*")):
        parts = cell.name.split("__")
        if len(parts) != 3:
            continue
        key = f"{parts[0]}__{parts[1]}"
        state[key]["cells"] += 1
        state[key]["pre_repair"] += 1 if list(cell.glob("*.pre-repair")) else 0
    return dict(state)


def _axis_key(row: dict[str, Any], axis: str) -> tuple:
    """Everything that identifies a comparison group EXCEPT the axis under test."""
    axes = row["axes"]
    fields = ["strategy", "max_tokens", "overlap_tokens", "tokenizer", "breakpoint_model"]
    return (row["corpus"], *(axes.get(f) for f in fields if f != axis))


def _compare(axis: str, levels: list[dict[str, Any]]) -> dict[str, Any]:
    """What varying one axis did to one otherwise-identical group of chunk sets."""
    shas = {lv["boundary_sha256"] for lv in levels}
    tokens = {lv["tokens_total"] for lv in levels}
    rows = [lv["rows"] for lv in levels]
    row_spread = (max(rows) - min(rows)) / max(rows) if max(rows) else 0.0
    if len(shas) == 1:
        status = "inert"
    elif len(tokens) == 1 and row_spread < 0.05:
        status = "weak"
    elif row_spread < 0.001:
        status = "content-only"
    else:
        status = "real"
    return {
        "axis": axis,
        "corpus": levels[0]["corpus"],
        "held_fixed": {k: v for k, v in levels[0]["axes"].items() if k not in (axis, "recipe")},
        "levels": [
            {
                "value": lv["axes"].get(axis),
                "profile": lv["profile"],
                "rows": lv["rows"],
                "tokens_total": lv["tokens_total"],
                "token_max": lv["token_max"],
                "boundary_sha256": lv["boundary_sha256"][:12],
            }
            for lv in sorted(levels, key=lambda entry: str(entry["axes"].get(axis)))
        ],
        "distinct_boundaries": len(shas),
        "distinct_token_mass": len(tokens),
        "row_count_spread_pct": round(row_spread * 100, 2),
        "status": status,
    }


def axis_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (axis, otherwise-identical group) that actually has two or more levels."""
    effects: list[dict[str, Any]] = []
    for axis in ("overlap_tokens", "max_tokens", "strategy", "breakpoint_model", "tokenizer"):
        groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["axes"].get("strategy") is None:
                continue
            groups[_axis_key(row, axis)].append(row)
        effects.extend(
            _compare(axis, levels)
            for levels in groups.values()
            if len({str(lv["axes"].get(axis)) for lv in levels}) > 1
        )
    return effects


def _conditions(effects: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """For each held-fixed field, which of its values produced which statuses.

    This is what turns "conditional" from a shrug into a usable statement: overlap on recursive
    is content-only at every max_tokens 256 group and real at several 512 ones, and that is
    visible here without opening the per-group list.
    """
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for effect in effects:
        for field, value in effect["held_fixed"].items():
            if field in ("strategy", "tokenizer"):
                continue
            statuses = out[field].setdefault(str(value), [])
            if effect["status"] not in statuses:
                statuses.append(effect["status"])
    return {field: dict(values) for field, values in out.items()}


def axis_status(effects: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-group effects up to one verdict per (axis, strategy).

    Deliberately NOT a worst-case rollup. Taking the strongest status any group showed reported
    every axis as "real", including overlap, whose boundaries provably never move at max_tokens
    256. A knob that only bites under some conditions is reported as conditional, with the counts
    and the conditions attached, because "it depends" is the finding.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        strategy = effect["held_fixed"].get("strategy") or "mixed"
        grouped[f"{effect['axis']}@{strategy}"].append(effect)

    summary: dict[str, Any] = {}
    for key, group in grouped.items():
        counts: dict[str, int] = {}
        for effect in group:
            counts[effect["status"]] = counts.get(effect["status"], 0) + 1
        statuses = sorted(counts, key=lambda s: -counts[s])
        summary[key] = {
            "axis": group[0]["axis"],
            "strategy": group[0]["held_fixed"].get("strategy") or "mixed",
            "groups": len(group),
            "status_counts": counts,
            "verdict": statuses[0] if len(counts) == 1 else "conditional",
            "conditions": _conditions(group) if len(counts) > 1 else {},
        }
    return summary


def _duplicate_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Profiles that differ in name but hold identical chunk boundaries."""
    by_sha: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_sha[f"{row['corpus']}|{row['boundary_sha256']}"].append(row["profile"])
    return [
        {"corpus": key.split("|", 1)[0], "profiles": sorted(profiles)}
        for key, profiles in sorted(by_sha.items())
        if len(profiles) > 1
    ]


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The checks whose failure invalidates a comparison rather than merely annotating it."""
    # Only overlap-0 profiles are comparable this way: with overlap>0 a chunk repeats its
    # predecessor's tail, so the same corpus legitimately yields more characters.
    by_corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (row["axes"].get("overlap_tokens") or 0) == 0:
            by_corpus[row["corpus"]].append(row)
    mass: list[dict[str, Any]] = []
    for corpus, group in sorted(by_corpus.items()):
        if len(group) < 2:
            continue
        chars = {r["profile"]: r["chars_total"] for r in group}
        tokens = {r["profile"]: r["tokens_total"] for r in group}
        cpt = {r["profile"]: r["chars_per_token"] for r in group if r["chars_per_token"]}
        char_delta = max(chars.values()) - min(chars.values())
        token_spread = (max(tokens.values()) - min(tokens.values())) / max(tokens.values())
        # A split can drop the delimiter or the whitespace it cut on, so a profile with more
        # chunks legitimately reconstructs a few characters fewer. That loss is bounded by the
        # chunk count, so the bound is DERIVED from this group rather than being a fixed
        # tolerance: anything above it is not boundary trimming and needs a real explanation.
        bound = max(r["rows"] for r in group) * _BOUNDARY_CHARS
        mass.append(
            {
                "corpus": corpus,
                # THE integrity check. Every overlap-0 profile reconstructs the same corpus
                # exactly once, so a character difference beyond the boundary bound means text
                # was lost or duplicated and the affected cells must not be scored. A token
                # spread with an in-bound character delta is only the counter-unit divergence,
                # which does not touch the text and so cannot bias retrieval.
                "char_delta_chars": char_delta,
                "char_delta_pct": round(char_delta / max(chars.values()) * 100, 6),
                "boundary_loss_bound_chars": bound,
                "text_coverage_ok": char_delta <= bound,
                "fewest_chars_profile": min(chars, key=lambda p: chars[p]),
                "token_spread_pct": round(token_spread * 100, 2),
                "chars_per_token_by_profile": cpt,
                "verdict": (
                    "MATERIAL text loss or duplication"
                    if char_delta > bound
                    else (
                        "token counter unit differs across strategies (text identical to within boundary trimming)"
                        if token_spread > 0.005
                        else "consistent"
                    )
                ),
            }
        )
    return {
        "cap_violations": [
            {
                "corpus": r["corpus"],
                "profile": r["profile"],
                "true_cap": r["true_cap"],
                "token_max": r["token_max"],
                "over": r["over_true_cap"],
            }
            for r in rows
            if r["cap_enforced"] is False
        ],
        "stale_meta_counts": [
            {"corpus": r["corpus"], "profile": r["profile"], "meta_count": r["meta_count"], "rows": r["rows"]}
            for r in rows
            if r["meta_count_stale"]
        ],
        "text_coverage_per_corpus": mass,
        "duplicate_configurations": _duplicate_configs(rows),
        "moved_boundaries_under_overlap": _moved_boundaries(rows),
    }


# Chunk-count spread beyond which a recursive overlap set has been re-cut rather than trimmed.
# Boundary trimming moves a count by a few chunks in a hundred thousand; a re-split moves it by
# a fifth. A tenth of a percent sits between them with a wide margin on both sides.
_CONTENT_ONLY_SPREAD = 0.001


def _moved_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recursive overlap sets whose chunk count moved against their overlap-0 sibling.

    ``recursive`` overlap is context appended to chunks whose boundaries are already fixed, so
    the count must not change with the level. A count that moves means a size-guard re-cut the
    set - four cap512 sets built under an earlier guard grew by a fifth and carried thousands of
    crumb chunks - and every cell embedded from such a set is void, not "overlap being real at
    that cap". The re-splitting strategies (``fast``, ``markdown``) move by design and are not
    judged here.
    """
    at_zero = {
        _overlap_sibling_key(r): r["rows"]
        for r in rows
        if r["axes"].get("strategy") == "recursive" and (r["axes"].get("overlap_tokens") or 0) == 0
    }
    moved: list[dict[str, Any]] = []
    for r in rows:
        if r["axes"].get("strategy") != "recursive" or (r["axes"].get("overlap_tokens") or 0) == 0:
            continue
        base = at_zero.get(_overlap_sibling_key(r))
        if base is None:
            continue
        spread = abs(r["rows"] - base) / base if base else 0.0
        if spread > _CONTENT_ONLY_SPREAD:
            moved.append(
                {
                    "corpus": r["corpus"],
                    "profile": r["profile"],
                    "rows": r["rows"],
                    "rows_at_overlap_0": base,
                    "spread_pct": round(spread * 100, 2),
                }
            )
    return moved


def _overlap_sibling_key(row: dict[str, Any]) -> tuple:
    """Everything that identifies a chunk set except its overlap level."""
    axes = row["axes"]
    return (row["corpus"], axes.get("max_tokens"), axes.get("tokenizer"), axes.get("breakpoint_model"))


def _print_summary(rows: list[dict[str, Any]], status: dict[str, Any], integrity: dict[str, Any]) -> None:
    print(f"\nscanned {len(rows)} chunk sets\n")
    print("axis status (per axis and strategy; conditional means the knob bites only sometimes):")
    for key, value in sorted(status.items()):
        counts = " ".join(f"{k}={v}" for k, v in sorted(value["status_counts"].items()))
        print(f"  {key:34s} {value['verdict']:13s} groups={value['groups']:<3d} [{counts}]")
    print(f"\ncap violations:            {len(integrity['cap_violations'])} chunk sets")
    print(f"stale meta counts:         {len(integrity['stale_meta_counts'])} chunk sets")
    print(f"duplicate configurations:  {len(integrity['duplicate_configurations'])}")
    print(f"moved boundaries (overlap): {len(integrity['moved_boundaries_under_overlap'])} chunk sets, VOID")
    print("\ntext coverage per corpus (overlap-0 profiles reconstruct the same corpus once):")
    for entry in integrity["text_coverage_per_corpus"]:
        flag = "OK  " if entry["text_coverage_ok"] else "LOSS"
        print(
            f"  {flag} {entry['corpus']:24s} char delta {entry['char_delta_chars']:>7,} "
            f"(bound {entry['boundary_loss_bound_chars']:>9,})  token spread {entry['token_spread_pct']:6.2f}%"
        )


def main() -> None:
    cache = _cache_root()
    wanted = [c for c in os.environ.get("CORPORA", "").split(",") if c]
    out = Path(os.environ.get("OUT", str(_ROOT / "tests" / "benchmarks" / "raw" / "chunk-dimension-audit.json")))

    directories = sorted((cache / "chunks").glob("*__*"))
    if wanted:
        directories = [d for d in directories if d.name.split("__", 1)[0] in wanted]
    if not directories:
        sys.exit(f"no chunk sets under {cache / 'chunks'}")

    repair = vector_repair_state(cache)
    rows: list[dict[str, Any]] = []
    for directory in directories:
        row = scan_chunk_set(directory)
        if row is None:
            continue
        row["vector_cells"] = repair.get(directory.name, {}).get("cells", 0)
        row["vector_cells_pre_repair"] = repair.get(directory.name, {}).get("pre_repair", 0)
        rows.append(row)
        print(
            f"[audit] {row['corpus']:24s} {row['profile']:46s} rows={row['rows']:>8d} "
            f"c/doc={row['chunks_per_doc']:>7} tok={row['tokens_total']:>11d} max={row['token_max']:>6d} "
            f"cap_ok={row['cap_enforced']}",
            flush=True,
        )

    effects = axis_effects(rows)
    status = axis_status(effects)
    integrity = _integrity(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "semdex_git_sha": _git_sha(),
                "cache_root": str(cache),
                "chunk_sets": rows,
                "axis_effects": effects,
                "axis_status": status,
                "integrity": integrity,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _print_summary(rows, status, integrity)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
