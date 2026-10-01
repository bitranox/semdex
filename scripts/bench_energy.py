#!/usr/bin/env python
"""What it costs in energy to embed a corpus, per provider.

Cost is the one axis of the component choice this repo never measured. It can be measured here
rather than modelled: the CPU package exposes a RAPL energy counter, the GPU reports power, and
the node hangs on a metered wall plug.

DOCUMENTS and BYTES are the denominator, never tokens. Token counts are not comparable across
chunking strategies - the semantic chunker counts in a different tokenizer and differs by up to 22
percent on identical text - so any per-token cost figure inherits that error before it starts. A
document is a document under every strategy.

Idle and work are INTERLEAVED (idle, work, idle, work, ...), never idle-once-then-work. This box is
shared and its idle draw swings 70 to 97 W, so a baseline taken once before the run absorbs
whatever a neighbouring container happened to be doing at that moment and is then subtracted from
every row. Each row carries the resulting signal-to-noise, and a row that cannot be resolved from
the idle swing says so instead of being quoted to three decimals.

Env: PROVIDERS, DOCS (default 2000), REPEATS (default 3), IDLE_SECONDS (default 20), TARIFF
(EUR/kWh, default 0.30), OUT. GPU cells additionally need GPU_HOST and read its nvidia-smi.
"""

# pyright: basic
# One-off benchmark harness; sudo/ssh probes and untyped deps make strict mode pure noise here.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _bench_energy import EnergyResult, cost_per_million, counter_delta_uj, idle_estimate, window_plan

_RAPL = Path("/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0")

DEFAULT_PROVIDERS = ("placeholder", "model2vec", "fastembed", "sentence_transformers")


def _sudo_read_int(path: Path) -> int:
    """RAPL is root-readable only, and srvadmin holds NOPASSWD sudo on this host."""
    out = subprocess.run(["sudo", "-n", "cat", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


class RaplCounter:
    """The CPU package energy counter, with its wrap point read once from the kernel."""

    def __init__(self) -> None:
        self.max_range_uj = _sudo_read_int(_RAPL / "max_energy_range_uj")

    def read_uj(self) -> int:
        return _sudo_read_int(_RAPL / "energy_uj")

    def measure(self, work: Any) -> tuple[float, float]:
        """Run ``work`` and return ``(joules, seconds)`` over exactly its execution.

        Two counter reads, because RAPL is cumulative: the sampling rate cannot affect an integral
        taken from a counter, which is the whole reason to prefer it over a power gauge.
        """
        before, started = self.read_uj(), time.perf_counter()
        work()
        elapsed, after = time.perf_counter() - started, self.read_uj()
        return counter_delta_uj(before, after, max_range_uj=self.max_range_uj) / 1e6, elapsed

    def idle_watts(self, seconds: float) -> float:
        joules, elapsed = self.measure(lambda: time.sleep(seconds))
        return joules / elapsed if elapsed else 0.0


def load_documents(n: int) -> list[str]:
    """Real corpus text, so the per-document figure maps onto the quality tables."""
    import ir_datasets

    out: list[str] = []
    for doc in ir_datasets.load("beir/nfcorpus/test").docs_iter():
        out.append((getattr(doc, "title", "") + " " + doc.text).strip())
        if len(out) >= n:
            break
    return out


# A work window shorter than this cannot be told apart from the baseline drifting under it. The
# fast providers embed 2,000 documents in well under a second, so the corpus is repeated until the
# window is filled - the same reason the throughput benchmark batches to a floor duration, and the
# same defect if skipped: a 25 ms window divided by a document count is arithmetic on drift.
MIN_WORK_SECONDS = 15.0


def _probe_rate(embedding: Any, documents: list[str]) -> float:
    """Documents per second from a short timed probe, used only to size the real window.

    Deliberately a SMALL sample: the probe on a slow CPU provider would otherwise be the most
    expensive part of its own measurement, and its only job is to pick a document count.
    """
    sample = documents[: min(64, len(documents))]
    started = time.perf_counter()
    embedding.embed_passages(sample)
    elapsed = time.perf_counter() - started
    return len(sample) / elapsed if elapsed > 0 else 0.0


def measure_provider(provider: str, documents: list[str], *, repeats: int, idle_seconds: float) -> dict[str, Any]:
    """Interleave idle and work windows for one provider and reduce them to one row."""
    from semdex.composition import build_embedding
    from semdex.domain.enums import EmbeddingBackend

    counter = RaplCounter()
    embedding = build_embedding(EmbeddingBackend(provider), allow_fallback=False)
    embedding.embed_passages(documents[:32])  # warm: model load must not land inside a window
    min_seconds = float(os.environ.get("MIN_WORK_SECONDS", MIN_WORK_SECONDS))
    per_pass, passes = window_plan(_probe_rate(embedding, documents), len(documents), min_seconds)
    work_docs = documents[:per_pass]
    print(f"    window: {per_pass} docs x {passes} passes", flush=True)

    def one_window() -> None:
        for _ in range(passes):
            embedding.embed_passages(work_docs)

    idles: list[float] = []
    work_j = 0.0
    work_s = 0.0
    for _ in range(repeats):
        idles.append(counter.idle_watts(idle_seconds))
        joules, seconds = counter.measure(one_window)
        work_j += joules
        work_s += seconds
    idles.append(counter.idle_watts(idle_seconds))

    idle_w, idle_spread = idle_estimate(idles)
    result = EnergyResult(
        total_j=work_j,
        seconds=work_s,
        idle_w=idle_w,
        idle_spread_w=idle_spread,
        units=per_pass * repeats * passes,
    )
    return _row(provider, embedding.model_id, embedding.dim, result, documents=work_docs, passes=passes)


def _row(
    provider: str,
    model_id: str,
    dim: int,
    result: EnergyResult,
    *,
    documents: list[str],
    passes: int,
    probe: str = "cpu-rapl",
) -> dict[str, Any]:
    tariff = float(os.environ.get("TARIFF", "0.30"))
    total_bytes = sum(len(d.encode()) for d in documents)
    return {
        "provider": provider,
        "model_id": model_id,
        "dim": dim,
        "probe": probe,
        "documents": result.units,
        "passes_per_window": passes,
        "mean_doc_bytes": round(total_bytes / len(documents), 1),
        "seconds": round(result.seconds, 2),
        "docs_per_s": round(result.units / result.seconds, 1) if result.seconds else 0.0,
        "average_w": round(result.average_w, 1),
        "idle_w": round(result.idle_w, 1),
        "idle_spread_w": round(result.idle_spread_w, 1),
        "marginal_j_per_doc": round(result.marginal_j_per_unit, 4),
        "total_j_per_doc": round(result.total_j_per_unit, 4),
        "signal_to_noise": round(result.signal_to_noise, 2),
        "resolved": result.resolved,
        "eur_per_million_docs_marginal": round(
            cost_per_million(result.marginal_j_per_unit, tariff_eur_per_kwh=tariff), 4
        ),
        "eur_per_million_docs_total": round(cost_per_million(result.total_j_per_unit, tariff_eur_per_kwh=tariff), 4),
    }


def main() -> None:
    docs = load_documents(int(os.environ.get("DOCS", "2000")))
    repeats = int(os.environ.get("REPEATS", "3"))
    idle_seconds = float(os.environ.get("IDLE_SECONDS", "20"))
    providers = os.environ.get("PROVIDERS", ",".join(DEFAULT_PROVIDERS)).split(",")
    print(f"{len(docs)} documents, {repeats} repeats, {idle_seconds:.0f}s idle windows interleaved", flush=True)

    rows: list[dict[str, Any]] = []
    for provider in providers:
        print(f"provider {provider}", flush=True)
        try:
            row = measure_provider(provider, docs, repeats=repeats, idle_seconds=idle_seconds)
        except Exception as exc:  # one unavailable provider must not end the sweep
            print(f"  SKIP {provider}: {type(exc).__name__}: {exc}", flush=True)
            continue
        flag = "" if row["resolved"] else "  NOT RESOLVED against idle swing"
        print(
            f"    {row['marginal_j_per_doc']} J/doc marginal, {row['docs_per_s']} docs/s, "
            f"idle {row['idle_w']}+/-{row['idle_spread_w']} W, s/n {row['signal_to_noise']}{flag}",
            flush=True,
        )
        rows.append(row)

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "cpu": _cpu_model(),
        "tariff_eur_per_kwh": float(os.environ.get("TARIFF", "0.30")),
        "corpus": "beir/nfcorpus/test",
        "note": "Denominated in DOCUMENTS, never tokens: token counts differ by up to 22 percent "
        "across chunking strategies, so a per-token cost inherits that error.",
        "rows": rows,
    }
    out_path = Path(os.environ.get("OUT", "energy.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


if __name__ == "__main__":
    main()
