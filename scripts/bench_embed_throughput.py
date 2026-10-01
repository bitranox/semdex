#!/usr/bin/env python
# pyright: basic
"""Controlled embedding-throughput micro-benchmark (#45).

Embed ONE fixed passage set with each model under DOCUMENTED, controlled conditions and report
docs/s. Unlike the quality-matrix `index_s` (which conflates chunk + embed + store-write over
mixed-length corpora and is reported as a range), this ISOLATES `embed_passages` on a single
fixed text set, warm (model load + a warmup batch excluded from timing), with the thread count,
batch size, and host recorded in the result - so the numbers are comparable across models and
reproducible.

Run on a QUIET host (no concurrent embedding) - CPU throughput is meaningless under contention,
and the ollama models share the GPU with any running pre-embed sweep.

Env:
  SEMDEX_TP_CORPUS        corpus.jsonl to read passages from
                          (default /corpora/miracl-slices/miracl_en_100k_slice/corpus.jsonl)
  SEMDEX_TP_N             passages to embed, timed (default 2000)
  SEMDEX_TP_BATCH         embed batch size (default 256, all backends)
  SEMDEX_TP_WARMUP        passages embedded + discarded before timing (default 64)
  SEMDEX_TP_THREADS       fastembed onnx thread count (default: all CPUs - always RECORDED)
  SEMDEX_TP_MODELS        comma list of embedding labels (default the 4 CPU embedders)
  SEMDEX_BENCH_OLLAMA_URL endpoint for the ollama (qwen3) models
  SEMDEX_TP_OUT           results json (default <cache>/scores/embed_throughput.json)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling driver helpers

from preembed_vectors import _EMBED_MODELS, _cache_root

from semdex.composition import build_embedding

_CPU_EMBEDDERS = [
    "model2vec:potion-base-8M",
    "model2vec:potion-retrieval-32M",
    "fastembed:bge-small",
    "fastembed:bge-base",
]
_DEFAULT_CORPUS = "/corpora/miracl-slices/miracl_en_100k_slice/corpus.jsonl"


def _host() -> dict:
    info: dict = {"hostname": platform.node(), "cores": os.cpu_count()}
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
        info["avx2"] = "avx2" in cpuinfo
    except OSError:
        pass
    return info


def _load_passages(corpus: Path, n: int) -> list[str]:
    texts: list[str] = []
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            texts.append((row.get("title", "") + "\n" + row.get("text", "")).strip())
            if len(texts) >= n:
                break
    return texts


def _measure(label: str, docs: list[str], *, batch: int, warmup: int, threads: int | None, endpoint: str | None):
    backend, model_id = _EMBED_MODELS[label]
    kwargs = {"model": model_id, "endpoint": endpoint}
    if label.startswith("fastembed:") and threads is not None:
        kwargs["threads"] = threads
    # allow_fallback=False: a placeholder silently standing in for an unloadable fastembed model
    # embeds ~1000x faster, so the run would report a fast, plausible, meaningless docs/s.
    embedding = build_embedding(backend, allow_fallback=False, **kwargs)
    if warmup:
        embedding.embed_passages(docs[:warmup])  # model load + warmup, excluded from timing
    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(docs), batch):
        embedding.embed_passages(docs[i : i + batch])
        done += len(docs[i : i + batch])
    elapsed = time.perf_counter() - t0
    mean_chars = round(sum(len(d) for d in docs) / len(docs)) if docs else 0
    return {
        "embedding": label,
        "docs": done,
        "batch": batch,
        "threads": threads if (label.startswith("fastembed:") and threads is not None) else "default",
        "elapsed_s": round(elapsed, 2),
        "docs_per_s": round(done / elapsed, 1) if elapsed else 0.0,
        "mean_chars": mean_chars,
    }


def main() -> None:
    corpus = Path(os.environ.get("SEMDEX_TP_CORPUS", _DEFAULT_CORPUS))
    n = int(os.environ.get("SEMDEX_TP_N", "2000"))
    batch = int(os.environ.get("SEMDEX_TP_BATCH", "256"))
    warmup = int(os.environ.get("SEMDEX_TP_WARMUP", "64"))
    threads_env = os.environ.get("SEMDEX_TP_THREADS")
    threads = int(threads_env) if threads_env else None
    models = os.environ.get("SEMDEX_TP_MODELS", ",".join(_CPU_EMBEDDERS)).split(",")
    endpoint = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
    out = Path(os.environ.get("SEMDEX_TP_OUT", str(_cache_root() / "scores" / "embed_throughput.json")))
    out.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_passages(corpus, n)
    report = {"host": _host(), "corpus": str(corpus), "n_docs": len(docs), "warmup": warmup, "rows": []}
    print(f"host: {report['host']}  corpus: {corpus.name}  n={len(docs)} batch={batch} warmup={warmup}", flush=True)
    for label in models:
        row = _measure(label, docs, batch=batch, warmup=warmup, threads=threads, endpoint=endpoint)
        report["rows"].append(row)
        print(
            f"[throughput] {label}: {row['docs_per_s']} docs/s "
            f"(threads={row['threads']}, batch={batch}, {row['elapsed_s']}s, ~{row['mean_chars']} chars/doc)",
            flush=True,
        )
        out.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("\n### Embedding throughput (controlled)\n")
    print(
        f"Host: {report['host'].get('cpu', '?')} ({report['host'].get('cores')} cores, "
        f"avx2={report['host'].get('avx2')}); {len(docs)} passages, batch {batch}, warm.\n"
    )
    print("| embedding | docs/s | threads | mean chars/doc |")
    print("|-----------|--------|---------|----------------|")
    for row in sorted(report["rows"], key=lambda r: -r["docs_per_s"]):
        print(f"| {row['embedding']} | {row['docs_per_s']} | {row['threads']} | {row['mean_chars']} |")


if __name__ == "__main__":
    main()
