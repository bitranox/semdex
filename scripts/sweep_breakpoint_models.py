#!/usr/bin/env python
"""Drive preembed_vectors.py across the SEMANTIC-chunker BREAKPOINT-MODEL axis.

Separate from sweep_chunk_profiles.py: that one varies the chunk PROFILE (recursive/semantic x
tokens x overlap) at a fixed default breakpoint model; THIS one holds the profile to the two
semantic settings and varies the breakpoint MODEL - the embedding model the SemanticChunker uses
to decide WHERE to cut. The retrieval embedder is held FIXED (default qwen3-8b) so any nDCG delta
across cells isolates the breakpoint model's effect, not the retriever's.

Question it answers: which breakpoint model gives the best semantic cuts (hence retrieval) on
GERMAN vs English long docs. chonkie's built-in default is potion-base-32M (English-distilled);
the alternatives here add multilingual coverage. The default-breakpoint baseline already lives in
the MLDR matrix as the plain ``semantic-*`` cells (no ``-bp`` suffix), so it is NOT re-run here.

Each breakpoint model routes to the right endpoint by its registry backend:
  MODEL2VEC -> local (no endpoint; must be in the run host's HF cache)
  OLLAMA    -> BP_OLLAMA_URL   (bge-m3, qwen3-0.6b)
  OPENAI    -> BP_SHIM_URL     (e5-large, jina-v3 on the sentence-transformers shim)

Endpoint PRE-FLIGHT: a model whose endpoint is unreachable is SKIPPED with a logged notice (never
silently), so a run today covers the ready models and a re-run once the shim is up fills the rest -
preembed skips any already-complete cell, so re-running is safe and resumes where it stopped.

One preembed invocation per (breakpoint_model x profile) covers ALL corpora/slices in that call.
Cell id: <slice>__semantic-t<tok>-o0-gpt2-bp<model>__<retrieval-embedder>.

Env:
  BP_MODELS       comma list of breakpoint registry labels
                  (default potion-multilingual-128M,bge-m3,qwen3-0.6b,e5-large,jina-v3)
  BP_PROFILES     comma list "<chunker>:<tokens>:<overlap>" (default semantic:256:0,semantic:512:0)
  BP_CORPORA      comma list of slices (default mldr_de_3k_slice,mldr_en_8k_slice)
  BP_RETRIEVAL    SEMDEX_PREEMBED_EMBEDDINGS value = retrieval embedder(s), comma list; each
                  chunk set is embedded by ALL of them (full matrix). Multilingual retrievers so
                  the de/en ranking is meaningful (default: qwen3-8b, qwen3-4b, bge-m3, e5-large).
  BP_OLLAMA_URL   ollama base URL, also the retrieval endpoint
                  (default http://px-semdex-test-embeddings:11434)
  BP_SHIM_URL     sentence-transformers shim /v1 URL
                  (default http://px-semdex-test-embeddings:7997/v1)
  BP_PHASE        chunk | embed | both (default both). chunk = produce chunk sets only (CPU +
                  light endpoint GPU); embed = embed already-chunked cells only (the heavy,
                  CPU-light GPU batch - runs full tilt regardless of dev-box CPU contention);
                  both = chunk+embed inline per cell. Decouples GPU embedding from CPU chunking.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_breakpoint_embeddings import BREAKPOINT_MODELS

from semdex.domain.enums import EmbeddingBackend

PREEMBED = str(Path(__file__).resolve().parent / "preembed_vectors.py")

_DEFAULT_MODELS = "potion-multilingual-128M,bge-m3,qwen3-0.6b,e5-large,jina-v3"
_DEFAULT_PROFILES = "semantic:256:0,semantic:512:0"
_DEFAULT_CORPORA = "mldr_de_3k_slice,mldr_en_8k_slice"


def _reachable(url: str) -> bool:
    """True if a GET on ``url`` returns any HTTP response within a short timeout."""
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:  # noqa: S310 - fixed internal host
            return 200 <= resp.status < 500
    except Exception:
        return False


def _endpoint_for(label: str, ollama_url: str, shim_url: str) -> tuple[str | None, bool]:
    """Return (endpoint, reachable) for a breakpoint model. Local model2vec -> (None, True)."""
    backend, _model_id, needs_endpoint = BREAKPOINT_MODELS[label]
    if not needs_endpoint:
        return None, True  # local static model2vec; no endpoint to probe
    if backend is EmbeddingBackend.OLLAMA:
        return ollama_url, _reachable(f"{ollama_url}/api/tags")
    if backend is EmbeddingBackend.OPENAI:
        # shim exposes /health next to /v1; probe that (strip a trailing /v1).
        health = shim_url[: -len("/v1")] + "/health" if shim_url.endswith("/v1") else shim_url
        return shim_url, _reachable(health)
    raise SystemExit(f"breakpoint model {label!r} has unroutable backend {backend}")


def _profiles(raw: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for spec in raw.split(","):
        chunker, tokens, overlap = spec.split(":")
        out.append((chunker, int(tokens), int(overlap)))
    return out


def main() -> int:
    models = [m.strip() for m in os.environ.get("BP_MODELS", _DEFAULT_MODELS).split(",") if m.strip()]
    profiles = _profiles(os.environ.get("BP_PROFILES", _DEFAULT_PROFILES))
    corpora = os.environ.get("BP_CORPORA", _DEFAULT_CORPORA)
    retrieval = os.environ.get(
        "BP_RETRIEVAL",
        "ollama:qwen3-embedding-8b,ollama:qwen3-embedding-4b,ollama:bge-m3,openai:e5-large",
    )
    ollama_url = os.environ.get("BP_OLLAMA_URL", "http://px-semdex-test-embeddings:11434")
    shim_url = os.environ.get("BP_SHIM_URL", "http://px-semdex-test-embeddings:7997/v1")

    # BP_PHASE decouples the CPU/light-GPU chunking from the heavy GPU retrieval embedding:
    #   chunk -> only produce chunk sets (MODE=chunk); embed -> only embed ready chunks
    #   (MODE=embed_only, the concentrated GPU batch); both (default) -> chunk+embed inline per cell.
    phase = os.environ.get("BP_PHASE", "both").strip().lower()
    phase_mode = {"chunk": "chunk", "embed": "embed_only", "both": "embed"}
    if phase not in phase_mode:
        raise SystemExit(f"BP_PHASE must be one of chunk|embed|both, got {phase!r}")
    mode = phase_mode[phase]

    for label in models:
        if label not in BREAKPOINT_MODELS:
            raise SystemExit(f"unknown breakpoint model {label!r}; known: {sorted(BREAKPOINT_MODELS)}")

    failures: list[str] = []
    skipped: list[str] = []
    done: list[str] = []
    for label in models:
        endpoint, ok = _endpoint_for(label, ollama_url, shim_url)
        # The breakpoint endpoint is only touched while CHUNKING; in the embed-only phase the
        # chunks already exist and the breakpoint model is never built, so its endpoint may be down.
        if phase != "embed" and not ok:
            skipped.append(f"{label} (endpoint {endpoint} unreachable)")
            print(
                f"\n##### SKIP breakpoint {label}: endpoint {endpoint} unreachable - re-run when it is up", flush=True
            )
            continue
        for chunker, tokens, overlap in profiles:
            tag = f"{label} | {chunker}-t{tokens}-o{overlap}"
            env = dict(os.environ)
            env.update(
                {
                    "SEMDEX_PREEMBED_CORPORA": corpora,
                    "SEMDEX_PREEMBED_EMBEDDINGS": retrieval,
                    "SEMDEX_PREEMBED_CHUNKER": chunker,
                    "SEMDEX_PREEMBED_MAX_TOKENS": str(tokens),
                    "SEMDEX_PREEMBED_OVERLAP": str(overlap),
                    "SEMDEX_PREEMBED_SEMANTIC_MODEL": label,
                    "SEMDEX_BENCH_OLLAMA_URL": ollama_url,  # ollama retrieval embedders (qwen3, bge-m3)
                    "SEMDEX_BENCH_OPENAI_URL": shim_url,  # openai-compat retrieval embedder (e5-large on the shim)
                    "MODE": mode,  # chunk | embed_only | embed (per BP_PHASE)
                }
            )
            if endpoint:
                env["SEMDEX_PREEMBED_SEMANTIC_ENDPOINT"] = endpoint
            else:
                env.pop("SEMDEX_PREEMBED_SEMANTIC_ENDPOINT", None)
            print(f"\n===== BP {tag} | corpora={corpora} | retrieval={retrieval} | ep={endpoint} =====", flush=True)
            t0 = time.perf_counter()
            rc = subprocess.call([sys.executable, PREEMBED], env=env)
            print(f"===== BP {tag} rc={rc} in {time.perf_counter() - t0:.0f}s =====", flush=True)
            (done if rc == 0 else failures).append(f"{tag} (rc={rc})")

    return _report_summary(done, skipped, failures)


def _report_summary(done: list[str], skipped: list[str], failures: list[str]) -> int:
    """Print the run summary and return the process exit code (kept out of main() for clarity)."""
    print("\n----- BREAKPOINT SWEEP SUMMARY -----", flush=True)
    print(f"  done:    {len(done)}", flush=True)
    for d in done:
        print(f"    ok  {d}", flush=True)
    if skipped:
        print(f"  skipped: {len(skipped)} (endpoint down - re-run to fill)", flush=True)
        for s in skipped:
            print(f"    --  {s}", flush=True)
    if failures:
        print(f"  FAILURES: {len(failures)}", flush=True)
        for f in failures:
            print(f"    XX  {f}", flush=True)
        return 1
    if skipped:
        print("BREAKPOINT-SWEEP-PARTIAL: ran the reachable models; re-run when skipped endpoints are up", flush=True)
        return 0
    print("BREAKPOINT-SWEEP-OK: all breakpoint models complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
