#!/usr/bin/env python
"""Drive preembed_vectors.py across the 8 #42 chunk profiles for a set of corpora.

One invocation per profile because the profile is env-driven. Chunk the corpora ONCE first
(run with a cheap embedder), then a CPU chain and a GPU chain can embed concurrently without
racing each other on the shared chunks.parquet.tmp.

One preembed invocation per profile (the profile is env-driven), sequential, resumable:
preembed itself skips any chunk dir / vector cell already complete, so a re-run continues
where it stopped. Env:

  SWEEP_CORPORA      comma list (default nfcorpus,scifact,cqadupstack)
  SWEEP_EMBEDDINGS   comma list passed to SEMDEX_PREEMBED_EMBEDDINGS
  SWEEP_PROFILES     comma list of "<chunker>:<max_tokens>:<overlap>" (default the 8)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# The 8 profiles of #42: RECURSIVE {256,512} x {0,10,15} + SEMANTIC {256,512} x {0}.
# SEMANTIC ignores overlap (the chonkie adapter applies OverlapRefinery only to RECURSIVE),
# so an overlap axis there would just re-embed identical chunks under a different name.
_DEFAULT_PROFILES = [
    ("recursive", 256, 0),
    ("recursive", 256, 10),
    ("recursive", 256, 15),
    ("recursive", 512, 0),
    ("recursive", 512, 10),
    ("recursive", 512, 15),
    ("semantic", 256, 0),
    ("semantic", 512, 0),
]

PREEMBED = str(Path(__file__).resolve().parent / "preembed_vectors.py")


def _profiles() -> list[tuple[str, int, int]]:
    raw = os.environ.get("SWEEP_PROFILES")
    if not raw:
        return _DEFAULT_PROFILES
    out: list[tuple[str, int, int]] = []
    for spec in raw.split(","):
        chunker, tokens, overlap = spec.split(":")
        out.append((chunker, int(tokens), int(overlap)))
    return out


def main() -> int:
    corpora = os.environ.get("SWEEP_CORPORA", "nfcorpus,scifact,cqadupstack")
    embeddings = os.environ["SWEEP_EMBEDDINGS"]
    profiles = _profiles()
    failures: list[str] = []
    for chunker, tokens, overlap in profiles:
        tag = f"{chunker}-t{tokens}-o{overlap}"
        env = dict(os.environ)
        env.update(
            {
                "SEMDEX_PREEMBED_CORPORA": corpora,
                "SEMDEX_PREEMBED_EMBEDDINGS": embeddings,
                "SEMDEX_PREEMBED_CHUNKER": chunker,
                "SEMDEX_PREEMBED_MAX_TOKENS": str(tokens),
                "SEMDEX_PREEMBED_OVERLAP": str(overlap),
            }
        )
        print(f"\n===== PROFILE {tag} | corpora={corpora} | emb={embeddings} =====", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, PREEMBED], env=env)
        print(f"===== PROFILE {tag} rc={rc} in {time.perf_counter() - t0:.0f}s =====", flush=True)
        if rc != 0:
            # Do not abort the whole sweep: a later profile may still be embeddable, and every
            # cell is independent + resumable. Collect and report at the end instead.
            failures.append(f"{tag} (rc={rc})")
    if failures:
        print("\nSWEEP-FAILURES: " + "; ".join(failures), flush=True)
        return 1
    print("\nSWEEP-OK: all profiles complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
