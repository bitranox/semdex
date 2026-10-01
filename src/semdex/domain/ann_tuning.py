"""ANN-recall preset resolution.

Map a portable :class:`AnnRecall` preset to a backend's native ANN search params, with optional
raw overrides winning field-by-field. ``BALANCED`` is empty (all params unset) so it keeps the
driver default and is non-breaking; the exact stores (``JSON``, ``SQLITE_VEC``) carry no ANN
params at any preset. The preset-to-value magic numbers live ONLY here (one place, calibrated by
the recall/latency bench), never hardcoded in the store adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import AnnRecall, StoreBackend

_ANN_STORES = frozenset({StoreBackend.LANCEDB, StoreBackend.PGVECTOR, StoreBackend.MARIADB})


@dataclass(frozen=True, slots=True)
class AnnParams:
    """Native ANN index/query params. ``None`` means "leave the driver default".

    Query-time (live, no reindex): ``nprobes`` / ``refine_factor`` (lancedb), ``ef_search``
    (pgvector / mariadb HNSW). Build-time (needs a reindex): ``m`` / ``ef_construction``
    (HNSW), ``num_partitions`` (lancedb IVF-PQ).
    """

    nprobes: int | None = None
    refine_factor: int | None = None
    ef_search: int | None = None
    m: int | None = None
    ef_construction: int | None = None
    num_partitions: int | None = None


# Preset -> native query-time params per ANN backend. BALANCED is each driver's own default for
# every backend EXCEPT lancedb, whose default the sweep measured as the worst point on its own
# frontier. These magic numbers are the ONLY place the
# preset values live, and they are now MEASURED rather than assumed: scripts/score_ann_frontier.py
# sweeps each store's knobs against the exact top-10 and the frontier is published in
# docs/benchmarks/05-vector-store.md. Reference cell: MLDR English, 148,008 chunks, 768 dimensions,
# 800 queries, where the exact scan scores nDCG@10 0.8447.
#
# Before that sweep these values were asserted rather than measured, and the sweep found lancedb's
# driver default keeps only 0.567 of the exact top-10, costing 0.0474 nDCG - the entire
# "approximation costs up to 5.6 percent" figure this repo used to publish. BALANCED no longer
# leaves lancedb at that default; see the note on it below. pgvector and mariadb are unchanged.
_PRESETS: dict[StoreBackend, dict[AnnRecall, AnnParams]] = {
    StoreBackend.LANCEDB: {
        # nprobes alone is nearly inert: recall is 0.561 at 5 probes and 0.567 at 80, flat from
        # 20 up. FAST therefore reproduces what the bare driver default used to do (recall 0.566,
        # 28.9 ms) and exists as the escape hatch for anyone who wants that behaviour back.
        AnnRecall.FAST: AnnParams(nprobes=10),
        # DELIBERATELY NOT THE DRIVER DEFAULT, unlike every other backend here. lancedb's own
        # default keeps just 0.567 of the exact top-10 and gives up 0.0474 nDCG; the refine
        # factor, which re-ranks candidates against the true vectors, is the knob that fixes it.
        # This setting measures 0.986 recall and nDCG 0.8412 - 0.4 percent under an exact scan -
        # for 32.0 ms against 29.4. Paying 9 percent latency for 42 points of recall is what
        # "balanced" ought to mean, and shipping a default nobody would choose after seeing the
        # frontier is worse than the upgrade surprise. Pick FAST to opt out.
        AnnRecall.BALANCED: AnnParams(nprobes=10, refine_factor=5),
        # recall 0.998 and nDCG 0.8447, matching the exact scan to four decimals, at 38.6 ms.
        AnnRecall.ACCURATE: AnnParams(nprobes=40, refine_factor=10),
    },
    # Measured and left unchanged: these three sit at 0.565 / 0.806 / 0.978 recall for 1.8 / 2.9 /
    # 11.1 ms, which is a well-separated ladder and the cleanest frontier of the three stores.
    StoreBackend.PGVECTOR: {
        AnnRecall.FAST: AnnParams(ef_search=20),
        AnnRecall.BALANCED: AnnParams(),
        AnnRecall.ACCURATE: AnnParams(ef_search=200),
    },
    # Kept for interface parity, but on the reference cell mariadb's recall does NOT respond to
    # this knob: 0.677 to 0.684 across ef_search 10 to 200, while latency rises from 63 to 76 ms.
    # That is not a dead knob in semdex - the session variable was verified set on the connection
    # the queries use - so the three levels are honest to request and simply do not separate.
    StoreBackend.MARIADB: {
        AnnRecall.FAST: AnnParams(ef_search=20),
        AnnRecall.BALANCED: AnnParams(),
        AnnRecall.ACCURATE: AnnParams(ef_search=200),
    },
}


def resolve_ann_params(backend: StoreBackend, recall: AnnRecall, overrides: AnnParams | None = None) -> AnnParams:
    """Resolve the effective ANN params: the backend's preset first, raw overrides winning.

    Exact stores carry no ANN params. An override field left ``None`` never clobbers the
    preset value for that field.
    """
    if backend not in _ANN_STORES:
        return AnnParams()
    base = _PRESETS[backend][recall]
    if overrides is None:
        return base
    return AnnParams(
        nprobes=base.nprobes if overrides.nprobes is None else overrides.nprobes,
        refine_factor=base.refine_factor if overrides.refine_factor is None else overrides.refine_factor,
        ef_search=base.ef_search if overrides.ef_search is None else overrides.ef_search,
        m=base.m if overrides.m is None else overrides.m,
        ef_construction=base.ef_construction if overrides.ef_construction is None else overrides.ef_construction,
        num_partitions=base.num_partitions if overrides.num_partitions is None else overrides.num_partitions,
    )
