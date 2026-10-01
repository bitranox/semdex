"""Application use cases orchestrating domain logic through the ports.

Each module is one use case (one actor). Use cases depend only on the narrow
port slices they need; the composition root supplies concrete adapters.

Contents:
    * :mod:`.indexing` - index_sources (extract -> chunk -> embed -> upsert)
    * :mod:`.reconciling` - reconcile (sync a collection to a connector listing)
    * :mod:`.searching` - search (single-target) + search_across_datasets (fan-out + RRF)
    * :mod:`.writing` - remember / forget (client-written knowledge datasets)
"""

from __future__ import annotations

from .indexing import IndexReport, index_sources
from .reconciling import ReconcileReport, reconcile
from .searching import DatasetSearchTarget, reciprocal_rank_fusion, search, search_across_datasets
from .writing import forget, remember

__all__ = [
    "DatasetSearchTarget",
    "IndexReport",
    "ReconcileReport",
    "forget",
    "index_sources",
    "reciprocal_rank_fusion",
    "reconcile",
    "remember",
    "search",
    "search_across_datasets",
]
