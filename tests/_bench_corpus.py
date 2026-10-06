"""Map a retrieval corpus's document ids onto the sources the index pipeline runs on.

The E2E benchmark matrices index labelled corpora (BEIR and friends) that live in memory
as ``doc id -> text``. The pipeline identifies a document by ``SourceRef.uri`` and the
in-memory extractor resolves that URI back to its document, so this module owns both
directions: building the sources and extractor for a corpus, and mapping search hits
back to the corpus's own doc ids so the metrics can compare them with the qrels.

It is a plain module rather than part of ``test_e2e_matrix.py`` so that a fast unit test
can drive the same plumbing: the matrix module is integration-marked and only runs on
the scheduled workflow, which is how a broken mapping once went unseen until that run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from urllib.parse import quote, unquote

from semdex.adapters.discovery.location import from_uri
from semdex.adapters.memory.index import InMemoryExtractor
from semdex.domain.models import Hit, SourceRef

__all__ = ["corpus_extractor", "corpus_sources", "doc_id_from_uri", "doc_uri", "ranked_doc_ids"]

# Every corpus document becomes one path segment under this root. The whole id is
# percent-encoded (slashes included), so the id survives as one segment whatever it holds,
# and the inverse is a string operation that never goes through path parsing.
_CORPUS_ROOT = "file:///corpus/"


def doc_uri(doc_id: str) -> str:
    """Return the ``file:`` URI that stands for corpus document *doc_id*.

    Examples:
        >>> doc_uri("<dbpedia:AC/DC>")
        'file:///corpus/%3Cdbpedia%3AAC%2FDC%3E'
    """
    return _CORPUS_ROOT + quote(doc_id, safe="")


def doc_id_from_uri(uri: str) -> str:
    """Return the corpus doc id a :func:`doc_uri` URI stands for (its exact inverse).

    Raises:
        ValueError: *uri* was not produced by :func:`doc_uri`.

    Examples:
        >>> doc_id_from_uri(doc_uri("MED-4972"))
        'MED-4972'
    """
    if not uri.startswith(_CORPUS_ROOT):
        raise ValueError(f"not a corpus document URI: {uri!r}")
    return unquote(uri[len(_CORPUS_ROOT) :])


def corpus_sources(docs: Mapping[str, str]) -> list[SourceRef]:
    """Return one source per corpus document, in corpus order."""
    return [SourceRef(uri=doc_uri(doc_id), label="", content_hash="", mtime=0.0) for doc_id in docs]


def corpus_extractor(docs: Mapping[str, str]) -> InMemoryExtractor:
    """Return an extractor that serves each corpus document to its source.

    The extractor looks a source up by the path its URI resolves to, so the keys are
    built through that same resolution rather than a second spelling of it.

    Raises:
        ValueError: two doc ids resolve to the same path (``a//b`` and ``a/b`` do), which
            would otherwise make one document silently replace the other.
    """
    by_path = {from_uri(doc_uri(doc_id)): text for doc_id, text in docs.items()}
    if len(by_path) != len(docs):
        raise ValueError(f"{len(docs) - len(by_path)} corpus doc id(s) resolve to the same in-memory path")
    return InMemoryExtractor(by_path)


def ranked_doc_ids(hits: Iterable[Hit], limit: int) -> list[str]:
    """Dedup chunk hits to corpus doc ids, best rank first, at most *limit* of them."""
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        doc_id = doc_id_from_uri(hit.uri)
        if doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
        if len(ranked) >= limit:
            break
    return ranked
