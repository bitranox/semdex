# semdex in short

semdex is a semantic search server and CLI built for **heterogeneous, multi-discipline
corpora**: one installation over material that shares no common format, language, subject
area, or sensible set of processing parameters. Source code, scanned contracts, research
papers, meeting notes, spreadsheets, and regulatory texts can sit in the same deployment and
stay reachable through a single query.

That is the design premise, not a side effect. Most search stacks assume a homogeneous
corpus and apply one global profile to everything, which costs the most on whichever
material fits that profile least. semdex assumes the opposite from the start, which is why
its pipeline is per-document rather than global, and why its index is plural rather than
single. Extraction, chunking, embedding, and the vector store each sit behind a port, and
not merely as a swappable choice: **several implementations of each run in parallel in one
deployment**. A single installation can drive two extractors, three chunk strategies, four
embedding models, and several vector databases at the same time, each serving the documents
it suits. There is no single global stack to compromise on, so no document has to be
processed by settings chosen for a different kind of document.

It indexes documents into local vector stores and watches the sources cross-platform.
Indexing is strictly one-way: semdex reads the sources and never writes back to them, so
pointing it at a live directory cannot alter, move, or reformat anything in it.

A **document router** sits in front of the pipeline and decides per document how it is
processed: which extractor reads it, how it is chunked, which embedding model represents
it, and which database it lands in. A scanned PDF, a Markdown note, a spreadsheet, and a
long prose document have very different optimal settings, and matching each to the
combination that suits it is what keeps retrieval quality high across a mixed corpus.
Routing is config-driven and reads three signals: where a document sits (its subtree), what
it is (its filetype), and what is actually inside it. Content is a first-class input, not a
fallback. Two files with the same extension can take entirely different paths because their
contents differ, so a text-layer PDF and a scanned one, or a prose document and a dense
table, are recognised and handled as the different things they are rather than being treated
alike because their names agree.

Every chunk carries **structured metadata** alongside its vector. Fields are not fixed by
semdex: a deployment defines whatever facets its corpus has, and any of them can be stored
at ingest, returned with each hit, and used as a **query-time filter**. A filter narrows the
candidate set by date range, author, status, version, language, source system, or any other
indexed field, and combines with semantic ranking rather than replacing it.

Because a heterogeneous corpus is deliberately spread across several stores, **search spans
them**: one query fans out to multiple databases at once and returns a single fused ranking.
A caller asks a question without knowing which discipline or which store holds the answer,
and a result set can draw on several at once.

Splitting the index is also a **performance lever, not only an organisational one**. Several
stores are several shards: each holds a fraction of the vectors, so each search runs over a
smaller index, and the fan-out queries them concurrently rather than in sequence. A corpus
can be partitioned along whatever axis suits it, by discipline, by age, by source system, or
purely by size.

Concurrency sets the terms, though: a fanned-out query costs the **slowest** shard, not the
sum of them. The gain comes from each shard being individually smaller, so it depends on the
shards being reasonably balanced, and one oversized or slow store sets the floor for every
query that includes it. So a query is scoped: it goes to the stores that can hold the answer
rather than to all of them.

Choosing those stores is not left to the caller. A **query router** mirrors the document
router on the read side: it interprets the incoming question, infers which collections are
relevant from the same facets the corpus is partitioned by, and dispatches only there. Asking
about an email searches the mail collections; asking about an API searches the code and
documentation collections. A caller can still name stores explicitly to override the
inference, and where the router is unsure it widens rather than narrows, because a query
sent to one store too many costs latency while a query sent to one store too few silently
returns an incomplete answer.

Scoping is what makes an uneven deployment work: because each store selects its own backend,
a frequently-hit partition and a large cold archive can run on entirely different vector
databases, and queries about live material never pay for the archive.

Three interfaces expose the same core, and all three can **ingest as well as search**: a
**CLI** for scripted indexing of files and folders, an **MCP server** for AI agents, and an
**HTTP REST API** for direct integration from any language. Ingestion accepts both a path to
index and text submitted directly, so a caller that already holds the content does not have
to write it to disk first.
