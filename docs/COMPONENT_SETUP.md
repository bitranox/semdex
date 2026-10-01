# Component setup guide

semdex's core is deliberately light: the default stack (embedded `text` extractor, `fastembed`
embeddings, `json` vector store, `recursive` chunker) needs no external service. Every heavier
option runs behind the same port as a **pip extra** and, for the container/server components, a
**separate service** you point semdex at by config. This guide sets up each one.

For *which* option to choose and *how each performs*, see [the benchmarks](benchmarks/README.md) (the
selection overview) and the per-dimension benchmark files -
[extraction](benchmarks/02-extraction.md), [chunking](benchmarks/03-chunking.md),
[embedding](benchmarks/04-embedding.md), [vector stores](benchmarks/05-vector-store.md); this document
is only the install/run mechanics. Switching any component is a config change plus its
extra - **reindex after switching**, because collections are keyed by (embedding model, dimension)
and stores do not migrate data.

Config lives in the `[extractor]`, `[embedding]`, `[vector_store]`, and `[chunker]` sections (see
`src/semdex/adapters/config/defaultconfig.d/`). Any key is overridable by env, e.g.
`SEMDEX___EMBEDDING__PROVIDER=ollama`.

> Examples use `localhost` / `<host>` placeholders. Put real hostnames in your own `.env` or config,
> never in the repo.

## Document extractors

`text` is built in and needs nothing. The rest run as containers and take an `[extractor].endpoint`.

| backend    | install                          | run                                                                               | `[extractor]`                                              |
|------------|----------------------------------|-----------------------------------------------------------------------------------|------------------------------------------------------------|
| text       | (built in)                       | -                                                                                 | `backend = "text"`                                         |
| markitdown | `pip install semdex[markitdown]` | `docker run -p 3001:3001 mcp/markitdown:latest --http --host 0.0.0.0 --port 3001` | `backend="markitdown"`, `endpoint="http://localhost:3001"` |
| xberg      | `pip install semdex[xberg]`      | `docker run -p 8000:8000 goldziher/kreuzberg:core`                                | `backend="xberg"`, `endpoint="http://localhost:8000"`      |
| docling    | `pip install semdex[docling]`    | `docker run -p 5001:5001 quay.io/docling-project/docling-serve:latest`            | `backend="docling"`, `endpoint="http://localhost:5001"`    |
| mineru     | `pip install semdex[mineru]`     | GPU image (see its docs); needs `--gpus all`                                      | `backend="mineru"`, `endpoint="http://localhost:8000"`     |

- **xberg** is the CPU OCR workhorse (tesseract, 18+ formats). **docling** is layout-aware
  PDF/Office. **mineru** is high-fidelity PDF/OCR but **requires an NVIDIA GPU**.

> **Naming:** the `xberg` backend runs the **kreuzberg v4** container - upstream, xberg is the
> polyglot framework brand, but the Docker image is still `goldziher/kreuzberg` (the `:core` tag is
> the smaller ~1-1.3 GB variant; both include Tesseract). semdex's own extra is httpx-only, so it
> installs no `kreuzberg`/`xberg` package.

**Enabling OCR (xberg):** the v4 image bundles Tesseract, but xberg returns only *metadata* for a
plain scanned image unless you force OCR. Set `[extractor].force_ocr = true` (and optionally
`ocr_language = "eng"`) to run OCR on scans/images. On born-digital documents leave it off - direct
extraction is faster and higher-fidelity than re-OCR (measured: `force_ocr` holds 100% on the fixture
documents and recovers the scanned image at 100%).

### Known issue: markitdown cannot process images (ExifTool CVE)

The pinned `mcp/markitdown:latest` image ships **ExifTool 12.16**, and markitdown's `ImageConverter`
runs ExifTool first for metadata. markitdown *refuses* ExifTool < 12.24 because 12.16 is vulnerable to
**CVE-2021-22204** (the ExifTool arbitrary-code-execution bug), so the image conversion aborts with:

```
ImageConverter threw RuntimeError: ExifTool version 12.16 is vulnerable to CVE-2021-22204.
Please upgrade to version 12.24 or later.
```

Consequences and options:
- This affects **image inputs only** (`.png`/`.jpg` scans). Office, HTML, and text-layer PDF are
  unaffected and extract at full fidelity.
- markitdown does image "OCR" only via an **LLM** (`llm_client`/`llm_model`), not a bundled OCR
  engine - so even with a fixed ExifTool it is not a scanned-text OCR tool.
- **For real scanned-text OCR use `xberg` (tesseract), `docling`, or `mineru` (GPU).**
- To unblock the ExifTool crash, build an image with a current ExifTool (pure-Perl, no compile):

  ```dockerfile
  FROM mcp/markitdown:latest
  USER root
  COPY exiftool.tar.gz /tmp/                       # from github.com/exiftool/exiftool tag >= 12.24
  RUN cd /tmp && tar xzf exiftool.tar.gz && d="$(ls -d exiftool-*/)" \
   && install -m 0755 "${d}exiftool" /usr/bin/exiftool \
   && cp -r "${d}lib/Image/"* /usr/share/perl5/Image/ && exiftool -ver
  ```

  Verified: this stops the crash, but markitdown then returns only image *metadata*
  (`ImageSize: 900x300`), not OCR text - it still has no OCR engine and markitdown-mcp exposes no LLM
  hook. So this is not worth it for OCR; use xberg/docling/mineru instead.

## Embedding providers

`fastembed` is the default (ONNX, no torch) and downloads its model on first use - no external
service. `[embedding].model` picks the model per provider (`None` = that provider's default); the
model choice is its own quality dimension (see docs/benchmarks/).

| provider              | install                         | external service                | `[embedding]`                                                                     |
|-----------------------|---------------------------------|---------------------------------|-----------------------------------------------------------------------------------|
| fastembed             | (core)                          | none (downloads model)          | `provider="fastembed"`, optional `model="BAAI/bge-base-en-v1.5"`                  |
| model2vec             | `pip install semdex[model2vec]` | none                            | `provider="model2vec"`                                                            |
| sentence_transformers | `pip install semdex[embed]`     | none (pulls torch; a GPU helps) | `provider="sentence_transformers"`, `model="all-MiniLM-L6-v2"`                    |
| ollama                | `pip install semdex[ollama]`    | an ollama server (below)        | `provider="ollama"`, `endpoint="http://<host>:11434"`, `model="nomic-embed-text"` |
| openai                | `pip install semdex[openai]`    | any OpenAI-compatible server    | `provider="openai"`, `endpoint="http://<host>:PORT/v1"`, `model="<model-id>"`     |

`ollama` and `openai` both keep the model OFF the semdex box (the server does the compute).
`ollama` speaks ollama's native `/api/embed`; `openai` speaks the standard `/v1/embeddings`
that hosted OpenAI, ollama's own `/v1`, llama.cpp's `llama-server`, and `infinity-emb` all
expose. Use `ollama` for models in the ollama registry; use `openai` for any other
OpenAI-compatible server (below).

### Setting up an ollama embedding server

```bash
# On the ollama host:
curl -fsSL https://ollama.com/install.sh | sh          # installs the binary + a systemd service
# Expose it on the network (default binds 127.0.0.1 only):
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment=OLLAMA_HOST=0.0.0.0:11434\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload && sudo systemctl enable --now ollama
# Pull embedding models (dims differ - part of the model dimension):
ollama pull nomic-embed-text      # 768-dim, general default
ollama pull all-minilm            # 384-dim, small/fast
ollama pull mxbai-embed-large     # 1024-dim, higher quality
ollama pull qwen3-embedding:4b    # GPU-tier, top MTEB-multilingual (large dim; needs VRAM)
ollama pull qwen3-embedding:8b    # GPU-tier, highest quality (needs more VRAM)
ollama pull bge-m3                # 1024-dim, strong multilingual (German); the standard multilingual pick
ollama pull qwen3-embedding:0.6b  # small multilingual, cheap enough to embed at chunk scale
```

Point semdex at it: `[embedding] provider="ollama"`, `endpoint="http://<host>:11434"`,
`model="nomic-embed-text"`. Query and index text must use the **same** provider+model.

For **non-English or mixed-language** corpora prefer a multilingual model. Measured on long
documents in English and German (see [the embedding benchmark](benchmarks/04-embedding.md)),
`qwen3-embedding:8b` leads both languages, `qwen3-embedding:4b` is close behind at a smaller
dimension, and `bge-m3` at 1024 is the cheapest of the three to store and search. Pick on the
language you actually have: `e5-large` all but ties the leader on English and comes last on
German. An English-only model on German content retrieves poorly.

**Set `[embedding] num_batch` if any text you embed can exceed 2048 tokens.** ollama's physical
batch defaults to 2048 while it serves these models with a 4096-token context, and it cannot
process an input larger than that batch. Its own recovery is to re-run the input truncated to
2048 and answer `HTTP 200`, so the text is embedded from its first 2048 tokens with no warning
and no error anywhere - a silent quality loss. In a request carrying many inputs that recovery
can fail instead and the whole call returns `400`, which is a client error and is therefore
never retried, so a long unattended index pass dies outright. Set `num_batch` to at least the
served context (4096 for `bge-m3` / `qwen3-embedding`) and confirm it took effect: the ollama
journal logs `llama_context: n_batch = 4096` when the model loads. This affects only the
`ollama` provider.

### Serving a model ollama does not have (infinity-emb)

ollama only serves models in its registry. To use any other HuggingFace embedding model
(for example `intfloat/multilingual-e5-large` or `jinaai/jina-embeddings-v3`), run
[`infinity-emb`](https://github.com/michaelfeil/infinity), a `pip`-installable
OpenAI-compatible embedding server (no Docker), and point semdex at it with `provider="openai"`.
This keeps torch and the model on the SERVER, not in the semdex install.

```bash
# On a GPU host (a CUDA-capable Python 3.11/3.12 - torch has no 3.14 wheels yet):
uv venv --python 3.12 /opt/infinity-venv
VIRTUAL_ENV=/opt/infinity-venv uv pip install "infinity-emb[all]"

# Serve one model on /v1/embeddings (OpenAI-compatible). jina-v3 needs trust-remote-code:
/opt/infinity-venv/bin/infinity_emb v2 \
  --model-id intfloat/multilingual-e5-large --port 7997 --device cuda
# jinaai/jina-embeddings-v3 additionally: --trust-remote-code
```

Point semdex at it (the base URL ends in `/v1`; semdex appends `/embeddings`):

```toml
[embedding]
provider = "openai"
endpoint = "http://<gpu-host>:7997/v1"
model    = "intfloat/multilingual-e5-large"
# api_key is not needed for a local infinity-emb; a secured gateway would set it via
# SEMDEX___EMBEDDING__API_KEY (never inline).
```

The same server also backs the **semantic chunker's** breakpoint model on a non-English
corpus - see the `semantic_model` note in `25-chunker.toml`; production `[chunker].semantic_model`
takes a LOCAL model id (model2vec / sentence-transformers), while the benchmark harness can point
the breakpoint model at one of these endpoints (`scripts/_bench_breakpoint_embeddings.py`).

## Vector stores

`json` is the default (embedded, exact, small corpora). `sqlite_vec` and `lancedb` are also embedded
(a file/dir); `pgvector` and `mariadb` are servers you pass a `[vector_store].dsn`.

| backend    | install                       | external service                                                                              | `[vector_store]`                                                                                 |
|------------|-------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| json       | (core)                        | none                                                                                          | `backend = "json"`                                                                               |
| sqlite_vec | `pip install semdex[sqlite]`  | none (needs a loadable-extension-capable sqlite)                                              | `backend = "sqlite_vec"`                                                                         |
| lancedb    | `pip install semdex[lance]`   | none                                                                                          | `backend = "lancedb"`                                                                            |
| pgvector   | `pip install semdex[pg]`      | `docker run -e POSTGRES_PASSWORD=pw -p 5432:5432 pgvector/pgvector:pg16`                      | `backend="pgvector"`, `dsn="host=localhost port=5432 user=postgres password=pw dbname=postgres"` |
| mariadb    | `pip install semdex[mariadb]` | `docker run -e MARIADB_ROOT_PASSWORD=pw -e MARIADB_DATABASE=semdex -p 3306:3306 mariadb:11.8` | `backend="mariadb"`, `dsn="mysql://root:pw@localhost:3306/semdex"`                               |

> `sqlite_vec` needs a Python whose sqlite was built with loadable-extension support
> (`--enable-loadable-sqlite-extensions`). Some distro/standalone builds disable it; if
> `enable_load_extension` is missing, use a Python that has it or a different backend.

## Chunkers

All are pip extras; none needs a service. `recursive`/`markdown` (chonkie) is the default;
`semantic`/`late` are embedding-driven (they load a small model on first use).

| strategy             | install                              | `[chunker]`                                  |
|----------------------|--------------------------------------|----------------------------------------------|
| whitespace           | (core, fallback)                     | `strategy = "whitespace"`                    |
| recursive / markdown | `pip install semdex[chunk]`          | `strategy = "recursive"` (or `"markdown"`)   |
| fast                 | `pip install semdex[chunk]`          | `strategy = "fast"` (semantic-text-splitter) |
| semantic / late      | `pip install semdex[chunk-semantic]` | `strategy = "semantic"` (or `"late"`)        |

## MCP server (`serve`)

`semdex serve` exposes the search / reindex / write use cases to MCP clients over a
transport. It serves the datasets defined in the `[[dataset]]` config and offers
these tools: `list_datasets`, `search` (one dataset), `search_datasets` (fan-out
over several), `reindex` (source datasets only), and `remember` / `forget` (writable
knowledge datasets only). Install the extra first:

```bash
pip install semdex[mcp-server]
```

### Transports

| transport | how clients reach it                                   | `[mcp]`               |
|-----------|--------------------------------------------------------|-----------------------|
| stdio     | the client spawns `semdex serve`; talks over stdin/out | `transport = "stdio"` |
| http      | listens on `host:port` (Streamable HTTP at `/mcp`)     | `transport = "http"`  |

```bash
semdex serve                              # stdio (default), local subprocess
semdex serve --transport http --port 9001 # Streamable HTTP on 127.0.0.1:9001
```

`--transport` / `--host` / `--port` override the `[mcp]` config. stdio ignores
host/port/auth (the launching process is the trust boundary).

### Auth (http only)

Set `[mcp].auth`. Secrets are NEVER written in config - the config names the
environment variable, and the value is read at runtime.

| auth   | set up                                                                   | secret env                       |
|--------|--------------------------------------------------------------------------|----------------------------------|
| none   | unauthenticated - use for stdio, localhost, or behind a proxy            | -                                |
| bearer | `auth = "bearer"`; export the token(s), comma-separated                  | `SEMDEX_MCP_BEARER_TOKENS`       |
| oauth  | `auth = "oauth"` + `oauth_jwks_uri` / `oauth_issuer` (+ proxy endpoints) | `SEMDEX_MCP_OAUTH_CLIENT_SECRET` |

```bash
export SEMDEX_MCP_BEARER_TOKENS="s3cret-a,s3cret-b"   # from your .env, never the repo
semdex serve --transport http --port 9001
```

See `defaultconfig.d/17-mcp.toml` for every `[mcp]` key documented in full.

### Fan-out search across datasets

`search_datasets(query, k, datasets)` searches several datasets at once and returns
one fused ranking. Omit `datasets` (or pass `["all"]`) to search every dataset the
server serves; otherwise pass a subset of names from `list_datasets`. Results from
different datasets are fused by Reciprocal Rank Fusion, so the returned `rrf_score`
orders them - never the per-dataset `score` (raw similarity is not comparable across
different embedding models). Each row is tagged with its source `dataset`.

The intended shape: keep a shared corpus on a database backend (its ACL enforced by
per-department DSN grants at the database) and layer department-specific datasets on
top, indexed once each. Accounting's server serves `shared` + `accounting`; a query
fans out over both and comes back as one ranking. `[mcp].rrf_k` (default 60) tunes
the fusion.

### Writable knowledge datasets (remember / forget)

Datasets come in two kinds, kept separate:

- **Source datasets** (the default): mirror an external source through a connector
  (filesystem now). Read-only over the source - the client can `search` and
  `reindex`, never write content.
- **Knowledge datasets** (`writable = true` in `[[dataset]]`, no `sources`, not
  `read_only`): a database-backed store the client writes to directly. The client
  deposits content with `remember(dataset, text, title, entry_id)` and removes it
  with `forget(dataset, entry_id)`; content is keyed by `semdex://<dataset>/<id>`.
  `reindex` refuses these (there is no source to reconcile), and source datasets
  refuse `remember`/`forget`.

The two never mix: a knowledge dataset has no connector, so the filewatcher's files
(and, later, mail) can never land in it. Put each writable dataset on its own
database; who may write is enforced by the DB GRANT on that dataset's DSN (a write
grant vs SELECT-only), with the `writable` flag as the advisory signal. `remember`
and `forget` take the dataset name, so an agent selects which allowed knowledge base
to write into; `list_datasets` reports each dataset's `writable` flag.

```bash
# In [[dataset]]: an agent-owned knowledge base on its own database.
# name = "agent_memory"
# backend = "pgvector"
# dsn = "postgresql://agent@db.internal:5432/semdex_agent_memory"  # DSN GRANT allows writes
# collection = "memory"
# writable = true
```

### Example client config (Claude Desktop, stdio)

```json
{
  "mcpServers": {
    "semdex": { "command": "semdex", "args": ["serve"] }
  }
}
```

## Benchmark prerequisites

Everything the benchmark matrices need, in one place:

1. **A capable venv with all extras** (a plain `pip install semdex` has only the light default
   stack):

   ```bash
   python -m venv ~/venvs/semdex && ~/venvs/semdex/bin/pip install \
     -e ".[dev,bench,markitdown,xberg,docling,mineru,model2vec,ollama,chunk,chunk-semantic,embed,sqlite,lance,pg,mariadb]"
   ```

   `sqlite_vec` needs a Python whose sqlite allows loadable extensions (see above); the
   benchmarks skip cells whose optional dependency is absent rather than failing.
2. **Docker** - the server stores (pgvector/mariadb) and the extractor containers are started
   as throwaway containers by the test fixtures; images are pulled on first use.
3. **Model + corpus caches** - fastembed/model2vec/sentence-transformers download their models
   on first use (HF cache); retrieval corpora land in `~/.ir_datasets`. First runs need
   network; later runs are cache-hits.
4. **An ollama server** (optional) - only the ollama embedding cells need it. Export
   `SEMDEX_BENCH_OLLAMA_URL=http://<your-ollama-host>:11434` (put the real host in your
   untracked `.env`, never in the repo); the cells skip cleanly when unset.

Then start whichever containers/servers you need (above) and run `make ti` (see
[the benchmarks](benchmarks/README.md) for the env knobs and the reproducible regression slice; each
`bench_*.md` file ends with its own re-run command; the `.claude/skills/bench-rerun` recipe
covers the full update-and-re-rank loop).
