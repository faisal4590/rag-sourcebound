# flp-rag

Question answering over the book *Front Line PHP* (PHP 8.2 edition). Answers come from the book only.
If the book does not contain the answer, the system returns the exact text `No information found`.

The full specification is `docs/front-line-php-rag-spec.md`. The architecture figure is `docs/rag-architecture-flowchart.png`.

## Start here

1. Copy `.env.example` to `.env`. Fill in one chat model key.
2. Run `uv sync --extra dev` (or `pip install -e ".[dev]"`).
3. Start Qdrant (:6333) and Phoenix (:6006). Pick one path:
   - **Docker:** `make up`. Uses `docker-compose.yml`.
   - **No Docker:** `uv sync --extra dev --extra local`, download the Qdrant binary once (below), then `make up-local`. Stop with `make down-local`.
4. `make status` checks that both services answer.
5. Do Milestone 0 from the spec, Section 11: one traced `hello` span visible at http://localhost:6006.

### Qdrant binary for the no-Docker path

```bash
mkdir -p .local/qdrant && cd .local/qdrant
curl -sSL -o q.tar.gz https://github.com/qdrant/qdrant/releases/download/v1.19.1/qdrant-aarch64-apple-darwin.tar.gz
tar xzf q.tar.gz && rm q.tar.gz && ./qdrant --version
```

Pick the `x86_64` asset on an Intel Mac. `.local/` is gitignored.

The release binary has no web UI. To get http://localhost:6333/dashboard, unzip the UI build so
that `.local/qdrant/static/index.html` exists, then restart with `make down-local && make up-local`:

```bash
curl -sSL -o /tmp/dist.zip https://github.com/qdrant/qdrant-web-ui/releases/latest/download/dist-qdrant.zip
unzip -qo /tmp/dist.zip -d /tmp/qdrant-ui && mv /tmp/qdrant-ui/dist .local/qdrant/static
``` Qdrant storage lives in `.local/qdrant/storage`, Phoenix data in `.local/phoenix`.

## Layout

| Path | Holds |
|---|---|
| `src/flp_rag/ingest/` | Stages 1-8, your own code |
| `src/flp_rag/graph/` | Stages 9-18 as a LangGraph, one node file per stage |
| `src/flp_rag/models.py` | The only file that names a model provider |
| `prompts/` | One file per prompt version |
| `eval/` | Golden dataset, smoke queries, harness, calibration |
| `data/raw/` | The PDF |
| `tests/` | One test file per "Done when" line |

Every Python file holds a one-line comment that names its stage and the spec section. No code exists yet.

## Milestones

M0 environment (1 evening) -> M1 parse and chunk (1 weekend) -> M2 embed and index (2 evenings) -> M3 answer and abstain (1 weekend) -> M4 hybrid, rerank, query understanding (1 weekend) -> M5 observability (1 weekend) -> M6 evaluation (1 weekend) -> M7 advanced (1-2 weekends).
