# LlamaParse / evidence-chain introspection (Python-only audit)

This report is based on the Python implementation under the repository, not on the markdown documentation set.

## 1. Executive summary

The repository already contains a strong evidence-centric pipeline, but it was split across a few different contracts:

- A direct parser integration path exists in the hybrid retrieval layer, with a preference for `llama_parse` when a key is available.
- The downstream graph stack already persists `SourceArtifact` evidence into Neo4j.
- The Copilot reference payload now includes `structured_claims` and `source_artifacts`, which is the key contract needed for evidence rendering in the UI.
- The graph-learning layer has been upgraded from a Node2Vec-only idea to a GraphSAGE/PyG-style embedder path while preserving the fallback Node2Vec branch.
- The large remaining risk is environmental and runtime-configuration dependent: the parser path is reachable, but the direct parser and embedding dependencies must be present in the active Python environment.

## 2. The parser plane

### Direct parser hook

The direct parser integration is centered in [app/pipeline/llamaindex_hybrid.py](app/pipeline/llamaindex_hybrid.py).

Key implementation points:

- `LlamaIndexHybrid.__init__()` reads environment/API settings and normalizes them for both LlamaIndex and LlamaParse.
- `_load_documents_from_llamaparse()` tries a direct `from llama_parse import LlamaParse` import and then calls `LlamaParse(...).load_data(file_path)`.
- `_load_documents_from_path()` has a strict preference order:
  1. direct `llama_parse`
  2. extension-specific loader candidates
  3. `SimpleDirectoryReader`
  4. structured JSON fallback

That means the ingestion path is intentionally defensive: it will still work in constrained environments, but it actively tries to go through the direct parser when the backend is available.

### Loader preference order and why it matters

The loader map in [app/pipeline/llamaindex_hybrid.py](app/pipeline/llamaindex_hybrid.py) supports common file types (`.txt`, `.md`, `.markdown`, `.html`, `.pdf`, `.csv`, `.json`) and the direct parser route is deliberately placed before the fallback readers.

This is the right design for a repo that needs broad corpus support because it makes the system more capable of handling additional source types without hardcoding every loader into the main execution path.

## 3. Retrieval and synthesis path

### Hybrid retrieval

The `LlamaIndexHybrid` class in [app/pipeline/llamaindex_hybrid.py](app/pipeline/llamaindex_hybrid.py) implements a retrieval flow around:

- `build_index()` for chunked text ingestion
- `build_index_from_text()` and `build_index_from_file()` for source-driven indexing
- `retrieve()` for evidence retrieval over the built vector index
- `citation_summarize()` for evidence-grounded synthesis

This is a classic LlamaIndex-style retrieval layer over the textual evidence produced by the rest of the pipeline.

### LLM-backed citation synthesis

The synthesis path in [app/pipeline/llamaindex_hybrid.py](app/pipeline/llamaindex_hybrid.py) has two important sides:

- a strong path that calls a Gemini endpoint through `_llm_synthesize_citation_json()`
- a deterministic fallback that constructs a heuristic JSON payload from evidence coverage and keyword signals

The important architectural point is that the LLM-backed path is now the preferred branch when `gemini_api_key` is configured, while the heuristic branch still preserves offline/degraded operation.

## 4. Graph-learning plane

### What changed in the graph embedder stack

The graph-learning implementation lives in [app/pipeline/advanced_models.py](app/pipeline/advanced_models.py).

The concrete additions are:

- `GraphSAGEGraphEmbedder`
- registration of `graphsage` in `initialize_advanced_models()`
- a fallback selection in [app/pipeline/advanced_pipeline.py](app/pipeline/advanced_pipeline.py) where `graphsage` is preferred before `node2vec`

This means the repository is no longer bound to the old single graph-learning implementation.

### GraphSAGE shape

The new `GraphSAGEGraphEmbedder` is designed as a PyG-style graph embedding bridge:

- it reads the current Neo4j graph
- converts it into a NetworkX graph
- builds a simple GraphSAGE-style conv stack over a node feature matrix
- emits node embeddings back to the pipeline

This is not a full production-grade GraphSAGE training stack yet, but it is a real implementation path that can be extended with true graph dataset construction and training loops.

### Why this matters

GraphRAG-style evidence benefits from richer node representations than a simple word-embed-only path. The new embedder makes the graph stage capable of representing relational structure directly rather than only through textual retrieval.

## 5. Source artifact evidence retention

The actual evidence retention contract is handled in [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py) and [app/pipeline/neo4j_store.py](app/pipeline/neo4j_store.py).

### Pipeline source artifact capture

In [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py), the vision-language stage constructs `source_artifacts` with fields such as:

- `artifact_id`
- `source_document`
- `kind`
- `page`
- `mime_type`
- `caption`
- `content`
- `metadata`

This is the exact evidence payload that flows later into graph persistence.

### Neo4j persistence contract

In [app/pipeline/neo4j_store.py](app/pipeline/neo4j_store.py), `persist_source_artifacts()`:

- validates incoming artifact dictionaries
- creates or updates a `Job` node
- creates or merges `SourceArtifact` nodes
- links these artifacts to the job using `HAS_SOURCE_ARTIFACT`
- stores the artifact content and provenance metadata in the graph

This is the key architectural step that makes image/page evidence queryable and persistent rather than just ephemeral output.

## 6. Copilot references rendering path

The final answer contract is assembled in [app/main.py](app/main.py).

### References builder

The `_build_references()` function now includes three evidence families:

1. neo4j records
2. `structured_claims`
3. `source_artifacts`

That is the most important contract fix in the repository: the final response path now has the same evidence objects that the graph and parser stages produce, instead of only exposing record-level graph facts.

### Why this closes the evidence loop

The ingestion chain now reads sources, extracts page/image evidence, persists it in Neo4j, and returns those artifacts through the references payload used in the Copilot UI. In practical terms, the evidence chain has become:

- source ingest
- parser extraction
- artifact capture
- graph persistence
- response payload surfacing
- UI reference rendering

## 7. Runtime dependencies and environment sensitivity

The codebase itself is structurally strong, but the environment still determines whether the full direct path is active.

A few dependencies are required for the newest behavior:

- `llama_parse`
- `llama-index-embeddings-openai`
- `torch_geometric`
- GraphSAGE-related PyG pieces

The repo is written defensively, so the fallback path keeps working even if these optional components are absent. However, the newer direct parser + GraphSAGE + LLM synthesis behavior is only fully exercised when those packages are installed into the active environment.

## 8. Verified behavior from the current codebase

The code path now verifies these behaviors:

- `source_artifacts` are accepted and persisted into Neo4j.
- `structured_claims` are surfaced into the Copilot references payload.
- `GraphSAGEGraphEmbedder` is available and instantiable in the advanced model stack.
- `citation_summarize()` can prefer a Gemini-backed synthesis route when a key is present.
- the fallback indexing path is now deterministic and test-friendly without requiring an external embedding service.

## 9. Remaining gap

The only meaningful remaining gap is end-to-end proof against a real 130+ source corpus under the direct parser backend, not just unit-level or regression-level proofs. The code path is ready for that proof, but the actual corpus sweep still needs to be run against the real environment and real artifacts.

## 10. Bottom line

The repository is no longer just a parser-plus-retriever shell. It now has a coherent evidence chain:

- parser ingestion with a direct `llama_parse` route
- graph-learning support via a GraphSAGE-style embedder
- LLM-backed citation synthesis
- persistent `SourceArtifact` evidence in Neo4j
- Copilot references UI surfacing for those artifacts and claims

That is a materially stronger evidence architecture than the earlier fallback-only design.
