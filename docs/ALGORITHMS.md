# Algorithms and Processing Methods

This document summarizes the major algorithmic families used in the repository, referencing the implementation modules that realize them.

## 1. OCR and Document Parsing

### Docling + Surya OCR path
Implemented in [app/pipeline/ocr_processor.py](app/pipeline/ocr_processor.py).

- `DoclingOCRProcessor` is the primary PDF extraction engine.
- It converts a PDF into a `doc.document.export_to_markdown()` style text representation.
- It also gathers document tables and layout signals.
- `Surya` is used to enrich the document with layout boxes, reading order, and table predictions.

Key algorithmic behaviors:

- PDF -> temporary file -> `DocumentConverter`
- Markdown extraction for text
- table extraction from Docling structures
- page-level layout analysis for spatial ordering
- formula candidate extraction using regex heuristics

## 2. Layout and Reading Order

The OCR processor builds a reading-order representation by sorting bounding boxes by page position. This produces a page-aware list of blocks ordered top-to-bottom and left-to-right.

This is useful for:

- downstream semantic chunking
- evidence ordering
- visual grounding between regions and entities

## 3. Visual Detection and Segmentation

Implemented in [app/pipeline/model_helpers.py](app/pipeline/model_helpers.py) and used by [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py).

### YOLO / PID detection
- `PIDSymbolDetector` performs industrial symbol detection for P&ID-style assets.
- It is a vision-first detection mechanism intended to locate diagram symbols or equipment references.

### GroundingDINO
- Zero-shot object detection over document images.
- Accepts natural language prompts and matches them to visual regions.

### SAM2
- Segmentation model for extracting instance masks around visually grounded objects.
- Produces region masks and quality metrics such as IoU and stability score.

## 4. Text Segmentation and Embedding

### Document segmentation
In [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py), text is chunked through `chunk_text(normalize_text(...))`.

The chunking logic is a hybrid of:

- normalization
- token/character windowing
- overlap to preserve nearby context

### Semantic indexing
The chunked text is converted into embeddings via `BgeEmbedder` and stored as vectors for downstream retrieval.

This stage is the bridge between raw document text and vector search / RAG reasoning.

## 5. Entity Extraction

Implemented in [app/pipeline/entity_extractor.py](app/pipeline/entity_extractor.py).

The repository uses an industrial NER stack, primarily centered on GLiNER-style extraction, with fallback heuristics 
when the model stack is unavailable.

Typical outputs are:

- entity name
- entity type
- confidence
- canonicalized identifier
- provenance or evidence span

## 6. Relation Extraction

Implemented in [app/pipeline/relation_extractor.py](app/pipeline/relation_extractor.py).

The relation pipeline uses a layered approach:

1. GLiREL-style relation extraction if available
2. REBEL fallback if GLiREL is missing or unstable
3. heuristic fallback based on co-occurrence and entity proximity

Typical relation types include equipment connectivity, control-flow, process-flow, and generalized `related_to` edges.

## 7. Entity Linking and Ontology Normalization

### Entity linking
Handled in [app/pipeline/entity_linker.py](app/pipeline/entity_linker.py) and [app/pipeline/models.py](app/pipeline/models.py).

The linker tries to normalize extracted names to stable IDs and disambiguate them through the knowledge base or fallback embedding-based heuristics.

### Ontology enrichment
The system uses `normalize_entity_payload()` and `normalize_relation_payload()` to ensure extracted entities and relations conform to an ontology-aware schema.

This means the pipeline can:

- propose new ontology types
- attach confidence and provenance
- keep stable IDs consistent across runs
- evolve a local schema for downstream graph construction

## 8. Re-ranking and Retrieval

The reranker path in [app/pipeline/reranker_v2.py](app/pipeline/reranker_v2.py) and [app/pipeline/model_helpers.py](app/pipeline/model_helpers.py) adds ranking to candidate evidence or entities.

Reranking is used to:

- prioritize relevant entities or relations
- improve answer quality in downstream retrieval or GraphRAG synthesis
- reduce noise from broad semantic retrieval

## 9. GraphRAG and Graph Reasoning

The graph reasoning path is a combination of:

- Neo4j persistence
- graph construction from entities and relations
- GraphRAG query synthesis and evidence selection
- retrieval over graph context and document chunks

The advanced stack in [app/pipeline/advanced_pipeline.py](app/pipeline/advanced_pipeline.py) adds:

- semantic vector indexing into Qdrant
- GraphRAG reasoning
- direct LLM synthesis using Qwen 3
- anomaly detection and forecasting
- RCA and failure prediction
- graph embedding and clustering

## 10. Advanced Analytics Algorithms

### Qdrant vector indexing
Stores entity embeddings and supports vector search.

### GraphRAG
Queries the graph and surrounding evidence for higher-order inference.

### LLM analysis
Uses Qwen 3-style analysis for deeper interpretation of extracted knowledge and text.

### Time-series forecasting
The advanced stage stack includes forecasting models used for anomaly and performance prediction.

### Clustering and lessons mining
The system also includes clustering and lessons-learned mining to identify repeated patterns across jobs or document collections.

## 11. Failure-Handling Strategy

The repository’s algorithmic design is not purely “all-or-nothing.” It intentionally uses a layered fallback hierarchy:

- strongest model path
- secondary model or alternate backend
- heuristic fallback
- graceful skip with structured output

This is a core design principle of the project and is visible throughout [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py).
