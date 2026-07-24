# Ingestion and Preprocessing

This document explains how the system accepts inputs, normalizes them, and prepares them for the main document analysis pipeline.

## 1. Supported Input Modes

The frontend-facing API in [app/main.py](app/main.py) accepts either:

- PDF file uploads
- raw document text
- arbitrary source files that can be parsed into text

Primary endpoints:

- `POST /api/v1/process-pdf`
- `POST /api/v1/process-doc`
- `POST /api/v1/workflows/analyze`
- `POST /api/v1/sources/ingest`

## 2. Ingestion Flow

### PDF ingestion
When a PDF is uploaded:

1. The file is validated for presence and extension.
2. The binary payload is captured.
3. A `JobResult` job record is created and marked as `processing`.
4. The PDF bytes are passed into the background execution path.

### Text ingestion
For text or text-like documents:

1. The input is accepted as a multipart form field.
2. If a file is uploaded, text is extracted through document parsing.
3. If text is already provided, it is normalized and sent through the pipeline.
4. The source metadata is optionally attached for downstream tracking.

## 3. Source Metadata Handling

The ingestion endpoints allow:

- `source_name`
- `source_type`
- `source_format`
- `parsed_source`
- `source_metadata`

This metadata is then persisted into the job payload and later surfaced through the workflow bundle.

## 4. Text Extraction and Normalization

The helper `_extract_text_from_document()` in [app/main.py](app/main.py) tries a document parser first, then falls back to UTF-8 decoding.

The main pipeline normalizes text through `normalize_text()` before segmentation and extraction.

Normalization is important because it:

- collapses inconsistent spacing
- improves chunk quality
- reduces downstream parsing noise
- supports more stable entity and relation extraction

## 5. Chunking and Segmentation

In [app/pipeline/engine_v2.py](app/pipeline/engine_v2.py), the pipeline creates text chunks from the normalized body using `chunk_text()`.

The chunker uses:

- a maximum chunk size
- overlap between adjacent chunks
- chunk metadata tracking for retrieval and indexing

These chunks are later used for:

- embedding generation
- LlamaIndex hybrid retrieval
- evidence selection for GraphRAG synthesis
- document-level reasoning and summarization

## 6. Visual and Layout Preprocessing

For PDFs, the pipeline includes a visual preprocessing stage:

- render pages to images
- run layout analysis
- collect bounding boxes and reading order
- gather tables, formulas, and page structures

This is a key part of the pipeline because the text extraction alone is often insufficient for industrial diagram understanding.

## 7. PID-Specific Preprocessing

The runtime performs PID-focused preprocessing to help detect industrial assets, diagrams, and equipment parts.

This includes:

- YOLO-based PID detection
- GroundingDINO object detection
- SAM2 segmentation
- PID component detection heuristics and canonicalization

The result is a richer spatial and semantic representation of the source document.

## 8. Preprocessing Failure Behavior

The pipeline is designed to continue even if some preprocessing stages fail.

In practice:

- required stages raise an exception and fail the job if they are critical
- optional stages are marked as `skipped` or downgraded to heuristic results
- downstream tasks still continue with the best available evidence

This resilience is one of the most important architecture patterns in the repository.
