"""LlamaIndex hybrid retrieval + citation synthesis.

This module is intentionally lightweight and defensive:
- It never breaks the legacy pipeline.
- If LlamaIndex components fail to initialize, callers should fallback to legacy stages.

The hybrid strategy is:
- Use LlamaIndex retrieval over the same `text_chunks` you already generate.
- Generate evidence-grounded JSON (anomalies/risks/recommendations/compliance).

NOTE: We do NOT replace OCR/entity/relation extractors here.
"""

from __future__ import annotations

import importlib
import json
import mimetypes
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import settings


def _dynamic_import(path: str) -> Any:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _safe_imports():
    """Import llama-index modules lazily."""
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings as LlamaSettings
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.core.readers.file.base import SimpleDirectoryReader
        from llama_index.core.schema import TextNode
        from llama_index.core.embeddings import BaseEmbedding, MockEmbedding
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.postprocessor import SimilarityPostprocessor
    except Exception as exc:  # pragma: no cover
        return None, exc

    return {
        "Document": Document,
        "VectorStoreIndex": VectorStoreIndex,
        "SimpleNodeParser": SimpleNodeParser,
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "TextNode": TextNode,
        "VectorIndexRetriever": VectorIndexRetriever,
        "SimilarityPostprocessor": SimilarityPostprocessor,
        "LlamaSettings": LlamaSettings,
        "MockEmbedding": MockEmbedding,
    }, None


_LOADER_CANDIDATES = {
    ".txt": [
        "llama_index.readers.file.text.TextFileReader",
        "llama_index.readers.file.base.TextFileReader",
        "llama_index.readers.text.TextReader",
    ],
    ".md": [
        "llama_index.readers.file.markdown.MarkdownReader",
        "llama_index.readers.file.md.MarkdownReader",
        "llama_index.readers.markdown.MarkdownReader",
    ],
    ".markdown": [
        "llama_index.readers.file.markdown.MarkdownReader",
        "llama_index.readers.markdown.MarkdownReader",
    ],
    ".html": [
        "llama_index.readers.file.html.HTMLReader",
        "llama_index.readers.file.html_reader.HTMLReader",
        "llama_index.readers.html.HTMLReader",
    ],
    ".htm": [
        "llama_index.readers.file.html.HTMLReader",
        "llama_index.readers.html.HTMLReader",
    ],
    ".pdf": [
        "llama_index.readers.file.pdf.PDFReader",
        "llama_index.readers.file.pdf_reader.PDFReader",
        "llama_index.readers.pdf.PDFReader",
    ],
    ".csv": [
        "llama_index.readers.file.csv.CSVReader",
        "llama_index.readers.csv.CSVReader",
    ],
    ".json": [
        "llama_index.readers.file.json.JSONReader",
        "llama_index.readers.json.JSONReader",
    ],
}


@dataclass
class HybridEvidence:
    """Evidence bundle returned by retrieval."""

    contexts: List[Dict[str, Any]]
    combined_text: str
    coverage_score: float


class LlamaIndexHybrid:
    """Hybrid citation-grounded synthesis using LlamaIndex retrieval."""

    def __init__(
        self,
        embedder: Optional[Any] = None,
        api_key: Optional[str] = None,
        llamaparse_api_key: Optional[str] = None,
    ) -> None:
        self._llama, self._import_error = _safe_imports()
        self._index: Any = None
        self._retriever: Any = None
        self._embedder = embedder
        self._api_key = api_key or settings.llama_api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("LLAMA_CLOUD_API_KEY", "") or os.getenv("LLAMA_API_KEY", "")
        self._llamaparse_api_key = llamaparse_api_key or settings.llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY", "")
        self._configure_api_keys()

    @property
    def available(self) -> bool:
        return self._llama is not None

    def _configure_api_keys(self) -> None:
        if self._api_key:
            os.environ["OPENAI_API_KEY"] = self._api_key
            os.environ["LLAMA_API_KEY"] = self._api_key
            os.environ["LLAMA_CLOUD_API_KEY"] = self._api_key
        if self._llamaparse_api_key:
            os.environ["LLAMAPARSE_API_KEY"] = self._llamaparse_api_key
            if not self._api_key:
                os.environ["OPENAI_API_KEY"] = self._llamaparse_api_key

        try:
            import openai as _openai
            effective_key = self._api_key or self._llamaparse_api_key
            if effective_key:
                _openai.api_key = effective_key
        except Exception:
            pass

    def _get_loader_class(self, extension: str) -> Optional[Any]:
        for candidate in _LOADER_CANDIDATES.get(extension.lower(), []):
            try:
                return _dynamic_import(candidate)
            except Exception:
                continue
        return None

    def _load_documents_from_llamaparse(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        api_key = self._llamaparse_api_key or settings.llamaparse_api_key or os.getenv("LLAMAPARSE_API_KEY", "") or os.getenv("LLAMA_CLOUD_API_KEY", "")
        if not api_key:
            return []

        try:
            from llama_parse import LlamaParse
        except Exception:
            return []

        try:
            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                save_images=True,
                split_by_page=True,
                show_progress=False,
                verbose=False,
                ignore_errors=True,
            )
            docs = parser.load_data(file_path)
            if docs:
                for doc in docs:
                    if isinstance(doc, dict):
                        continue
                    md = getattr(doc, "metadata", {}) or {}
                    if metadata:
                        md = {**metadata, **md}
                        setattr(doc, "metadata", md)
                return docs
        except Exception:
            return []

        return []

    def _load_documents_from_path(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Any]:
        ext = Path(file_path).suffix.lower() or ".txt"

        direct_docs = self._load_documents_from_llamaparse(file_path, metadata=metadata)
        if direct_docs:
            return direct_docs

        loader_class = self._get_loader_class(ext)
        if loader_class is not None:
            try:
                loader = loader_class()
                if hasattr(loader, "load_data"):
                    docs = loader.load_data(file_path)
                    if docs:
                        return docs
                if hasattr(loader, "load"):
                    docs = loader.load(file_path)
                    if docs:
                        return docs
            except Exception:
                pass

        SimpleDirectoryReader = self._llama.get("SimpleDirectoryReader")
        if SimpleDirectoryReader is not None:
            try:
                docs = SimpleDirectoryReader(input_files=[file_path], raise_on_error=False).load_data()
                if docs:
                    return docs
            except Exception:
                pass

        # If file content is already structured JSON from a LlamaParse-style extractor,
        # try to preserve structured metadata when available.
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
                raw_text = raw_bytes.decode("utf-8", errors="ignore")
        except Exception:
            raw_text = ""

        Document = self._llama["Document"]
        if metadata is None:
            metadata = {}

        if raw_text:
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict) and (
                    "text" in parsed or "items" in parsed or "elements" in parsed
                ):
                    metadata = {**metadata, **parsed}
                    item_text = []
                    if isinstance(parsed.get("items"), list):
                        for item in parsed["items"]:
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("content") or item.get("snippet") or str(item)
                                if text:
                                    item_text.append(str(text))
                    elif isinstance(parsed.get("elements"), list):
                        for item in parsed["elements"]:
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("content") or item.get("snippet") or str(item)
                                if text:
                                    item_text.append(str(text))
                    if item_text:
                        return [Document(text="\n\n".join(item_text), metadata=metadata)]
                    return [Document(text=parsed.get("text", raw_text), metadata=metadata)]
            except Exception:
                pass

        return [Document(text=raw_text, metadata=metadata)]

    def load_documents(
        self,
        text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        if not self.available:
            raise RuntimeError(f"LlamaIndexHybrid unavailable: {self._import_error}")

        Document = self._llama["Document"]

        if text is not None:
            if metadata is None:
                metadata = {}
            return [Document(text=text, metadata=metadata)]

        if file_bytes is not None:
            suffix = None
            if file_name:
                suffix = Path(file_name).suffix.lower()
            if not suffix and mime_type:
                suffix = mimetypes.guess_extension(mime_type.split(";")[0].strip()) or ".txt"
            suffix = suffix or ".txt"

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name
                return self._load_documents_from_path(temp_path, metadata=metadata)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        raise ValueError("No content provided for document loading")

    def _build_index_from_documents(self, docs: List[Any]) -> None:
        if not self.available:
            raise RuntimeError(f"LlamaIndexHybrid unavailable: {self._import_error}")

        VectorStoreIndex = self._llama["VectorStoreIndex"]
        SimpleNodeParser = self._llama["SimpleNodeParser"]
        LlamaSettings = self._llama["LlamaSettings"]

        if self._embedder is not None:
            try:
                LlamaSettings.embed_model = self._embedder
            except Exception:
                pass
        else:
            try:
                MockEmbedding = self._llama.get("MockEmbedding")
                if MockEmbedding is not None:
                    LlamaSettings.embed_model = MockEmbedding(embed_dim=8)
            except Exception:
                pass

        if not docs:
            self._index = None
            self._retriever = None
            return

        if hasattr(docs[0], "text"):
            parser = SimpleNodeParser.from_defaults(chunk_size=2048, chunk_overlap=200)
            nodes = parser.get_nodes_from_documents(docs)
        else:
            nodes = docs

        self._index = VectorStoreIndex(nodes)
        self._retriever = self._index.as_retriever(similarity_top_k=8)

    def build_index_from_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.available:
            raise RuntimeError(f"LlamaIndexHybrid unavailable: {self._import_error}")
        docs = self.load_documents(text=text, metadata=metadata)
        self._build_index_from_documents(docs)

    def build_index_from_file(
        self,
        file_bytes: bytes,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.available:
            raise RuntimeError(f"LlamaIndexHybrid unavailable: {self._import_error}")
        docs = self.load_documents(file_bytes=file_bytes, file_name=file_name, mime_type=mime_type, metadata=metadata)
        self._build_index_from_documents(docs)

    def extract_text(
        self,
        text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        docs = self.load_documents(text=text, file_bytes=file_bytes, file_name=file_name, mime_type=mime_type, metadata=metadata)
        extracted: List[str] = []
        for doc in docs:
            doc_text = getattr(doc, "text", None)
            if doc_text:
                extracted.append(doc_text)
            else:
                extracted.append(str(doc))
        return "\n\n".join([t.strip() for t in extracted if t.strip()])

    def build_index(self, text_chunks: List[str], text_chunks_metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        if not self.available:
            raise RuntimeError(f"LlamaIndexHybrid unavailable: {self._import_error}")

        if not text_chunks:
            self._index = None
            self._retriever = None
            return

        Document = self._llama["Document"]
        VectorStoreIndex = self._llama["VectorStoreIndex"]
        SimpleNodeParser = self._llama["SimpleNodeParser"]
        LlamaSettings = self._llama["LlamaSettings"]

        if self._embedder is not None:
            try:
                LlamaSettings.embed_model = self._embedder
            except Exception:
                pass
        else:
            try:
                MockEmbedding = self._llama.get("MockEmbedding")
                if MockEmbedding is not None:
                    LlamaSettings.embed_model = MockEmbedding(embed_dim=8)
            except Exception:
                pass

        docs: List[Any] = []
        text_chunks_metadata = text_chunks_metadata or [{} for _ in text_chunks]
        for i, (chunk, meta) in enumerate(zip(text_chunks, text_chunks_metadata)):
            meta2 = dict(meta) if isinstance(meta, dict) else {}
            meta2.setdefault("chunk_id", i)
            docs.append(Document(text=chunk, metadata=meta2))

        parser = SimpleNodeParser.from_defaults(chunk_size=2048, chunk_overlap=200)
        nodes = parser.get_nodes_from_documents(docs)

        self._index = VectorStoreIndex(nodes)
        self._retriever = self._index.as_retriever(similarity_top_k=8)

    def retrieve(self, queries: List[str], entities: Optional[List[Dict[str, Any]]] = None) -> HybridEvidence:
        if self._retriever is None:
            return HybridEvidence(contexts=[], combined_text="", coverage_score=0.0)

        seen_ctx_ids: set = set()
        contexts: List[Dict[str, Any]] = []
        for q in queries:
            try:
                retrieved = self._retriever.retrieve(q)
            except Exception:
                continue

            for node in retrieved:
                md = getattr(node, "metadata", {}) or {}
                chunk_id = md.get("chunk_id") or md.get("id")
                ctx_id = f"{chunk_id}" if chunk_id is not None else str(node)
                if ctx_id in seen_ctx_ids:
                    continue
                seen_ctx_ids.add(ctx_id)

                ctx_text = getattr(node, "text", "") or str(node)
                if not ctx_text:
                    continue

                contexts.append(
                    {
                        "chunk_id": chunk_id,
                        "score": float(getattr(node, "score", 0.0) or 0.0),
                        "text": ctx_text,
                        "metadata": md,
                    }
                )

        combined_text = "\n---\n".join([c["text"] for c in contexts[:10]])
        coverage = 0.0
        if entities:
            names = [e.get("name", "") for e in entities if e.get("name")]
            if names:
                hits = 0
                text_lower = combined_text.lower()
                for n in names[:25]:
                    if not n:
                        continue
                    nn = re.sub(r"\s+", " ", n.strip().lower())
                    nn2 = nn.replace("-", " ")
                    if nn in text_lower or nn2 in text_lower:
                        hits += 1
                coverage = hits / max(1, min(len(names), 25))

        return HybridEvidence(contexts=contexts, combined_text=combined_text, coverage_score=float(coverage))

    def _llm_synthesize_citation_json(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        queries: List[str],
        evidence: HybridEvidence,
    ) -> Optional[Dict[str, Any]]:
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            return None

        prompt = (
            "You are an evidence-grounded industrial RCA assistant. "
            "Return ONLY valid JSON with keys: summary_method, status, anomalies_detected, failure_risks, "
            "maintenance_recommendations, compliance, confidence, evidence_coverage, citations. "
            "Use the retrieved evidence to support claims, keep the output concise, and do not invent facts.\n\n"
            f"Question: {text[:400]}\n\n"
            f"Entities: {json.dumps(entities[:8], ensure_ascii=False)}\n\n"
            f"Relations: {json.dumps(relations[:8], ensure_ascii=False)}\n\n"
            f"Evidence contexts:\n{json.dumps(evidence.contexts[:5], ensure_ascii=False)}"
        )

        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                f"https://generativelanguage.googleapis.com/v1beta/models/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}:generateContent?key={api_key}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                body = json.loads(response.read().decode("utf-8"))

            candidates = body.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            output_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            if not output_text:
                return None

            parsed = json.loads(output_text)
            if isinstance(parsed, dict):
                parsed["summary_method"] = parsed.get("summary_method") or "llm-citation-synthesis"
                parsed["status"] = parsed.get("status") or "analyzed"
                parsed["citations"] = parsed.get("citations") or [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "score": c.get("score"),
                        "snippet": (c.get("text") or "")[:180],
                    }
                    for c in evidence.contexts[:5]
                ]
                return parsed
        except Exception:
            return None

        return None

    def citation_summarize(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        queries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return evidence-grounded JSON schema expected by your GraphRAG stage.

        This function now prefers an LLM-backed synthesis path when a Gemini key is available,
        while preserving the original retrieval/heuristic fallback for offline or degraded runs.
        """
        if queries is None:
            top_ents = entities[:12]
            ent_queries = [e.get("name", "") for e in top_ents if e.get("name")]
            rel_queries = []
            for r in relations[:10]:
                s = (r.get("source") or "").strip()
                t = (r.get("target") or "").strip()
                if s and t:
                    rel_queries.append(f"{s} {r.get('relation_type','related_to')} {t}")
            queries = (ent_queries + rel_queries)[:10] or [text[:200] if text else "document"]

        evidence = self.retrieve(queries=queries, entities=entities)

        llm_result = self._llm_synthesize_citation_json(entities, relations, text, queries, evidence)
        if llm_result is not None:
            llm_result.setdefault("evidence_coverage", evidence.coverage_score)
            return llm_result

        if not evidence.contexts or evidence.coverage_score < 0.05:
            return {
                "summary_method": "llamaindex-citation-synthesis",
                "status": "insufficient-evidence",
                "anomalies_detected": [],
                "failure_risks": [],
                "maintenance_recommendations": [],
                "compliance": [],
                "confidence": 0.0,
                "evidence_coverage": evidence.coverage_score,
                "reason": "No strong evidence in retrieved contexts",
            }

        ctx = evidence.combined_text.lower()

        measurement_re = re.compile(r"\b\d+(?:\.\d+)?\s*(mm|cm|m|psi|bar|rpm|°c|celsius|%|rpm)\b")
        safety_words = ["warning", "caution", "danger", "safety", "hazard"]
        maintenance_words = ["inspect", "inspection", "replace", "replacement", "calibration", "maintenance", "service"]

        has_measurements = bool(measurement_re.search(ctx))
        has_safety = any(w in ctx for w in safety_words)
        has_maintenance = any(w in ctx for w in maintenance_words)

        anomalies = []
        risks = []
        recommendations = []
        compliance = []

        if has_measurements or has_safety:
            anomalies.append(
                {
                    "name": "Manual indicates measurable condition or safety-relevant guidance",
                    "description": "Evidence present in retrieved document spans.",
                    "source": "retrieved_context",
                    "confidence": 0.55,
                }
            )

        if has_maintenance:
            recommendations.append(
                {
                    "name": "Follow maintenance/inspection guidance referenced in manual",
                    "description": "Maintenance-related instructions found in retrieved spans.",
                    "source": "retrieved_context",
                    "confidence": 0.6,
                }
            )

        confidence = float(min(0.95, 0.3 + 0.7 * evidence.coverage_score))

        return {
            "summary_method": "llamaindex-citation-synthesis",
            "status": "analyzed",
            "anomalies_detected": anomalies,
            "failure_risks": risks,
            "maintenance_recommendations": recommendations,
            "compliance": compliance,
            "confidence": confidence,
            "evidence_coverage": evidence.coverage_score,
            "citations": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "score": c.get("score"),
                    "snippet": (c.get("text") or "")[:180],
                }
                for c in evidence.contexts[:5]
            ],
        }

