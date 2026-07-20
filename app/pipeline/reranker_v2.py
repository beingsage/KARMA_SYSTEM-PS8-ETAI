"""Enhanced reranking with fallbacks, distillation, and cross-encoder support."""

import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings


class LexicalReranker:
    """Fast lexical reranking using BM25 and token overlap."""

    def __init__(self):
        self.corpus: List[str] = []
        self.inverted_index: Dict[str, List[int]] = {}
        self.doc_freqs: List[Counter] = []
        self.idf_cache: Dict[str, float] = {}
        self.backend = "lexical"

    def index_documents(self, documents: List[str]) -> None:
        self.corpus = documents
        self.inverted_index = {}
        self.doc_freqs = []

        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            freq = Counter(tokens)
            self.doc_freqs.append(freq)

            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_idx)

    def score_pair(self, query: str, candidate: str) -> float:
        """Score using BM25-inspired formula."""
        query_tokens = self._tokenize(query)
        candidate_tokens = self._tokenize(candidate)

        if not query_tokens or not candidate_tokens:
            return 0.0

        k1, b = 1.5, 0.75
        avg_doc_len = sum(len(freq) for freq in self.doc_freqs) / max(len(self.doc_freqs), 1) if self.doc_freqs else 1
        candidate_len = len(candidate_tokens)

        score = 0.0
        for token in set(query_tokens):
            token_freq = candidate_tokens.count(token)
            if token_freq == 0:
                continue

            doc_count = len(self.inverted_index.get(token, []))
            idf = (len(self.corpus) - doc_count + 0.5) / max(doc_count + 0.5, 1)
            bm25_freq = (token_freq * (k1 + 1)) / (token_freq + k1 * (1 - b + b * (candidate_len / max(avg_doc_len, 1))))
            score += idf * bm25_freq

        return min(1.0, score / max(len(query_tokens), 1))

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rerank candidates using BM25 and return (index, text, score) tuples."""
        if not candidates:
            return []

        scores = [self.score_pair(query, candidate) for candidate in candidates]
        ranked = [(idx, candidates[idx], scores[idx]) for idx in range(len(candidates))]
        ranked.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if len(t) > 2]


class EmbeddingReranker:
    """Lightweight embedding-based reranking using cosine similarity."""

    def __init__(self):
        self.embedder = None
        self.backend = "embedding"
        self._initialize()

    def _initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            candidate_models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ]

            for model_name in candidate_models:
                try:
                    self.embedder = SentenceTransformer(model_name, device="cpu")
                    print(f"✓ Embedding reranker ready: {model_name}")
                    return
                except Exception:
                    continue

            print("✗ Embedding reranker init failed; using lexical fallback")
            self.embedder = None
        except Exception as e:
            print(f"✗ Embedding reranker init failed: {e}")
            self.embedder = None

    def score_pair(self, query: str, candidate: str) -> float:
        if self.embedder is None:
            return 0.0

        try:
            query_emb = self.embedder.encode([query], convert_to_tensor=False)[0]
            candidate_emb = self.embedder.encode([candidate], convert_to_tensor=False)[0]

            dot = sum(q * c for q, c in zip(query_emb, candidate_emb))
            query_norm = sum(q * q for q in query_emb) ** 0.5
            candidate_norm = sum(c * c for c in candidate_emb) ** 0.5

            if query_norm == 0.0 or candidate_norm == 0.0:
                return 0.0

            return float(dot / (query_norm * candidate_norm))
        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rerank candidates using embeddings and return (index, text, score) tuples."""
        if not candidates or self.embedder is None:
            return []

        scores = [self.score_pair(query, candidate) for candidate in candidates]
        ranked = [(idx, candidates[idx], scores[idx]) for idx in range(len(candidates))]
        ranked.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


class CrossEncoderReranker:
    """Cross-encoder reranking (long-term research path)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model = None
        self.backend = "cross_encoder"
        self.model_name = model_name or os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
        self._initialize()

    def _initialize(self) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name, max_length=512)
            print(f"✓ Cross-encoder reranker ready: {self.model_name}")
        except Exception as e:
            print(f"⚠ Cross-encoder reranker init failed: {e}; will use fallback")
            self.model = None

    def score_pair(self, query: str, candidate: str) -> float:
        if self.model is None:
            return 0.0

        try:
            scores = self.model.predict([[query, candidate]], convert_to_tensor=False)
            return float(scores[0]) if scores is not None and len(scores) > 0 else 0.0
        except Exception:
            return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rerank candidates using cross-encoder and return (index, text, score) tuples."""
        if not candidates or self.model is None:
            return []

        scores = [self.score_pair(query, candidate) for candidate in candidates]
        ranked = [(idx, candidates[idx], scores[idx]) for idx in range(len(candidates))]
        ranked.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked


class EnhancedReranker:
    """Multi-strategy reranker with fallback chain and telemetry."""

    def __init__(self):
        self.primary: Optional[Any] = self._load_primary()
        self.secondary: Optional[Any] = None
        self.tertiary: Optional[Any] = None
        self.telemetry: Dict[str, int] = {"primary": 0, "secondary": 0, "tertiary": 0}
        self._setup_fallbacks()

    def _load_primary(self) -> Optional[Any]:
        mode = os.getenv("RERANKER_MODE", "bge").lower()
        if mode == "cross_encoder":
            return CrossEncoderReranker()
        else:
            from app.pipeline.model_helpers import BgeReranker

            return BgeReranker()

    def _setup_fallbacks(self) -> None:
        if self.primary is None or (hasattr(self.primary, "model") and self.primary.model is None):
            self.secondary = EmbeddingReranker()
            if self.secondary.embedder is None:
                self.tertiary = LexicalReranker()

    def score_pair(self, query: str, candidate: str) -> float:
        try:
            if self.primary is not None:
                if hasattr(self.primary, "model") and self.primary.model is not None:
                    self.telemetry["primary"] += 1
                    return self.primary.score_pair(query, candidate)
                elif hasattr(self.primary, "score_pair"):
                    try:
                        score = self.primary.score_pair(query, candidate)
                        if score > 0.0:
                            self.telemetry["primary"] += 1
                            return score
                    except Exception:
                        pass
        except Exception:
            pass

        if self.secondary is not None:
            try:
                score = self.secondary.score_pair(query, candidate)
                if score > 0.0:
                    self.telemetry["secondary"] += 1
                    return score
            except Exception:
                pass

        if self.tertiary is not None:
            try:
                score = self.tertiary.score_pair(query, candidate)
                self.telemetry["tertiary"] += 1
                return score
            except Exception:
                pass

        return 0.0

    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, str, float]]:
        """Rerank candidates and return (index, text, score) tuples."""
        if not candidates:
            return []

        scores = [self.score_pair(query, candidate) for candidate in candidates]
        ranked = [(idx, candidates[idx], scores[idx]) for idx in range(len(candidates))]
        ranked.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        return ranked

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "backend": getattr(self.primary, "backend", "unknown"),
            "calls": self.telemetry,
            "total_calls": sum(self.telemetry.values()),
        }


def compute_ndcg(ranked_list: List[float], ideal_ranked_list: List[float], k: Optional[int] = None) -> float:
    """Compute NDCG@k (Normalized Discounted Cumulative Gain)."""
    if not ranked_list or not ideal_ranked_list:
        return 0.0

    if k is not None:
        ranked_list = ranked_list[:k]
        ideal_ranked_list = ideal_ranked_list[:k]

    dcg = sum(relevance / (1.0 + i) for i, relevance in enumerate(ranked_list))
    idcg = sum(relevance / (1.0 + i) for i, relevance in enumerate(ideal_ranked_list))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
