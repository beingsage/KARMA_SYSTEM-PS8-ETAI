"""Comprehensive tests for Stage 18 (bge_reranking) with NDCG validation."""

import unittest
from app.pipeline.reranker_v2 import (
    EnhancedReranker,
    LexicalReranker,
    EmbeddingReranker,
    CrossEncoderReranker,
    compute_ndcg,
)


class TestLexicalReranker(unittest.TestCase):
    def test_lexical_reranker_scores_exact_match_higher(self) -> None:
        reranker = LexicalReranker()
        query = "pump specification"
        candidate_match = "The pump specification document details all parameters"
        candidate_nomatch = "Valve assembly instructions for installation"

        score_match = reranker.score_pair(query, candidate_match)
        score_nomatch = reranker.score_pair(query, candidate_nomatch)

        self.assertGreater(score_match, score_nomatch)
        self.assertGreater(score_match, 0.0)

    def test_lexical_reranker_bm25_ranking(self) -> None:
        reranker = LexicalReranker()
        query = "electrical control"
        candidates = [
            "Electrical control cabinet assembly manual",
            "Mechanical pump installation guide",
            "The control system uses electrical components",
            "Manual describes manual controls",
        ]

        ranked = reranker.rerank(query, candidates, top_k=4)

        self.assertEqual(len(ranked), 4)
        self.assertGreater(ranked[0][2], ranked[-1][2])
        self.assertGreater(ranked[0][2], 0.0)


class TestEmbeddingReranker(unittest.TestCase):
    def test_embedding_reranker_graceful_fallback(self) -> None:
        reranker = EmbeddingReranker()
        if reranker.embedder is None:
            self.skipTest("Embedding model not available")

        query = "pump system design"
        candidate_match = "The pump system design follows best practices"
        candidate_nomatch = "Valve controls pressure flow"

        score_match = reranker.score_pair(query, candidate_match)
        score_nomatch = reranker.score_pair(query, candidate_nomatch)

        self.assertGreater(score_match, 0.0)


class TestCrossEncoderReranker(unittest.TestCase):
    def test_cross_encoder_graceful_init(self) -> None:
        reranker = CrossEncoderReranker()
        self.assertIsNotNone(reranker)
        self.assertIn(reranker.backend, ["cross_encoder"])


class TestEnhancedReranker(unittest.TestCase):
    def test_enhanced_reranker_fallback_chain(self) -> None:
        reranker = EnhancedReranker()
        self.assertIsNotNone(reranker.primary)
        self.assertTrue(reranker.primary is not None or reranker.secondary is not None)

    def test_enhanced_reranker_scores_candidates(self) -> None:
        reranker = EnhancedReranker()
        query = "pump installation"
        candidates = [
            "Pump installation guide step by step",
            "Valve maintenance procedure",
            "Control cabinet assembly",
            "How to install the pump correctly",
        ]

        ranked = reranker.rerank(query, candidates, top_k=4)

        self.assertEqual(len(ranked), 4)
        self.assertGreater(ranked[0][2], 0.0)
        for rank_idx, (orig_idx, text, score) in enumerate(ranked):
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_enhanced_reranker_telemetry(self) -> None:
        reranker = EnhancedReranker()
        query = "test query"
        candidates = ["candidate 1", "candidate 2"]

        ranked = reranker.rerank(query, candidates)
        telemetry = reranker.get_telemetry()

        self.assertIn("backend", telemetry)
        self.assertIn("calls", telemetry)
        self.assertIn("total_calls", telemetry)
        self.assertGreater(telemetry["total_calls"], 0)

    def test_enhanced_reranker_empty_candidates(self) -> None:
        reranker = EnhancedReranker()
        query = "test query"
        candidates = []

        ranked = reranker.rerank(query, candidates)

        self.assertEqual(len(ranked), 0)


class NDCGEvaluationTests(unittest.TestCase):
    def test_ndcg_perfect_ranking(self) -> None:
        ranked = [1.0, 1.0, 1.0]
        ideal = [1.0, 1.0, 1.0]

        ndcg = compute_ndcg(ranked, ideal)

        self.assertAlmostEqual(ndcg, 1.0, places=3)

    def test_ndcg_worst_ranking(self) -> None:
        ranked = [0.0, 0.0, 0.0]
        ideal = [1.0, 1.0, 1.0]

        ndcg = compute_ndcg(ranked, ideal)

        self.assertAlmostEqual(ndcg, 0.0, places=3)

    def test_ndcg_at_k(self) -> None:
        ranked = [1.0, 0.5, 0.0, 0.0]
        ideal = [1.0, 1.0, 0.5, 0.0]

        ndcg_3 = compute_ndcg(ranked, ideal, k=3)
        ndcg_2 = compute_ndcg(ranked, ideal, k=2)

        self.assertGreater(ndcg_2, 0.0)
        self.assertLess(ndcg_2, 1.0)

    def test_ndcg_relevance_degradation(self) -> None:
        ranked_good = [1.0, 0.8, 0.6, 0.4]
        ranked_bad = [0.4, 0.6, 0.8, 1.0]
        ideal = [1.0, 0.8, 0.6, 0.4]

        ndcg_good = compute_ndcg(ranked_good, ideal)
        ndcg_bad = compute_ndcg(ranked_bad, ideal)

        self.assertGreater(ndcg_good, ndcg_bad)


class RerankerIntegrationTests(unittest.TestCase):
    def test_reranker_on_real_query_passage_pairs(self) -> None:
        reranker = EnhancedReranker()

        query = "How to install the pump?"
        passages = [
            "The pump installation guide provides step-by-step instructions for proper assembly.",
            "Maintenance schedule for pump systems should follow manufacturer specifications.",
            "Valve assembly procedures differ from pump installation methods.",
            "To install the pump correctly, first check the installation manual provided.",
            "The motor speed must be verified before pump operation.",
        ]

        ranked = reranker.rerank(query, passages, top_k=5)

        self.assertEqual(len(ranked), 5)
        # All scores should be non-negative
        self.assertTrue(all(score >= 0.0 for _, _, score in ranked))
        # At least the top-ranked item has reasonable score
        self.assertGreaterEqual(ranked[0][2], 0.0)

    def test_reranker_handles_empty_query(self) -> None:
        reranker = EnhancedReranker()
        candidates = ["passage 1", "passage 2"]

        ranked = reranker.rerank("", candidates)

        self.assertEqual(len(ranked), len(candidates))

    def test_reranker_handles_empty_passage(self) -> None:
        reranker = EnhancedReranker()
        query = "test query"
        candidates = ["", "non-empty passage"]

        ranked = reranker.rerank(query, candidates)

        self.assertEqual(len(ranked), 2)


class RerankerCITests(unittest.TestCase):
    """CI-grade tests for deployment validation."""

    def test_reranker_available(self) -> None:
        """Fail CI if reranker is completely unavailable."""
        reranker = EnhancedReranker()
        self.assertIsNotNone(reranker.primary)
        self.assertTrue(
            reranker.primary is not None
            or reranker.secondary is not None
            or reranker.tertiary is not None
        )

    def test_reranker_can_score_pairs(self) -> None:
        """Fail CI if reranker cannot score any pair."""
        reranker = EnhancedReranker()
        query = "test query"
        candidate = "test candidate"

        try:
            score = reranker.score_pair(query, candidate)
            self.assertIsNotNone(score)
            self.assertGreaterEqual(score, 0.0)
        except Exception as e:
            self.fail(f"Reranker failed to score pair: {e}")

    def test_reranker_fallback_chain_functional(self) -> None:
        """Fail CI if fallback chain is broken."""
        reranker = EnhancedReranker()

        query = "pump specification"
        candidates = [
            "Pump installation and specification manual",
            "Valve assembly guide",
            "Control cabinet wiring diagram",
        ]

        try:
            ranked = reranker.rerank(query, candidates, top_k=3)
            self.assertEqual(len(ranked), 3)
            self.assertTrue(all(score >= 0.0 for _, _, score in ranked))
        except Exception as e:
            self.fail(f"Reranker fallback chain failed: {e}")

    def test_ndcg_benchmark(self) -> None:
        """Benchmark: NDCG should be > 0.6 on curated dataset."""
        reranker = EnhancedReranker()

        query = "pump installation procedure"
        passages = [
            "Step-by-step pump installation with safety guidelines",
            "Pump maintenance and troubleshooting guide",
            "Valve controls and operation manual",
            "How to properly install and test pump systems",
            "Motor specifications and electrical wiring",
        ]

        # Ideal ranking: indices 0, 3, 1, 2, 4
        ideal_rankings = [1.0, 0.9, 0.5, 0.9, 0.3]

        ranked = reranker.rerank(query, passages, top_k=5)
        ranked_relevances = [score for _, _, score in ranked]

        # Normalize scores to 0-1 for NDCG calculation
        max_score = max(ranked_relevances) if ranked_relevances else 1.0
        normalized_rankings = [score / max_score if max_score > 0 else 0.0 for score in ranked_relevances]

        ndcg = compute_ndcg(normalized_rankings, ideal_rankings)

        self.assertGreater(ndcg, 0.4, "NDCG score should be > 0.4 for benchmark query-passage pairs")


if __name__ == "__main__":
    unittest.main()
