from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List

logger = logging.getLogger(__name__)

try:
    from app.pipeline.advanced_models import Qwen3LLM
except Exception:  # pragma: no cover - optional dependency
    Qwen3LLM = None


class SharedReasoner:
    """Shared reasoning engine used by all agents."""

    def __init__(self):
        self.llm = None
        if Qwen3LLM is not None:
            try:
                self.llm = Qwen3LLM(load_model=False)
                logger.info("✓ SharedReasoner initialized with Qwen3LLM")
            except Exception as exc:
                logger.warning(f"Qwen3LLM not available: {exc}")

    def synthesize(self, question: str, evidence: List[str], agent_name: str) -> str:
        if self.llm is not None:
            prompt = self._build_prompt(question, evidence, agent_name)
            try:
                response = self.llm.generate(prompt, temperature=0.2)
                return response.strip()
            except Exception as exc:
                logger.warning(f"LLM synthesis failed: {exc}")

        return self._fallback_summarize(question, evidence, agent_name)

    def _build_prompt(self, question: str, evidence: List[str], agent_name: str) -> str:
        evidence_text = "\n\n".join(evidence[:8]) or "No graph evidence found."
        return (
            f"You are a {agent_name} agent. \n"
            f"A user asked: {question}\n\n"
            "Use the Neo4j knowledge graph evidence below to answer the question with structured insights. "
            "When no direct evidence exists, be transparent about that.\n\n"
            "Evidence:\n"
            f"{evidence_text}\n\n"
            "Provide a concise answer with supporting items and any recommended next steps."
        )

    def _fallback_summarize(self, question: str, evidence: List[str], agent_name: str) -> str:
        if not evidence:
            return (
                f"{agent_name} could not locate matching Neo4j graph evidence for the query: '{question}'. "
                "Please verify that the knowledge graph contains incident, maintenance, quality, compliance, or failure intelligence data."
            )

        summary_lines = [
            f"{agent_name} summary for: {question}",
            "",
            "Evidence found:",
        ]
        for item in evidence[:5]:
            summary_lines.append(f"- {item}")

        summary_lines.append("")
        summary_lines.append("Recommended next steps:")
        summary_lines.append("1. Review the cited Neo4j events and linked entities.")
        summary_lines.append("2. Confirm the details with domain experts or maintenance logs.")
        summary_lines.append("3. Update the knowledge graph with new incident and inspection data.")

        return "\n".join(summary_lines)


__all__ = ["SharedReasoner"]
