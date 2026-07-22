from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class QualityAgent(BaseAgent):
    agent_name = "Quality"
    search_labels = [
        "Inspection",
        "Batch",
        "SOP",
        "Calibration",
        "CAPA",
        "Quality",
        "Defect",
        "Sample",
    ]

    def build_query(self, question: str, entities: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        if question:
            return self.neo4j.search_relationships(
                source_labels=self.search_labels,
                target_labels=self.search_labels,
                keyword=question,
                limit=self.default_limit,
            )
        return self.neo4j.search_nodes(self.search_labels, question, limit=self.default_limit)

    def reason(self, question: str, intent: str, evidence: List[str]) -> str:
        if not evidence:
            return (
                "Quality agent could not find inspection, CAPA, batch, or calibration evidence in Neo4j. "
                "Ensure quality deviations are recorded as structured entities and relations in the graph."
            )
        return super().reason(question, intent, evidence)


def run_quality(state: dict[str, Any]) -> dict[str, Any]:
    question = str(state.get("user_text") or state.get("question") or "").strip()
    agent = QualityAgent()
    result = agent.run(question)
    return {"status": "quality_complete", **result}


__all__ = ["QualityAgent", "run_quality"]
