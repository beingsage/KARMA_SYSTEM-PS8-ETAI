from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class RegulationAgent(BaseAgent):
    agent_name = "Regulation"
    search_labels = [
        "Regulation",
        "Standard",
        "Audit",
        "SOP",
        "Compliance",
        "ISO",
        "PESO",
        "OISD",
        "FactoryAct",
        "API",
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
                "Regulation agent found no compliance or audit evidence in Neo4j. "
                "Make sure regulatory obligations are captured in the knowledge graph."
            )
        return super().reason(question, intent, evidence)


def run_regulation(state: dict[str, Any]) -> dict[str, Any]:
    question = str(state.get("user_text") or state.get("question") or "").strip()
    agent = RegulationAgent()
    result = agent.run(question)
    return {"status": "regulation_complete", **result}


__all__ = ["RegulationAgent", "run_regulation"]
