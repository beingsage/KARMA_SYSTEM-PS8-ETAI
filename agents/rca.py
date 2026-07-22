from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class RootCauseAnalysisAgent(BaseAgent):
    agent_name = "Root Cause Analysis"
    search_labels = [
        "Incident",
        "Failure",
        "Alarm",
        "Sensor",
        "Inspection",
        "Maintenance",
        "Equipment",
        "Problem",
        "Issue",
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
                "Root Cause Analysis could not retrieve matching failure or incident evidence from Neo4j. "
                "Please verify the knowledge graph contains failure history, alarms, sensors, and inspections."
            )
        return super().reason(question, intent, evidence)


def run_rca(state: dict[str, Any]) -> dict[str, Any]:
    question = str(state.get("user_text") or state.get("question") or "").strip()
    agent = RootCauseAnalysisAgent()
    result = agent.run(question)
    return {"status": "rca_complete", **result}


__all__ = ["RootCauseAnalysisAgent", "run_rca"]
