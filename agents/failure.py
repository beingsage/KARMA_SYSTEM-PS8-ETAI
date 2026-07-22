from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class FailureIntelligenceAgent(BaseAgent):
    agent_name = "Failure Intelligence"
    search_labels = [
        "Incident",
        "NearMiss",
        "Lesson",
        "FailurePattern",
        "Trend",
        "RootCause",
        "Event",
        "Report",
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
                "Failure Intelligence agent did not find incident or near-miss evidence in Neo4j. "
                "Load past failure reports and lessons learned into the graph for pattern detection."
            )
        return super().reason(question, intent, evidence)


def run_failure_intelligence(state: dict[str, Any]) -> dict[str, Any]:
    question = str(state.get("user_text") or state.get("question") or "").strip()
    agent = FailureIntelligenceAgent()
    result = agent.run(question)
    return {"status": "failure_intelligence_complete", **result}


__all__ = ["FailureIntelligenceAgent", "run_failure_intelligence"]
