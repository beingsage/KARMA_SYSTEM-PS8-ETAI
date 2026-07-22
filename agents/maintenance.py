from __future__ import annotations

from typing import Any, Dict, List

from agents.base_agent import BaseAgent


class MaintenanceAgent(BaseAgent):
    agent_name = "Maintenance"
    search_labels = [
        "Maintenance",
        "Equipment",
        "SparePart",
        "Service",
        "Schedule",
        "WorkOrder",
        "Inspection",
        "OEM",
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
                "Maintenance agent found no Neo4j evidence for maintenance history or equipment health. "
                "Check that maintenance records and OEM documents are stored in the knowledge graph."
            )
        return super().reason(question, intent, evidence)


def run_maintenance(state: dict[str, Any]) -> dict[str, Any]:
    question = str(state.get("user_text") or state.get("question") or "").strip()
    agent = MaintenanceAgent()
    result = agent.run(question)
    return {"status": "maintenance_complete", **result}


__all__ = ["MaintenanceAgent", "run_maintenance"]
