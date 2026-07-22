from types import SimpleNamespace

from agents.base_agent import BaseAgent
from agents.reasoning import SharedReasoner


class StubNeo4j:
    def search_nodes(self, labels, keyword="", limit=20):
        return [
            {
                "n": {
                    "name": "Compressor",
                    "description": "Critical compressor in the plant",
                    "type": "Equipment",
                },
                "labels": ["Equipment"],
            }
        ]

    def search_relationships(self, source_labels, target_labels, keyword="", limit=20):
        return [
            {
                "a": {"name": "Compressor", "type": "Equipment"},
                "r": {"type": "MAINTAINED_BY"},
                "b": {"name": "Maintenance Window", "type": "Maintenance"},
            }
        ]


def test_planner_and_structured_output():
    agent = BaseAgent(neo4j_client=StubNeo4j(), reasoner=SharedReasoner())
    result = agent.run("Why did the compressor fail after the maintenance window?")

    assert result["agent"] == "base"
    assert result["plan"]["intent"] == "rca"
    assert result["plan"]["use_internet"] is True
    assert result["summary"]
    assert result["key_findings"]
    assert result["evidence"]
    assert result["recommendations"]
    assert result["citations"]
    assert result["confidence"] >= 0.0
