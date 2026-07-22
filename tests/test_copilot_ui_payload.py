from app.main import copilot_chat


class StubAgent:
    def __call__(self, state):
        return {
            "agent": "Root Cause Analysis",
            "answer": "The compressor likely failed due to maintenance-related stress.",
            "plan": {
                "intent": "rca",
                "equipment": "P101",
                "document_type": "incident_report",
                "regulations": [],
                "use_internet": True,
            },
            "evidence": ["Evidence one", "Evidence two"],
            "summary": "A likely root cause was identified.",
            "key_findings": ["Maintenance window overlapped with failure"],
            "recommendations": ["Review the maintenance history"],
            "confidence": 0.87,
            "risk_level": "high",
            "related_assets": ["P101"],
            "related_documents": ["Neo4j knowledge graph"],
            "next_actions": ["Inspect the incident timeline"],
            "citations": [{"id": "c1", "title": "Evidence 1", "summary": "Maintenance evidence"}],
            "records": [{"n": {"name": "P101"}}],
        }


def test_copilot_chat_surfaces_plan_and_evidence(monkeypatch):
    monkeypatch.setattr("app.main._pick_agent", lambda question: (StubAgent(), "rca"))
    monkeypatch.setattr("app.main._call_tavily", lambda query: [])
    monkeypatch.setattr("app.main._call_gemini", lambda question, evidence, agent_name, context="": None)

    payload = copilot_chat({"question": "Why did the compressor fail?", "job_id": None, "use_web": False})

    assert payload["plan"]["intent"] == "rca"
    assert payload["evidence"] == ["Evidence one", "Evidence two"]
    assert payload["summary"] == "A likely root cause was identified."
    assert payload["recommendations"] == ["Review the maintenance history"]
    assert payload["citations"][0]["id"] == "c1"
