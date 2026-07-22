from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List

from agents.neo4j_client import Neo4jClient
from agents.reasoning import SharedReasoner

logger = logging.getLogger(__name__)

ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_\-]+)\b")


class BaseAgent:
    agent_name = "base"
    search_labels: List[str] = []
    default_limit = 20

    def __init__(self, neo4j_client: Neo4jClient | None = None, reasoner: SharedReasoner | None = None):
        self.neo4j = neo4j_client or Neo4jClient()
        self.reasoner = reasoner or SharedReasoner()
        self.memory: Dict[str, Any] = {
            "current_equipment": None,
            "current_plant": None,
            "previous_queries": [],
            "previous_cypher": [],
            "current_incident": None,
            "current_hypothesis": None,
        }

    def run(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        question_text = str(question or "").strip()
        plan = self.plan(question_text)
        entities = self.extract_entities(question_text)
        records = self.build_query(question_text, entities, plan=plan, **kwargs)
        graph_evidence = self.merge_evidence(records)
        web_evidence = self.retrieve_web_evidence(question_text, plan)
        evidence = self.rank_evidence(question_text, graph_evidence + web_evidence, records, plan)
        answer = self.reason(question_text, plan, evidence)
        summary = self.reflect(question_text, answer, evidence, plan)
        citations = self.generate_citations(question_text, evidence, records, plan)

        self.memory["previous_queries"] = (self.memory.get("previous_queries", []) + [question_text])[-8:]
        self.memory["current_equipment"] = plan.get("equipment") or (entities[0] if entities else None)
        self.memory["current_hypothesis"] = summary

        return {
            "agent": self.agent_name,
            "plan": plan,
            "intent": plan.get("intent", self.agent_name),
            "question": question_text,
            "entities": entities,
            "records": records,
            "evidence": evidence,
            "answer": answer,
            "summary": summary,
            "key_findings": self._extract_key_findings(evidence),
            "recommendations": self.recommend(question_text, evidence, plan),
            "confidence": self.estimate_confidence(evidence, records, plan),
            "risk_level": self.estimate_risk_level(evidence, plan),
            "related_assets": self.related_assets(entities, evidence),
            "related_documents": self.related_documents(plan, evidence),
            "next_actions": self.recommend(question_text, evidence, plan),
            "citations": citations,
        }

    def plan(self, question: str) -> Dict[str, Any]:
        text = (question or "").strip().lower()
        entities = self.extract_entities(question)
        equipment = entities[0] if entities else None
        intent = "rca"
        if any(keyword in text for keyword in ["compliance", "regulation", "audit", "standard", "iso", "oisd", "factory act", "policy"]):
            intent = "regulation"
        elif any(keyword in text for keyword in ["quality", "inspection", "capa", "calibration", "defect", "batch", "sop"]):
            intent = "quality"
        elif any(keyword in text for keyword in ["failure", "incident", "near miss", "lesson", "trend", "root cause", "risk", "fail", "failed", "cause", "caused", "why"]):
            intent = "rca"
        elif any(keyword in text for keyword in ["maintenance", "repair", "service", "downtime", "spare", "pm", "schedule"]):
            intent = "maintenance"

        use_internet = bool(
            intent in {"regulation", "failure", "quality", "rca"}
            or any(keyword in text for keyword in ["latest", "recent", "update", "current", "why", "cause", "caused", "trend", "external", "vendor", "bulletin"])
        )

        document_type = "incident_report" if "report" in text or intent == "failure" else "maintenance_log"
        regulations = [
            regulation for regulation in ["ISO", "PESO", "OISD", "API"]
            if regulation.lower() in text or intent == "regulation"
        ]

        plan = {
            "intent": intent,
            "equipment": equipment,
            "document_type": document_type,
            "regulations": regulations,
            "use_internet": use_internet,
        }
        self.memory["current_equipment"] = equipment or self.memory.get("current_equipment")
        return plan

    def extract_entities(self, question: str) -> List[str]:
        if not question:
            return []

        candidates = ENTITY_PATTERN.findall(question)
        lower = question.lower()
        tokens = [token for token in candidates if token.lower() in lower]
        return list(dict.fromkeys(tokens))[:8]

    def build_query(self, question: str, entities: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        if self.search_labels:
            return self.neo4j.search_relationships(self.search_labels, self.search_labels, question, limit=self.default_limit)
        return self.neo4j.search_nodes(self.search_labels, question, limit=self.default_limit)

    def retrieve_web_evidence(self, question: str, plan: Dict[str, Any]) -> List[str]:
        if not plan.get("use_internet"):
            return []

        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("TAVILY") or ""
        if not api_key:
            return []

        try:
            body = json.dumps({
                "api_key": api_key,
                "query": question,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 3,
            }).encode("utf-8")
            request = urllib.request.Request(
                "https://api.tavily.com/search",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            results = payload.get("results") or []
            return [
                f"Web result: {item.get('title') or 'search result'} - {item.get('content') or item.get('snippet') or ''}".strip()
                for item in results
            ]
        except Exception:
            return []

    def merge_evidence(self, records: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        for record in records:
            if not record:
                continue

            if "a" in record and "r" in record and "b" in record:
                source = record["a"]
                relation = record["r"]
                target = record["b"]
                evidence.append(self._format_relation_evidence(source, relation, target))
                continue

            if "n" in record:
                evidence.append(self._format_node_evidence(record["n"], record.get("labels", [])))
                continue

            # Fallback generic flatten
            evidence.append(self._format_generic_record(record))

        return evidence

    def reason(self, question: str, intent: Any, evidence: List[str]) -> str:
        plan = intent if isinstance(intent, dict) else {"intent": intent}
        agent_name = plan.get("intent") or self.agent_name
        return self.reasoner.synthesize(question, evidence, str(agent_name))

    def rank_evidence(self, question: str, evidence: List[str], records: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[str]:
        if not evidence:
            return []

        scored: List[tuple[float, str]] = []
        question_terms = {token for token in re.findall(r"[a-zA-Z0-9_]+", question.lower()) if len(token) > 2}
        for item in evidence:
            text = (item or "").lower()
            overlap = sum(1 for term in question_terms if term in text)
            score = overlap * 0.3
            if plan.get("intent") in {"rca", "failure"} and any(keyword in text for keyword in ["incident", "failure", "alarm", "sensor", "maintenance", "root", "cause"]):
                score += 0.4
            if plan.get("intent") == "maintenance" and any(keyword in text for keyword in ["maintenance", "service", "downtime", "spare", "schedule"]):
                score += 0.4
            if plan.get("intent") == "quality" and any(keyword in text for keyword in ["inspection", "capa", "calibration", "quality", "defect"]):
                score += 0.4
            if plan.get("intent") == "regulation" and any(keyword in text for keyword in ["standard", "regulation", "audit", "iso", "oisd", "peso", "api"]):
                score += 0.4
            if "web result" in text:
                score += 0.1
            scored.append((score, item))

        ranked = [item for _, item in sorted(scored, key=lambda entry: entry[0], reverse=True)]
        return ranked[:8]

    def reflect(self, question: str, answer: str, evidence: List[str], plan: Dict[str, Any]) -> str:
        if not evidence:
            return (
                f"No direct evidence was found for the request. "
                f"The best next step is to verify the graph and add supporting incident or maintenance records."
            )

        reflection = f"{answer}\n\nReflection: The evidence indicates a likely {plan.get('intent', 'analysis')} hypothesis; confirm with operational records and add missing supporting evidence if needed."
        return reflection

    def generate_citations(self, question: str, evidence: List[str], records: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for index, item in enumerate(evidence[:5], start=1):
            citations.append({
                "id": f"citation-{index}",
                "title": f"Evidence {index}",
                "source": "neo4j" if not item.lower().startswith("web result") else "web",
                "summary": item,
                "url": "https://neo4j.com/docs/" if not item.lower().startswith("web result") else "",
            })
        return citations

    def recommend(self, question: str, evidence: List[str], plan: Dict[str, Any]) -> List[str]:
        intent = plan.get("intent", "analysis")
        if not evidence:
            return ["Inspect the graph and add missing entities or relations before making a final decision."]
        if intent == "maintenance":
            return ["Schedule follow-up maintenance, review the affected equipment, and confirm spare parts availability."]
        if intent == "quality":
            return ["Open a CAPA task, verify calibration history, and inspect the related batch or sample records."]
        if intent == "regulation":
            return ["Gather the cited standards and prepare an audit package with supporting evidence."]
        if intent == "failure":
            return ["Review the incident timeline, verify related alarms and sensors, and document the preventive recommendation."]
        return ["Review the ranked evidence, validate the findings with domain experts, and capture missing details."]

    def estimate_confidence(self, evidence: List[str], records: List[Dict[str, Any]], plan: Dict[str, Any]) -> float:
        base = 0.45 + min(0.4, 0.05 * max(0, len(evidence)))
        if records:
            base += 0.1
        if plan.get("use_internet"):
            base += 0.05
        return round(min(0.99, base), 2)

    def estimate_risk_level(self, evidence: List[str], plan: Dict[str, Any]) -> str:
        if not evidence:
            return "low"
        if plan.get("intent") in {"rca", "failure"}:
            return "high"
        if plan.get("intent") == "regulation":
            return "medium"
        return "medium"

    def related_assets(self, entities: List[str], evidence: List[str]) -> List[str]:
        assets = [entity for entity in entities if entity]
        for item in evidence:
            for token in re.findall(r"[A-Z][a-zA-Z0-9_\-]+", item):
                if token not in assets:
                    assets.append(token)
        return assets[:8]

    def related_documents(self, plan: Dict[str, Any], evidence: List[str]) -> List[str]:
        documents = ["Neo4j knowledge graph"]
        if plan.get("use_internet"):
            documents.append("Web search results")
        if evidence:
            documents.append("Ranked evidence snippets")
        return documents

    def _extract_key_findings(self, evidence: List[str]) -> List[str]:
        return [item for item in evidence[:3] if item]

    def _format_relation_evidence(self, source: Dict[str, Any], relation: Dict[str, Any], target: Dict[str, Any]) -> str:
        source_name = source.get("name") or source.get("canonical_name") or source.get("type") or "source"
        relation_type = relation.get("type") or relation.get("relation_type") or "related_to"
        target_name = target.get("name") or target.get("canonical_name") or target.get("type") or "target"
        source_labels = ",".join(source.get("labels", []))
        target_labels = ",".join(target.get("labels", []))
        return f"{source_name} ({source_labels}) {relation_type} {target_name} ({target_labels})"

    def _format_node_evidence(self, node: Dict[str, Any], labels: List[str]) -> str:
        name = node.get("name") or node.get("canonical_name") or node.get("type") or "node"
        description = node.get("description") or node.get("notes") or ""
        label_text = ",".join(labels)
        details = description.strip()
        return f"[{label_text}] {name}: {details}" if details else f"[{label_text}] {name}"

    def _format_generic_record(self, record: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key, value in record.items():
            if isinstance(value, dict):
                parts.append(self._format_node_evidence(value, value.get("labels", [])))
            else:
                parts.append(f"{key}: {value}")
        return " | ".join(parts)


__all__ = ["BaseAgent"]
