"""
Industrial Copilot Agent - Qwen-based reasoning engine.
Handles RCA, maintenance recommendations, compliance checks, and structured LLM analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json
import re

from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None  # type: ignore
    LogisticRegression = None  # type: ignore
    cosine_similarity = None  # type: ignore


class FailureMode(Enum):
    """Common industrial failure modes."""
    CAVITATION = "cavitation"
    CORROSION = "corrosion"
    FOULING = "fouling"
    SEAL_FAILURE = "seal_failure"
    BEARING_WEAR = "bearing_wear"
    VIBRATION_EXCESSIVE = "vibration_excessive"
    TEMPERATURE_DEVIATION = "temperature_deviation"
    PRESSURE_DEVIATION = "pressure_deviation"


class EvidenceCritic:
    """Verify that a claim is supported by retrieved evidence chunks."""

    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    def train(self, training_data: List[Dict[str, Any]]) -> None:
        if TfidfVectorizer is None or LogisticRegression is None:
            return

        texts = [f"{sample['claim']} {sample['evidence']}" for sample in training_data]
        labels = [int(sample["supported"]) for sample in training_data]
        if not texts:
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict_proba(self, claim: str, evidence: str) -> float:
        if self.is_trained and self.vectorizer is not None and self.model is not None:
            text = f"{claim} {evidence}"
            features = self.vectorizer.transform([text])
            return float(self.model.predict_proba(features)[0][1])
        return self._heuristic_score(claim, evidence)

    @staticmethod
    def _heuristic_score(claim: str, evidence: str) -> float:
        claim_lower = claim.lower()
        evidence_lower = evidence.lower()
        if any(phrase in claim_lower for phrase in ["serious personal injury", "safety concern", "unknown anomaly"]):
            return 0.05
        if any(term in evidence_lower for term in ["wear", "corrosion", "leak", "failure", "overpressure"]):
            return 0.7
        overlap = len(set(claim_lower.split()) & set(evidence_lower.split()))
        return min(1.0, max(0.0, 0.2 + 0.05 * overlap))


class SemanticRetriever:
    """Lite semantic index for retrieval-augmented generation."""

    def __init__(self) -> None:
        self.vectorizer = None
        self.matrix = None
        self.chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        self.chunks = chunks
        if TfidfVectorizer is None:
            return
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        if self.vectorizer is None or self.matrix is None or cosine_similarity is None:
            return self._lexical_retrieve(query, top_k)

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.matrix)[0]
            ranked = sorted(
                enumerate(similarities), key=lambda item: item[1], reverse=True
            )[:top_k]
            return [
                {"id": f"chunk:{idx+1}", "text": self.chunks[idx], "score": float(score)}
                for idx, score in ranked
                if score > 0.0
            ]
        except Exception:
            return self._lexical_retrieve(query, top_k)

    def _lexical_retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        scored = []
        for idx, chunk in enumerate(self.chunks):
            score = sum(1 for token in query_lower.split() if token in chunk.lower())
            scored.append((idx, float(score)))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            {"id": f"chunk:{idx+1}", "text": self.chunks[idx], "score": score}
            for idx, score in ranked
            if score > 0.0
        ]


class MaintenanceType(Enum):
    """Types of maintenance interventions."""
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


class IndustrialCopilotAgent:
    """AI agent for industrial system analysis and decision support."""

    def __init__(self, load_model: bool = False) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = settings.qwen_model
        self._load_attempted = False
        self.retriever = SemanticRetriever()
        self.critic = EvidenceCritic()
        self._train_default_critic()

        if load_model:
            self._initialize()

    def _ensure_model(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True

        if self._load_attempted:
            return False

        return self._initialize()

    def _initialize(self) -> bool:
        self._load_attempted = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            with allow_trusted_torch_pickle():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            print("✓ Industrial Copilot Agent Qwen LLM initialized")
            return True
        except Exception as exc:
            print(f"✗ Industrial Copilot initialization failed: {exc}")
            self.model = None
            self.tokenizer = None
            return False

    def reason(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        text_chunks: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        rca_result = self.root_cause_analysis(entities, relations, text)
        maintenance_result = self.get_maintenance_plan(entities, relations)
        compliance_result = self.compliance_check(entities)
        risk_result = self.risk_assessment(entities, relations)
        llm_result = self._generate_llm_insights(entities, relations, text, text_chunks, query)

        return {
            "agent": "industrial-copilot",
            "reasoning_chain": {
                "root_cause_analysis": rca_result,
                "maintenance_plan": maintenance_result,
                "compliance_status": compliance_result,
                "risk_assessment": risk_result,
                "llm_insights": llm_result,
            },
            "executive_summary": llm_result.get(
                "summary",
                self._build_summary(rca_result, maintenance_result, compliance_result),
            ),
            "confidence": llm_result.get("confidence", 0.0) or 0.75,
        }

    def root_cause_analysis(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
    ) -> Dict[str, Any]:
        failure_indicators = self._detect_failure_indicators(entities, text)
        failure_chains = self._trace_failure_chains(entities, relations, failure_indicators)

        rca = {
            "failure_indicators": failure_indicators,
            "root_causes": [],
            "contributing_factors": [],
            "failure_chain": failure_chains,
        }

        if failure_indicators:
            rca["root_causes"] = self._identify_root_causes(failure_indicators)
            rca["contributing_factors"] = self._identify_factors(entities, failure_indicators)

        return rca

    def get_maintenance_plan(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        equipment = [e for e in entities if e.get("entity_type") == "equipment"]

        plan = {
            "maintenance_type": MaintenanceType.PREVENTIVE.value,
            "priority_equipment": [],
            "schedule": [],
            "estimated_cost": 0,
            "estimated_downtime_hours": 0,
        }

        for eq in equipment[:5]:
            criticality = self._assess_criticality(eq, relations)
            interval = self._calculate_interval(eq, criticality)
            plan["priority_equipment"].append(
                {
                    "equipment": eq["name"],
                    "priority": criticality,
                    "interval_days": interval,
                    "tasks": self._generate_tasks(eq, criticality),
                }
            )

        plan["schedule"] = self._generate_schedule(plan["priority_equipment"])
        return plan

    def compliance_check(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        compliance_status = {
            "iso_55000_asset_management": self._check_asset_management(entities),
            "osha_safety": self._check_safety_compliance(entities),
            "environmental": self._check_environmental_compliance(entities),
            "overall_compliance_score": 0.0,
            "violations": [],
            "recommendations": [],
        }

        scores = [
            compliance_status["iso_55000_asset_management"]["score"],
            compliance_status["osha_safety"]["score"],
            compliance_status["environmental"]["score"],
        ]
        compliance_status["overall_compliance_score"] = sum(scores) / len(scores)

        if compliance_status["overall_compliance_score"] < 0.8:
            compliance_status["recommendations"].append(
                "Strengthen asset management documentation and safety checks."
            )

        return compliance_status

    def risk_assessment(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        risks: List[Dict[str, Any]] = []
        equipment = [e for e in entities if e.get("entity_type") == "equipment"]

        for eq in equipment:
            score = self._calculate_risk_score(eq, relations)
            if score > 0.6:
                risks.append(
                    {
                        "equipment": eq["name"],
                        "risk_score": score,
                        "risk_category": self._categorize_risk(score),
                        "mitigation_strategy": self._recommend_mitigation(eq),
                    }
                )

        average_score = sum(r["risk_score"] for r in risks) / max(len(risks), 1)
        return {
            "total_risk_score": average_score,
            "identified_risks": sorted(risks, key=lambda r: r["risk_score"], reverse=True),
            "risk_level": self._assess_risk_level(average_score),
        }

    def _detect_failure_indicators(self, entities: List[Dict[str, Any]], text: str) -> List[str]:
        indicators: List[str] = []
        failure_keywords = [
            "cavitation",
            "corrosion",
            "fouling",
            "seal",
            "bearing",
            "vibration",
            "temperature",
            "pressure",
            "alarm",
            "fault",
            "degradation",
            "wear",
            "leakage",
            "inefficiency",
        ]
        text_lower = text.lower()
        for keyword in failure_keywords:
            if keyword in text_lower:
                indicators.append(keyword)

        return list(dict.fromkeys(indicators))

    def _trace_failure_chains(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        indicators: List[str],
    ) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        for relation in relations:
            relation_text = str(relation).lower()
            if any(ind in relation_text for ind in indicators):
                chains.append(
                    {
                        "from": relation.get("source"),
                        "to": relation.get("target"),
                        "relation": relation.get("relation_type"),
                    }
                )
        return chains

    def _identify_root_causes(self, indicators: List[str]) -> List[str]:
        cause_map = {
            "cavitation": "low inlet pressure or high outlet pressure",
            "corrosion": "material incompatibility or corrosive fluid exposure",
            "fouling": "sediment buildup or filter bypass",
            "vibration": "shaft imbalance or misalignment",
            "temperature": "cooling failure or excessive friction",
        }
        return [cause_map[key] for key in indicators if key in cause_map]

    def _identify_factors(self, entities: List[Dict[str, Any]], indicators: List[str]) -> List[str]:
        factors: List[str] = []
        for entity in entities:
            name = entity.get("name", "").lower()
            for indicator in indicators:
                if indicator in name:
                    factors.append(f"{entity.get('name')} is associated with {indicator}")
        return list(dict.fromkeys(factors))

    def _assess_criticality(self, equipment: Dict[str, Any], relations: List[Dict[str, Any]]) -> str:
        dependency_count = sum(
            1
            for r in relations
            if str(r.get("source", "")).lower() == str(equipment.get("name", "")).lower()
        )
        if dependency_count > 3:
            return "critical"
        if dependency_count > 1:
            return "high"
        return "medium"

    def _calculate_interval(self, equipment: Dict[str, Any], criticality: str) -> int:
        intervals = {"critical": 7, "high": 30, "medium": 90}
        return intervals.get(criticality, 90)

    def _generate_tasks(self, equipment: Dict[str, Any], criticality: str) -> List[str]:
        eq_name = str(equipment.get("name", "")).lower()
        tasks = ["Visual inspection", "Performance testing"]
        if "pump" in eq_name:
            tasks.extend(["Check inlet pressure", "Verify outlet flow"])
        if "valve" in eq_name:
            tasks.extend(["Test sealing integrity", "Check actuation"])
        if "motor" in eq_name:
            tasks.extend(["Inspect vibration", "Measure insulation resistance"])
        return list(dict.fromkeys(tasks))

    def _generate_schedule(self, equipment_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        schedule: List[Dict[str, Any]] = []
        for index, eq in enumerate(equipment_list):
            schedule.append(
                {
                    "order": index + 1,
                    "equipment": eq["equipment"],
                    "planned_date": f"Day {eq['interval_days']}",
                    "estimated_duration_hours": 2 if eq["priority"] == "critical" else 1,
                }
            )
        return schedule

    def _check_asset_management(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        equipment_count = len([e for e in entities if e.get("entity_type") == "equipment"])
        score = min(0.9, equipment_count / 10)
        return {
            "standard": "ISO 55000",
            "score": score,
            "status": "compliant" if score > 0.7 else "non-compliant",
        }

    def _check_safety_compliance(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"standard": "OSHA", "score": 0.85, "status": "compliant"}

    def _check_environmental_compliance(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"standard": "Environmental", "score": 0.80, "status": "compliant"}

    def _calculate_risk_score(self, equipment: Dict[str, Any], relations: List[Dict[str, Any]]) -> float:
        return min(1.0, 0.5 * float(equipment.get("confidence", 0.7)))

    def _categorize_risk(self, score: float) -> str:
        if score > 0.8:
            return "critical"
        if score > 0.6:
            return "high"
        return "medium"

    def _recommend_mitigation(self, equipment: Dict[str, Any]) -> str:
        return f"Increase monitoring and implement predictive maintenance for {equipment.get('name', 'equipment')}"

    def _assess_risk_level(self, average_score: float) -> str:
        if average_score > 0.8:
            return "critical"
        if average_score > 0.6:
            return "high"
        return "medium"

    def vision_language_analysis(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        layout: Optional[List[Dict[str, Any]]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        reading_order: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_agent_prompt(
            entities,
            relations,
            text,
            query="Perform a vision-language interpretation of the document, including layout structure, tables, and reading order.",
            layout=layout,
            tables=tables,
            reading_order=reading_order,
        )
        response_text = self._query_llm(prompt)
        parsed = self._parse_json_response(response_text)
        return {
            "summary": parsed.get("summary", response_text),
            "anomalies": parsed.get("anomalies", []),
            "risks": parsed.get("risks", []),
            "recommendations": parsed.get("recommendations", []),
            "compliance": parsed.get("compliance", []),
            "confidence": parsed.get("confidence", 0.0),
            "raw_response": response_text,
        }

    def _generate_llm_insights(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        text_chunks: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        evidence_query = query or "Analyze the system and identify anomalies, risks, and maintenance recommendations."
        chunks = text_chunks or []
        self.retriever.build(chunks)
        retrieved_chunks = self._retrieve_candidate_evidence(evidence_query, chunks)

        prompt = self._build_rag_prompt(entities, relations, retrieved_chunks, evidence_query)
        response_text = self._query_llm(prompt)
        parsed = self._parse_json_response(response_text)

        processed = self._post_process_agent_output(parsed, retrieved_chunks)
        processed["raw_response"] = response_text
        return processed

    def _build_agent_prompt(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        query: Optional[str] = None,
        layout: Optional[List[Dict[str, Any]]] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        reading_order: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        entity_str = "\n".join(
            f"- {e.get('name', '')} (type: {e.get('entity_type', 'unknown')})"
            for e in entities[:12]
        )
        relation_str = "\n".join(
            f"- {r.get('source', '')} {r.get('relation_type', '')} {r.get('target', '')}"
            for r in relations[:12]
        )
        excerpt = text[:1000].replace("\n", " ")
        query_text = f"\nUSER QUERY: {query}" if query else ""
        layout_text = "\nLAYOUT STRUCTURE:\n" + str(layout[:6]) if layout else ""
        tables_text = "\nTABLES:\n" + str(tables[:4]) if tables else ""
        reading_text = "\nREADING ORDER:\n" + str(reading_order[:12]) if reading_order else ""
        return (
            "You are an industrial operations expert. Analyze the extracted entities, relations, document excerpt, and document structure then return ONLY valid JSON. "
            "Provide a JSON object with keys: summary, anomalies, risks, recommendations, compliance, confidence."
            f"\n\nENTITIES:\n{entity_str}\n\n"
            f"RELATIONS:\n{relation_str}\n\n"
            f"DOCUMENT_EXCERPT:\n{excerpt}\n"
            f"{layout_text}{tables_text}{reading_text}{query_text}\n"
            "Do not provide markdown or any commentary outside the JSON object."
        )

    def _build_rag_prompt(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        query: str,
    ) -> str:
        entity_str = "\n".join(
            f"- {e.get('name', '')} (type: {e.get('entity_type', 'unknown')})"
            for e in entities[:12]
        )
        relation_str = "\n".join(
            f"- {r.get('source', '')} {r.get('relation_type', '')} {r.get('target', '')}"
            for r in relations[:12]
        )
        chunk_str = "\n".join(
            f"{chunk['id']}: {chunk['text']}"
            for chunk in retrieved_chunks[:6]
        )
        return (
            "You are an industrial operations and maintenance expert. Use only the candidate evidence chunks provided below to answer the query. "
            "If a claim cannot be supported with a referenced chunk, do not include it. "
            "For every high-impact claim, include a 'provenance' field referencing the chunk id or a page/image span. "
            "Do not hallucinate or invent unsupported actions. Return valid JSON only. "
            "Use a structured schema with claim id, type, text, provenance, confidence, and verification_score."
            f"\n\nQUERY:\n{query}\n\n"
            f"ENTITIES:\n{entity_str}\n\n"
            f"RELATIONS:\n{relation_str}\n\n"
            f"CANDIDATE_EVIDENCE_CHUNKS:\n{chunk_str}\n\n"
            "Output a JSON object with keys: summary, anomalies, risks, recommendations, compliance, confidence, structured_claims. "
            "Each recommendation must include provenance and a verification_score."
        )

    def _retrieve_candidate_evidence(self, query: str, chunks: List[str]) -> List[Dict[str, Any]]:
        if not chunks:
            return []
        if not self.retriever.chunks or self.retriever.chunks != chunks:
            self.retriever.build(chunks)
        results = self.retriever.retrieve(query, top_k=6)
        if not results and chunks:
            # fallback: include the first few chunks if retrieval failed
            return [{"id": f"chunk:{i+1}", "text": chunk, "score": 0.0} for i, chunk in enumerate(chunks[:3])]
        return results

    def _derive_provenance_from_chunk(self, item: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> Optional[str]:
        if item.get("provenance"):
            return item.get("provenance")
        text = f"{item.get('name', '')} {item.get('description', '')}".lower()
        for chunk in retrieved_chunks:
            if chunk["text"].lower() in text or any(term in chunk["text"].lower() for term in text.split() if len(term) > 4):
                return chunk["id"]
        return None

    def _is_high_impact_claim(self, item: Dict[str, Any]) -> bool:
        text = f"{item.get('name', '')} {item.get('description', '')}".lower()
        return any(keyword in text for keyword in ["critical", "urgent", "immediate", "high priority", "safety", "failure"])

    def _post_process_agent_output(
        self,
        parsed: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        anomalies = self._dedupe_items(parsed.get("anomalies", []))
        risks = self._dedupe_items(parsed.get("risks", []))
        recommendations = self._dedupe_items(parsed.get("recommendations", []))

        annotations = []
        human_review_queue = []
        structured_claims = []

        for item_type, items in [
            ("anomaly", anomalies),
            ("risk", risks),
            ("recommendation", recommendations),
        ]:
            for item in items:
                if not isinstance(item, dict):
                    continue
                provenance = self._derive_provenance_from_chunk(item, retrieved_chunks)
                if item_type == "recommendation" and not provenance:
                    continue
                item["provenance"] = provenance
                verification_score = self._critic_verify_claim(item, retrieved_chunks)
                item["verification_score"] = round(verification_score, 2)
                item["verified"] = verification_score >= 0.5
                item["impact"] = "critical" if self._is_high_impact_claim(item) else "normal"

                if item["impact"] == "critical" and not item["provenance"]:
                    human_review_queue.append(
                        {
                            "claim": item.get("name", ""),
                            "issue": "Missing provenance for high-impact claim",
                            "required_action": "Verify the recommendation before execution.",
                        }
                    )
                    continue

                if item["impact"] == "critical" and item["verification_score"] < 0.7:
                    human_review_queue.append(
                        {
                            "claim": item.get("name", ""),
                            "issue": "Low verification score for high-impact claim",
                            "verification_score": item["verification_score"],
                        }
                    )

                structured_claims.append(
                    {
                        "id": f"claim:{len(structured_claims)+1}",
                        "type": item_type,
                        "text": item.get("name", ""),
                        "description": item.get("description", ""),
                        "provenance": item["provenance"],
                        "verification_score": item["verification_score"],
                        "verified": item["verified"],
                        "impact": item["impact"],
                        "confidence": float(item.get("confidence", parsed.get("confidence", 0.0)) or 0.0),
                    }
                )

        explanation_chains = self._build_chain_of_evidence(structured_claims, retrieved_chunks)
        return {
            "summary": parsed.get("summary", ""),
            "anomalies": anomalies,
            "risks": risks,
            "recommendations": recommendations,
            "compliance": parsed.get("compliance", []),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "structured_claims": structured_claims,
            "evidence_retrieval": retrieved_chunks,
            "human_review_queue": human_review_queue,
            "explanation_chains": explanation_chains,
        }

    def _dedupe_items(self, items: List[Any]) -> List[Dict[str, Any]]:
        unique = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            text = f"{item.get('name', '')} {item.get('description', '')}".lower().strip()
            normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text))
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(item)
        return unique

    def _critic_verify_claim(self, claim: Dict[str, Any], retrieved_chunks: List[Dict[str, Any]]) -> float:
        text = f"{claim.get('name', '')} {claim.get('description', '')}".strip()
        best_score = 0.0
        for chunk in retrieved_chunks:
            score = self.critic.predict_proba(text, chunk["text"])
            best_score = max(best_score, score)
        return best_score

    def _build_chain_of_evidence(
        self,
        structured_claims: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        for claim in structured_claims:
            evidence_nodes = []
            if claim["provenance"]:
                evidence_nodes.append(
                    {
                        "id": claim["provenance"],
                        "type": "evidence_chunk" if claim["provenance"].startswith("chunk:") else "provenance",
                        "confidence": claim["verification_score"],
                        "text": next(
                            (chunk["text"] for chunk in retrieved_chunks if chunk["id"] == claim["provenance"]),
                            claim["provenance"],
                        ),
                    }
                )
            chains.append(
                {
                    "claim_id": claim["id"],
                    "claim_text": claim["text"],
                    "provenance_nodes": evidence_nodes,
                    "edges": [
                        {
                            "from": claim["id"],
                            "to": evidence_nodes[0]["id"] if evidence_nodes else "unknown",
                            "relation": "supported_by",
                            "edge_confidence": claim["verification_score"],
                        }
                    ] if evidence_nodes else [],
                }
            )
        return chains

    def _train_default_critic(self) -> None:
        training_data = [
            {
                "claim": "Pump bearing wear detected",
                "evidence": "Pump A shows 2.5mm bearing wear on page 4",
                "supported": True,
            },
            {
                "claim": "Image 5 shows corrosion on the flange",
                "evidence": "Flange corrosion is visible in the inspection report",
                "supported": True,
            },
            {
                "claim": "Serious personal injury",
                "evidence": "No specific evidence provided",
                "supported": False,
            },
            {
                "claim": "Unknown anomaly without evidence",
                "evidence": "Document does not provide supporting details",
                "supported": False,
            },
        ]
        self.critic.train(training_data)

    def _query_llm(self, prompt: str) -> str:
        if not self._ensure_model():
            return "{\"summary\": \"LLM unavailable\", \"anomalies\": [], \"risks\": [], \"recommendations\": [], \"compliance\": [], \"confidence\": 0.0}"

        import torch
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                num_beams=1,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )
        response_ids = outputs[0][encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        body = self._extract_json_object(response)
        if not body:
            return {
                "summary": response,
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
            }

        try:
            parsed = json.loads(body)
            return {
                "summary": parsed.get("summary", response),
                "anomalies": parsed.get("anomalies", []),
                "risks": parsed.get("risks", []),
                "recommendations": parsed.get("recommendations", []),
                "compliance": parsed.get("compliance", []),
                "confidence": parsed.get("confidence", 0.0),
            }
        except json.JSONDecodeError:
            return {
                "summary": response,
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
            }

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        if start < 0:
            return ""
        depth = 0
        for index, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

    def _build_summary(
        self,
        rca: Dict[str, Any],
        maintenance: Dict[str, Any],
        compliance: Dict[str, Any],
    ) -> str:
        return (
            "INDUSTRIAL SYSTEM ANALYSIS SUMMARY\n"
            f"Root Causes: {', '.join(rca.get('root_causes', ['None identified']))}\n"
            f"Priority Equipment: {len(maintenance.get('priority_equipment', []))}\n"
            f"Compliance Score: {compliance.get('overall_compliance_score', 0):.1%}\n"
            "Execute recommended maintenance and safety actions to improve reliability."
        )
