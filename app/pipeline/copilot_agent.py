"""
Industrial Copilot Agent - Qwen-based reasoning engine.
Handles RCA, maintenance recommendations, compliance checks, and structured LLM analysis.
"""

from typing import Any, Dict, List, Optional
from enum import Enum
import json
import re

from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle


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
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        rca_result = self.root_cause_analysis(entities, relations, text)
        maintenance_result = self.get_maintenance_plan(entities, relations)
        compliance_result = self.compliance_check(entities)
        risk_result = self.risk_assessment(entities, relations)
        llm_result = self._generate_llm_insights(entities, relations, text, query)

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
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = self._build_agent_prompt(entities, relations, text, query)
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

    def _query_llm(self, prompt: str) -> str:
        if not self._ensure_model():
            return "{\"summary\": \"LLM unavailable\", \"anomalies\": [], \"risks\": [], \"recommendations\": [], \"compliance\": [], \"confidence\": 0.0}"

        import torch
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
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
