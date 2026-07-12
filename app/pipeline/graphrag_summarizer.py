"""
Microsoft GraphRAG Integration
Uses LLM-based reasoning over the knowledge graph for industrial insights.
"""

import json
import re
from typing import Any, Dict, List, Optional
from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle
from app.pipeline.document_utils import chunk_text


class GraphRAGSummarizer:
    """Generate industrial insights using GraphRAG and LLM reasoning."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = settings.qwen_model
        self._load_attempted = False

    def _ensure_llm(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True

        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            with allow_trusted_torch_pickle():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            print("✓ Qwen LLM initialized for GraphRAG reasoning")
            return True
        except Exception as exc:
            print(f"✗ GraphRAG Qwen initialization failed: {exc}")
            self.model = None
            self.tokenizer = None
            return False

    def generate_summary(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        text_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self._ensure_llm():
            return {
                "summary_method": "unavailable",
                "reasoning": "GraphRAG model unavailable.",
                "anomalies_detected": [],
                "failure_risks": [],
                "maintenance_recommendations": [],
                "compliance": [],
                "confidence": 0.0,
            }

        prompt = self._build_reasoning_prompt(entities, relations, text, text_chunks)
        response_text = self._query_llm(prompt)
        parsed = self._parse_json_response(response_text)

        return {
            "summary_method": "qwen-graphrag",
            "reasoning": response_text,
            "anomalies_detected": parsed.get("anomalies", []),
            "failure_risks": parsed.get("risks", []),
            "maintenance_recommendations": parsed.get("recommendations", []),
            "compliance": parsed.get("compliance", []),
            "confidence": parsed.get("confidence", 0.0) or 0.9,
        }

    def _build_reasoning_prompt(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        text_chunks: Optional[List[str]] = None,
    ) -> str:
        entity_str = "\n".join(
            f"- {e['name']} (type: {e.get('entity_type', 'unknown')}, confidence: {e.get('confidence', 0.0)})"
            for e in entities[:12]
        )
        relation_str = "\n".join(
            f"- {r.get('source')} {r.get('relation_type')} {r.get('target')}"
            for r in relations[:12]
        )
        if text_chunks:
            excerpt = " \n---\n ".join(text_chunks[:3])
        else:
            excerpt = text[:1200].replace("\n", " ")

        return (
            "You are an industrial operations and maintenance expert. Analyze the extracted entities, relations, and document excerpt. "
            "Return ONLY valid JSON with the exact keys: anomalies, risks, recommendations, compliance, confidence. "
            "Do not include any markdown, prose headers, or explanation outside the JSON object."
            f"\n\nEXTRACTED_ENTITIES:\n{entity_str}\n\n"
            f"RELATIONSHIPS:\n{relation_str}\n\n"
            f"DOCUMENT_EXCERPT:\n{excerpt}\n\n"
            "Output format example: {\"anomalies\": [], \"risks\": [], \"recommendations\": [], \"compliance\": [], \"confidence\": 0.0}"
        )

    def _query_llm(self, prompt: str) -> str:
        if not self._ensure_llm():
            return ""

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

        generated = outputs[0][encoded["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        body = self._extract_json_object(response)
        if not body:
            return {
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
            }

        try:
            parsed = json.loads(body)
            return {
                "anomalies": parsed.get("anomalies", []),
                "risks": parsed.get("risks", []),
                "recommendations": parsed.get("recommendations", []),
                "compliance": parsed.get("compliance", []),
                "confidence": parsed.get("confidence", 0.0),
            }
        except json.JSONDecodeError:
            return {
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
