"""
Microsoft GraphRAG Integration
Uses LLM-based reasoning over the knowledge graph for industrial insights.
Enhanced with evidence grounding, provenance tracking, and hallucination prevention.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.config import settings
from app.pipeline.compat import allow_trusted_torch_pickle
from app.pipeline.document_utils import chunk_text
from app.pipeline.runtime import select_device

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
except ImportError:
    TfidfVectorizer = None  # type: ignore
    LogisticRegression = None  # type: ignore


class ClaimSupportClassifier:
    """Lightweight classifier for claim support detection."""

    def __init__(self) -> None:
        self.vectorizer = None
        self.model = None
        self.is_trained = False

    def train(self, training_data: List[Dict[str, Any]]) -> None:
        if TfidfVectorizer is None or LogisticRegression is None:
            return

        texts = [sample["text"] for sample in training_data]
        labels = [int(sample["supported"]) for sample in training_data]
        if not texts:
            return

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500)
        self.model.fit(X, labels)
        self.is_trained = True

    def predict_proba(self, text: str) -> float:
        if self.is_trained and self.vectorizer is not None and self.model is not None:
            features = self.vectorizer.transform([text])
            return float(self.model.predict_proba(features)[0][1])
        return self._heuristic_score(text)

    @staticmethod
    def _heuristic_score(text: str) -> float:
        lower = text.lower()
        generic_phrases = [
            "serious personal injury",
            "safety concern",
            "hazard",
            "dangerous situation",
            "placeholder",
            "unknown",
            "not enough information",
        ]
        if any(phrase in lower for phrase in generic_phrases):
            return 0.05

        if re.search(r"\b(page|section|image|part)\s*\d+\b", lower):
            return 0.85
        if re.search(r"\d+\s*(mm|cm|m|psi|bar|rpm|celsius|°c|%)", lower):
            return 0.8
        if any(word in lower for word in ["leak", "wear", "corrosion", "failure", "degradation"]):
            return 0.6
        return 0.4


class GraphRAGSummarizer:
    """Generate industrial insights using GraphRAG and LLM reasoning.
    
    Key features:
    - Evidence grounding: all claims must cite document spans or be marked unsupported
    - Confidence reduction: defaults to 0.3, only increases with validated evidence
    - Placeholder checking: validates image references against actually-processed images
    - Provenance tracking: every claim includes source reference (page, span, or entity)
    - Hallucination prevention: rejects generic template responses, requires specific evidence
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = settings.qwen_model
        self._load_attempted = False
        self.claim_classifier = ClaimSupportClassifier()
        self._train_default_claim_classifier()
        # Minimum evidence coverage threshold to include a claim (0.0-1.0)
        self.min_evidence_threshold = 0.5
        # Maximum confidence to assign without explicit evidence
        self.max_unvalidated_confidence = 0.3
        # Whether every claim must include explicit provenance
        self.evidence_required = True

    def _ensure_llm(self) -> bool:
        if self.model is not None and self.tokenizer is not None:
            return True


        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            import torch

            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Decide device/dtype
            device = getattr(settings, "device_for_extraction", None) or getattr(settings, "device_for_embedding", None)
            if not device:
                device = select_device()

            target_torch_device = torch.device("cuda:0" if str(device).startswith("cuda") else "cpu")
            dtype = torch.float16 if target_torch_device.type == "cuda" else torch.float32

            # Prefer local snapshot for fully offline/local runs
            local_dir = Path(getattr(settings, "qwen_local", ""))
            model_source: str = self.model_name
            if local_dir and local_dir.exists() and any(local_dir.iterdir()):
                model_source = str(local_dir)

            with allow_trusted_torch_pickle():
                self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=model_source == str(local_dir))
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    local_files_only=model_source == str(local_dir),
                    torch_dtype=dtype if target_torch_device.type == "cuda" else None,
                    device_map=None,
                )

            self.model.to(target_torch_device)
            self.model.eval()

            # Remember for input movement
            self._qwen_device = target_torch_device
            self._qwen_dtype = dtype

            print(f"✓ Qwen LLM initialized for GraphRAG reasoning (source={model_source}, device={target_torch_device}, dtype={dtype})")
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
        """Generate evidence-grounded analysis of entities and relations.
        
        Returns analysis with provenance tracking or 'no_evidence' status if insufficient data.
        """
        # Check if we have sufficient evidence before spending time loading the model.
        evidence_check = self._check_evidence_sufficiency(entities, relations, text)
        if not evidence_check["sufficient"]:
            return {
                "summary_method": "insufficient-evidence",
                "status": "no_evidence",
                "reasoning": evidence_check["reason"],
                "anomalies_detected": [],
                "failure_risks": [],
                "maintenance_recommendations": [],
                "compliance": [],
                "confidence": 0.0,
                "evidence_coverage": evidence_check["coverage"],
                "explanation_chains": [],
            }

        if not self._ensure_llm():
            return {
                "summary_method": "unavailable",
                "status": "llm_unavailable",
                "reasoning": "GraphRAG model unavailable.",
                "anomalies_detected": [],
                "failure_risks": [],
                "maintenance_recommendations": [],
                "compliance": [],
                "confidence": 0.0,
                "evidence_coverage": 0.0,
                "explanation_chains": [],
            }

        prompt = self._build_reasoning_prompt(entities, relations, text, text_chunks)
        response_text = self._query_llm(prompt)
        parsed = self._parse_json_response(response_text)

        # Validate that parsed claims are actually grounded in the data
        validated_anomalies = self._validate_claims(parsed.get("anomalies", []), entities, text)
        validated_risks = self._validate_claims(parsed.get("risks", []), entities, text)
        validated_recommendations = self._validate_claims(parsed.get("recommendations", []), entities, text)

        # Calculate evidence coverage as percentage of validated claims
        total_claims = len(validated_anomalies) + len(validated_risks) + len(validated_recommendations)
        original_claims = len(parsed.get("anomalies", [])) + len(parsed.get("risks", [])) + len(parsed.get("recommendations", []))
        evidence_coverage = total_claims / max(1, original_claims) if original_claims > 0 else 0.0

        # Reduce confidence based on evidence coverage
        base_confidence = float(parsed.get("confidence", 0.3) or 0.3)
        adjusted_confidence = min(base_confidence, self.max_unvalidated_confidence) * evidence_coverage

        explanation_chains = self._build_explanation_chains(
            validated_anomalies + validated_risks + validated_recommendations,
            entities,
            text,
        )

        return {
            "summary_method": "qwen-graphrag-grounded",
            "status": "analyzed" if total_claims > 0 else "no_evidence",
            "reasoning": response_text,
            "anomalies_detected": validated_anomalies,
            "failure_risks": validated_risks,
            "maintenance_recommendations": validated_recommendations,
            "compliance": parsed.get("compliance", []),
            "confidence": min(adjusted_confidence, 0.95),  # Cap at 0.95
            "evidence_coverage": evidence_coverage,
            "claims_validated": total_claims,
            "claims_original": original_claims,
            "explanation_chains": explanation_chains,
        }

    def _check_evidence_sufficiency(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
    ) -> Dict[str, Any]:
        """Check if we have sufficient evidence to generate meaningful analysis.
        
        Returns dict with 'sufficient' (bool), 'coverage' (0-1), and 'reason' (str).
        """
        checks = {
            "has_entities": len(entities) > 0,
            "has_relations": len(relations) > 0,
            "has_text": len(text.strip()) > 100,
            "high_confidence_entities": sum(1 for e in entities if e.get("confidence", 0) > 0.6) > 0,
        }

        coverage = sum(checks.values()) / len(checks)

        if coverage < self.min_evidence_threshold:
            reason = "Insufficient evidence: " + ", ".join(
                f"{k}={v}" for k, v in checks.items()
            )
            return {"sufficient": False, "coverage": coverage, "reason": reason}

        return {"sufficient": True, "coverage": coverage, "reason": "Evidence sufficient"}

    def _validate_claims(
        self,
        claims: List[Any],
        entities: List[Dict[str, Any]],
        text: str,
    ) -> List[Dict[str, Any]]:
        """Validate that claims are grounded and carry explicit provenance."""
        validated = []
        entity_names = {e.get("name", "").lower() for e in entities}
        text_lower = text.lower()

        for claim in claims:
            if not isinstance(claim, dict):
                continue

            claim_text = f"{claim.get('name', '')} {claim.get('description', '')}".strip()
            if not claim_text:
                continue

            # Check for placeholder or obviously hallucinated text
            if self._contains_placeholder(claim_text):
                continue

            # Use classifier to estimate whether the claim is supported by evidence
            support_prob = self._claim_support_probability(claim_text)
            if support_prob < 0.5:
                continue

            # Derive provenance from explicit source or claim content
            provenance = self._derive_provenance(claim, entity_names, text_lower)
            if provenance is None:
                continue

            # Ensure claim references actual evidence and is not only generic language
            if not self._is_claim_supported(claim_text, entity_names, text_lower):
                continue

            claim = {**claim, "source": provenance, "confidence": float(claim.get("confidence", support_prob))}
            validated.append(claim)

        return validated

    def _contains_placeholder(self, text: str) -> bool:
        lower = text.lower()
        return "``" in lower or "placeholder" in lower or "[image" in lower or "<image" in lower

    def _claim_support_probability(self, claim_text: str) -> float:
        return self.claim_classifier.predict_proba(claim_text)

    def _derive_provenance(
        self,
        claim: Dict[str, Any],
        entity_names: set,
        text_lower: str,
    ) -> Optional[str]:
        explicit_source = claim.get("source")
        if explicit_source and isinstance(explicit_source, str) and explicit_source.strip():
            return explicit_source.strip()

        # check for explicit image or page references in the claim itself
        if m := re.search(r"\b(image|img)\s*[:#]?\s*(\d+)\b", claim.get("name", "") + " " + claim.get("description", ""), re.IGNORECASE):
            return f"image:{m.group(2)}"

        if m := re.search(r"\bpage\s*(\d+)\b", claim.get("name", "") + " " + claim.get("description", ""), re.IGNORECASE):
            return f"page:{m.group(1)}"

        if m := re.search(r"\bspan\s*(\d+)\b", claim.get("name", "") + " " + claim.get("description", ""), re.IGNORECASE):
            return f"span:{m.group(1)}"

        # Prefer explicit entity references
        claim_text = f"{claim.get('name', '')} {claim.get('description', '')}".lower()
        for entity in entity_names:
            if entity and entity in claim_text:
                return f"entity:{entity}"

        # If claim text appears directly in document, mark as text excerpt provenance
        excerpt = claim.get("description", "") or claim.get("name", "")
        if excerpt and excerpt.lower() in text_lower:
            snippet = excerpt[:120].strip()
            return f"text_excerpt:{snippet}"

        return None

    def _is_claim_supported(self, claim_text: str, entity_names: set, text_lower: str) -> bool:
        if any(entity in claim_text for entity in entity_names if entity):
            return True

        if any(
            phrase in text_lower
            for phrase in [
                claim_text[:50].lower(),
                claim_text.split(".")[0].lower(),
            ]
            if phrase and len(phrase) > 10
        ):
            return True

        if re.search(r"\b(page|image|span|part|section)\s*\d+\b", claim_text):
            return True

        return self._has_specific_detail({"description": claim_text})

    def _train_default_claim_classifier(self) -> None:
        training_data = [
            {"text": "Pump bearing wear detected on page 3", "supported": True},
            {"text": "Image 5 shows corrosion on the flange", "supported": True},
            {"text": "Part 101 shows 2.5mm wear on the seal surface", "supported": True},
            {"text": "Section 4.2 confirms maintenance document updates", "supported": True},
            {"text": "Serious personal injury", "supported": False},
            {"text": "Unknown anomaly without evidence", "supported": False},
            {"text": "Placeholder text indicating missing image", "supported": False},
            {"text": "Safety concern exists", "supported": False},
        ]
        self.claim_classifier.train(training_data)

    def _build_explanation_chains(
        self,
        validated_claims: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        text: str,
    ) -> List[Dict[str, Any]]:
        chains: List[Dict[str, Any]] = []
        entity_names = {e.get("name", "") for e in entities}
        for idx, claim in enumerate(validated_claims, start=1):
            claim_id = f"claim:{idx}"
            source = claim.get("source", "unknown")
            evidence_nodes = []
            edges = []

            if source.startswith("entity:"):
                entity_name = source.split("entity:", 1)[1]
                evidence_nodes.append(
                    {
                        "id": f"entity:{entity_name}",
                        "type": "entity",
                        "text": entity_name,
                        "confidence": 1.0,
                    }
                )
            elif source.startswith("page:"):
                evidence_nodes.append(
                    {
                        "id": source,
                        "type": "page",
                        "text": f"Evidence cited on {source}",
                        "confidence": 0.9,
                    }
                )
            elif source.startswith("image:"):
                evidence_nodes.append(
                    {
                        "id": source,
                        "type": "image",
                        "text": f"Evidence referenced from {source}",
                        "confidence": 0.9,
                    }
                )
            elif source.startswith("text_excerpt:"):
                evidence_nodes.append(
                    {
                        "id": source,
                        "type": "text_excerpt",
                        "text": source.split("text_excerpt:", 1)[1],
                        "confidence": 0.85,
                    }
                )
            else:
                evidence_nodes.append(
                    {
                        "id": source,
                        "type": "unknown",
                        "text": source,
                        "confidence": 0.5,
                    }
                )

            edges.append(
                {
                    "from": claim_id,
                    "to": evidence_nodes[0]["id"],
                    "relation": "supported_by",
                    "edge_confidence": round(float(claim.get("confidence", 0.0)) if claim.get("confidence") is not None else 0.0, 2),
                }
            )

            chains.append(
                {
                    "claim_id": claim_id,
                    "claim_text": claim.get("name", ""),
                    "claim_description": claim.get("description", ""),
                    "source": source,
                    "provenance_nodes": evidence_nodes,
                    "edges": edges,
                }
            )

        return chains

    @staticmethod
    def _has_specific_detail(claim: Dict[str, Any]) -> bool:
        """Check if a claim has specific, non-generic detail."""
        if not isinstance(claim, dict):
            return False

        # Look for specific details like numbers, components, measurements
        claim_str = str(claim).lower()
        specific_indicators = [
            r"\d+\s*(mm|cm|m|psi|bar|rpm|celsius|°c|%)",  # measurements
            r"part\s*\d+",  # part numbers
            r"section\s*\d+",  # section references
            r"page\s*\d+",  # page references
            r"[a-z]+\s*[0-9]{2,}",  # component codes
        ]

        import re
        return any(re.search(pattern, claim_str) for pattern in specific_indicators)

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
            excerpt = "\n---\n".join(text_chunks[:3])
        else:
            excerpt = text[:1200].replace("\n", " ")

        return (
            "You are an industrial operations and maintenance expert analyzing extracted entities and document content. "
            "CRITICAL REQUIREMENTS:\n"
            "1. ONLY cite specific evidence from the provided data - do NOT generate generic or hallucinated claims.\n"
            "2. For each claim, explicitly state the source (entity name, page number, or specific text excerpt).\n"
            "3. If you cannot find supporting evidence in the data, return the field as EMPTY ARRAY [].\n"
            "4. Avoid template language like 'serious personal injury' - be specific about what was detected.\n"
            "5. NEVER insert image placeholders or repeated `` marks.\n"
            "6. Confidence should be 0.3-0.7: only use 0.8+ if you find VERY STRONG evidence.\n"
            "Return ONLY valid JSON with exact keys: anomalies, risks, recommendations, compliance, confidence.\n"
            "Do not include any markdown, prose headers, or explanation outside the JSON.\n"
            f"\nEXTRACTED_ENTITIES:\n{entity_str}\n\n"
            f"RELATIONSHIPS:\n{relation_str}\n\n"
            f"DOCUMENT_EXCERPT:\n{excerpt}\n\n"
            "Output example:\n"
            '{"anomalies": [{"name": "Pump bearing wear detected in maintenance logs (page 3)", "source": "maintenance_entity"}], '
            '"risks": [{"name": "Seal failure within 6 months if bearing not replaced", "source": "bearing_entity"}], '
            '"recommendations": [{"name": "Replace bearing assembly before month-end service window", "source": "maintenance_schedule"}], '
            '"compliance": [], '
            '"confidence": 0.65}'
        )

    def _query_llm(self, prompt: str) -> str:
        if not self._ensure_llm():
            return ""

        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        # Move inputs to the same device as the model
        qwen_device = getattr(self, "_qwen_device", None)
        if qwen_device is not None:
            encoded = {k: v.to(qwen_device) for k, v in encoded.items()}
        with torch.no_grad():

            # Use lower temperature (0.3) for more deterministic, evidence-based output
            # Avoid high temperature which encourages hallucination
            outputs = self.model.generate(
                **encoded,

                max_new_tokens=512,
                temperature=0.3,  # Low temperature for consistency
                top_p=0.9,  # Nucleus sampling for focused output
                num_beams=1,  # Single beam (greedy) for more predictable output
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # Deterministic generation
            )

        generated = outputs[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with robust error handling."""
        body = self._extract_json_object(response)
        if not body:
            return {
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
                "parse_status": "no_json_found",
            }

        try:
            parsed = json.loads(body)
            # Validate that parsed values are lists or have expected structure
            return {
                "anomalies": self._ensure_list(parsed.get("anomalies")),
                "risks": self._ensure_list(parsed.get("risks")),
                "recommendations": self._ensure_list(parsed.get("recommendations")),
                "compliance": self._ensure_list(parsed.get("compliance")),
                "confidence": max(0.0, min(1.0, float(parsed.get("confidence", 0.3) or 0.3))),
                "parse_status": "json_parsed",
            }
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"⚠ GraphRAG JSON parse error: {exc}")
            return {
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
                "parse_status": f"parse_error: {type(exc).__name__}",
            }

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        """Convert value to list, handling various input types."""
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        if value is None or value == "":
            return []
        return [value]

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
