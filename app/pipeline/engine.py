import asyncio
import re
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from app.config import settings
from app.pipeline.models import detect_pid_components, get_model_stage_manifest
from app.storage import create_job, update_job

# Import real models
try:
    from app.pipeline.ocr_processor import DoclingOCRProcessor
except ImportError:
    DoclingOCRProcessor = None

try:
    from app.pipeline.entity_extractor import GlinerEntityExtractor
except ImportError:
    GlinerEntityExtractor = None

try:
    from app.pipeline.relation_extractor import RebelRelationExtractor
except ImportError:
    RebelRelationExtractor = None

try:
    from app.pipeline.neo4j_store import Neo4jGraphStore
except ImportError:
    Neo4jGraphStore = None

try:
    from app.pipeline.graphrag_summarizer import GraphRAGSummarizer
except ImportError:
    GraphRAGSummarizer = None

try:
    from app.pipeline.copilot_agent import IndustrialCopilotAgent
except ImportError:
    IndustrialCopilotAgent = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class IndustrialGraphPipeline:
    def __init__(self) -> None:
        self.ocr_processor = None
        self.entity_extractor = None
        self.relation_extractor = None
        self.graph_store = None
        self.rag_summarizer = None
        self.copilot_agent = None
        self.yolo_model = None
        self.embedding_model = None
        self.model_mode = "initializing"
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all model components"""
        
        # OCR processor
        if DoclingOCRProcessor is not None:
            try:
                self.ocr_processor = DoclingOCRProcessor()
                print("✓ Docling OCR processor loaded")
            except Exception as e:
                print(f"✗ Docling OCR failed: {e}")
        
        # Entity extraction
        if GlinerEntityExtractor is not None:
            try:
                self.entity_extractor = GlinerEntityExtractor()
                print("✓ GLiNER entity extractor loaded")
                self.model_mode = "local-transformers"
            except Exception as e:
                print(f"✗ GLiNER failed: {e}")
        
        # Relation extraction
        if RebelRelationExtractor is not None:
            try:
                self.relation_extractor = RebelRelationExtractor()
                if not getattr(self.relation_extractor, "is_ready", False):
                    raise RuntimeError("Relation extractor did not initialize")
                print("✓ GLiREL relation extractor loaded")
            except Exception as e:
                self.relation_extractor = None
                print(f"✗ Relation extractor failed: {e}")
        
        # Neo4j store
        if Neo4jGraphStore is not None:
            try:
                self.graph_store = Neo4jGraphStore()
                print("✓ Neo4j graph store initialized")
            except Exception as e:
                print(f"✗ Neo4j store failed: {e}")
        
        # GraphRAG
        if GraphRAGSummarizer is not None:
            try:
                self.rag_summarizer = GraphRAGSummarizer()
                print("✓ GraphRAG summarizer loaded")
            except Exception as e:
                print(f"✗ GraphRAG failed: {e}")
        
        # Copilot agent
        if IndustrialCopilotAgent is not None:
            try:
                self.copilot_agent = IndustrialCopilotAgent()
                print("✓ Industrial Copilot Agent loaded")
            except Exception as e:
                print(f"✗ Copilot Agent failed: {e}")
        
        # YOLO detector
        if YOLO is not None:
            try:
                self.yolo_model = YOLO("yolov8n.pt")
                print("✓ YOLO P&ID detector loaded")
            except Exception as e:
                print(f"✗ YOLO failed: {e}")
        
        # Embeddings
        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(settings.embedding_model)
                print("✓ Embedding model loaded")
            except Exception as e:
                print(f"✗ Embedding failed: {e}")

        # Determine runtime mode
        core_ready = all([
            self.ocr_processor,
            self.entity_extractor,
            self.relation_extractor,
            self.rag_summarizer,
            self.copilot_agent,
        ])
        if core_ready:
            self.model_mode = "best-model-stack"
        elif self.entity_extractor and self.relation_extractor:
            self.model_mode = "partial-stack"
        else:
            self.model_mode = "unavailable"
        print(f"✓ Pipeline initialized in '{self.model_mode}' mode")

    async def run(self, uploaded_filename: Optional[str], pdf_bytes: bytes) -> Dict[str, Any]:
        """Run full industrial pipeline"""
        
        job = create_job(uploaded_filename)
        job_id = job["job_id"]
        update_job(job_id, {"status": "processing", "message": "Starting pipeline..."})
        
        try:
            # Stage 1: OCR
            ocr_result = await self._process_ocr(pdf_bytes, job_id)
            text = ocr_result["text"]
            
            # Stage 2: YOLO P&ID Detection
            pid_result = self._run_yolo_pid_detector(text)
            
            # Stage 3: Entity Extraction (GLiNER)
            entities = self._extract_entities(text)
            
            # Stage 4: Relation Extraction (GLiREL / heuristic fallback)
            relations = self._extract_relations(text, entities)
            
            # Stage 5: Entity Linking & Disambiguation
            resolved_entities = self._link_entities(entities)
            
            # Stage 6: Neo4j Persistence
            neo4j_result = self._persist_to_graph(resolved_entities, relations, job_id)
            
            # Stage 7: GraphRAG Summarization
            rag_summary = self._summarize_with_rag(resolved_entities, relations, text)
            
            # Stage 8: Copilot Reasoning
            copilot_analysis = self._run_copilot_analysis(resolved_entities, relations, text)
            
            # Compile results
            result = {
                "job_id": job_id,
                "status": "completed",
                "message": "Pipeline completed successfully",
                "uploaded_filename": uploaded_filename,
                "timestamp": datetime.now().isoformat(),
                
                # Extraction results
                "entities": self._format_entities(resolved_entities),
                "relations": relations,
                "pid_components": pid_result.get("components", []),
                
                # Analysis results
                "graph_summary": rag_summary.get("summary_method", "N/A"),
                "anomalies": rag_summary.get("anomalies_detected", []),
                "failure_risks": rag_summary.get("failure_risks", []),
                "maintenance_recommendations": rag_summary.get("maintenance_recommendations", []),
                
                # Copilot insights
                "copilot_analysis": copilot_analysis,
                
                # Metadata
                "pipeline_metadata": {
                    "stages": [
                        "docling_surya_ocr",
                        "yolo_pid_detector",
                        "gliner_entity_extraction",
                        "rebel_relation_extraction",
                        "entity_linking",
                        "neo4j_persist",
                        "graphrag_summary",
                        "copilot_agent",
                    ],
                    "model_mode": self.model_mode,
                    "processors": {
                        "ocr": self.ocr_processor is not None,
                        "entity": self.entity_extractor is not None,
                        "relation": self.relation_extractor is not None,
                        "neo4j": self.graph_store is not None,
                        "rag": self.rag_summarizer is not None,
                        "copilot": self.copilot_agent is not None,
                    },
                    "neo4j_status": neo4j_result,
                },
            }
            
            update_job(job_id, result)
            return result
            
        except Exception as e:
            error_result = {
                "job_id": job_id,
                "status": "failed",
                "message": str(e),
                "error": type(e).__name__,
            }
            update_job(job_id, error_result)
            raise

    async def _process_ocr(self, pdf_bytes: bytes, job_id: str) -> Dict[str, Any]:
        """Process PDF with OCR"""
        
        if self.ocr_processor:
            return await self.ocr_processor.process(pdf_bytes)
        raise RuntimeError("Docling+Surya OCR processor unavailable. Install docling and surya-ocr and ensure the OCR stage is enabled.")

    def _run_yolo_pid_detector(self, text: str) -> Dict[str, Any]:
        """Detect P&ID components with YOLO"""
        
        components = detect_pid_components(text)
        
        return {
            "stage": "yolo_pid_detector",
            "status": "model-backed" if self.yolo_model else "available",
            "components": components,
        }

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities with GLiNER"""
        
        if self.entity_extractor:
            return self.entity_extractor.extract(text)
        raise RuntimeError("GLiNER entity extractor unavailable. Install GLiNER and ensure the entity extraction stage is available.")

    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations with the active relation extractor."""
        
        if self.relation_extractor:
            return self.relation_extractor.extract(text, entities)
        raise RuntimeError("Relation extractor unavailable. Ensure GLiREL or the heuristic fallback is available.")

    def _link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Link and disambiguate entities"""
        
        resolved = []
        seen = set()
        
        for entity in entities:
            canonical = re.sub(r"\s+", "_", entity.get("name", "").strip().lower())
            
            if canonical not in seen:
                seen.add(canonical)
                resolved.append({
                    **entity,
                    "canonical_name": canonical,
                })
        
        return resolved

    def _persist_to_graph(self, entities: List[Dict], relations: List[Dict], job_id: str) -> str:
        """Persist to Neo4j"""
        
        if not self.graph_store:
            raise RuntimeError("Neo4j persistence unavailable. Ensure Neo4j is running and available at the configured URI.")
        
        try:
            self.graph_store.persist_entities(entities, job_id)
            self.graph_store.persist_relations(relations, job_id)
            return "persisted"
        except Exception as e:
            print(f"✗ Neo4j persistence failed: {e}")
            return f"failed: {e}"

    def _summarize_with_rag(self, entities: List[Dict], relations: List[Dict], text: str) -> Dict[str, Any]:
        """Generate insights with GraphRAG"""
        
        if self.rag_summarizer:
            return self.rag_summarizer.generate_summary(entities, relations, text)
        raise RuntimeError("GraphRAG summarizer unavailable. Install or enable GraphRAG reasoning.")

    def _run_copilot_analysis(self, entities: List[Dict], relations: List[Dict], text: str) -> Dict[str, Any]:
        """Run copilot reasoning"""
        
        if self.copilot_agent:
            return self.copilot_agent.reason(entities, relations, text)
        raise RuntimeError("Copilot agent unavailable. Install or enable the Qwen reasoning engine.")

    def _format_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format entities for output"""
        
        return [
            {
                "name": e.get("name"),
                "entity_type": e.get("entity_type"),
                "confidence": e.get("confidence"),
                "canonical_name": e.get("canonical_name"),
            }
            for e in entities
        ]

    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction"""
        
        domain_terms = ["pump", "motor", "sensor", "valve", "compressor", "tank"]
        entities = []
        
        for term in domain_terms:
            if term in text.lower():
                entities.append({
                    "name": term.title(),
                    "entity_type": "equipment",
                    "confidence": 0.6,
                })
        
        return entities

    def _fallback_relation_extraction(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Fallback relation extraction"""
        
        relations = []
        entity_names = [e["name"].lower() for e in entities]
        
        for i, e1 in enumerate(entity_names):
            for e2 in entity_names[i+1:]:
                pattern = rf"\b{e1}\b.*?\b{e2}\b"
                if re.search(pattern, text.lower()):
                    relations.append({
                        "source": entity_names[i].title(),
                        "target": e2.title(),
                        "relation_type": "related_to",
                        "confidence": 0.6,
                    })
        
        return relations


_default_pipeline = None

def get_pipeline() -> IndustrialGraphPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = IndustrialGraphPipeline()
    return _default_pipeline


def run_pipeline(uploaded_filename: Optional[str], pdf_bytes: bytes) -> Dict[str, Any]:
    pipeline = get_pipeline()
    return asyncio.run(pipeline.run(uploaded_filename, pdf_bytes))
