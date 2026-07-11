"""
Industrial PDF-to-Graph Pipeline Engine
Full production-grade pipeline with adaptive fallback, structured stage tracking, and robust persistence.
"""

import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.config import settings
from app.pipeline.document_utils import chunk_text, normalize_text
from app.pipeline.entity_linker import BlinkEntityLinker
from app.pipeline.model_helpers import (
    BgeEmbedder,
    BgeReranker,
    GroundingDinoDetector,
    PIDSymbolDetector,
    SamSegmenter,
)
from app.pipeline.advanced_pipeline import AdvancedPipelineStages
from app.pipeline.models import canonicalize_entity_name, detect_pid_components
from app.storage import create_job, update_job


class IndustrialGraphPipeline:
    """Full-featured industrial analysis pipeline."""

    def __init__(self) -> None:
        self.ocr_processor = None
        self.entity_extractor = None
        self.relation_extractor = None
        self.graph_store = None
        self.rag_summarizer = None
        self.copilot_agent = None
        self.yolo_model = None
        self.pid_symbol_detector = None
        self.grounding_dino_detector = None
        self.sam_segmenter = None
        self.embedding_model = None
        self.reranker_model = None
        self.blink_linker = None
        self.doclayout_yolo_detector = None
        self.table_transformer_pipeline = None
        self.table_transformer_processor = None
        self.table_transformer_model = None
        self.model_mode = "initializing"
        self.stage_status: List[Dict[str, Any]] = []
        self._initialize_all_models()

    def _initialize_all_models(self) -> None:
        """Initialize all model and inference components."""

        print("Initializing Industrial PDF-to-Graph Pipeline...")

        try:
            from app.pipeline.ocr_processor import get_best_ocr_processor

            self.ocr_processor = get_best_ocr_processor()
            print(f"✓ OCR processor ready ({self.ocr_processor.__class__.__name__})")
        except Exception as exc:
            print(f"⚠ OCR initialization failed: {type(exc).__name__} - {exc}")

        try:
            from app.pipeline.entity_extractor import GlinerEntityExtractor

            self.entity_extractor = GlinerEntityExtractor()
            print(f"✓ Entity extractor ready ({self.entity_extractor.__class__.__name__})")
        except Exception as exc:
            print(f"⚠ Entity extractor initialization failed: {type(exc).__name__} - {exc}")

        try:
            from app.pipeline.relation_extractor import RebelRelationExtractor

            relation_extractor = RebelRelationExtractor()
            if not getattr(relation_extractor, "is_ready", False):
                raise RuntimeError("Relation extractor did not initialize")
            self.relation_extractor = relation_extractor
            print(f"✓ Relation extractor ready ({self.relation_extractor.__class__.__name__})")
        except Exception as exc:
            self.relation_extractor = None
            print(f"⚠ Relation extractor initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.pid_symbol_detector = PIDSymbolDetector()
            print(f"✓ PID symbol detector ready ({self.pid_symbol_detector.source})")
        except Exception as exc:
            self.pid_symbol_detector = None
            print(f"⚠ PID symbol detector initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.grounding_dino_detector = GroundingDinoDetector()
            print("✓ GroundingDINO detector ready")
        except Exception as exc:
            self.grounding_dino_detector = None
            print(f"⚠ GroundingDINO initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.sam_segmenter = SamSegmenter()
            print("✓ SAM2 segmenter ready")
        except Exception as exc:
            self.sam_segmenter = None
            print(f"⚠ SAM2 initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.embedding_model = BgeEmbedder()
            print("✓ BGE embedding model ready")
        except Exception as exc:
            self.embedding_model = None
            print(f"⚠ BGE embedding initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.reranker_model = BgeReranker()
            print("✓ BGE reranker ready")
        except Exception as exc:
            self.reranker_model = None
            print(f"⚠ BGE reranker initialization failed: {type(exc).__name__} - {exc}")

        try:
            self.blink_linker = BlinkEntityLinker()
            print("✓ BLINK linker ready")
        except Exception as exc:
            self.blink_linker = None
            print(f"⚠ BLINK linker initialization failed: {type(exc).__name__} - {exc}")

        try:
            from app.pipeline.neo4j_store import Neo4jGraphStore

            self.graph_store = Neo4jGraphStore()
            if getattr(self.graph_store, "connected", False):
                print("✓ Neo4j store ready")
            else:
                print("✗ Neo4j store unavailable")
        except Exception as exc:
            print(f"⚠ Graph store initialization failed: {type(exc).__name__} - {exc}")

        try:
            from app.pipeline.graphrag_summarizer import GraphRAGSummarizer

            self.rag_summarizer = GraphRAGSummarizer()
            print(f"✓ GraphRAG summarizer ready ({self.rag_summarizer.__class__.__name__})")
        except Exception as exc:
            print(f"⚠ GraphRAG initialization failed: {type(exc).__name__} - {exc}")

        try:
            from app.pipeline.copilot_agent import IndustrialCopilotAgent

            self.copilot_agent = IndustrialCopilotAgent()
            print(f"✓ Copilot agent ready ({self.copilot_agent.__class__.__name__})")
        except Exception as exc:
            print(f"⚠ Copilot agent initialization failed: {type(exc).__name__} - {exc}")

        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO("yolov8n.pt")
            print("✓ YOLO model loaded")
        except Exception as exc:
            print(f"⚠ YOLO initialization failed: {type(exc).__name__} - {exc}")

        try:
            import doclayout_yolo

            model_path = Path(__file__).resolve().parents[2] / "yolov8n.pt"
            self.doclayout_yolo_detector = doclayout_yolo.YOLO(str(model_path))
            print("✓ DocLayout-YOLO detector loaded")
        except Exception as exc:
            print(f"⚠ DocLayout-YOLO initialization failed: {type(exc).__name__} - {exc}")

        self.model_mode = self._resolve_model_mode()
        print(f"✓ Pipeline initialized in '{self.model_mode}' mode")

    def _resolve_model_mode(self) -> str:
        if all(
            [
                self.ocr_processor,
                self.entity_extractor,
                self.relation_extractor,
                self.rag_summarizer,
                self.copilot_agent,
            ]
        ):
            return "best-model-stack"

        if self.ocr_processor and self.entity_extractor and self.relation_extractor:
            return "hybrid-fallback-stack"

        if self.ocr_processor and self.entity_extractor:
            return "partial-stack"

        if self.ocr_processor:
            return "ocr-only"

        return "unavailable"

    async def run(self, uploaded_filename: Optional[str], pdf_bytes: bytes, job_id: Optional[str] = None) -> Dict[str, Any]:
        if job_id is None:
            job = create_job(uploaded_filename)
            job_id = job["job_id"]
        else:
            update_job(job_id, {"status": "processing", "message": "Pipeline started."})
        self.stage_status = []

        try:
            ocr_result = await self._run_stage(
                "docling_surya_ocr",
                self._process_ocr,
                required=True,
                pdf_bytes=pdf_bytes,
            )
            text = ocr_result.get("text", "")

            doclayout_yolo_info = await self._run_stage(
                "doclayout_yolo_analysis",
                self._analyze_doclayout_yolo,
                required=False,
                ocr_result=ocr_result,
                pdf_bytes=pdf_bytes,
            )

            layout_info = await self._run_stage(
                "surya_layout_understanding",
                self._extract_layout,
                required=False,
                ocr_result=ocr_result,
            )

            table_info = await self._run_stage(
                "table_extraction",
                self._extract_tables,
                required=False,
                ocr_result=ocr_result,
            )

            table_transformer_info = await self._run_stage(
                "table_transformer_extraction",
                self._extract_tables_with_transformer,
                required=False,
                ocr_result=ocr_result,
                pdf_bytes=pdf_bytes,
            )

            groundingdino_info = await self._run_stage(
                "groundingdino_detection",
                self._detect_groundingdino_objects,
                required=False,
                pdf_bytes=pdf_bytes,
                text=text,
            )

            sam_segmentation_info = await self._run_stage(
                "sam2_segmentation",
                self._segment_with_sam,
                required=False,
                pdf_bytes=pdf_bytes,
            )

            formulas = await self._run_stage(
                "nougat_formula_recognition",
                self._recognize_formulas,
                required=False,
                text=text,
            )

            reading_order = await self._run_stage(
                "docling_reading_order",
                self._build_reading_order,
                required=False,
                ocr_result=ocr_result,
            )

            yolo_pid_insights = await self._run_stage(
                "yolo_pid_detector",
                self._detect_pid_with_yolo,
                required=False,
                pdf_bytes=pdf_bytes,
            )

            pid_symbol_insights = await self._run_stage(
                "pid_symbol_detection",
                self._detect_pid_symbols,
                required=False,
                pdf_bytes=pdf_bytes,
            )

            pid_components = await self._run_stage(
                "pid_component_detection",
                self._detect_pid_components,
                required=False,
                text=text,
            )

            text_chunks = await self._run_stage(
                "document_segmentation",
                self._segment_document,
                required=False,
                text=text,
            )

            semantic_index = await self._run_stage(
                "semantic_indexing",
                self._index_text_chunks,
                required=False,
                text_chunks=text_chunks,
            )

            entities = await self._run_stage(
                "entity_extraction",
                self._extract_entities,
                required=True,
                text=text,
            )

            relations = await self._run_stage(
                "relation_extraction",
                self._extract_relations,
                required=False,
                text=text,
                entities=entities,
            )

            resolved_entities = await self._run_stage(
                "entity_linking",
                self._link_entities,
                required=False,
                entities=entities,
            )

            bge_ranking = await self._run_stage(
                "bge_reranking",
                self._rerank_entities,
                required=False,
                text=text,
                entities=resolved_entities,
                relations=relations,
            )

            vision_language_insights = await self._run_stage(
                "qwen2_5_vl",
                self._vision_language_understanding,
                required=False,
                entities=resolved_entities,
                relations=relations,
                text=text,
                layout=layout_info.get("layout", []),
                tables=table_info.get("tables", []),
                reading_order=reading_order,
            )

            rag_analysis = await self._run_stage(
                "graphrag_analysis",
                self._graphrag_analyze,
                required=False,
                entities=resolved_entities,
                relations=relations,
                text=text,
                text_chunks=text_chunks,
            )

            copilot_analysis = await self._run_stage(
                "copilot_analysis",
                self._copilot_analyze,
                required=False,
                entities=resolved_entities,
                relations=relations,
                text=text,
            )

            neo4j_status = await self._run_stage(
                "neo4j_persistence",
                self._persist_graph,
                required=False,
                entities=resolved_entities,
                relations=relations,
                job_id=job_id,
            )

            # -----------------------------------------------------------------
            # Advanced pipeline stages (Qdrant, GraphRAG, Qwen3, TimesFM, TFT)
            # Run these after the Neo4j persistence so the graph is available
            # -----------------------------------------------------------------
            try:
                advanced = AdvancedPipelineStages()

                adv_input = {
                    "job_id": job_id,
                    "embeddings": semantic_index.get("embeddings") if isinstance(semantic_index, dict) else [],
                    "entities": resolved_entities or [],
                    "relations": relations or [],
                    "text": text,
                    "text_chunks": text_chunks or [],
                    "sensor_data": {},
                    "machine_id": None,
                    "logs": [],
                }

                adv_semantic = await self._run_stage(
                    "stage_8_semantic_indexing",
                    advanced.stage_semantic_indexing,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_graph = await self._run_stage(
                    "stage_9_graph_reasoning",
                    advanced.stage_graph_reasoning,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_llm = await self._run_stage(
                    "stage_10_llm_analysis",
                    advanced.stage_llm_analysis,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_anomaly = await self._run_stage(
                    "stage_11_anomaly_detection",
                    advanced.stage_anomaly_detection,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_rul = await self._run_stage(
                    "stage_12_rul_prediction",
                    advanced.stage_rul_prediction,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_rca = await self._run_stage(
                    "stage_13_root_cause_analysis",
                    advanced.stage_root_cause_analysis,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_failure = await self._run_stage(
                    "stage_14_failure_prediction",
                    advanced.stage_failure_prediction,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_graph_embeddings = await self._run_stage(
                    "stage_15_graph_embeddings",
                    advanced.stage_graph_embeddings,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_clustering = await self._run_stage(
                    "stage_16_embedding_clustering",
                    advanced.stage_embedding_clustering,
                    required=False,
                    pipeline_result=adv_input,
                )

                adv_lessons = await self._run_stage(
                    "stage_17_lessons_learned",
                    advanced.stage_lessons_learned,
                    required=False,
                    pipeline_result=adv_input,
                )

            except Exception as exc:
                print(f"⚠ Advanced pipeline initialization failed: {exc}")

            result = {
                "job_id": job_id,
                "status": "completed",
                "message": "Pipeline completed successfully",
                "uploaded_filename": uploaded_filename,
                "timestamp": datetime.now().isoformat(),
                "text": text,
                "entities": self._format_entities(resolved_entities),
                "relations": relations,
                "pid_components": pid_components or [],
                "semantic_indexing": semantic_index or {},
                "document_segments": text_chunks or [],
                "layout": layout_info.get("layout", []),
                "tables": table_info.get("tables", []),
                "formulas": formulas or [],
                "doclayout_yolo": doclayout_yolo_info or {},
                "reading_order": reading_order or [],
                "table_transformer": table_transformer_info or {},
                "groundingdino": groundingdino_info or {},
                "sam_segments": sam_segmentation_info or {},
                "yolo_pid_insights": yolo_pid_insights or {},
                "pid_symbol_insights": pid_symbol_insights or {},
                "bge_ranking": bge_ranking or {},
                "vision_language": vision_language_insights or {},
                "neo4j_status": neo4j_status,
                "rag_analysis": rag_analysis or {},
                "copilot_analysis": copilot_analysis or {},
                "advanced_semantic": locals().get("adv_semantic") or {},
                "advanced_graph": locals().get("adv_graph") or {},
                "advanced_llm": locals().get("adv_llm") or {},
                "anomaly_detection": locals().get("adv_anomaly") or {},
                "rul_prediction": locals().get("adv_rul") or {},
                "root_cause_analysis": locals().get("adv_rca") or {},
                "failure_prediction": locals().get("adv_failure") or {},
                "advanced_graph_embeddings": locals().get("adv_graph_embeddings") or {},
                "advanced_clustering": locals().get("adv_clustering") or {},
                "advanced_lessons_learned": locals().get("adv_lessons") or {},
                "pipeline_metadata": {
                    "model_mode": self.model_mode,
                    "runtime_mode": self.model_mode,
                    "stage_status": self.stage_status,
                    "stages": [stage["stage"] for stage in self.stage_status],
                    "loaded_components": {
                        "ocr": self.ocr_processor is not None,
                        "entity_extractor": self.entity_extractor is not None,
                        "relation_extractor": self.relation_extractor is not None,
                        "neo4j": self.graph_store is not None and getattr(self.graph_store, "connected", False),
                        "rag": self.rag_summarizer is not None,
                        "copilot": self.copilot_agent is not None,
                        "yolo": self.yolo_model is not None,
                        "pid_symbol_detector": self.pid_symbol_detector is not None,
                        "groundingdino": self.grounding_dino_detector is not None,
                        "sam": self.sam_segmenter is not None,
                        "embeddings": self.embedding_model is not None,
                        "reranker": self.reranker_model is not None,
                        "blink": self.blink_linker is not None,
                    },
                },
            }

            update_job(job_id, result)
            return result

        except Exception as exc:
            failure_payload = {
                "job_id": job_id,
                "status": "failed",
                "message": str(exc),
                "error": type(exc).__name__,
                "pipeline_metadata": {"stage_status": self.stage_status},
            }
            update_job(job_id, failure_payload)
            raise

    async def _run_stage(
        self,
        stage_name: str,
        func: Callable[..., Any],
        required: bool,
        **kwargs: Any,
    ) -> Any:
        status = "completed"
        message = ""
        result: Any = None

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
        except Exception as exc:
            message = str(exc)
            if required:
                status = "failed"
                self.stage_status.append(
                    {
                        "stage": stage_name,
                        "status": status,
                        "message": message,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                raise
            status = "skipped"
            result = [] if stage_name in ["entity_extraction", "relation_extraction"] else {}

        self.stage_status.append(
            {
                "stage": stage_name,
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        return result

    def _process_ocr(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if self.ocr_processor is None:
            raise RuntimeError("OCR processor unavailable.")
        return self.ocr_processor.process(pdf_bytes)

    def _render_pdf_pages(self, pdf_bytes: bytes) -> List[Any]:
        try:
            import fitz
            from PIL import Image

            document = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            for page in document:
                pix = page.get_pixmap(dpi=200)
                mode = "RGBA" if pix.alpha else "RGB" if pix.n == 3 else "L"
                image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                pages.append(image)
            return pages
        except Exception as exc:
            print(f"⚠ PDF rendering failed: {exc}")
            return []

    def _segment_document(self, text: str) -> List[str]:
        if not text:
            return []
        return chunk_text(normalize_text(text), max_chars=1400, overlap=220)

    def _index_text_chunks(self, text_chunks: List[str]) -> Dict[str, Any]:
        if self.embedding_model is None:
            return {"status": "skipped", "indexed_chunks": 0}

        if not text_chunks:
            return {"status": "skipped", "indexed_chunks": 0}

        try:
            embeddings = []
            for chunk in text_chunks:
                embedding = self.embedding_model.encode(chunk)
                vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                embeddings.append(vec)
            return {"status": "indexed", "indexed_chunks": len(text_chunks), "embeddings": embeddings}
        except Exception as exc:
            print(f"⚠ Semantic indexing failed: {exc}")
            return {"status": "failed", "indexed_chunks": 0}

    def _detect_pid_components(self, text: str) -> List[str]:
        return detect_pid_components(text)

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        if self.entity_extractor is None:
            raise RuntimeError("Entity extractor unavailable.")
        return self.entity_extractor.extract(text)

    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.relation_extractor is None:
            return []
        return self.relation_extractor.extract(text, entities)

    def _extract_layout(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "layout": ocr_result.get("layout", []),
            "source": "surya" if ocr_result.get("layout") else "docling",
        }

    def _extract_tables(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        tables = ocr_result.get("tables", [])
        return {
            "tables": tables,
            "table_count": len(tables) if isinstance(tables, list) else 0,
        }

    def _extract_tables_with_transformer(self, ocr_result: Dict[str, Any], pdf_bytes: bytes) -> Dict[str, Any]:
        tables = ocr_result.get("tables", [])
        transformer_detections = []
        transformer_tables: List[Dict[str, Any]] = []
        images = self._render_pdf_pages(pdf_bytes)

        if images:
            try:
                from table_transformer import TableExtractionPipeline

                if self.table_transformer_pipeline is None:
                    self.table_transformer_pipeline = TableExtractionPipeline(det_device="cpu", str_device="cpu")

                pipeline = self.table_transformer_pipeline
                if pipeline.det_model is not None and pipeline.str_model is not None:
                    for page_number, image in enumerate(images, start=1):
                        extracted_tables = pipeline.extract(
                            image,
                            out_objects=True,
                            out_cells=True,
                            out_html=False,
                            out_csv=False,
                        )
                        for table_index, raw_table in enumerate(extracted_tables, start=1):
                            transformer_tables.append(
                                {
                                    "page": page_number,
                                    "table_index": table_index,
                                    "objects": raw_table.get("objects", []),
                                    "cells": raw_table.get("cells", []),
                                    "structure": raw_table.get("structure", {}),
                                }
                            )
                            if raw_table.get("objects"):
                                transformer_detections.extend(raw_table.get("objects", []))
                else:
                    raise RuntimeError("table_transformer package models are not loaded; falling back to Hugging Face TableTransformer")
            except Exception as package_exc:
                print(f"⚠ Table Transformer package extraction failed: {package_exc}")
                try:
                    import warnings
                    from transformers import AutoImageProcessor
                    from transformers.models.table_transformer import TableTransformerForObjectDetection
                    import torch

                    model_name = "microsoft/table-transformer-detection"
                    if self.table_transformer_processor is None:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*num_batches_tracked.*")
                            self.table_transformer_processor = AutoImageProcessor.from_pretrained(model_name)
                    if self.table_transformer_model is None:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*num_batches_tracked.*")
                            self.table_transformer_model = TableTransformerForObjectDetection.from_pretrained(model_name)

                    processor = self.table_transformer_processor
                    model = self.table_transformer_model.to(settings.device_for_extraction)

                    for page_number, image in enumerate(images, start=1):
                        inputs = processor(images=image, return_tensors="pt").to(settings.device_for_extraction)
                        outputs = model(**inputs)
                        target_sizes = torch.tensor([[image.height, image.width]]).to(settings.device_for_extraction)
                        results = processor.post_process_object_detection(
                            outputs,
                            threshold=0.5,
                            target_sizes=target_sizes,
                        )
                        if results:
                            result = results[0]
                            for score, label, box in zip(result.get("scores", []), result.get("labels", []), result.get("boxes", [])):
                                detection = {
                                    "page": page_number,
                                    "label": model.config.id2label[int(label)],
                                    "confidence": float(score),
                                    "bbox": [float(coord) for coord in box.tolist()],
                                }
                                transformer_detections.append(detection)
                                transformer_tables.append(detection)
                except Exception as exc:
                    print(f"⚠ Hugging Face Table Transformer stage failed: {exc}")
                    transformer_detections = []
                    transformer_tables = []

        return {
            "tables": tables,
            "table_count": len(tables) if isinstance(tables, list) else 0,
            "table_transformer_tables": transformer_tables,
            "table_transformer_detections": transformer_detections,
        }

    def _analyze_doclayout_yolo(self, ocr_result: Dict[str, Any], pdf_bytes: bytes) -> Dict[str, Any]:
        layout_objects: List[Dict[str, Any]] = []
        source = "surya"
        images = self._render_pdf_pages(pdf_bytes)

        if images:
            try:
                import doclayout_yolo  # type: ignore

                if self.doclayout_yolo_detector is None:
                    root_model = Path(__file__).resolve().parents[2] / "yolov8n.pt"
                    self.doclayout_yolo_detector = doclayout_yolo.YOLO(str(root_model))

                detector = self.doclayout_yolo_detector
                for page_number, image in enumerate(images, start=1):
                    results = detector(image)
                    for result in results:
                        if not hasattr(result, "boxes"):
                            continue
                        coords = result.boxes.xyxy.tolist()
                        classes = result.boxes.cls.tolist()
                        scores = result.boxes.conf.tolist()
                        for bbox, class_id, score in zip(coords, classes, scores):
                            label = (
                                result.names[int(class_id)]
                                if isinstance(result.names, (list, dict))
                                else str(int(class_id))
                            )
                            layout_objects.append(
                                {
                                    "page": page_number,
                                    "label": label,
                                    "confidence": float(score),
                                    "bbox": [float(coord) for coord in bbox],
                                }
                            )
                source = "doclayout_yolo"
            except Exception as exc:
                print(f"⚠ DocLayout-YOLO analysis failed: {exc}")
                layout_objects = []

        if not layout_objects:
            layout_objects = ocr_result.get("layout", [])
            source = "surya" if layout_objects else "docling"

        return {
            "layout": layout_objects,
            "source": source,
            "detected_objects": len(layout_objects),
        }

    def _detect_groundingdino_objects(self, pdf_bytes: bytes, text: str) -> Dict[str, Any]:
        if self.grounding_dino_detector is None:
            return {"detections": [], "source": "groundingdino_unavailable"}

        images = self._render_pdf_pages(pdf_bytes)
        if not images:
            return {"detections": [], "source": "groundingdino_unavailable"}

        prompt = text.strip()[:256] if text else "industrial equipment layout"
        try:
            detections = self.grounding_dino_detector.detect(images, prompt)
            return {
                "detections": detections,
                "source": "groundingdino",
                "prompt": prompt,
                "count": len(detections),
            }
        except Exception as exc:
            print(f"⚠ GroundingDINO detection failed: {exc}")
            return {"detections": [], "source": "groundingdino_failed", "error": str(exc)}

    def _segment_with_sam(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if self.sam_segmenter is None:
            return {"segments": [], "source": "sam_unavailable"}

        images = self._render_pdf_pages(pdf_bytes)
        if not images:
            return {"segments": [], "source": "sam_unavailable"}

        segments: List[Dict[str, Any]] = []
        try:
            for image in images:
                segments.extend(self.sam_segmenter.segment(image))
            return {"segments": segments, "source": "sam2", "count": len(segments)}
        except Exception as exc:
            print(f"⚠ SAM2 segmentation failed: {exc}")
            return {"segments": [], "source": "sam_failed", "error": str(exc)}

    def _detect_pid_symbols(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if self.pid_symbol_detector is None:
            return {"symbols": [], "source": "pid_symbol_detector_unavailable"}

        images = self._render_pdf_pages(pdf_bytes)
        if not images:
            return {"symbols": [], "source": "pid_symbol_detector_unavailable"}

        try:
            symbols = self.pid_symbol_detector.detect(images)
            return {"symbols": symbols, "source": self.pid_symbol_detector.source, "count": len(symbols)}
        except Exception as exc:
            print(f"⚠ PID symbol detection failed: {exc}")
            return {"symbols": [], "source": "pid_symbol_detector_failed", "error": str(exc)}

    def _detect_pid_with_yolo(self, pdf_bytes: bytes) -> Dict[str, Any]:
        if self.yolo_model is None:
            return {"detected_objects": [], "source": "yolo_unavailable"}

        try:
            images = self._render_pdf_pages(pdf_bytes)
            detections = []
            for page_number, image in enumerate(images, start=1):
                results = self.yolo_model(image)
                for result in results:
                    if not hasattr(result, "boxes"):
                        continue
                    boxes = result.boxes.xyxy.tolist()
                    labels = [
                        result.names[int(label)] if isinstance(result.names, (list, dict)) else str(int(label))
                        for label in result.boxes.cls.tolist()
                    ]
                    confidences = result.boxes.conf.tolist()
                    detections.append(
                        {
                            "page": page_number,
                            "boxes": boxes,
                            "labels": labels,
                            "confidences": confidences,
                        }
                    )
            return {"detected_objects": detections, "source": "ultralytics_yolo"}
        except Exception as exc:
            print(f"⚠ PID detection stage failed: {exc}")
            return {"detected_objects": [], "source": "yolo_fallback_failed"}

    def _recognize_formulas(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"formulas": []}
        # Simple heuristic extraction of formula-like text snippets.
        import re

        candidates = []
        patterns = [
            r"\b[A-Z][A-Za-z]?\d+(?:[A-Z][A-Za-z]?\d+)+\b",
            r"\b[A-Za-z0-9\)\]\}]+\s*=\s*[A-Za-z0-9\(\[\{]+\b",
        ]
        for pattern in patterns:
            for match in re.findall(pattern, text):
                normalized = match.strip()
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
        return {"formulas": candidates[:20]}

    def _build_reading_order(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        reading_order = ocr_result.get("reading_order", [])
        if not isinstance(reading_order, list):
            reading_order = []
        return {"reading_order": reading_order}

    def _vision_language_understanding(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        layout: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        reading_order: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.copilot_agent is None:
            return {
                "summary": "vision-language reasoning unavailable",
                "anomalies": [],
                "risks": [],
                "recommendations": [],
                "compliance": [],
                "confidence": 0.0,
            }

        return self.copilot_agent.vision_language_analysis(
            entities,
            relations,
            text,
            layout=layout,
            tables=tables,
            reading_order=reading_order,
        )

    def _link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.blink_linker is not None:
            try:
                return self.blink_linker.link_entities(entities)
            except Exception as exc:
                print(f"⚠ BLINK linking failed: {exc}")

        linked: List[Dict[str, Any]] = []
        seen = set()

        for entity in entities:
            canonical = canonicalize_entity_name(entity.get("name", ""))
            if canonical and canonical not in seen:
                seen.add(canonical)
                entity["canonical_name"] = canonical
                entity["link_source"] = "canonical_fallback"
                linked.append(entity)

        return linked

    def _rerank_entities(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if self.reranker_model is None:
            return {"ranked": [], "source": "reranker_unavailable"}

        candidates = []
        for entity in entities[:10]:
            name = entity.get("name", "")
            if name:
                candidates.append(name)

        for relation in relations[:10]:
            source = relation.get("source", "")
            target = relation.get("target", "")
            if source and target:
                candidates.append(f"{source} {relation.get('relation_type', '')} {target}".strip())

        if not candidates:
            return {"ranked": [], "source": "reranker_empty_candidates"}

        ranked = []
        try:
            for candidate in candidates[:10]:
                score = self.reranker_model.score_pair(text[:512], candidate)
                ranked.append({"candidate": candidate, "score": float(score)})
            ranked.sort(key=lambda item: item["score"], reverse=True)
            return {"ranked": ranked, "source": "bge_reranker", "count": len(ranked)}
        except Exception as exc:
            print(f"⚠ BGE reranking failed: {exc}")
            return {"ranked": [], "source": "reranker_failed", "error": str(exc)}

    def _graphrag_analyze(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        text: str,
        text_chunks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.rag_summarizer is None:
            return {
                "summary_method": "unavailable",
                "anomalies_detected": [],
                "failure_risks": [],
                "maintenance_recommendations": [],
                "confidence": 0.0,
            }
        return self.rag_summarizer.generate_summary(entities, relations, text, text_chunks)

    def _copilot_analyze(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        if self.copilot_agent is None:
            return {
                "agent": "unavailable",
                "reasoning_chain": {},
                "executive_summary": "Copilot reasoning unavailable.",
                "confidence": 0.0,
            }
        return self.copilot_agent.reason(entities, relations, text)

    def _persist_graph(self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]], job_id: str) -> str:
        if self.graph_store is None:
            return "skipped (neo4j unavailable)"

        entity_success = self.graph_store.persist_entities(entities, job_id)
        relation_success = self.graph_store.persist_relations(relations, job_id)

        if entity_success and relation_success:
            return "persisted"
        if entity_success or relation_success:
            return "partial"
        return "skipped"

    def _format_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for entity in entities:
            formatted.append(
                {
                    "name": entity.get("name", ""),
                    "entity_type": entity.get("entity_type", "unknown"),
                    "confidence": float(entity.get("confidence", 0.0) or 0.0),
                    "canonical_name": entity.get("canonical_name", canonicalize_entity_name(entity.get("name", ""))),
                }
            )
        return formatted


_pipeline: Optional[IndustrialGraphPipeline] = None


def get_pipeline() -> IndustrialGraphPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = IndustrialGraphPipeline()
    return _pipeline


async def run_pipeline(filename: str, pdf_bytes: bytes, job_id: Optional[str] = None) -> Dict[str, Any]:
    pipeline = get_pipeline()
    return await pipeline.run(filename, pdf_bytes, job_id=job_id)
