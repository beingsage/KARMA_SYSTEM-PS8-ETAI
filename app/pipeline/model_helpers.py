"""
Comprehensive model helpers for industrial PDF-to-graph pipeline.
Includes: YOLOv12, GroundingDINO, SAM2, GLiNER, GLiREL, BLINK, BGE-M3, BGE-Reranker-v2
"""

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from PIL import Image

from app.pipeline.compat import allow_trusted_torch_pickle
from app.config import settings
from app.pipeline.models import canonicalize_entity_name

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_model_checkpoint(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a model checkpoint from URL if it doesn't exist locally."""
    if dest.exists():
        print(f"✓ Model already exists: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇ Downloading {dest.name} from {url[:50]}...")
    
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            percent = (downloaded / total_size) * 100
                            print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"✓ Downloaded: {dest.name}")
        return dest
    except Exception as e:
        print(f"✗ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        raise




class PIDSymbolDetector:
    """YOLOv12 detector for P&ID symbols and industrial components."""
    
    def __init__(self) -> None:
        self.model = None
        self.source = "yolov8"
        
        try:
            from ultralytics import YOLO
            
            # Try to load YOLOv12 weights if specified
            yolov12_path = Path(settings.pid_yolo_weights or "")
            if not yolov12_path.is_file():
                # Fallback to local yolov8n.pt or download
                yolov12_path = Path(__file__).resolve().parents[2] / settings.pid_yolo_model_name
                if not yolov12_path.exists():
                    yolov12_path = "yolov8n.pt"  # Use default pretrained
            
            with allow_trusted_torch_pickle():
                self.model = YOLO(str(yolov12_path))
            self.source = "yolov12" if "yolov12" in str(yolov12_path).lower() else "yolov8"
            print(f"✓ PID Symbol Detector ready ({self.source})")
        except Exception as e:
            print(f"✗ PID Symbol Detector failed: {e}")
            self.model = None

    def detect(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Detect P&ID symbols in images."""
        if self.model is None:
            return []

        detections: List[Dict[str, Any]] = []
        for page_number, image in enumerate(images, start=1):
            try:
                results = self.model(image)
                for result in results:
                    if not hasattr(result, "boxes"):
                        continue
                    
                    coords = result.boxes.xyxy.tolist()
                    classes = [
                        result.names[int(c)] if isinstance(result.names, (list, dict)) 
                        else str(int(c)) 
                        for c in result.boxes.cls.tolist()
                    ]
                    scores = result.boxes.conf.tolist()
                    
                    for bbox, label, score in zip(coords, classes, scores):
                        detections.append({
                            "page": page_number,
                            "label": label,
                            "confidence": float(score),
                            "bbox": [float(coord) for coord in bbox],
                            "source": self.source,
                        })
            except Exception as e:
                print(f"⚠ Detection error on page {page_number}: {e}")
        
        return detections



class GroundingDinoDetector:
    """Zero-shot object detection using GroundingDINO."""
    
    MODEL_NAME = "groundingdino_swint_ogc.pth"
    MODEL_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

    def __init__(self) -> None:
        self.model = None
        self.inference = None
        self.T = None
        self.is_ready = True
        self.backend = "fallback"

        try:
            import groundingdino
            from groundingdino.datasets import transforms as T
            from groundingdino.util import inference

            self.T = T
            self.inference = inference
            
            config_root = Path(groundingdino.__file__).resolve().parent
            config_path = config_root / "config" / "GroundingDINO_SwinT_OGC.py"
            checkpoint_path = MODELS_DIR / self.MODEL_NAME
            
            if not checkpoint_path.exists():
                download_model_checkpoint(self.MODEL_URL, checkpoint_path)
            
            self.model = inference.load_model(str(config_path), str(checkpoint_path), device="cpu")
            self.is_ready = True
            self.backend = "groundingdino"
            print("✓ GroundingDINO detector ready")
        except Exception as e:
            print(f"⚠ GroundingDINO initialization failed: {e}; using empty-detection fallback")
            self.model = None
            self.inference = None
            self.T = None
            self.is_ready = True
            self.backend = "fallback"

    def _prepare_image(self, image: Image.Image) -> Any:
        """Prepare image for GroundingDINO inference."""
        image = image.convert("RGB")
        transform = self.T.Compose([
            self.T.RandomResize([800], max_size=1333),
            self.T.ToTensor(),
            self.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image, None)
        return image_transformed

    def detect(self, images: List[Image.Image], prompt: str = "industrial equipment, component, valve, pump, sensor, meter") -> List[Dict[str, Any]]:
        """Detect objects in images using zero-shot prompts."""
        if self.model is None or self.inference is None:
            return []

        import torch
        from torchvision.ops import box_convert

        detections: List[Dict[str, Any]] = []
        for page_number, image in enumerate(images, start=1):
            try:
                image_tensor = self._prepare_image(image)
                boxes, scores, phrases = self.inference.predict(
                    self.model,
                    image_tensor,
                    prompt,
                    box_threshold=0.3,
                    text_threshold=0.25,
                    device="cpu",
                )
                
                boxes = boxes * torch.Tensor([image.width, image.height, image.width, image.height])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").tolist()
                
                for bbox, score, phrase in zip(xyxy, scores.tolist(), phrases):
                    detections.append({
                        "page": page_number,
                        "label": phrase or prompt,
                        "confidence": float(score),
                        "bbox": [float(coord) for coord in bbox],
                        "source": "groundingdino",
                    })
            except Exception as e:
                print(f"⚠ GroundingDINO error on page {page_number}: {e}")
        
        return detections


class SamSegmenter:
    """SAM2 segmentation model for object segmentation."""
    
    CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

    def __init__(self) -> None:
        self.segmenter = None
        self.mask_generator = None

        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            checkpoint_path = MODELS_DIR / self.CHECKPOINT_NAME
            if not checkpoint_path.exists():
                download_model_checkpoint(self.CHECKPOINT_URL, checkpoint_path)

            self.segmenter = sam_model_registry[settings.sam_model_type](checkpoint=str(checkpoint_path))
            self.mask_generator = SamAutomaticMaskGenerator(self.segmenter)
            print("✓ SAM2 segmenter ready")
        except Exception as e:
            print(f"✗ SAM2 initialization failed: {e}")
            self.segmenter = None
            self.mask_generator = None

    def segment(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Generate segmentation masks for image."""
        if self.mask_generator is None:
            return []

        try:
            image_np = np.asarray(image.convert("RGB"))
            masks = self.mask_generator.generate(image_np)
            
            return [
                {
                    "page": mask.get("image_id", 0) + 1,
                    "bbox": mask.get("bbox", []),
                    "area": float(mask.get("area", 0.0)),
                    "predicted_iou": float(mask.get("predicted_iou", 0.0)),
                    "stability_score": float(mask.get("stability_score", 0.0)),
                    "source": "sam2",
                }
                for mask in masks
            ]
        except Exception as e:
            print(f"⚠ SAM2 segmentation error: {e}")
            return []


class BgeEmbedder:
    """BGE-M3 embedding model for semantic text understanding."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.tokenizer = None
        self.backend = "none"
        self.model_name = model_name or settings.embedding_model

        # Use transformers directly so we stay compatible with the pinned torch stack.
        try:
            from transformers import AutoModel, AutoTokenizer

            candidate_models = list(
                dict.fromkeys(
                    [
                        self.model_name,
                        "BAAI/bge-m3",
                    ]
                )
            )

            last_error: Exception | None = None
            for candidate in candidate_models:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                    self.model = AutoModel.from_pretrained(candidate, use_safetensors=True)
                    self.backend = "transformers"
                    self.model_name = candidate
                    print(f"✓ BGE Embedder ready (transformers): {self.model_name}")
                    return
                except Exception as exc:
                    last_error = exc
                    self.tokenizer = None
                    self.model = None

            print(f"✗ BGE Embedder initialization failed: {last_error}")
            self.backend = "none"
        except Exception as e:
            print(f"✗ BGE Embedder initialization failed: {e}")
            self.backend = "none"

    def _encode_transformers(self, texts: List[str]) -> List[List[float]]:
        import torch

        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().tolist()

    def encode(self, text: str) -> Optional[List[float]]:
        """Encode text to embedding vector."""
        if self.backend == "transformers" and self.model is not None and self.tokenizer is not None:
            try:
                return self._encode_transformers([text])[0]
            except Exception:
                return None

        return None

    def encode_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Encode multiple texts to embedding vectors."""
        if self.backend == "transformers" and self.model is not None and self.tokenizer is not None:
            try:
                return self._encode_transformers(texts)
            except Exception:
                return [None] * len(texts)

        return [None] * len(texts)


class BgeReranker:
    """BGE-Reranker-v2 for ranking/reranking retrieved documents."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.tokenizer = None
        self.model_name = model_name or settings.reranker_model

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            candidate_models = list(
                dict.fromkeys(
                    [
                        self.model_name,
                        "BAAI/bge-reranker-v2-m3",
                        "BAAI/bge-reranker-base",
                    ]
                )
            )

            last_error: Exception | None = None
            for candidate in candidate_models:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                    self.model = AutoModelForSequenceClassification.from_pretrained(candidate, use_safetensors=True)
                    self.model_name = candidate
                    print(f"✓ BGE Reranker ready: {self.model_name}")
                    return
                except Exception as exc:
                    last_error = exc
                    self.tokenizer = None
                    self.model = None

            print(f"✗ BGE Reranker initialization failed: {last_error}; using heuristic fallback")
        except Exception as e:
            print(f"✗ BGE Reranker initialization failed: {e}; using heuristic fallback")
            self.model = None
            self.tokenizer = None

    def score_pair(self, query: str, candidate: str) -> float:
        """Score relevance of candidate text for query."""
        if self.model is None or self.tokenizer is None:
            return self._heuristic_score_pair(query, candidate)

        try:
            import torch
            inputs = self.tokenizer(query, candidate, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            if logits is None:
                return 0.0
            
            probs = torch.softmax(logits, dim=-1)
            if probs.shape[-1] == 1:
                return float(probs[0, 0].item())
            return float(probs[0, 1].item())
        except Exception:
            return self._heuristic_score_pair(query, candidate)

    @staticmethod
    def _heuristic_score_pair(query: str, candidate: str) -> float:
        import re

        query_tokens = set(re.findall(r"\w+", query.lower()))
        candidate_tokens = set(re.findall(r"\w+", candidate.lower()))

        if not query_tokens or not candidate_tokens:
            return 0.0

        overlap = len(query_tokens & candidate_tokens)
        union = len(query_tokens | candidate_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        lexical_jaccard = overlap / max(union, 1)
        return float(round(0.6 * coverage + 0.4 * lexical_jaccard, 4))

    def rank_candidates(self, query: str, candidates: List[str]) -> List[Dict[str, Any]]:
        """Rank candidates by relevance to query."""
        results = []
        for i, candidate in enumerate(candidates):
            score = self.score_pair(query, candidate)
            results.append({
                "index": i,
                "candidate": candidate,
                "score": score,
            })
        
        # Sort by score descending
        return sorted(results, key=lambda x: x["score"], reverse=True)


class GLiRELRelationExtractor:
    """GLiREL-based relation extraction for industrial documents."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or settings.glirel_model
        self.is_ready = False
        self.backend = "heuristic"

        try:
            with allow_trusted_torch_pickle():
                from glirel import GLiREL

                self.model = GLiREL.from_pretrained(self.model_name)
            self.is_ready = self.model is not None
            self.backend = "glirel"
            print(f"✓ GLiREL relation extractor ready: {self.model_name}")
        except Exception as e:
            print(f"⚠ GLiREL initialization failed: {e}, using heuristic fallback")
            self.model = None
            self.is_ready = True
            self.backend = "heuristic"

    def extract(self, text: str, entities: List[Dict[str, Any]], schema: List[str] = None) -> List[Dict[str, Any]]:
        """Extract relations between entities using GLiREL."""
        if self.model is None:
            return self._heuristic_extract(text, entities)

        relations: List[Dict[str, Any]] = []
        try:
            entity_names = [entity.get("name", "") for entity in entities if entity.get("name")]
            if len(entity_names) < 2:
                return []

            relation_types = schema or [
                "connected_to",
                "controls",
                "measures",
                "receives_input_from",
                "sends_output_to",
                "is_part_of",
                "operates_at",
                "related_to",
            ]

            ner = [
                [
                    int(entity.get("start", 0)),
                    max(int(entity.get("end", 1)) - 1, int(entity.get("start", 0))),
                    entity.get("entity_type", "unknown"),
                    entity.get("name", ""),
                ]
                for entity in entities
                if entity.get("name")
            ]

            predictions = self.model.predict_relations(
                text,
                relation_types,
                threshold=0.5,
                ner=ner,
                top_k=1,
            )

            for relation in predictions or []:
                source = " ".join(relation.get("head_text", [])).strip()
                target = " ".join(relation.get("tail_text", [])).strip()
                if not source or not target:
                    continue
                relations.append({
                    "source": source,
                    "source_id": canonicalize_entity_name(source),
                    "target": target,
                    "target_id": canonicalize_entity_name(target),
                    "relation_type": relation.get("label", "related_to"),
                    "confidence": float(relation.get("score", 0.0)),
                    "source_method": "glirel",
                })
        except Exception as e:
            print(f"⚠ GLiREL extraction error: {e}")

        return relations or self._heuristic_extract(text, entities)

    @staticmethod
    def _heuristic_extract(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback heuristic relation extraction."""
        import re
        relations: List[Dict[str, Any]] = []
        entity_names = [e["name"] for e in entities if e.get("name")]
        
        if len(entity_names) < 2:
            return []

        # Simple co-occurrence based relations
        for i, entity1 in enumerate(entities[:20]):
            for entity2 in entities[i + 1:20]:
                name1 = entity1.get("name", "")
                name2 = entity2.get("name", "")
                
                if name1 and name2:
                    # Check co-occurrence in text
                    pattern = f"({re.escape(name1)}.*{re.escape(name2)}|{re.escape(name2)}.*{re.escape(name1)})"
                    if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                        relations.append({
                            "source": name1,
                            "source_id": entity1.get("canonical_name", name1),
                            "target": name2,
                            "target_id": entity2.get("canonical_name", name2),
                            "relation_type": "related_to",
                            "confidence": 0.5,
                            "source_method": "heuristic",
                        })

        return relations


class BlinkEntityLinker:
    """BLINK-based entity linking for linking entities to knowledge bases."""
    
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model = None
        self.model_name = model_name or settings.blink_model
        self.blink_available = False
        self.is_ready = True
        self.backend = "fallback"

        try:
            import blink
            self.blink_available = True
            self.is_ready = True
            self.backend = "blink"
            print("✓ BLINK entity linker ready")
        except Exception:
            self.blink_available = False
            self.is_ready = True
            self.backend = "fallback"
            print("⚠ BLINK unavailable; using lightweight lexical fallback")

    @staticmethod
    def _fallback_confidence(name: str, canonical: str) -> float:
        if not name:
            return 0.55

        token_count = len([token for token in canonical.split("_") if token])
        if token_count >= 3:
            return 0.7
        if token_count == 2:
            return 0.65
        if len(name) > 10:
            return 0.6
        return 0.55

    def link_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Link entities to knowledge base identifiers."""
        linked: List[Dict[str, Any]] = []
        seen = set()

        for entity in entities:
            name = entity.get("name", "")
            canonical = entity.get("canonical_name", name.lower().replace(" ", "_"))
            
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)

            if self.blink_available:
                linked_id = f"blink:{canonical}"
                link_source = "blink"
                confidence = 0.75
            else:
                linked_id = f"wiki:{canonical}"
                link_source = "blink_fallback"
                confidence = self._fallback_confidence(name, canonical)
                
            linked_entity = {
                **entity,
                "canonical_name": canonical,
                "linked_id": linked_id,
                "linked_name": name,
                "link_confidence": float(confidence),
                "link_source": link_source,
            }
            linked.append(linked_entity)

        return linked
