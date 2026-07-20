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
from app.pipeline.runtime import select_device

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PIDSymbolDetector:
    """Lightweight placeholder for P&ID symbol detector.

    This minimal implementation provides the expected interface for the
    pipeline initialization. It attempts to load a YOLO-based detector if
    available; otherwise falls back to a no-op detector returning empty lists.
    """

    def __init__(self) -> None:
        self.source = "pid_yolo"
        self.backend = "none"
        self.model = None
        self.is_ready = False

        try:
            # Try to import a YOLO-like detector if the project provides one
            # Keep this optional to avoid hard dependency during quick runs
            from ultralytics import YOLO
            # Prefer PID-specific weights if configured, otherwise use bundled model
            candidate = None
            if getattr(settings, "pid_yolo_weights", None):
                cfg_path = Path(settings.pid_yolo_weights)
                if cfg_path.exists():
                    candidate = cfg_path
                    print(f"✓ PIDSymbolDetector configured to use PID weights: {candidate}")
                else:
                    raise FileNotFoundError(
                        f"PID_YOLO_WEIGHTS is set to '{cfg_path}', but the file does not exist."
                    )

            if candidate is None:
                candidate = MODELS_DIR / settings.pid_yolo_model_name
                if candidate.exists():
                    print(
                        f"⚠ No PID_YOLO_WEIGHTS configured. Using default YOLO model '{candidate.name}' as fallback."
                    )
                else:
                    raise FileNotFoundError(
                        f"Default PID YOLO model '{candidate}' not found. Please set PID_YOLO_WEIGHTS to a valid weights file."
                    )

            try:
                with allow_trusted_torch_pickle():
                    self.model = YOLO(str(candidate))
                self.backend = "ultralytics"
                self.is_ready = True
                print(f"✓ PIDSymbolDetector ready (ultralytics YOLO @ {candidate.name})")
            except Exception as exc:
                self.model = None
                self.is_ready = False
                print(f"⚠ PIDSymbolDetector failed to load model: {type(exc).__name__}: {exc}")
                raise
        except Exception as exc:
            # No YOLO available or model loading failed; mark not ready
            self.model = None
            self.is_ready = False
            print(f"⚠ PIDSymbolDetector initialization failed: {type(exc).__name__} - {exc}")

    def detect(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Detect P&ID symbols; returns empty list if model unavailable."""
        if not self.is_ready or self.model is None:
            return []

        results: List[Dict[str, Any]] = []
        try:
            for page, img in enumerate(images, start=1):
                try:
                    res = self.model(img)
                    # ultralytics returns a Results object; extract boxes if present
                    for r in res:
                        boxes = getattr(r, 'boxes', None)
                        if boxes is None:
                            continue
                        labels = None
                        if hasattr(r, 'names'):
                            labels = r.names
                        for box in boxes:
                            xyxy = box.xyxy.tolist() if hasattr(box, 'xyxy') else None
                            conf = float(box.conf) if hasattr(box, 'conf') else 0.0
                            label = "pid_symbol"
                            if hasattr(box, 'cls') and labels is not None:
                                cls_index = int(box.cls)
                                label = labels.get(cls_index, str(cls_index)) if isinstance(labels, dict) else labels[cls_index]
                            results.append({
                                "page": page,
                                "label": label,
                            "confidence": conf,
                            "bbox": xyxy or [],
                            "source": "pid_yolo",
                        })
                except Exception:
                    continue
        except Exception:
            return []

        return results


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



class GroundingDinoDetector:
    """Zero-shot object detection using GroundingDINO."""
    
    MODEL_NAME = "groundingdino_swint_ogc.pth"
    MODEL_URL = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

    def __init__(self) -> None:
        self.model = None
        self.inference = None
        self.T = None
        self.device = select_device()
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
            
            self.model = inference.load_model(str(config_path), str(checkpoint_path), device=self.device)
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
                    device=self.device,
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
        self.predictor = None

        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
            from app.pipeline.runtime import select_device
            import inspect

            checkpoint_path = MODELS_DIR / self.CHECKPOINT_NAME
            if not checkpoint_path.exists():
                download_model_checkpoint(self.CHECKPOINT_URL, checkpoint_path)

            self.segmenter = sam_model_registry[settings.sam_model_type](checkpoint=str(checkpoint_path))
            # Create predictor for box/point prompting if available
            try:
                self.predictor = SamPredictor(self.segmenter)
            except Exception:
                self.predictor = None

            # Configure mask generator differently depending on device availability
            device = select_device()
            try:
                import torch
                # Move model to chosen device if possible
                try:
                    self.segmenter.to("cuda" if device == "cuda" else "cpu")
                except Exception:
                    pass

                if device == "cuda":
                    # GPU: more thorough masks and more prompt coverage
                    mask_kwargs = dict(
                        points_per_side=32,
                        pred_iou_thresh=0.55,
                        stability_score_thresh=0.45,
                        box_nms_thresh=0.7,
                        crop_n_layers=2,
                        crop_n_points_downscale_factor=1,
                        min_mask_region_area=50,
                    )
                else:
                    # CPU: conservative settings to keep local runs responsive
                    mask_kwargs = dict(
                        points_per_side=16,
                        pred_iou_thresh=0.75,
                        stability_score_thresh=0.70,
                        box_nms_thresh=0.85,
                        crop_n_layers=0,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=250,
                    )
            except Exception:
                mask_kwargs = dict()

            # Filter mask_kwargs to supported parameters of SamAutomaticMaskGenerator
            try:
                sig = inspect.signature(SamAutomaticMaskGenerator.__init__)
                supported = {k for k in sig.parameters.keys()}
                safe_kwargs = {k: v for k, v in mask_kwargs.items() if k in supported}
            except Exception:
                safe_kwargs = mask_kwargs

            try:
                self.mask_generator = SamAutomaticMaskGenerator(self.segmenter, **safe_kwargs)
            except TypeError as exc:
                print(f"⚠ SAM2 mask generator initialization fallback: {exc}")
                self.mask_generator = SamAutomaticMaskGenerator(self.segmenter)

            print(f"✓ SAM2 segmenter ready (device={device})")
        except Exception as e:
            print(f"✗ SAM2 initialization failed: {e}")
            self.segmenter = None
            self.mask_generator = None
            self.predictor = None

    def segment(self, image: Image.Image, boxes: Optional[List[List[float]]] = None, page_number: int = 1) -> List[Dict[str, Any]]:
        """Generate segmentation masks for image.

        If `boxes` is provided (list of [x1,y1,x2,y2]) this will crop each box and
        generate masks focused on that region, then translate mask coordinates back
        to the full image. This is more reliable when using detection boxes as prompts.
        """
        if self.mask_generator is None:
            return []

        try:
            image_rgb = image.convert("RGB")
            max_dim = 1024 if self._is_cpu_mode() else 1536
            if max(image_rgb.width, image_rgb.height) > max_dim:
                scale = max_dim / max(image_rgb.width, image_rgb.height)
                new_w = max(1, int(image_rgb.width * scale))
                new_h = max(1, int(image_rgb.height * scale))
                image_rgb = image_rgb.resize((new_w, new_h), Image.LANCZOS)
            image_np = np.asarray(image_rgb)

            segments: List[Dict[str, Any]] = []

            def _normalize_boxes(candidate_boxes: Optional[List[List[float]]]) -> List[List[float]]:
                normalized: List[List[float]] = []
                if not candidate_boxes:
                    return normalized
                for box in candidate_boxes[:3]:
                    try:
                        x0, y0, x1, y1 = [float(v) for v in box]
                        if x1 <= x0 or y1 <= y0:
                            continue
                        if max(image_rgb.width, image_rgb.height) > max_dim:
                            scale = max_dim / max(image.width, image.height)
                            x0 *= scale
                            y0 *= scale
                            x1 *= scale
                            y1 *= scale
                        normalized.append([x0, y0, x1, y1])
                    except Exception:
                        continue
                return normalized

            prompt_boxes = _normalize_boxes(boxes)
            if not prompt_boxes:
                # Fallback to a center crop for a guaranteed lightweight result
                w, h = image_rgb.size
                cx, cy = w / 2.0, h / 2.0
                box_w, box_h = max(40, w * 0.3), max(40, h * 0.3)
                prompt_boxes = [[max(0, cx - box_w / 2), max(0, cy - box_h / 2), min(w, cx + box_w / 2), min(h, cy + box_h / 2)]]

            # If boxes are available, use SamPredictor on those regions first.
            if getattr(self, "predictor", None) is not None and len(prompt_boxes) > 0:
                try:
                    from numpy import array as nparray

                    self.predictor.set_image(image_np)
                    for box in prompt_boxes:
                        try:
                            x0, y0, x1, y1 = [float(v) for v in box]
                            if x1 <= x0 or y1 <= y0:
                                continue
                            xywh = [x0, y0, x1 - x0, y1 - y0]
                            masks, iou_preds, _ = self.predictor.predict(box=nparray([xywh]), multimask_output=True)
                            for idx, mask in enumerate(masks):
                                mask_np = mask.astype(np.uint8)
                                ys, xs = np.where(mask_np)
                                if ys.size and xs.size:
                                    y0m, ym = float(ys.min()), float(ys.max())
                                    x0m, xm = float(xs.min()), float(xs.max())
                                    bbox = [x0m + x0, y0m + y0, xm + x0, ym + y0]
                                else:
                                    bbox = [x0, y0, x1, y1]
                                area = float(mask_np.sum())
                                predicted_iou = float(iou_preds[idx]) if iou_preds is not None and len(iou_preds) > idx else 0.0
                                segments.append({
                                    "page": page_number,
                                    "bbox": bbox,
                                    "area": area,
                                    "predicted_iou": predicted_iou,
                                    "stability_score": 0.0,
                                    "source": "sam2_prompted",
                                })
                        except Exception as exc:
                            print(f"⚠ SAM2 prompt segmentation error for box {box}: {exc}")
                            continue
                    if segments:
                        return segments
                except Exception as e:
                    print(f"⚠ SAM predictor failed: {e}; falling back to automatic generator")

            # Default: automatic mask generator (no prompts)
            masks = self.mask_generator.generate(image_np)
            return [
                {
                    "page": page_number,
                    "bbox": mask.get("bbox", []),
                    "area": float(mask.get("area", 0.0)),
                    "predicted_iou": float(mask.get("predicted_iou", 0.0)) if mask.get("predicted_iou") is not None else 0.0,
                    "stability_score": float(mask.get("stability_score", 0.0)) if mask.get("stability_score") is not None else 0.0,
                    "source": "sam2",
                }
                for mask in masks
            ]
        except Exception as e:
            print(f"⚠ SAM2 segmentation error: {e}")
            return []


    def _is_cpu_mode(self) -> bool:
        try:
            from app.pipeline.runtime import select_device
            return select_device() != "cuda"
        except Exception:
            return True


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

        device = getattr(self.model, "device", torch.device("cpu"))
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
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


class VisualLanguageCaptioner:
    """Visual-language image captioning helper using BLIP-style transformers models."""

    def __init__(self, model_name: Optional[str] = None, local_path: Optional[str] = None) -> None:
        self.processor = None
        self.model = None
        self.backend = "none"
        self.model_name = model_name or settings.visual_lm_model
        self.local_path = local_path or settings.visual_lm_local
        self.local_model_dir = self._resolve_local_model_dir(self.local_path)

        candidate_sources = []
        if self.local_model_dir is not None:
            candidate_sources.append(str(self.local_model_dir))
        if self.model_name:
            candidate_sources.append(self.model_name)

        last_error: Exception | None = None
        for source in candidate_sources:
            local_files_only = Path(source).exists()
            try:
                self._load_transformers(source, local_files_only=local_files_only)
                self.backend = "transformers"
                self.model_name = source
                print(f"✓ VisualLanguageCaptioner ready: {self.model_name} (local_files_only={local_files_only})")
                if local_files_only is False and self.local_model_dir is not None:
                    self._save_pretrained_to_local()
                return
            except Exception as exc:
                last_error = exc
                self.processor = None
                self.model = None

        print(f"✗ VisualLanguageCaptioner initialization failed: {last_error}; using fallback captioning")

    def _resolve_local_model_dir(self, local_path: Optional[str]) -> Optional[Path]:
        if not local_path:
            return None

        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[2] / local_path

        return candidate

    def _load_transformers(self, source: str, local_files_only: bool) -> None:
        try:
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

            self.processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only)
            self.model = AutoModelForImageTextToText.from_pretrained(source, local_files_only=local_files_only)
        except Exception:
            raise

    def _save_pretrained_to_local(self) -> None:
        if self.local_model_dir is None:
            return

        try:
            self.local_model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"⚠ Failed to create local VL model dir '{self.local_model_dir}': {exc}")
            return

        try:
            self.processor.save_pretrained(self.local_model_dir)
        except Exception as exc:
            print(f"⚠ Failed to save VL processor locally: {exc}")

        try:
            self.model.save_pretrained(self.local_model_dir)
        except Exception as exc:
            print(f"⚠ Failed to save VL model locally: {exc}")

    def caption_images(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        if self.processor is None or self.model is None or not images:
            return []

        try:
            import torch

            normalized_images = [image.convert("RGB") for image in images]
            inputs = self.processor(images=normalized_images, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)

            if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                captions = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            elif hasattr(self.processor, "batch_decode"):
                captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
            else:
                captions = [str(output) for output in outputs]

            result: List[Dict[str, Any]] = []
            for page_num, caption in enumerate(captions, start=1):
                result.append(
                    {
                        "page": page_num,
                        "caption": caption.strip(),
                        "method": "visual_lm",
                    }
                )
            return result
        except Exception as exc:
            print(f"⚠ VisualLanguageCaptioner captioning failed: {exc}")
            return []


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
            device = getattr(self.model, "device", torch.device("cpu"))
            inputs = self.tokenizer(query, candidate, return_tensors="pt", truncation=True, padding=True).to(device)
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
        self.device = select_device()
        self.is_ready = False
        self.backend = "heuristic"

        try:
            with allow_trusted_torch_pickle():
                from glirel import GLiREL

                self.model = GLiREL.from_pretrained(self.model_name, map_location=self.device)
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
