# Pipeline Stage Failure Analysis and Remediation

Generated: 2026-07-15

Purpose
-------
This document inspects the pipeline stages that produced the least useful outputs for the test document `1_Hydro MPC-1-10 (1).pdf` and provides an in-depth analysis for each stage labeled as weak or failed. For each stage we include: observed failures (evidence), likely root causes, the ideal goal and expected outputs, short/medium/long-term remediation steps, measurable metrics, testing suggestions, and example configuration or prompt ideas.

Bad stages audited
------------------
- Stage 7 — `sam2_segmentation`
- Stage 10 — `yolo_pid_detector`
- Stage 11 — `pid_symbol_detection`
- Stage 12 — `pid_component_detection`
- Stage 16 — `relation_extraction`
- Stage 18 — `bge_reranking`
- Stage 19 — `qwen2_5_vl`
- Stage 20 — `graphrag_analysis`
- Stage 21 — `copilot_analysis`

Methodology
-----------
Inputs: the PDF `1_Hydro MPC-1-10 (1).pdf` (a 10-page Hydro MPC installation manual) and the JSON outputs located in `data/pipeline/*` produced by a single run. Evidence snippets are taken from the stage JSONs captured during the run.

Scope and assumptions
---------------------
- The document is technical, primarily text with headings and a few images/figures; it is not formula-heavy.
- The pipeline is a mix of OCR, vision models and LLM/graph stages. Some stages are optional and may be skipped when a model is unavailable.

Per-stage deep analysis
=======================

Stage 7 — sam2_segmentation
---------------------------
Observed failure (evidence)
- Output: `text_length=0`, `full_output.segments` contains 3 very small segment boxes, `stage4_context=0`.
- Segments appear tiny (example bbox values) and do not correspond to page-level meaningful regions.

What went wrong (likely causes)
- Prompting/context mismatch: segmentation was probably asked to produce masks for region types that don't exist or were not supplied useful prompts.
- Weak or insufficient prompts/clicks for a prompt-driven SAM run.
- Poor integration between layout/OCR and segmentation: coordinates and context (page-scale geometry, anchors) not passed.
- Model scale or timeout limits caused truncated outputs.

Ideal / goal outputs
- Page-level meaningful segments: full-page text regions, figures, tables, headers, and other regions of interest.
- Masks and bounding boxes with confidence and stable IoU values; segmentation maps that map to OCR text blocks.
- Per-segment metadata: page id, approximate area, suggested label (text/table/figure).

Metrics to measure
- Coverage: percent of text tokens covered by segments overlapping OCR blocks.
- Average segment IoU with annotated ground truth regions.
- Mean number of useful segments per page (vs noise segments).

Short-term fixes (quick wins)
- Pass page-scale context (page width/height, layout bbox) to SAM prompts so masks are generated at correct scale.
- Use deterministic seeds and prompt templates so SAM is invoked reproducibly.
- Limit SAM to two prompt modes: automatic grid of boxes + layout-driven anchors (rather than open prompting) to increase coverage.

Medium-term improvements
- Implement a layout→segmentation adapter that converts layout bounding boxes and text blocks into SAM prompts (boxes or points) automatically.
- Train a lightweight classifier that labels SAM segments into `text/table/figure/figure_caption` using a small annotated set from the document corpus.

Long-term research & upgrades
- Consider fine-tuning a mask-diffusion model or a document-specialized SAM variant on synthetic document crops to improve region detection on manuals.
- Combine segmentation masks with OCR token anchors to produce token-aligned masks (for fine-grained table extraction).

Validation and tests
- Unit test: for a small annotated sample, assert coverage > 95% of annotated text tokens.
- Integration test: segment count per page compared to expected distribution (sanity check for empty/oversized sets).

Example config/prompt
- Use layout bbox center points as `prompt_points` to SAM plus a small radius adapted to page DPI.

---

Stage 10 — yolo_pid_detector
---------------------------
Observed failure (evidence)
- Output contained mostly `book` detections on pages 3 and 5 with low confidence (~0.40) and empty detections on many pages.

What went wrong (likely causes)
- Model mismatch: the YOLO model used is likely trained on general objects (e.g., book) rather than domain-specific PID symbols/equipment imagery.
- Low-quality pre/post-processing: box thresholds and NMS may be misconfigured, yielding only low-confidence detections.
- Domain shift: manual pages are document photos/scans — object appearance differs from model training images.

Ideal / goal outputs
- Detect domain-relevant elements (diagrams, schematics, pumps, valves, electrical panels, figures) with high precision/recall, per-page localization, and confidence scores.

Metrics to measure
- Precision/Recall for domain labels on a labeled test set. mAP@0.5 as a standard.
- False positive rate on pages with only text.

Short-term fixes
- Lower detection threshold to capture more proposals, then filter using a downstream classifier (reduce false negatives).
- Replace the generic YOLO weights with a fine-tuned model trained on document-region labels (book, figure, diagram, schematic, pump, valve).

Medium-term improvements
- Collect a small labeled dataset of pages with annotated components (pump, valve, gauge, schematic) and fine-tune the detector.
- Add an ensemble step: run a general detector + document-specific detector and fuse results with a lightweight classifier.

Long-term research
- Train detectors on synthetic rendered diagrams and real scans to close domain gap; consider multi-scale detectors and text-aware object models.

Validation and tests
- mAP, per-class recall for equipment symbols. Create CI test to run detector on a held-out set and assert mAP>0.6 for critical classes.

---

Stage 11 — pid_symbol_detection
-------------------------------
Observed failure (evidence)
- Output listed two symbols labeled as `book` (same as YOLO), count=2, with confidences ≈ 0.40–0.45.

What went wrong (likely causes)
- Upstream detector (YOLO) produced poor proposals; symbol detector simply relabeled those proposals.
- Symbol classifier is likely generic (trained on COCO-like data) and not PID-specific.

Ideal / goal outputs
- A curated set of symbol detections relevant to P&ID and manuals: icons (warning, danger), schematic symbols, connectors, and localized captions.

Short-term fixes
- Add heuristic filters: ignore detections with label `book` when document layout suggests text blocks; instead, infer `figure`.
- Re-run symbol classification only on high-confidence region proposals (>0.65) to avoid relabeling noise.

Medium-term improvements
- Build a small PID-symbol classifier (transfer learning) using 200–500 symbol images per label.
- Augment training with synthetic symbol renderings over document backgrounds.

Validation and tests
- Per-class precision on a labeled symbol set; require precision>0.75 for critical symbols like safety/warning.

---

Stage 12 — pid_component_detection
----------------------------------

### Observed failure (evidence)

**Current output:**
```json
{
  "full_output": [
    "pump",
    "valve",
    "motor",
    "sensor",
    "line",
    "control",
    "tank"
  ]
}
```

**Critical deficiencies:**
- Canned list with zero spatial grounding (no page ids, bounding boxes, or character offsets).
- No confidence scores or frequency counts.
- No provenance: which text span or visual region yielded each component?
- No linking to canonical part numbers or equipment identifiers.
- Identical list regardless of document content (stateless fallback behavior).

**Evidence from document:** The test PDF `1_Hydro MPC-1-10 (1).pdf` contains multiple component mentions:
- "Hydro MPC booster systems" (page 3)
- "CR 120 or CR 150 pumps" (page 3, specific model variants)
- "manifolds", "sealing compound", "control cabinet" (pages 4–5)
- "expansion joints", "vibration dampeners" (page 4)
- These should be extracted as localized instances, not generic tokens.

---

### Root cause analysis (deep dive)

**1. Architecture / integration issue (primary cause)**
- Stage 12 likely receives empty or unprocessed input from stage 10–11 (YOLO detector failed with generic "book" labels).
- Without valid visual detections upstream, the fallback is a hardcoded component list (no real linking logic).
- **Evidence:** Stage 10 output shows only "book" labels on pages 3 and 5 (false positives). Stage 11 then relabels those same detections. By stage 12, the proposal set is compromised.

**2. Missing text-to-component linking engine**
- No bridge between OCR text (stage 1) and component aggregation.
- If visual detection is disabled/failed, there is no fallback to extract components from text entities (stage 15).
- **Impact:** Component detection becomes a static list rather than a dynamic, document-specific extraction.

**3. Absence of component taxonomy and canonical mapping**
- No predefined list of valid part/component types specific to Hydro MPC systems (e.g., "CR 120 pump", "FlexCon expansion tank", "control variant").
- Current list ["pump", "valve", "motor", ...] is too generic (applies to any industrial device).
- No mapping from detected entity name to canonical part ID, SKU, or equipment catalog reference.

**4. No signal propagation from earlier stages**
- Stage 15 (entity extraction) extracted 239 entities including "Hydro MPC", "pump", "valve", "control variant" with confidence scores and entity types.
- Stage 12 does not consume stage 15 output; it reinvents the wheel with a hardcoded list.
- This is a design flaw: stages should be orchestrated to pass refined signals downstream.

---

### Ideal / goal outputs (detailed specification)

**Expected JSON structure (for the test PDF):**
```json
{
  "timestamp": "2026-07-15T12:00:00.000Z",
  "stage": "pid_component_detection",
  "status": "completed",
  "full_output": {
    "components": [
      {
        "name": "Hydro MPC booster system",
        "canonical_id": "hydro_mpc_booster",
        "entity_type": "equipment",
        "detected_via": "text_entity",
        "confidence": 0.95,
        "occurrences": [
          {
            "page": 3,
            "position": "start",
            "char_offset": [142, 169],
            "ocr_snippet": "These installation and operating instructions apply to the Grundfos Hydro MPC booster systems.",
            "context_window": 50
          },
          {
            "page": 5,
            "position": "mid",
            "char_offset": [450, 477],
            "ocr_snippet": "The Hydro MPC booster systems with CR 120 or CR 150 pumps are secured by means of transport straps.",
            "context_window": 50
          }
        ],
        "variants": ["CR 120", "CR 150"],
        "related_entities": ["pump", "motor", "control_variant"]
      },
      {
        "name": "CR 120 pump",
        "canonical_id": "cr_120_pump",
        "entity_type": "equipment_variant",
        "detected_via": "text_entity",
        "confidence": 0.88,
        "occurrences": [
          {
            "page": 3,
            "position": "mid",
            "char_offset": [487, 495],
            "ocr_snippet": "The Hydro MPC booster systems with CR 120 or CR 150 pumps...",
            "context_window": 50
          }
        ],
        "parent_equipment": "hydro_mpc_booster",
        "model_number": "CR 120",
        "specifications": {
          "pump_type": "centrifugal",
          "max_pressure_bar": "10"
        }
      },
      {
        "name": "expansion joint",
        "canonical_id": "expansion_joint_generic",
        "entity_type": "accessory",
        "detected_via": ["text_entity", "layout_context"],
        "confidence": 0.82,
        "occurrences": [
          {
            "page": 4,
            "position": "mid",
            "char_offset": [1203, 1220],
            "ocr_snippet": "...we recommend that you fit expansion joints on the inlet and outlet pipes...",
            "context_window": 80,
            "associated_image": "Fig. 1"
          }
        ],
        "quantity_mentioned": 2,
        "installation_notes": "inlet and outlet"
      },
      {
        "name": "manifold",
        "canonical_id": "manifold_generic",
        "entity_type": "component",
        "detected_via": "text_entity",
        "confidence": 0.79,
        "occurrences": [
          {
            "page": 4,
            "position": "mid",
            "char_offset": [890, 910],
            "ocr_snippet": "Connect the pipes to the manifolds of the booster system.",
            "context_window": 50
          }
        ]
      }
    ],
    "summary": {
      "total_unique_components": 4,
      "total_mentions": 6,
      "detection_methods": {
        "text_entity": 4,
        "visual_detection": 0,
        "layout_context": 1
      },
      "coverage_notes": "Strong text extraction; visual detection (stage 10) produced no valid proposals, so text extraction was primary signal."
    },
    "quality_flags": [
      "low_visual_detection_confidence",
      "fallback_to_text_extraction_only"
    ]
  }
}
```

**Key improvements over current output:**
1. **Spatial grounding:** page id, character offsets, OCR snippets
2. **Confidence scores** and multiple occurrence instances
3. **Canonical IDs** and taxonomies (e.g., `cr_120_pump` vs. just `pump`)
4. **Multimodal provenance:** text_entity, visual_detection, layout_context
5. **Component relationships:** parent equipment, variants, related entities
6. **Actionable metadata:** specifications, installation notes, quantities
7. **Quality summary:** detection methods and fallback signals

---

### Metrics to measure (rigorous evaluation)

**1. Extraction coverage (recall)**
```
Recall = (# correctly extracted components) / (# ground truth components in document)
Goal: > 90%
Test: manually annotate 10–20 documents with all component mentions; compute recall per document.
```

**2. Localization accuracy (precision)**
```
Precision = (# components with valid page id + character offset) / (# total components returned)
Goal: > 95%
Sub-metric: offset accuracy within ±20 chars
```

**3. Canonical linking accuracy**
```
Linking F1 = harmonic mean of precision and recall for canonical_id predictions
Goal: F1 > 0.85
Test: for each extracted component, verify canonical_id matches a part catalog.
```

**4. Multimodal integration (confidence in fallback)**
```
When visual detection fails, text extraction should trigger with high confidence.
Metric: precision of text-only components (should exceed 0.8 on document corpus)
```

**5. Component class distribution (sanity check)**
```
Expected: diversity of component types (pump, motor, valve, control, accessories)
Anomaly: if all returns are "pump" → likely hardcoded fallback
Threshold: ≥ 3 distinct component types per non-trivial document
```

---

### Short-term fixes (1–2 weeks, low-code)

#### Fix 1: Inject entity extraction as fallback
**Goal:** Stage 12 should consume stage 15 output (entities) and fallback text extraction when visual detection is weak.

**Code sketch:**
```python
def pid_component_detection_v2(stage10_detections, stage15_entities, ocr_full_text):
    """
    Improved component detection with fallback to text entity extraction.
    """
    components = {}
    
    # 1. Try visual detections first (from stage 10)
    if stage10_detections and len(stage10_detections) > 0:
        for detection in stage10_detections:
            if detection['confidence'] > 0.65:  # high-confidence threshold
                comp_name = detection['label']
                if comp_name not in components:
                    components[comp_name] = {
                        'name': comp_name,
                        'detected_via': 'visual',
                        'confidence': detection['confidence'],
                        'occurrences': [detection]
                    }
    
    # 2. Fallback: extract components from text entities (stage 15)
    component_keywords = {
        'pump': ['pump', 'CR 120', 'CR 150'],
        'valve': ['valve', 'check valve', 'expansion valve'],
        'motor': ['motor', 'electric motor', 'drive'],
        'sensor': ['sensor', 'transmitter', 'gauge', 'pressure gauge'],
        'control': ['control', 'control variant', 'control cabinet'],
        'manifold': ['manifold', 'intake manifold', 'outlet manifold'],
        'expansion_joint': ['expansion joint', 'vibration dampener', 'damper'],
        'sealing': ['sealing compound', 'gasket', 'seal'],
    }
    
    # Extract component mentions from OCR text using entity offsets
    for entity in stage15_entities:
        entity_name = entity['name'].lower()
        entity_text = entity.get('canonical_name', '')
        
        # Match entity against known component keywords
        for comp_type, keywords in component_keywords.items():
            if any(kw.lower() in entity_name or kw.lower() in entity_text for kw in keywords):
                if comp_type not in components:
                    components[comp_type] = {
                        'name': entity_name,
                        'detected_via': 'text_entity',
                        'confidence': entity.get('confidence', 0.5),
                        'occurrences': [],
                        'canonical_id': f'{comp_type}_{entity.get("canonical_name", "unknown")}'
                    }
                
                # Append occurrence with page reference
                components[comp_type]['occurrences'].append({
                    'page': entity.get('page', 'unknown'),
                    'char_offset': [entity.get('start', -1), entity.get('end', -1)],
                    'entity_id': entity.get('canonical_name'),
                    'confidence': entity.get('confidence', 0)
                })
    
    return {
        'timestamp': datetime.now().isoformat(),
        'stage': 'pid_component_detection',
        'status': 'completed',
        'full_output': {
            'components': list(components.values()),
            'summary': {
                'total_unique_components': len(components),
                'total_mentions': sum(len(c['occurrences']) for c in components.values()),
                'detection_methods': {
                    'visual': sum(1 for c in components.values() if c['detected_via'] == 'visual'),
                    'text_entity': sum(1 for c in components.values() if c['detected_via'] == 'text_entity')
                }
            }
        }
    }
```

**Integration:**
- Modify the pipeline orchestrator so stage 12 receives both stage 10 and stage 15 outputs.
- Add configuration file (`component_taxonomy.json`) with predefined component keywords and canonical IDs.

**Expected impact:** Output moves from 7 generic items to 8–12 localized, evidence-backed components with page references.

---

#### Fix 2: Add simple canonical mapping
**Goal:** Link detected component names to a canonical part/equipment taxonomy.

**Data file: `app/data/component_taxonomy.json`**
```json
{
  "pump": {
    "canonical_id": "pump_generic",
    "variants": ["CR 120", "CR 150", "centrifugal pump"],
    "category": "mechanical",
    "typical_pressures": "1–10 bar"
  },
  "CR_120_pump": {
    "canonical_id": "cr_120_pump",
    "parent": "pump_generic",
    "model_number": "CR 120",
    "manufacturer": "Grundfos",
    "specifications": {
      "type": "centrifugal",
      "max_pressure_bar": 10,
      "nominal_flow_m3h": 2.0
    }
  },
  "expansion_joint": {
    "canonical_id": "expansion_joint_generic",
    "category": "accessory",
    "purpose": "vibration damping and thermal expansion",
    "keywords": ["expansion joint", "damper", "vibration dampener"]
  }
}
```

**Usage in stage 12:**
```python
with open('app/data/component_taxonomy.json') as f:
    taxonomy = json.load(f)

def map_to_canonical(component_name):
    for key, data in taxonomy.items():
        if component_name.lower() in [key.lower()] + [v.lower() for v in data.get('variants', [])]:
            return data['canonical_id']
    return f'unknown_{component_name}'
```

---

#### Fix 3: Add page-level linkage
**Goal:** Every component should record which page(s) it appears on.

**Strategy:**
- Store OCR text as a dict mapping `page_id -> text` during stage 1.
- In stage 12, search for component keywords within each page's OCR text.
- Record occurrence with page id and approximate character offset.

---

### Medium-term improvements (4–12 weeks, moderate effort)

#### Improvement 1: Build a component linking classifier
**Objective:** Train a lightweight text classifier that maps component mentions to canonical IDs.

**Dataset construction:**
- Collect 500–1000 labeled (mention, canonical_id) pairs from manuals in your document corpus.
- Examples:
  - `"CR 120 pump" → cr_120_pump`
  - `"Hydro MPC booster system" → hydro_mpc_booster`
  - `"expansion joint" → expansion_joint_generic`

**Model architecture:**
- Use a fine-tuned DistilBERT or similar lightweight transformer.
- Input: component mention text + context (surrounding 50 chars).
- Output: canonical_id class (50–100 classes depending on catalog size).

**Training code sketch:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json

# Load labeled data
with open('data/component_mentions_labeled.json') as f:
    data = json.load(f)  # [{mention, canonical_id, context}, ...]

# Build dataset
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(d['canonical_id'] for d in data)))

# Tokenize and prepare
def tokenize_fn(example):
    text = example['mention'] + ' [SEP] ' + example['context']
    tokens = tokenizer(text, truncation=True, max_length=128, padding='max_length')
    tokens['labels'] = label_to_id[example['canonical_id']]
    return tokens

dataset = HFDataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()}).map(tokenize_fn)

# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir='./models/component_linker', num_train_epochs=5),
    train_dataset=dataset
)
trainer.train()
```

#### Improvement 2: Multimodal component fusion
**Objective:** Fuse text entities + visual detections to produce higher-confidence component instances.

**Strategy:**
- For each text-extracted component, search for related visual detections (using NLP embedding similarity or rule-based matching).
- If a visual detection is nearby (same page, overlapping region), increase confidence and record both modalities.

**Implementation:**
```python
def fuse_multimodal_components(text_components, visual_detections):
    """
    Fuse text-extracted and visually-detected components.
    """
    fused = {}
    
    # Start with text components
    for comp in text_components:
        fused[comp['canonical_id']] = {**comp, 'detected_via': 'text_entity'}
    
    # Try to match visual detections
    for vdet in visual_detections:
        if vdet['confidence'] > 0.65:
            # Find nearest text component on same page
            page_id = vdet['page']
            best_match = None
            best_sim = 0.0
            
            for text_comp in text_components:
                if text_comp['page'] == page_id:
                    # Simple keyword matching (could use embeddings for sophistication)
                    if any(kw in vdet['label'].lower() for kw in text_comp['keywords']):
                        best_match = text_comp['canonical_id']
                        best_sim = 0.9
                        break
            
            if best_match and best_sim > 0.7:
                # Upgrade confidence and record both sources
                fused[best_match]['detected_via'] = ['text_entity', 'visual']
                fused[best_match]['confidence'] = max(fused[best_match].get('confidence', 0.5), vdet['confidence'])
                fused[best_match]['visual_bboxes'] = fused[best_match].get('visual_bboxes', []) + [vdet['bbox']]
    
    return list(fused.values())
```

#### Improvement 3: Component relationship graph
**Objective:** Build a knowledge graph of component relationships (e.g., "CR 120 pump is part of Hydro MPC booster").

**Data structure:**
```json
{
  "component_graph": {
    "hydro_mpc_booster": {
      "has_pump": ["cr_120_pump", "cr_150_pump"],
      "has_motor": ["electric_motor_generic"],
      "has_accessory": ["expansion_joint_generic", "manifold_generic"]
    },
    "cr_120_pump": {
      "part_of": "hydro_mpc_booster",
      "model": "CR 120",
      "max_pressure": 10
    }
  }
}
```

**Usage in stage 12:**
- When extracting "Hydro MPC", automatically populate expected sub-components (pump, motor) with lower confidence if not explicitly mentioned.
- This helps recover missed components and improves coverage.

---

### Long-term research & architecture (3–6 months)

#### Research 1: Document-specific component ontology learning
**Goal:** Automatically learn a component taxonomy from a corpus of documents (unsupervised or semi-supervised).

**Approach:**
- Use topic modeling (LDA or BERTopic) to cluster component mentions across documents.
- Build a hierarchy of component types automatically (e.g., pump → centrifugal pump → CR 120 pump).
- Link clusters to canonical catalogs via entity linking models.

#### Research 2: Multimodal component recognition
**Goal:** Train end-to-end model that recognizes components from image + OCR jointly.

**Approach:**
- Collect labeled dataset of component instances (visual region + text mention pairs).
- Train a vision transformer + text encoder jointly using contrastive loss (e.g., CLIP-like architecture).
- Fine-tune on domain-specific manuals.

#### Research 3: Component relationship extraction via graph neural networks
**Goal:** Predict component relationships (part-of, requires, etc.) using GNN.

**Approach:**
- Build a graph of components as nodes; entity relationships as edges.
- Train a GNN to predict missing edges using textual and visual features.
- Integrate with stage 16 (relation extraction) for synergy.

---

### Validation & testing (comprehensive suite)

#### Unit tests
```python
import pytest
from stage_12 import pid_component_detection_v2

def test_component_detection_returns_list():
    """Ensure output is a list of component dicts, not a canned list."""
    result = pid_component_detection_v2(stage10=[], stage15=[])
    assert isinstance(result['full_output']['components'], list)

def test_component_detection_with_entities():
    """Test fallback extraction from stage 15 entities."""
    mock_entities = [
        {'name': 'Hydro MPC', 'canonical_name': 'hydro_mpc', 'page': 3, 'start': 100, 'end': 110, 'confidence': 0.95},
        {'name': 'CR 120 pump', 'canonical_name': 'cr_120_pump', 'page': 3, 'start': 150, 'end': 165, 'confidence': 0.88}
    ]
    result = pid_component_detection_v2(stage10=[], stage15=mock_entities)
    components = result['full_output']['components']
    
    assert len(components) >= 2
    assert any('hydro_mpc' in c['canonical_id'].lower() for c in components)
    assert any('pump' in c['name'].lower() for c in components)

def test_component_has_page_id():
    """Ensure all components have spatial grounding."""
    result = pid_component_detection_v2(stage10=[], stage15=mock_entities)
    components = result['full_output']['components']
    
    for comp in components:
        assert len(comp['occurrences']) > 0
        for occ in comp['occurrences']:
            assert 'page' in occ
            assert 'char_offset' in occ or 'bbox' in occ

def test_canonical_id_coverage():
    """Ensure all components are mapped to canonical IDs."""
    result = pid_component_detection_v2(stage10=[], stage15=mock_entities)
    components = result['full_output']['components']
    
    for comp in components:
        assert 'canonical_id' in comp
        assert comp['canonical_id'] != ''

def test_no_hardcoded_fallback():
    """Detect if stage 12 is returning a hardcoded list."""
    result1 = pid_component_detection_v2(stage10=[], stage15=mock_entities_1)
    result2 = pid_component_detection_v2(stage10=[], stage15=mock_entities_2)
    
    comps1 = [c['name'] for c in result1['full_output']['components']]
    comps2 = [c['name'] for c in result2['full_output']['components']]
    
    assert comps1 != comps2, "Output should vary based on input, not be hardcoded"
```

#### Integration tests (on real documents)
```python
def test_stage12_on_hydro_mpc_manual():
    """Test on the test PDF: 1_Hydro MPC-1-10 (1).pdf"""
    ocr_text = load_ocr_output('data/pipeline/1.docling_surya_ocr/ocr_output.json')
    entities = load_entity_output('data/pipeline/15.entity_extraction/stage15_output.json')
    detections = load_detections('data/pipeline/10.yolo_pid_detector/stage10_output.json')
    
    result = pid_component_detection_v2(stage10=detections, stage15=entities)
    components = result['full_output']['components']
    
    # Expected: at least these components mentioned in the Hydro MPC manual
    expected_keywords = ['hydro_mpc', 'pump', 'valve', 'control', 'motor', 'expansion']
    found_keywords = set()
    
    for comp in components:
        for kw in expected_keywords:
            if kw in comp['canonical_id'].lower() or kw in comp['name'].lower():
                found_keywords.add(kw)
    
    # Require coverage > 80% of expected
    assert len(found_keywords) / len(expected_keywords) > 0.8, f"Missing: {expected_keywords - found_keywords}"
```

#### Performance benchmarks
- **Throughput:** stage 12 should complete in < 500ms per document (no heavy model inference).
- **Memory:** < 200 MB peak for 100+ component instances.
- **Coverage:** ≥ 90% recall on a labeled evaluation set of 20 documents.

---

### Developer checklist for improvement

- [ ] Reproduce current failure locally with the test PDF.
- [ ] Implement Fix 1: fallback to stage 15 entities.
- [ ] Add component taxonomy config file.
- [ ] Write unit tests and ensure all pass.
- [ ] Test on 5–10 additional manuals to verify generalization.
- [ ] Add telemetry: log detection method (visual vs. text) per stage run.
- [ ] Plan medium-term improvement: component linking classifier.
- [ ] Document canonical_id schema and taxonomy in README.

---

### Integration Points (orchestration notes)

1. **Upstream dependencies (fixed):**
   - Stage 10 (visual detection) → may produce weak proposals.
   - Stage 15 (entity extraction) → primary fallback signal; **must be passed to stage 12**.

2. **Downstream consumers (affected):**
   - Stage 20 (graphrag_analysis) and stage 21 (copilot_analysis) should consume stage 12 output to ground their reasoning (e.g., "which components are mentioned in maintenance recommendations?").

3. **Recommended orchestration change:**
   ```
   OLD: Stage 10 → Stage 11 → Stage 12 (chain, weak)
   NEW: Stage 10 + Stage 15 → Stage 12 (fusion, strong)
   ```

---

### Example output after improvements (test PDF)

```json
{
  "full_output": {
    "components": [
      {
        "canonical_id": "hydro_mpc_booster",
        "name": "Hydro MPC booster system",
        "confidence": 0.95,
        "occurrences": [
          {"page": 3, "char_offset": [142, 169], "context_snippet": "...Grundfos Hydro MPC booster systems."},
          {"page": 3, "char_offset": [450, 477], " ": "...CR 120 or CR 150 pumps..."}
        ],
        "detected_via": "text_entity",
        "category": "primary_equipment"
      },
      {
        "canonical_id": "cr_120_pump",
        "name": "CR 120 pump",
        "confidence": 0.88,
        "occurrences": [
          {"page": 3, "char_offset": [487, 495], "context_snippet": "CR 120 or CR 150 pumps..."}
        ],
        "detected_via": "text_entity",
        "parent_component": "hydro_mpc_booster",
        "model": "CR 120"
      },
      {
        "canonical_id": "expansion_joint",
        "name": "expansion joint",
        "confidence": 0.82,
        "occurrences": [
          {"page": 4, "char_offset": [1203, 1220], "context_snippet": "...fit expansion joints on the inlet and outlet pipes..."}
        ],
        "detected_via": "text_entity",
        "quantity": 2,
        "installation_location": ["inlet", "outlet"]
      }
    ],
    "summary": {
      "total_components": 12,
      "coverage_pct": 92,
      "detection_methods": {"text_entity": 12, "visual": 0}
    }
  }
}
```

**Improvement over current:** from 7 generic, non-grounded items → 12 localized, canonical, and evidence-backed component instances.

---

Stage 16 — relation_extraction
-----------------------------
Observed failure (evidence)
- Output: empty list `full_output = []` — no relations found despite multiple entities extracted (stage15 produced ~239 entities).

What went wrong (likely causes)
- Relation model may not have been invoked correctly or lacks prompt/context (entity spans not passed properly).
- Task mismatch: the chosen relation extraction approach may require document-aware context windows that were not provided.
- Model capacity or thresholding could be too strict, discarding all low-confidence relations.

Ideal / goal outputs
- Typed relations between entities (e.g., `part_of(Hydro MPC, pump)`, `requires(electrical_installation, control_variant)`) with span references, confidence scores, and provenance (text snippet or page id).

Metrics
- Precision/Recall on a labeled relation corpus; F1 preferred. Average precision for top-k relations.

Short-term fixes
- Verify entity offsets and ensure the relation model receives entity spans or canonical ids.
- Lower inference thresholds temporarily and log candidates to inspect false positives.

Medium-term improvements
- Use joint modeling: train a relation extractor that uses both local sentence windows and cross-sentence attention across the document.
- Add weak supervision rules (distant supervision from headings, bullet lists, and co-occurrence) to bootstrap relations.

Long-term research
- Explore graph neural networks on entity embeddings to predict relation edges, and fine-tune on domain relation labels.

Validation and tests
- Relation F1 on held-out labeled documents; add unit tests to ensure relations are non-empty for documents that contain known relations.

---

Stage 18 — bge_reranking
------------------------
Observed failure (evidence)
- Output: `full_output`: {"ranked": [], "source": "reranker_unavailable", "reason": "intentionally_skipped"}

What went wrong (likely causes)
- Reranker model is unavailable on this environment (missing weights, disabled service, or license limits). The pipeline intentionally skipped reranking.

Ideal / goal outputs
- Reranked candidate passages or document chunks by relevance to given queries (for retrieval tasks). Output should include ranking scores and IDs.

Short-term fixes
- Detect missing model early and fall back to a simple lexical scoring (BM25 or cosine on embeddings) to provide partial ranking.
- Add explicit telemetry and actionable logs when reranker unavailable.

Medium-term improvements
- Package a lightweight reranker (distilled) and include in containerized dependencies so reranker is available offline.
- Add an evaluation harness that checks reranker availability during CI and runs a small ranking test.

Long-term research
- Explore cross-encoder rerankers and distillation strategies so a high-quality ranker can run in constrained environments.

Validation and tests
- NDCG@k on a labeled query→passage dataset; add integration tests to fail the pipeline if reranker is essential but missing.

---

Stage 19 — qwen2_5_vl
---------------------
Observed failure (evidence)
- Output: `images_processed=10`, `status="vl_model_unavailable"` (visual-language model unavailable).

What went wrong (likely causes)
- Visual-LM backend/model is not installed or not reachable. Could be licensing, missing container, or GPU resource availability.

Ideal / goal outputs
- Per-image multimodal outputs: image captions grounded in document context, object descriptions, figure-level summaries, and alignment to OCR text.

Short-term fixes
- Provide an offline fallback: simple image captioning model (open-source) or extract surrounding OCR text as proxy captions.
- Add clear runtime check and fallback pipeline to keep downstream stages functional.

Medium-term improvements
- Containerize and include a smaller multimodal model suitable for CPU or single-GPU inference (distilled vision-language model).
- Add orchestrator logic to allocate GPU resources when available and degrade gracefully to CPU.

Long-term research
- Fine-tune a vision-language model for document images and figure captions to reduce hallucinations and improve grounding.

Validation and tests
- Captioning BLEU/ROUGE against a small gold set; grounding accuracy for linking caption phrases to OCR tokens.

---

Stage 20 — graphrag_analysis
---------------------------
Observed failure (evidence)
- Output contains a partially hallucinated reasoning snippet beginning with: "dangerou s situation which, if not avoided, will result in serious personal injury." followed by many repeated `<!-- image -->` placeholders. `anomalies_detected` and `failure_risks` were empty lists and `confidence=0.9`.

What went wrong (likely causes)
- The analysis model produced high-confidence generic safety text (likely triggered by hazard words in the manual) instead of a grounded, evidence-based analysis.
- The stage likely concatenated template safety language with image placeholders because visual models were unavailable.

Ideal / goal outputs
- Evidence-backed analysis: detected anomalies (with page references and text or image snippets), failure risk estimations with supporting evidence, and concrete maintenance recommendations tied to parts/sections.

Short-term fixes
- Reduce model temperature and avoid overly confident default templates; force answers to cite document spans or return `no_evidence` when lacking data.
- Replace placeholder insertion logic with explicit checks: only include image placeholders when images are actually processed and available.

Medium-term improvements
- Use a pipeline that requires evidence: enforce that every claim must include provenance (span/page id or image id).
- Train a small classifier to detect whether a given claim is supported by document text or images, and block unproven claims.

Long-term research
- Build a multimodal explanation model that produces structured reasoning chains with verifiable provenance nodes and edge confidences.

Validation and tests
- Grounding precision: percent of claims with valid provenance. Add a benchmark of deliberate trap prompts to detect hallucination.

---

Stage 21 — copilot_analysis
--------------------------
Observed failure (evidence)
- Output contains a long `reasoning_chain` with repeated and overlapping suggestions (e.g., many repeated associations around `vibration` and `pressure`) and a `maintenance_plan` with generic entries. The content is not tightly grounded to document evidence and appears noisy.

What went wrong (likely causes)
- The high-level agent produced a best-effort plan using entity lists without being constrained by evidence; this produced repetitive and generic results.
- Lack of grounding constraints and missing fact-checking/disambiguation steps.

Ideal / goal outputs
- Concise executive summary of the document's operational/maintenance instructions, a prioritized maintenance plan tied to specific sections and page references, and a small set of high-confidence root-cause hypotheses with supporting evidence.

Short-term fixes
- Post-process the agent output to deduplicate repeated items and require evidence links for each recommended action.
- Add a hallucination-check: require the Copilot stage to return a `provenance` field for every high-impact claim.

Medium-term improvements
- Use a retrieval-augmented generation approach: Copilot should retrieve candidate evidence chunks (via semantic index) and base reasoning only on those chunks through a chain-of-evidence routine.
- Add critic models that verify each claim against the retrieved chunks.

Long-term research
- Build an internal RAG orchestration that supports structured outputs (JSON schema) and instrument a human-in-the-loop verification workflow for safety-critical recommendations.

Validation and tests
- Use a small annotated set of documents with expected executive summaries and maintenance plans; score Copilot outputs using ROUGE and evidence-precision metrics. Add CI checks to ensure deduplication and provenance coverage.

Appendix: cross-cutting recommendations
=====================================

1. Fail-safe & graceful degradation
- Every stage that depends on an external model must declare availability at startup and provide a deterministic fallback (lexical, heuristic, or distilled model). Skipping silently reduces usefulness.

2. Provenance-first policy
- Enforce that high-level stages (graphrag, copilot, rerankers) must attach provenance (page id, OCR span, image id) for each claim. When provenance is absent, output a conservative "no evidence" response.

3. Telemetry & early detection
- Add a healthcheck that asserts model presence and minimal performance on a few toy inputs (unit sample) before running the full pipeline. Emit clear logs when stepping into fallback modes.

4. Small labeled evaluation sets
- Build a compact per-stage validation set (50–200 examples) that covers the most common document types you process (manuals, datasheets, reports). Run per-stage checks in CI.

5. Multimodal fusion patterns
- Prefer late fusion where high-confidence visual detections are cross-checked with text extractions. Avoid letting low-confidence visual output directly trigger high-impact textual reasoning stages.

6. Developer checklist for each remediation
- Reproduce locally with the same PDF and JSON outputs.
- Run focused unit tests for the stage (detect missing inputs, bad types, empty arrays).
- Add instrumentation to report fraction of empty outputs per stage across production runs.

Closing notes
=============
This audit is written to be actionable. Start with the short-term fixes to get immediate improvement (fallbacks, threshold tuning, provenance enforcement) and follow with medium-term data collection and model adaptation. Long-term research is recommended for stages that require domain adaptation (symbol detection, reranking, multimodal reasoning).

If you want, I can:
- Convert each "short/medium/long-term" list into tracked issues or PR checklists.
- Produce a minimal set of unit tests that assert non-empty, provably-grounded outputs for stages 13–21.























## Rating of the 21 pipeline stages on the test PDF

I compared the generated outputs against the content of 1_Hydro MPC-1-10 (1).pdf.pdf). The PDF is a Hydro MPC installation manual, so the stages that captured structure, headings, and document meaning were the ones that performed well.

### Overall verdict
- Strong / good: OCR, layout, segmentation, entity extraction, and basic structural stages
- Weak / bad: PID-specific detection, empty relation extraction, skipped reranking, and later “reasoning” stages that became generic or irrelevant

### Stage-by-stage rating

| Stage | Verdict | Why |
|---|---|---|
| 1. OCR | Good | Extracted the manual text well and captured real headings/sections from the PDF |
| 2. Layout analysis | Good | Identified document-style headings and layout structure |
| 3. Layout understanding | Good | Confirmed the document structure and heading flow |
| 4. Table structure analysis | Mixed | No meaningful tables were detected, which is acceptable for this PDF |
| 5. GroundingDINO detection | Good | Detected relevant document-region objects tied to the manual |
| 6. Cross-stage synthesis | Good | Produced a sensible overall quality summary and structure overview |
| 7. SAM2 segmentation | Bad | Produced very limited/weak segmentation and did not meaningfully help |
| 8. Nougat formula recognition | Bad | Returned no formulas, which is fine for this document but not useful here |
| 9. Reading order | Mixed | The output was effectively empty, so it did not add much value |
| 10. YOLO PID detector | Bad | Detected mostly “book” objects, which is not meaningful for this document |
| 11. PID symbol detection | Bad | The detected symbols were not relevant to the instruction manual |
| 12. PID component detection | Bad | The output was generic and not tied to the actual PDF content |
| 13. Document segmentation | Good | Split the document into meaningful chunks/sections from the manual |
| 14. Semantic indexing | Good | Indexed the document into chunks, which is useful for retrieval |
| 15. Entity extraction | Good | Extracted meaningful entities such as “Hydro MPC”, “installation”, “service”, etc. |
| 16. Relation extraction | Bad | Returned zero relations, so it failed to add value |
| 17. Entity linking | Good | Linked the extracted entities in a consistent way |
| 18. BGE reranking | Bad | Was skipped because the reranker was unavailable |
| 19. Qwen2.5-VL | Bad | The model was unavailable, so no useful visual reasoning was produced |
| 20. GraphRAG analysis | Bad | The output became generic and partly hallucinated rather than grounded |
| 21. Copilot analysis | Bad | The reasoning/maintenance summary was too generic and not well grounded in the PDF |

## Best-performing stages
These are the ones I would trust most for this PDF:
- 1, 2, 3, 5, 6, 13, 14, 15, 17

## Weakest stages
These were the least useful for this particular test:
- 7, 10, 11, 12, 16, 18, 19, 20, 21

## Bottom line
For this PDF, the pipeline is strongest at:
- OCR and document structure understanding

It is weakest at:
- PID-specific detection
- relation extraction
- later generative reasoning stages

If you want, I can next turn this into a simple “good / average / bad” scorecard or rank the stages from 1 to 21 with numeric scores.


































































































































Deep introspection (source-only) + final stage-by-stage rating
What I inspected (source files)
From app/pipeline/ I read the actual implementation of the core execution flow and key stages:

engine_v2.py (orchestrator + stage gating + fallback logic)
ocr_processor.py (OCR + formula/“reading_order” generation inside OCR)
entity_extractor.py (GLiNER chunking + labeling)
relation_extractor.py (GLiREL + heuristic + weak supervision + graph inference)
graphrag_summarizer.py (GraphRAG grounding + evidence gating + claim validation)
plus model_helpers.py (SAM2 prompt/segmentation logic, GroundingDINO prompting, YOLO PID detector interface, captioning fallback mechanics)
This lets me rate not just outcomes, but why those outcomes are mechanically plausible.

Final rating of each stage (1–21)
I’m rating against “useful output for the given industrial instruction manual PDF” and “does it materially improve end results beyond text-only extraction”.

1. OCR — 8.8/10
Code evidence: DoclingOCRProcessor uses Docling markdown export + Surya layout/table extraction when available; otherwise TesseractOCRProcessor fallback returns text.

Why not 10: The OCR “reading_order” is produced as a layout sorting heuristic (page/position), not true instruction-step sequencing. Also formula detection is text-pattern only.

2. Layout analysis (DocLayout-YOLO / Surya layout objects) — 8.0/10
Code evidence: engine_v2.py uses _analyze_doclayout_yolo with a fallback to Surya/ocr layout boxes (_build_ocr_layout_fallback).

Why Mixed in practice: Layout objects can be generic; stage doesn’t ensure semantic usefulness for downstream unless headings/entities line up.

3. Layout understanding (heading flow / structural summary) — 7.8/10
Code evidence: In engine_v2.py, synthesis stages derive headings mostly from labels heading in layout items (or from fallback heuristics).

Why not 9: “Layout understanding” is largely derived from OCR layout labels; there’s no robust document semantic model ensuring order like “Step 1/2/3”.

4. Table structure analysis — 5.5/10
Code evidence: _extract_tables_with_transformer uses heavy table-transformer detection and may detect boxes that aren’t tables; it returns tables from OCR first then transformer “detections”.

Matches your outcome: “No meaningful tables detected” is consistent.

5. GroundingDINO detection — 7.6/10
Code evidence: Prompt is built mainly from headings/text snippet; GroundingDINO runs over rendered page images.

Why “Good” but not full: DINO detects “phrases” tied to prompt, not necessarily the true instruction components for your document. It’s plausible it finds “document-region objects” but not necessarily useful symbols.

6. Cross-stage synthesis — 7.9/10
Code evidence: _build_structural_stage6_summary and _build_structural_stage3/5 compute quality heuristics like presence of headings/detections and generate a structured narrative.

Why not 9: It’s more “structural packaging” than true multi-modal reasoning.

7–12 Visual/PID pipeline: why they perform poorly on this PDF
7. SAM2 segmentation — 2.5/10
Code evidence (big deal):

_segment_with_sam only processes images[:1] when CUDA is not available:

pages_to_process = images[:1] if not is_cuda else images
SAM prompt-box segmentation quality depends completely on good boxes from GroundingDINO.
If boxes_by_page is empty, it falls back to a single center crop prompt (guaranteed but often meaningless for technical diagrams).
Why “Bad” matches code: For an instruction manual PDF (text + diagrams), random center crops won’t map to meaningful objects. Even with detections, SAM’s output is bbox/areas, but there’s no downstream logic that uses masks to improve ordering/relations.

8. Nougat formula recognition — 2.0/10
Code evidence: In engine_v2.py, _recognize_formulas is not Nougat at all—it’s just regex heuristics on OCR text patterns.

So for your manual PDF it returns empty (expected).

9. Reading order — 3.0/10
Code evidence: _build_reading_order in engine_v2.py simply returns ocr_result.get("reading_order", []).

In ocr_processor.py, reading_order is produced from Surya layout boxes sorted by (y, x). That’s not step-by-step instruction order; it’s “visual reading order”.

10. YOLO PID detector — 1.5/10
Code evidence:

YOLO runs on rendered images, but labels are whatever the YOLO model returns.
There’s no mapping from YOLO classes → instruction manual semantics.
Output is stored as raw detections without using them to filter entity extraction/relations.
11. PID symbol detection — 1.5/10
Code evidence: PIDSymbolDetector.detect() labels everything as "pid_symbol" and depends on YOLO weights:

If weights are missing/unloaded, returns [].
Even when loaded, it doesn’t interpret symbol categories—just generic symbol boxes.
12. PID component detection — 2.0/10
Code evidence: engine_v2.py calls:


result = detect_pid_components_enhanced(text)
But I did not see an integration path where detected components are used to:

improve relation extraction,
improve grounding prompts,
or enforce schema edge creation.
So even if it detects something, it typically doesn’t propagate into reasoning.

13–17 Knowledge graph construction (text side) works
13. Document segmentation — 8.2/10
Code evidence: _segment_document uses chunking (max_chars=1400, overlap=220) on normalized OCR markdown.

This directly supports semantic indexing and evidence-grounded reasoning.

14. Semantic indexing — 7.7/10
Code evidence: _index_text_chunks does embedding for each chunk if BGE exists.

Why not 9: It returns all embeddings in-memory as a list (not necessarily Qdrant-upsert in this stage), so it may not be used effectively by later “graph rag” unless advanced pipeline runs.

15. Entity extraction — 8.5/10
Code evidence: GLiNER chunking with max_chars=350, overlap=80 prevents truncation. Entities are merged by canonical_name and kept by highest confidence.

This explains your “meaningful entities” success.

16. Relation extraction — 2.0/10
Code evidence: This is where your output (“returned zero relations”) most likely comes from.

Even though relation_extractor.py has strong fallback logic, the inputs likely don’t satisfy its preconditions:

It returns [] early if:


if not text or len(entities) < 2: return []
It relies on _entity_appears_in_text(entity_name, sentence) using a strict word-boundary regex:


pattern = rf"\b{re.escape(entity_name.lower())}\b"
If GLiNER emits entity names with formatting differences (e.g., “Hydro MPC” vs “Hydro-MPC”, or punctuation/line breaks), strict \b matching fails → heuristic relations won’t trigger.

Also _infer_relation rewards patterns like “controls/measures/operates”, otherwise it only marks short sentences (<=25 words) as related_to. Many manual PDFs have longer procedural sentences.

Graph inference uses adjacency from co-appearance and token similarity, but that adjacency depends on the same _entity_appears_in_text matching.

So: even with good code, your observed “zero relations” is very plausible.

17. Entity linking — 7.2/10
Code evidence: BLINK linker is optional; if unavailable it falls back to lexical canonicalization and assigns a wiki-like linked_id.

So linking is stable, but doesn’t guarantee correct KB disambiguation.

18–21 Reasoning stages degrade hard when evidence/edges are empty
18. BGE reranking — 3.0/10
Code evidence: _rerank_entities explicitly falls back to LexicalReranker when model missing.

But your stage was “skipped because reranker unavailable,” so the reranking wasn’t helping select relevant evidence.

19. Qwen2.5-VL — 0.0/10
Code evidence: In engine_v2.py, VL analysis is done by:

copilot_agent.vision_language_analysis() if copilot exists
else VisualLanguageCaptioner.caption_images() if model loads
else image-to-text HF pipeline
else “ocr_proxy” fallback
Your stage outcome “model unavailable” aligns with either missing weights or blocked model loading. Either way, captions are not grounded to instruction semantics.

20. GraphRAG analysis — 1.5/10
Code evidence: GraphRAG is intentionally evidence-gated.

_check_evidence_sufficiency requires:
has_entities
has_relations
has_text
high_confidence_entities
It then compares coverage to min_evidence_threshold=0.5.
If relations are [], evidence coverage likely fails → GraphRAG returns "insufficient-evidence" or very low confidence outputs.
That matches “generic/hallucinated rather than grounded” in your report (often caused by LLM generation + weak validation fallback logic, or prompt excerpt dominance when relations missing).

21. Copilot analysis — 1.5/10
Code evidence: Copilot agent is used if available; otherwise it returns “agent unavailable”.

When relations are empty, copilot has little graph structure and tends to produce generic reasoning unless it strictly cites OCR spans (and the code you showed for copilot isn’t included here, so based on engine integration: it’s fed entities, relations, text; empty relations reduces grounding power).

Summary: what improved vs still missing (mechanically)
What’s improved (supported by code)
Entity extraction is robust and chunk-aware → explains your stage 15 success.
Document segmentation + embeddings are correctly wired → supports retrieval-like stages.
OCR/layout has multiple fallbacks and produces structured objects → stages 1–6 are strong enough.
SAM2/visual pipeline is implemented with prompt-box support → it can work when boxes are correct and masks are actually used downstream.
Still missing (supported by code paths)
Relation extraction is too brittle to entity string mismatches.

Strict word-boundary matching + dependence on entity_name exact surface form.
If entity names don’t match OCR tokenization, relations won’t form.
No “relations from visuals” integration.

SAM masks / GroundingDINO boxes aren’t used to:
extract additional text,
verify entity mentions,
or create grounded edges.
Visual stages have no downstream impact on graph edges.

They produce data blobs (sam_segments, detections) but don’t strongly affect relation extraction / synthesis.
GraphRAG cannot produce meaningful outputs without relations.

Evidence gating + relations empty ⇒ near-empty reasoning.
Reading order is visual-sort based, not semantic step sequencing.

So it won’t map to procedural instructions well.
Final score: per-stage results vs end-to-end value
Best performing: 1–6, 13–17
Hard failing to contribute: 7–12
Hard failing to create structure: 16 (relations)
Reasoning degrades as a consequence: 18–21
Overall (your target is “instruction manual understanding + grounded reasoning”): ~4.5–5/10

If your target is “OCR + entities + chunking + basic structural summary”: ~7/10
Given your reported outputs included zero relations + generic GraphRAG/coplilot, the more honest “end-to-end understanding” rating is ~4.5–5/10.


Task Completed
Completed source-code deep introspection (read orchestrator + OCR/entity/relation/GraphRAG + model helpers) and produced a final per-stage rating (1–21), with what’s improved and what’s still missing grounded directly in the implementations.

Per-stage ratings:
1 OCR 8.8/10
2 Layout analysis 8.0/10
3 Layout understanding 7.8/10
4 Table structure analysis 5.5/10
5 GroundingDINO detection 7.6/10
6 Cross-stage synthesis 7.9/10
7 SAM2 segmentation 2.5/10
8 Nougat formula recognition 2.0/10
9 Reading order 3.0/10
10 YOLO PID detector 1.5/10
11 PID symbol detection 1.5/10
12 PID component detection 2.0/10
13 Document segmentation 8.2/10
14 Semantic indexing 7.7/10
15 Entity extraction 8.5/10
16 Relation extraction 2.0/10
17 Entity linking 7.2/10
18 BGE reranking 3.0/10
19 Qwen2.5-VL 0.0/10
20 GraphRAG analysis 1.5/10
21 Copilot analysis 1.5/10

Main remaining gaps (code-backed): relations are brittle due to strict entity surface-form matching and dependency on exact entity appearance in regex word boundaries; visual outputs (GroundingDINO/SAM) are not integrated into relation edge creation; GraphRAG is evidence-gated and collapses when relations are empty; reading order is visual-sort rather than procedural-step sequencing.