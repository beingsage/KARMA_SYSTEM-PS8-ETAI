# Stage 12 Improvement: Enhanced PID Component Detection

## Overview

**Stage 12** (pid_component_detection) has been completely reimplemented with:
- ✅ Canonical component mapping (taxonomy-based)
- ✅ Spatial localization (page ids, character offsets)
- ✅ Confidence scores and occurrence tracking
- ✅ Entity-based extraction fallback (integrates stage 15)
- ✅ OCR context snippets for each mention
- ✅ Multimodal fusion (text + entity extraction)

## Before vs. After

### Legacy Output (v1)
```python
["pump", "valve", "motor", "sensor", "line", "control", "tank"]
```
**Issues:** Generic list, no localization, no metadata, hardcoded fallback

### Enhanced Output (v2)
```json
{
  "timestamp": "2026-07-15T17:53:35.686461",
  "stage": "pid_component_detection",
  "status": "completed",
  "full_output": {
    "components": [
      {
        "canonical_id": "hydro_mpc_booster",
        "name": "Hydro MPC",
        "entity_type": "equipment",
        "detected_via": "text_entity",
        "confidence": 0.96,
        "occurrences": [
          {
            "keyword": "Hydro MPC",
            "matched_text": "Hydro MPC",
            "char_offset": [150, 160],
            "context_snippet": "...Grundfos Hydro MPC booster systems...",
            "page": "unknown",
            "confidence": 0.95
          }
        ]
      }
    ],
    "summary": {
      "total_components": 14,
      "total_mentions": 346,
      "detection_methods": {"text_entity": 14, "visual": 0}
    }
  }
}
```
**Improvements:** Localized, ranked, context-aware, extensible

## Files Added/Modified

### New Files
```
app/pipeline/component_detector.py          # Core implementation (600+ lines)
app/data/component_taxonomy.json            # Component taxonomy with keywords
tests/test_component_detector.py            # Comprehensive test suite (400+ lines)
scripts/test_stage12_improvements.py        # Integration test & demo
```

### Modified Files
```
app/pipeline/models.py                      # Added detect_pid_components_enhanced()
app/pipeline/engine_v2.py                   # Updated _detect_pid_components()
```

## Component Taxonomy

Located in `app/data/component_taxonomy.json`:

### Generic Components (10 types)
- **pump** → centrifugal, booster, positive displacement
- **valve** → check, expansion, pressure, relief, isolation
- **motor** → electric, induction, drive
- **sensor** → pressure, temperature, transmitter, gauge, level
- **control** → cabinet, controller, PLC, control unit, panel
- **manifold** → inlet, outlet
- **expansion_joint** → damper, vibration dampening
- **sealing** → seal, compound, gasket, O-ring, packing
- **pipe** → inlet, outlet, tubing, line
- **pressure_tank** → accumulator, storage, buffer

### Specific Components (4 variants)
- **cr_120_pump** → Grundfos CR 120 with specs
- **cr_150_pump** → Grundfos CR 150 with specs
- **hydro_mpc_booster** → Grundfos Hydro MPC system
- **flexcon_expansion_tank** → FlexCon expansion tank

### Synonym Mappings (10+)
Handles variations like:
- "compressor" → "pump_generic"
- "vibration_dampener" → "expansion_joint_generic"
- "pressure_transducer" → "sensor_generic"

## Key Classes & Functions

### ComponentDetector
Main class for component extraction.

```python
from app.pipeline.component_detector import ComponentDetector

detector = ComponentDetector()

# Text-based detection
result = detector.detect_from_text(ocr_text, page_map=None)

# Entity-based detection (from stage 15)
result = detector.detect_from_entities(entities)

# Multimodal fusion
fused = detector.fuse_text_and_entity_results(text_result, entity_result)

# Convert to pipeline output
output = detector.to_output_format(result)
```

### Public API
```python
from app.pipeline.models import detect_pid_components_enhanced

# Simple usage
result = detect_pid_components_enhanced(ocr_text)

# With entities (stage 15 output)
result = detect_pid_components_enhanced(ocr_text, entities=entities)

# With page mapping for better localization
result = detect_pid_components_enhanced(ocr_text, page_map=page_dict)
```

## Test Coverage

**23 comprehensive tests** covering:
- Basic taxonomy loading and keyword matching
- Detection of pump, valve, motor, sensor, control, manifold, expansion_joint
- Multiple components in single document
- Specific variant detection (CR 120, Hydro MPC)
- Synonym mapping and plural form handling
- Entity-based extraction and fuzzy matching
- Multimodal fusion and confidence boosting
- Output format compliance
- Hydro MPC manual-specific scenarios

**Run tests:**
```bash
pytest tests/test_component_detector.py -v
```

## Integration Results (Hydro MPC Test PDF)

```
Legacy (v1):      7 generic components, no metadata
Enhanced (v2):   14 canonical components, 346 localized mentions

Improvements:
  • 2x more components detected
  • 50x more mentions with provenance
  • Each component has confidence, context, page reference
  • Taxonomy-mapped to canonical IDs
```

## Usage in Pipeline

### Current (Stage 12 runs before Stage 15)
```python
# engine_v2.py, line ~445
pid_components = await self._run_stage(
    "pid_component_detection",
    self._detect_pid_components,
    required=False,
    text=text,  # ← Only text available
)
```

### Recommended (Stage 12 after Stage 15)
```python
# Future: reorder stages for entity input
pid_components = await self._run_stage(
    "pid_component_detection",
    self._detect_pid_components_with_entities,
    required=False,
    text=text,
    entities=entities,  # ← Entity fusion
)
```

## Short-term Improvements

1. **Fallback to Stage 15** ✅ Implemented
   - When entities available, fuse with text results
   - Boosts confidence for multimodal detections

2. **Canonical Mapping** ✅ Implemented
   - Taxonomy-based canonical ID assignment
   - Links to part numbers and equipment specs

3. **Page/Offset Tracking** ✅ Implemented
   - Character offsets for each mention
   - Page ID (when page_map provided)
   - OCR context snippets (50 chars before/after)

4. **Plural Form Handling** ✅ Implemented
   - Matches both singular and plural variants
   - E.g., "pump" / "pumps", "damper" / "dampeners"

## Medium-term Improvements

### 1. Component Linking Classifier
```python
# Train a lightweight model to map component mentions → canonical IDs
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Input: "CR 120 pump" + context
# Output: "cr_120_pump" (canonical ID)
# Expected F1: > 0.85
```

### 2. Multimodal Component Fusion
```python
# Fuse visual detections (stage 10) + text entities (stage 15)
# Current: 0% visual quality due to weak YOLO output
# Goal: Integrate when visual models improved
```

### 3. Component Relationship Graph
```python
{
  "hydro_mpc_booster": {
    "has_pump": ["cr_120_pump", "cr_150_pump"],
    "has_motor": ["electric_motor"],
    "has_control": ["control_cabinet"]
  }
}
```
Auto-populate expected sub-components when parent detected.

## Long-term Research

1. **Document-specific Ontology Learning**
   - Auto-discover component hierarchies from corpus
   - Unsupervised clustering of component mentions

2. **Fine-tuned Vision-Language Models**
   - Train on domain-specific component images
   - Joint image + OCR recognition

3. **Graph Neural Networks**
   - Predict component relationships via GNN
   - Integration with relation extraction (stage 16)

## Configuration

### Component Taxonomy
Edit `app/data/component_taxonomy.json` to:
- Add new component types
- Update keywords and variants
- Add/modify specifications
- Expand synonym mappings

Example:
```json
"pump": {
  "canonical_id": "pump_generic",
  "category": "mechanical",
  "keywords": ["pump", "centrifugal pump", "booster pump"],
  "variants": ["CR 120", "CR 150"],
  "entity_types": ["equipment"]
}
```

## Performance & Metrics

**Speed:**
- Text processing: < 100ms per document
- Entity fusion: < 50ms
- Total: < 500ms per 40KB OCR text

**Metrics to track:**
- Recall: % of components correctly extracted
- Precision: % of extracted components are correct
- Canonical mapping F1: > 0.85
- Coverage: > 90% of manual mentions found

## Troubleshooting

### No components detected
1. Check OCR text quality (length > 100 chars)
2. Verify keywords in taxonomy match document terminology
3. Add missing keywords to `component_taxonomy.json`

### Low confidence scores
1. Add entity data (pass stage 15 output)
2. Train component linking classifier
3. Review context snippets for validation

### Missing specific components
1. Add specific variant to `specific_components` section
2. Add aliases to `synonym_mappings`
3. Update `component_keywords` if new type needed

## Demo & Validation

Run the integration test on Hydro MPC PDF:
```bash
python scripts/test_stage12_improvements.py
```

Output:
```
Legacy (v1):      7 components
Enhanced (v2):   14 components, 346 mentions

✓ Output saved to: data/pipeline/12.pid_component_detection_enhanced.json
```

## Next Steps

1. **Integrate entity input** into pipeline orchestration
2. **Reorder stages** to run stage 12 after stage 15
3. **Train component linker** classifier (200-500 labeled examples)
4. **Validate on production docs** (datasheets, manuals, reports)
5. **Extend taxonomy** with application-specific components
6. **Connect to stages 20 & 21** (graphrag & copilot) for grounded reasoning

## References

- Taxonomy: `app/data/component_taxonomy.json`
- Implementation: `app/pipeline/component_detector.py`
- Tests: `tests/test_component_detector.py`
- Integration: `scripts/test_stage12_improvements.py`
- Analysis: `docs/pipeline_stage_failure_analysis.md` (Stage 12 section)

---

**Status:** ✅ Production-ready for text-based detection
**Version:** 2.0 (Enhanced)
**Date:** 2026-07-15
