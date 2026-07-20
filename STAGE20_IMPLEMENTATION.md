# Stage 20 — GraphRAG Analysis Implementation

## Overview

Stage 20 has been enhanced with **evidence grounding**, **hallucination prevention**, and **confidence calibration** to address the issues of generic safety text, placeholder insertion, and unrealistic confidence scores.

## Problem Statement (Fixed)

**Original Issues:**
- Output contained hallucinated reasoning with repeated placeholder markers
- `anomalies_detected` and `failure_risks` were empty despite high confidence=0.9
- Generic safety language ("serious personal injury") without specific evidence
- Model produced high-confidence generic templates instead of grounded analysis

## Key Enhancements

### 1. Evidence Grounding
- **Requirement**: All claims must cite specific evidence (entity, text span, measurement)
- **Implementation**: Prompt explicitly requires "ONLY cite specific evidence" and "supporting evidence must be provided"
- **Validation**: Claims are filtered if they reference non-existent entities or are too generic
- **Result**: Only evidence-backed claims are included in output

### 2. Hallucination Prevention
- **Placeholder Checking**: Filters out claims containing placeholder markers (``, `[image`, etc.)
- **Generic Phrase Detection**: Removes generic safety language without specifics
- **Measurement Detection**: Accepts claims with specific details (numbers, components, measurements)
- **Entity Grounding**: Validates claims reference actual entities or document content

### 3. Confidence Calibration
- **Default**: Reduced from 0.9 to 0.3 (more realistic for unvalidated claims)
- **Adjustment**: Confidence multiplied by evidence coverage ratio
- **Maximum**: Capped at 0.95 even with perfect evidence
- **Threshold**: Unvalidated claims max 0.3 confidence

### 4. Evidence Sufficiency Checking
- **Pre-Analysis**: Checks if entities, relations, and text meet minimum thresholds
- **Coverage Metric**: Scores evidence availability (0.0-1.0)
- **Fallback**: Returns `no_evidence` status if coverage < 0.5
- **Transparency**: Returns coverage metrics in response

### 5. Provenance Tracking
- **Source Attribution**: Every claim includes `source` field (entity name, document span, or "graphrag-validated")
- **Page References**: Claims can include page numbers and text excerpts
- **Traceability**: Full chain from raw data to conclusion

## Implementation Details

### File: `app/pipeline/graphrag_summarizer.py`

#### Key Methods

1. **`generate_summary()`** - Main entry point
   - Checks evidence sufficiency first
   - Validates all claims before returning
   - Calculates evidence coverage
   - Adjusts confidence based on validation

2. **`_check_evidence_sufficiency()`** - Pre-analysis gate
   - Verifies entities, relations, text presence
   - Returns coverage metric and pass/fail decision
   - Blocks analysis if data insufficient

3. **`_validate_claims()`** - Core hallucination prevention
   - Filters generic safety language
   - Removes placeholder markers
   - Checks entity references
   - Accepts specific details (measurements, components)

4. **`_has_specific_detail()`** - Detail detection
   - Regex patterns for measurements (2.5mm, 3500 rpm)
   - Component codes (Part 101, Section 4.2)
   - Page references (Page 15, Line 3)

5. **`_build_reasoning_prompt()`** - Enhanced LLM prompt
   - Explicit evidence grounding requirements
   - Confidence guidance (0.3-0.7 normal, 0.8+ requires strong evidence)
   - Example output format
   - Prohibition on placeholder insertion

6. **`_query_llm()`** - Temperature reduction
   - Lowered temperature from default to 0.3 (deterministic)
   - Removed beam search (more hallucination potential)
   - Greedy decoding with single beam
   - No sampling (reproducible output)

7. **`_parse_json_response()`** - Robust parsing
   - Handles missing/invalid JSON gracefully
   - Clamps confidence to [0.0, 1.0]
   - Returns parse status for diagnostics
   - Never crashes on malformed input

### Response Structure

```json
{
  "summary_method": "qwen-graphrag-grounded|insufficient-evidence|unavailable",
  "status": "analyzed|no_evidence|llm_unavailable",
  "reasoning": "Full LLM reasoning chain",
  "anomalies_detected": [
    {
      "name": "Specific anomaly with measurement or entity reference",
      "source": "entity_name or text_excerpt or graphrag-validated"
    }
  ],
  "failure_risks": [
    {
      "name": "Risk with evidence",
      "source": "source_reference"
    }
  ],
  "maintenance_recommendations": [
    {
      "name": "Specific actionable recommendation",
      "source": "source_reference"
    }
  ],
  "compliance": [],
  "confidence": 0.35,
  "evidence_coverage": 0.85,
  "claims_validated": 3,
  "claims_original": 5
}
```

## Validation & Testing

### Test Suite: `tests/test_stage20_graphrag.py`

**21 comprehensive tests covering:**

1. **Evidence Grounding** (6 tests)
   - Sufficiency checking with good/bad data
   - Generic phrase filtering
   - Specific detail acceptance
   - Placeholder removal
   - Measurement detection

2. **Confidence Calibration** (3 tests)
   - Maximum unvalidated confidence enforcement
   - Evidence coverage reduction
   - Confidence capping at 0.95

3. **JSON Parsing** (6 tests)
   - Valid JSON parsing
   - Invalid JSON handling
   - Missing field defaults
   - Invalid confidence value bounds
   - Type conversion (`_ensure_list`)

4. **Prompt Construction** (3 tests)
   - Evidence requirement inclusion
   - Confidence guidance presence
   - Example output format

5. **Integration Tests** (3 tests)
   - Expected response structure
   - No-evidence fallback
   - Evidence coverage tracking

### Running Tests

```bash
# All Stage 20 tests
pytest tests/test_stage20_graphrag.py -v

# Specific test class
pytest tests/test_stage20_graphrag.py::TestGraphRAGEvidenceGrounding -v

# Single test
pytest tests/test_stage20_graphrag.py::TestGraphRAGEvidenceGrounding::test_validate_claims_filters_generic_phrases -v
```

## Configuration Parameters

### In `GraphRAGSummarizer` class:

```python
self.min_evidence_threshold = 0.5      # Coverage needed to analyze (0.0-1.0)
self.max_unvalidated_confidence = 0.3  # Max confidence without evidence
```

### Environment Variables (via `.env.local`):

```bash
QWEN_MODEL=Qwen/Qwen2.5-0.5B-Instruct  # Model used for reasoning
```

## Short-Term Fixes Applied

✅ **Temperature reduction** - From default to 0.3 for deterministic output
✅ **Placeholder checking** - Explicit filters for `` and `[image` markers
✅ **Confidence reduction** - Default 0.9 → 0.3, scaled by evidence
✅ **Generic phrase detection** - Filters unspecific safety language
✅ **Evidence sufficiency gate** - Pre-checks data before analysis
✅ **Evidence-backed claims** - All claims must cite sources

## Medium-Term Improvements Available

- [ ] Claim classification model to detect hallucinations
- [ ] Entity grounding database for reference verification
- [ ] Structured reasoning chains with provenance edges
- [ ] Confidence confidence (meta-confidence) scoring
- [ ] Human feedback loop for hallucination detection

## Long-Term Research Directions

- Multimodal explanation models with verifiable provenance
- Graph-based reasoning with edge confidences
- Adversarial testing for hallucination robustness
- Fine-tuned models for industrial maintenance analysis

## Deployment Notes

1. **Backward Compatibility**: Returns `no_evidence` instead of hallucinations if data insufficient
2. **Output Changes**: Response structure extended with evidence metrics
3. **Confidence Interpretation**: Now conservative (0.0-0.3 default) instead of optimistic (0.9)
4. **Migration**: Systems consuming Stage 20 output should handle new `status` and `evidence_coverage` fields

## Example Usage

```python
from app.pipeline.graphrag_summarizer import GraphRAGSummarizer

summarizer = GraphRAGSummarizer()

entities = [
    {"name": "Pump A", "entity_type": "equipment", "confidence": 0.85},
    {"name": "Bearing", "entity_type": "component", "confidence": 0.92},
]

relations = [
    {"source": "Pump A", "relation_type": "has_component", "target": "Bearing"}
]

text = "Pump A bearing shows 2.5mm wear on the seal surface per inspection report page 3."

result = summarizer.generate_summary(entities, relations, text)

# Result includes only evidence-backed claims with sources
print(f"Status: {result['status']}")
print(f"Confidence: {result['confidence']}")
print(f"Evidence Coverage: {result['evidence_coverage']}")
for anomaly in result['anomalies_detected']:
    print(f"- {anomaly['name']} (source: {anomaly['source']})")
```

## Monitoring & Observability

### Metrics to Track

1. **Evidence Coverage**: % of analyzed documents with > 0.5 coverage
2. **Claims Validated Ratio**: claims_validated / claims_original
3. **Confidence Distribution**: Histogram of returned confidence scores
4. **Status Distribution**: % of analyzed vs no_evidence vs unavailable
5. **Parse Status**: JSON parsing success/failure rates

### Alerts to Set

- If average confidence > 0.7 (possible over-confidence)
- If evidence_coverage < 0.3 (weak analysis)
- If claims_validated_ratio < 0.5 (many hallucinations detected)
- If parse failures > 5% (model output quality degradation)
