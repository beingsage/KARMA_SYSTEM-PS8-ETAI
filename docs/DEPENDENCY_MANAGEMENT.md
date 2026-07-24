## Layered Dependency System

This document explains how the new dependency architecture works and how to maintain it for both local and Kaggle deployments.

---

## Architecture Overview

The system uses a **production-grade layered dependency model** to prevent version conflicts:

```
requirements/
├── base.txt              ← Core infrastructure (never changes)
├── vision.txt            ← Computer vision only
├── document.txt          ← OCR and document parsing
├── nlp.txt               ← Language models and NLP
├── graph.txt             ← Neo4j and vector stores
├── analytics.txt         ← Analytics and forecasting
├── agent.txt             ← LLM agents and reasoning
└── all.txt               ← Aggregate (imports all above)

requirements.txt          ← Points to all.txt
requirements.lock         ← Frozen versions for reproducibility
```

---

## Layer Details

### Layer 0: Base Infrastructure
**File:** `requirements/base.txt`

Core packages that rarely need updates:
- FastAPI, Uvicorn, Pydantic
- Requests, HTTP clients
- NumPy, PyArrow, Pillow
- Core I/O and data structures

**Stability:** ★★★★★ (Essential, rarely changed)

### Layer 1: Vision Stack
**File:** `requirements/vision.txt`

Computer vision libraries:
- PyTorch 2.10.0 + TorchVision + TorchAudio
- UltraLytics (YOLOv8)
- SAM2, GroundingDINO
- DocLayout YOLO

**Stability:** ★★★★ (Stable, tested with PyTorch ecosystem)

### Layer 2: Document Understanding
**File:** `requirements/document.txt`

OCR and document processing:
- Docling, Surya OCR, Nougat
- PyMuPDF, Tesseract integration
- Albumentations (augmentation)

**Stability:** ★★★ (Good, but may need OCR system dependencies)

### Layer 3: NLP
**File:** `requirements/nlp.txt`

Language models and extraction:
- Transformers 4.57.6
- Sentence-Transformers 5.6.0
- GLiNER, GLiREL, BLINK
- Seqeval

**Stability:** ★★★★ (Transformers ecosystem is well-maintained)
**Note:** Only ONE package controls `transformers`. No other layer touches it.

### Layer 4: Knowledge Graph
**File:** `requirements/graph.txt`

Graph databases and embeddings:
- Neo4j Python driver
- Qdrant client
- NetworkX

**Stability:** ★★★★ (Databases are stable)
**Note:** Neo4j is optional on Kaggle.

### Layer 5: Analytics & Forecasting
**File:** `requirements/analytics.txt`

Time series and clustering:
- TimesFM 1.2.7
- PyTorch Lightning 2.2.0
- BERTopic, HDBSCAN, UMAP
- Scikit-learn, Pandas, SciPy

**Stability:** ★★★ (Interdependent packages, careful version management)

### Layer 6: Agents
**File:** `requirements/agent.txt`

LLM orchestration and reasoning:
- LangGraph 0.2.76
- LangChain 0.3.0
- GraphRAG 3.1.0

**Stability:** ★★★ (Rapidly evolving, but pinned versions)

---

## Installation Order

**CRITICAL: Always install in this order to prevent conflicts:**

```bash
./scripts/install_base.sh
./scripts/install_vision.sh
./scripts/install_document.sh
./scripts/install_nlp.sh
./scripts/install_graph.sh
./scripts/install_analytics.sh
./scripts/install_agents.sh
```

**Or use the convenience wrapper:**
```bash
./scripts/install_all.sh
```

---

## Version Pinning Strategy

Every package is fully pinned to prevent silent breakage:

✓ Good (locked version):
```
transformers==4.57.6
pandas==2.2.3
torch==2.10.0
```

✗ Bad (unpinned or loose):
```
transformers>=4.0
pandas==2.*
torch  # No version
```

### Why Pinning Matters

When you run `pip install -r requirements.txt`:
- **Pinned:** Exact same versions every time (reproducible)
- **Loose:** Could install any matching version (unpredictable)

Example with one loose pin:
```
pip install torch==2.10.0 pandas
# Could install pandas 3.0, which breaks many packages
```

---

## Updating Dependencies

### When to Update

1. **Security patches:** Update immediately (e.g., `transformers 4.57.6` → `4.57.7`)
2. **Bug fixes:** Update when you encounter the bug
3. **Features:** Only if needed for new functionality
4. **Major versions:** Requires full regression testing

### How to Update

1. **Edit the specific layer file:**
   ```bash
   # Example: Update transformers in NLP layer
   vim requirements/nlp.txt
   # Change: transformers==4.57.6 → transformers==4.57.7
   ```

2. **Test the layer in isolation:**
   ```bash
   pip install --upgrade -r requirements/nlp.txt
   python -c "import transformers; print(transformers.__version__)"
   ```

3. **Test the pipeline:**
   ```bash
   python scripts/validate_pipeline.py
   ```

4. **Update all.txt** (it auto-includes layers):
   ```bash
   # No edit needed; all.txt already aggregates changes
   ```

5. **Regenerate lockfile:**
   ```bash
   pip freeze > requirements.lock.new
   # Review changes, then move to requirements.lock
   mv requirements.lock.new requirements.lock
   ```

---

## Conflict Resolution

If you see dependency conflicts:

1. **Read the error message** - it usually says what conflicts
2. **Identify the layers involved** - find which layers have conflicting packages
3. **Check if it's an inter-layer issue:**
   - If yes: Update both layers to compatible versions
   - If no: The package has conflicting sub-dependencies

Example conflict:
```
ERROR: pip's dependency resolver does not currently take into account all the packages
that are installed... scikit-learn 1.5.2 requires scipy>=1.13.1
```

**Solution:**
```bash
# In requirements/analytics.txt, ensure:
scipy>=1.13.1  # At least 1.13.1
# Or upgrade scipy in all files that need it
```

---

## Kaggle-Specific Considerations

### What Works Well on Kaggle

✓ All layers install correctly
✓ No venv needed (uses Kaggle's Python)
✓ GPU auto-detected
✓ PyTorch wheels pre-built for Kaggle runtime

### What Needs Adaptation

⚠ Neo4j: Optional (Docker not available)
⚠ Redis: Optional (Kaggle provides session storage)
⚠ Some packages require system libs (Tesseract, Java)

### Kaggle Installation

```bash
# In Kaggle Notebook:
!bash run_all.sh
```

This automatically detects Kaggle and uses corrected dependencies.

---

## Verification Checklist

After installing all layers, verify:

```bash
# ✓ Core infrastructure
python -c "import fastapi, pydantic; print('✓ Core OK')"

# ✓ Vision stack
python -c "import torch, ultralytics; print('✓ Vision OK')"

# ✓ Document processing
python -c "import docling, surya; print('✓ Document OK')"

# ✓ NLP
python -c "import transformers, gliner; print('✓ NLP OK')"

# ✓ Graph
python -c "import neo4j, qdrant_client; print('✓ Graph OK')"

# ✓ Analytics
python -c "import timesfm, bertopic; print('✓ Analytics OK')"

# ✓ Agents
python -c "import langgraph, graphrag; print('✓ Agents OK')"

# Full health check
python scripts/verify_environment.py
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `No module named X` | Layer not installed | Run `./scripts/install_X.sh` |
| `Version conflict: A 1.0 vs B 2.0` | Incompatible pins | Update to compatible versions in both files |
| `CUDA out of memory` | Wrong torch version for GPU | Verify `torch==2.10.0` not `torch+cpu` |
| `ImportError in Kaggle` | Missing Kaggle-specific package | Use `!pip install` in notebook cell |
| `All layers installed but pipeline fails` | Version mismatch at runtime | Run `python scripts/validate_pipeline.py` |

---

## Best Practices

1. **Always pin versions** - No wildcards or loose constraints
2. **Test each layer independently** - Catch conflicts early
3. **Keep base stable** - Only update base for security
4. **Document breaks** - If you update a layer, document why
5. **Version the lockfile** - Commit `requirements.lock` to git
6. **Test on Kaggle early** - Different package availability

---

## Future Maintenance

For ongoing updates:
- Check for security advisories monthly
- Pin new versions when adopting new features
- Run `validate_pipeline.py` after any update
- Regenerate lockfile every 3 months
