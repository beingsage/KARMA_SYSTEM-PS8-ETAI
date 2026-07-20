# Pipeline Bug & Issue Tracker

> Last Updated: 2025-01-19

---

## 🟢 Fixed Issues

### Bug 1: Neo4j Cypher Syntax Error (Fixed ✓)
- **File:** `app/pipeline/neo4j_store.py`
- **Fix:** Added `WITH e` between `FOREACH` and `MATCH` to properly scope entity variable
- **Status:** ✅ Fixed

### Bug 2: Pydantic `formulas` ValidationError (Fixed ✓)
- **File:** `app/pipeline/engine_v2.py`
- **Fix:** `run()` and `run_from_text()` now safely extract `formulas` list from dict wrapper
- **Status:** ✅ Fixed

### Bug 3: Pydantic `reading_order` ValidationError (Fixed ✓)
- **File:** `app/pipeline/engine_v2.py`
- **Fix:** `run()` and `run_from_text()` now safely extract `reading_order` list from dict wrapper
- **Status:** ✅ Fixed

---

## 🔴 High Priority (Crashes/Runtime Failures)

### 1. Qwen Initialization: `name 'torch' is not defined`
- **Evidence:** `Qwen 3 initialization failed: name 'torch' is not defined`
- **Root Cause:** `torch` imported inside the `try` block for Qwen model loading, but used before import in some code paths
- **Files:** `app/pipeline/graphrag_summarizer.py`, `app/pipeline/advanced_models.py`, `app/pipeline/copilot_agent.py`
- **Fix:** Ensure `import torch` is at the top of the file or at least before any code that uses `torch.float16`, `torch.device`, etc.

```python
# Fix: Move import to top of file
import torch
```

### 2. HDBSCAN ↔ scikit-learn API Mismatch
- **Evidence:** `check_array() got an unexpected keyword argument 'ensure_all_finite'`
- **Root Cause:** Newer scikit-learn replaced `ensure_all_finite` with `force_all_finite`, but older hdbscan uses the old name
- **File:** `app/pipeline/advanced_models.py` (lines 606-627)
- **Current State:** Already has DBSCAN fallback, but HDBSCAN still fails
- **Fix Options:**
  - Option A: Pin scikit-learn to older version: `pip install "scikit-learn<1.5"`
  - Option B: Upgrade hdbscan: `pip install --upgrade hdbscan`
