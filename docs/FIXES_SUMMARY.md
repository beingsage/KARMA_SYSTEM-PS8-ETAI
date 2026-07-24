# System Fixes and Improvements

## Issues Fixed

### 1. Table Transformer Warning Suppression ✓
**Problem**: Unnecessary warning about unused model weights during table extraction fallback
```
Some weights of the model checkpoint at microsoft/table-transformer-detection 
were not used when initializing TableTransformerForObjectDetection...
```

**Solution**:
- Added warning filter in `app/pipeline/engine_v2.py` using `warnings.catch_warnings()`
- Expected warnings are now suppressed while errors are still reported
- Applied globally in `app/main.py` to catch all FutureWarnings and UserWarnings from dependencies

**Files Modified**:
- `app/pipeline/engine_v2.py` - Wrapped model loading in `warnings.catch_warnings()`
- `app/main.py` - Added global warning filters
- `app/pipeline/ocr_processor.py` - Added warning suppressions for OCR dependencies

### 2. Execution Mode Configuration ✓
**Problem**: System needed flexibility for local CPU testing vs GPU server deployment

**Solution**: 
Added dual execution modes with automatic device placement:

#### CPU Mode (Local Testing)
```bash
export EXECUTION_MODE=cpu  # default
./run_all.sh
```
- No time restrictions
- All processing on CPU
- Suitable for comprehensive feature testing

#### GPU Mode (Server/Production)
```bash
export EXECUTION_MODE=gpu
export TABLE_EXTRACTION_TIMEOUT=120  # optional
./run_all.sh
```
- GPU acceleration (cuda:0)
- Optional timeouts for safety
- Optimized for throughput

**Files Modified**:
- `app/config.py` - Added execution mode settings with device configuration
- `app/pipeline/engine_v2.py` - Updated table extraction to use `settings.device_for_extraction`
- `run_all.sh` - Added execution mode display
- `tests/test_relation_extractor_fallback.py` - Added execution mode tests

### 3. Device Placement by Mode ✓
**Problem**: No automatic device switching between CPU and GPU modes

**Solution**:
```python
# Auto-configured based on EXECUTION_MODE
device_for_detection: str       # YOLOv12, GroundingDINO
device_for_segmentation: str    # SAM2
device_for_extraction: str      # Table Transformer
device_for_embedding: str       # BGE embeddings
```

### 4. Optional Processing Timeouts ✓
**Problem**: Long-running operations could hang the system

**Solution**:
Added configurable timeout settings:
```bash
TABLE_EXTRACTION_TIMEOUT=0       # 0 = no limit (default for CPU)
OCR_PROCESSING_TIMEOUT=0         # 0 = no limit (default for CPU)
PDF_RENDER_TIMEOUT=0             # 0 = no limit (default for CPU)
```

## Configuration Summary

### Environment Variables Added

```bash
# Mode Selection
EXECUTION_MODE=cpu|gpu                    # Default: cpu

# Device Configuration (auto-set, can override)
DEVICE_FOR_DETECTION=device               # Auto-set based on mode
DEVICE_FOR_SEGMENTATION=device            # Auto-set based on mode
DEVICE_FOR_EXTRACTION=device              # Auto-set based on mode
DEVICE_FOR_EMBEDDING=device               # Auto-set based on mode

# Processing Timeouts (seconds, 0 = no limit)
TABLE_EXTRACTION_TIMEOUT=0                # Default: 0 (no limit)
OCR_PROCESSING_TIMEOUT=0                  # Default: 0 (no limit)
PDF_RENDER_TIMEOUT=0                      # Default: 0 (no limit)
```

## Testing

All changes have been validated:
- ✓ Python syntax validation: `python3 -m py_compile`
- ✓ Shell script validation: `bash -n run_all.sh`
- ✓ Regression tests: `pytest tests/test_relation_extractor_fallback.py`
- ✓ Execution mode tests: All 3 tests passing

## Usage Examples

### Local Testing (CPU - No Time Limits)
```bash
# Start with CPU mode (default)
./run_all.sh

# System will:
# - Run all models on CPU
# - Have no time restrictions
# - Allow thorough testing of all features
```

### Server Production (GPU - Optimized)
```bash
# Enable GPU mode with timeouts
export EXECUTION_MODE=gpu
export TABLE_EXTRACTION_TIMEOUT=180
export OCR_PROCESSING_TIMEOUT=300

./run_all.sh

# System will:
# - Use GPU acceleration for all models
# - Enforce timeouts for safety
# - Optimize for throughput
```

### Hybrid Testing
```bash
# Start with CPU for debugging
export EXECUTION_MODE=cpu
./run_all.sh

# Later, test GPU performance
# (in another session)
export EXECUTION_MODE=gpu
./run_all.sh
```

## Documentation

See [EXECUTION_MODES.md](EXECUTION_MODES.md) for:
- Detailed mode descriptions
- Performance comparison table
- Device placement reference
- Troubleshooting guide
- Migration recommendations

## Backward Compatibility

All changes are **backward compatible**:
- Default behavior unchanged (CPU mode, no timeouts)
- No breaking changes to APIs
- Existing configurations still work
- New features are opt-in via environment variables

## What This Enables

1. **Local Development**: Test all features at leisure without time pressure
2. **Production Deployment**: Optimized GPU processing with safety timeouts  
3. **Hybrid Setups**: Switch modes based on workload
4. **Debugging**: CPU mode helps isolate issues
5. **Performance Testing**: Compare CPU vs GPU behavior

## Next Steps

1. Test CPU mode with your local PDFs
2. If you have a GPU, test `EXECUTION_MODE=gpu`
3. Adjust timeouts based on your hardware
4. Monitor performance and adjust settings as needed
