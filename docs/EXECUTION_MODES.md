# Execution Modes Guide

## Overview

The Industrial PDF-to-Graph Pipeline supports two execution modes:

1. **CPU Mode** (default) - For local testing with no time restrictions
2. **GPU Mode** - For optimized server deployment with GPU acceleration

## CPU Mode (Development/Testing)

### When to Use
- **Local development and testing**
- **Testing all pipeline features without time pressure**
- **Debugging and troubleshooting**
- **Single-user or small-scale processing**

### Configuration
```bash
# Default - no need to set anything
export EXECUTION_MODE=cpu

# Or simply run:
./run_all.sh
```

### Characteristics
- All models run on CPU
- **No timeout restrictions** - processing can take as long as needed
- Suitable for comprehensive feature testing
- Lower hardware requirements
- May be slower for large documents

### Behavior
- Table extraction uses CPU only
- OCR processing runs on CPU
- Image rendering and detection on CPU
- Good for validating results without time constraints

## GPU Mode (Production/Server)

### When to Use
- **Production server deployment**
- **High-throughput processing**
- **Real-time API usage**
- **Multi-user scenarios**
- **Performance optimization**

### Configuration
```bash
# Enable GPU mode
export EXECUTION_MODE=gpu

# Then run:
./run_all.sh
```

### Device Placement
When GPU mode is enabled, the pipeline uses:
- **Detection models** → `cuda:0` (GPU 0)
- **Segmentation models** → `cuda:0` (GPU 0)
- **Text extraction** → `cuda:0` (GPU 0)
- **Embeddings** → `cuda:0` (GPU 0)

### Characteristics
- GPU acceleration for all heavy models
- **Optional timeouts** for table extraction (default: no limit)
- Significantly faster processing
- Requires GPU hardware (CUDA-capable device)
- Higher throughput for batch processing

### Optional Timeouts
To add timeouts for GPU mode, set environment variables:

```bash
export EXECUTION_MODE=gpu
export TABLE_EXTRACTION_TIMEOUT=120      # 120 seconds for table extraction
export OCR_PROCESSING_TIMEOUT=300        # 300 seconds for OCR
export PDF_RENDER_TIMEOUT=60             # 60 seconds for PDF rendering

./run_all.sh
```

Set to `0` (default) for no timeout.

## Performance Comparison

| Feature | CPU Mode | GPU Mode |
|---------|----------|----------|
| Large documents (50+ pages) | Slow | Fast |
| Table extraction | ~30-50 sec | ~5-10 sec |
| Entity recognition | Slower | Faster |
| Memory usage | Lower | Higher |
| Startup time | Normal | Normal + GPU warmup |
| Time limit | None | Configurable |
| Best for | Testing | Production |

## Environment Variables Reference

### Mode Configuration
```bash
EXECUTION_MODE=cpu|gpu           # Execution mode (default: cpu)
```

### Device Configuration (Auto-set based on EXECUTION_MODE)
```bash
# Auto-set to:
# - cpu: all CPU
# - gpu: all cuda:0
# Override if needed:
# DEVICE_FOR_DETECTION=cuda:0
# DEVICE_FOR_SEGMENTATION=cuda:0
# DEVICE_FOR_EXTRACTION=cuda:0
# DEVICE_FOR_EMBEDDING=cuda:0
```

### Timeout Configuration (GPU mode only)
```bash
TABLE_EXTRACTION_TIMEOUT=0       # Table extraction (0 = no limit)
OCR_PROCESSING_TIMEOUT=0         # OCR processing (0 = no limit)
PDF_RENDER_TIMEOUT=0             # PDF rendering (0 = no limit)
```

## How to Test

### Test CPU Mode Locally
```bash
# Start the app in CPU mode
export EXECUTION_MODE=cpu
./run_all.sh

# In another terminal, upload a test PDF
curl -X POST http://127.0.0.1:8001/api/v1/process-pdf \
  -F "file=@sample.pdf" \
  -F "filename=sample.pdf"

# Monitor progress without time pressure
# Check logs for table extraction progress
```

### Test GPU Mode on Server
```bash
# On GPU-enabled server
export EXECUTION_MODE=gpu
./run_all.sh

# With timeouts for safety
export TABLE_EXTRACTION_TIMEOUT=180
./run_all.sh
```

## Troubleshooting

### CPU Mode Issues
- **Slow processing**: Normal behavior; consider GPU for production
- **System freeze**: Reduce concurrent uploads or add timeouts
- **High memory**: Reduce PDF page count or split documents

### GPU Mode Issues
- **Out of memory**: Reduce batch size or switch to CPU mode
- **CUDA errors**: Ensure GPU drivers are installed and up-to-date
- **No GPU detected**: Check NVIDIA drivers and CUDA installation
- **Timeout errors**: Increase timeout values or disable them

### Checking Current Mode
```bash
# Check configuration in logs
grep "\[CONFIG\] Execution Mode" logs/run_all.log

# Or check settings at runtime:
curl http://127.0.0.1:8001/api/v1/models/status | grep execution_mode
```

## Migration from CPU to GPU

1. **Verify GPU availability**:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

2. **Update configuration**:
   ```bash
   export EXECUTION_MODE=gpu
   ```

3. **Restart services**:
   ```bash
   ./run_all.sh
   ```

4. **Monitor performance**:
   - Watch GPU utilization: `nvidia-smi -l 1`
   - Check throughput improvement
   - Adjust timeouts if needed

## Recommendations

### For Development
```bash
# No time pressure, test all features
export EXECUTION_MODE=cpu
./run_all.sh
```

### For Production
```bash
# GPU-accelerated with safety timeouts
export EXECUTION_MODE=gpu
export TABLE_EXTRACTION_TIMEOUT=180
export OCR_PROCESSING_TIMEOUT=300
./run_all.sh
```

### For Hybrid (CPU primary, GPU fallback)
```bash
# Start in CPU mode, monitor performance
export EXECUTION_MODE=cpu
./run_all.sh

# If too slow, switch to GPU:
export EXECUTION_MODE=gpu
./run_all.sh
```
