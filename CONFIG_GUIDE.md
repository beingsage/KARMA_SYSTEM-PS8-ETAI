# Configuration System Guide

## Overview

The application now uses a centralized configuration file (`.env.local`) in the root directory instead of environment variables. This makes it easier to:

- Manage all settings in one place
- Switch between different execution modes (CPU/GPU)
- Keep local settings out of version control
- Override defaults without command-line exports

## Quick Start

### 1. Create Your Local Configuration

Copy the example configuration file:

```bash
cd /media/sagesujal/DEV1/bytes/structured
cp .env.example .env.local
```

### 2. Edit `.env.local` for Your Setup

Open `.env.local` in your editor and customize settings:

```bash
# Local CPU testing (default)
EXECUTION_MODE=cpu
TABLE_EXTRACTION_TIMEOUT=0      # No limits on local

# Or for GPU server:
EXECUTION_MODE=gpu
TABLE_EXTRACTION_TIMEOUT=180    # 3 minutes max on server
```

### 3. Run the Application

Simply run as usual - settings load from `.env.local` automatically:

```bash
./run_all.sh
```

No need for `export` commands! The config file is loaded automatically.

## Configuration Priority

Settings are loaded in this order (first match wins):

1. **Environment Variables** (allows command-line override)
   ```bash
   EXECUTION_MODE=gpu ./run_all.sh  # Overrides .env.local
   ```

2. **`.env.local` file** (your local configuration)
   ```
   EXECUTION_MODE=cpu
   ```

3. **Default values** (fallback if neither above exists)

## Configuration File Structure

### `.env.example`
- Template with all available settings
- Well-commented with descriptions
- Includes optimization profiles
- **Never edited directly** - use for reference

### `.env.local`
- **Your local configuration** - edit this file
- Automatically created from `.env.example`
- Ignored by git (in `.gitignore`)
- Not shared with other developers

## Common Configurations

### Local Development (CPU, No Limits)
```ini
EXECUTION_MODE=cpu
TABLE_EXTRACTION_TIMEOUT=0
OCR_PROCESSING_TIMEOUT=0
PDF_RENDER_TIMEOUT=0
LOG_LEVEL=DEBUG
VERBOSE=true
```

### GPU Server (Production)
```ini
EXECUTION_MODE=gpu
TABLE_EXTRACTION_TIMEOUT=180
OCR_PROCESSING_TIMEOUT=300
PDF_RENDER_TIMEOUT=60
LOG_LEVEL=INFO
VERBOSE=false
```

### Quick Demo (GPU with Short Timeouts)
```ini
EXECUTION_MODE=gpu
TABLE_EXTRACTION_TIMEOUT=10
OCR_PROCESSING_TIMEOUT=15
PDF_RENDER_TIMEOUT=5
LOG_LEVEL=INFO
```

### Custom GPU Device
```ini
EXECUTION_MODE=gpu
DEVICE_FOR_DETECTION=cuda:1     # Use GPU 1 instead of 0
DEVICE_FOR_EXTRACTION=cuda:1
DEVICE_FOR_EMBEDDING=cuda:1
```

## Supported Settings

### Execution & Performance
```ini
EXECUTION_MODE=cpu|gpu                  # Mode: cpu or gpu
DEVICE_FOR_DETECTION=AUTO|cuda:0|cpu   # Detection device (auto-set by mode)
DEVICE_FOR_SEGMENTATION=AUTO|cuda:0|cpu # Segmentation device
DEVICE_FOR_EXTRACTION=AUTO|cuda:0|cpu   # Extraction device
DEVICE_FOR_EMBEDDING=AUTO|cuda:0|cpu    # Embedding device
TABLE_EXTRACTION_TIMEOUT=0              # Seconds (0 = unlimited)
OCR_PROCESSING_TIMEOUT=0                # Seconds (0 = unlimited)
PDF_RENDER_TIMEOUT=0                    # Seconds (0 = unlimited)
```

### Model Configuration
```ini
GLINER_MODEL=urchade/gliner_medium-v2.1
GLIREL_MODEL=jackboyla/glirel-large-v0
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
# ... and many more model paths
```

### Database Configuration
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=industrial_graph_password
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=industrial_documents
```

### Feature Flags
```ini
GRAPHRAG_ENABLED=true
LANGGRAPH_ENABLED=true
ENABLE_PM=true
ENABLE_RUL=true
ENABLE_ANOMALY=true
ENABLE_RCA=true
```

### Logging
```ini
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR
VERBOSE=true|false
```

## Viewing Active Configuration

When the application starts, it logs which config file is loaded:

```
[CONFIG] Loaded configuration from: /media/sagesujal/DEV1/bytes/structured/.env.local
[CONFIG] Execution Mode: CPU
[CONFIG] Running in CPU mode (no time restrictions, for local testing)
```

## Troubleshooting

### Config File Not Found
```
[CONFIG] Using defaults (.env.local not found, create one from .env.example)
```

**Solution**: Create `.env.local` from `.env.example`
```bash
cp .env.example .env.local
```

### Settings Not Changing
Remember the priority order:
1. Environment variables override everything
2. `.env.local` overrides defaults
3. If a setting isn't in `.env.local`, the default is used

Check if environment variable is set:
```bash
echo $EXECUTION_MODE  # If set, it overrides .env.local
```

### Timeout Not Working
- Set timeout to > 0: `TABLE_EXTRACTION_TIMEOUT=180`
- Value 0 means no timeout (unlimited)
- Timeouts only apply to CPU mode defaults; in GPU mode you control it

### Device Not Using GPU
Check your configuration:
```bash
grep "DEVICE_FOR" .env.local
grep "EXECUTION_MODE" .env.local
```

If `EXECUTION_MODE=gpu`, devices should auto-switch to `cuda:0`.

## Switching Between Modes

### From CPU to GPU
Edit `.env.local`:
```bash
# Change this line:
EXECUTION_MODE=cpu
# To:
EXECUTION_MODE=gpu

# And set reasonable timeouts:
TABLE_EXTRACTION_TIMEOUT=180
```

Then run: `./run_all.sh`

### From GPU Back to CPU
```bash
# Change back:
EXECUTION_MODE=cpu
# Remove timeouts:
TABLE_EXTRACTION_TIMEOUT=0
```

Then run: `./run_all.sh`

## Performance Tuning

### For Faster Processing
```ini
EXECUTION_MODE=gpu
DEVICE_FOR_DETECTION=cuda:0
DEVICE_FOR_SEGMENTATION=cuda:0
DEVICE_FOR_EXTRACTION=cuda:0
DEVICE_FOR_EMBEDDING=cuda:0
```

### For Lower Memory Usage
```ini
EXECUTION_MODE=cpu
# Smaller batch sizes (if supported):
# BATCH_SIZE=1 (if you add this config)
```

### For Debugging
```ini
LOG_LEVEL=DEBUG
VERBOSE=true
EXECUTION_MODE=cpu
TABLE_EXTRACTION_TIMEOUT=0
```

## Best Practices

1. **Always start from `.env.example`**
   ```bash
   cp .env.example .env.local
   ```

2. **Never commit `.env.local`**
   - It's in `.gitignore`
   - It contains your local settings

3. **Document team configs**
   - Keep `.env.example` updated
   - Update comments in `.env.example` for new settings

4. **Use profiles for common scenarios**
   - See the "OPTIMIZATION PROFILES" section in `.env.example`
   - Copy/paste entire profile blocks into `.env.local`

5. **Test mode switches**
   ```bash
   # Test CPU mode
   EXECUTION_MODE=cpu ./run_all.sh
   
   # Test GPU mode
   EXECUTION_MODE=gpu ./run_all.sh
   ```

## Migration from Environment Variables

If you were previously using:
```bash
export EXECUTION_MODE=gpu
export TABLE_EXTRACTION_TIMEOUT=180
./run_all.sh
```

Now do this instead:

1. Create `.env.local`:
   ```bash
   cp .env.example .env.local
   ```

2. Edit `.env.local` with your settings
3. Run normally:
   ```bash
   ./run_all.sh
   ```

The environment variables will **still work** (for backward compatibility), but `.env.local` is now the preferred approach.

## File Location Reference

```
/media/sagesujal/DEV1/bytes/structured/
├── .env.example      ← Template (version controlled)
├── .env.local        ← Your settings (NOT version controlled)
├── app/
│   ├── config.py     ← Loads from .env.local automatically
│   └── ...
└── run_all.sh        ← No changes needed
```

## What Changed

- **Before**: `export EXECUTION_MODE=gpu && ./run_all.sh`
- **After**: Edit `.env.local`, then `./run_all.sh`

That's it! Everything else works the same way.
