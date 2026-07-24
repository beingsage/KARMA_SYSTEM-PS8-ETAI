<!-- Generated: 2026-07-03 -->
<!-- This file documents the complete one-command setup for the Industrial PDF-to-Graph Pipeline -->

# 📋 Setup Verification Checklist

## ✅ Pre-Requisites (Check These First)

- [ ] **Python 3.8+** installed
  ```bash
  python3 --version
  ```

- [ ] **Docker installed**
  ```bash
  docker --version
  docker ps  # Test connection
  ```

- [ ] **Docker Compose available**
  ```bash
  docker compose --version
  ```

- [ ] **Internet connection** (for downloading ~2.5GB models)
  ```bash
  ping google.com
  ```

- [ ] **~3-5GB disk space free** (for models + Docker)
  ```bash
  df -h | grep -E "^/dev"
  ```

- [ ] **In the project directory**
  ```bash
  pwd  # Should show: /media/sagesujal/DEV1/bytes/structured
  ```

---

## 🚀 Execution Checklist

### Step 1: Run the Setup Script

```bash
cd /media/sagesujal/DEV1/bytes/structured
bash run_all.sh
```

The script will automatically:

- [ ] **Create virtual environment** (if needed)
  - Location: `./.venv/`
  - Verification: `ls -la .venv/bin/activate`

- [ ] **Install dependencies** (~2-3 minutes)
  - Packages: FastAPI, torch, transformers, ultralytics, etc.
  - Log should show: `Successfully installed ...`

- [ ] **Create directories**
  - [ ] `./data/jobs/` - Job storage
  - [ ] `./data/uploads/` - PDF uploads
  - [ ] `./models/` - AI model cache

- [ ] **Download AI models** (~5-30 minutes depending on internet)
  - [ ] YOLOv12 (~100MB)
  - [ ] GroundingDINO (~500MB)
  - [ ] SAM2 (~400MB)
  - [ ] GLiNER (~600MB)
  - [ ] GLiREL (~400MB)
  - [ ] BLINK (~500MB)
  - [ ] BGE-M3 (~450MB)
  - [ ] BGE-Reranker (~300MB)
  - **Total:** ~2.5GB

- [ ] **Start Neo4j and Qdrant containers**
  - Check: `docker compose ps` should show `neo4j` and `qdrant` as `Up`

- [ ] **Launch FastAPI server**
  - Expected output: `Uvicorn running on 0.0.0.0:8001`

---

## 🔍 Verification Steps

### After Script Completes

1. **Check Server is Running**
   ```bash
   curl http://localhost:8001/health
   # Expected response: {"status": "ok"}
   ```

2. **Access API Documentation**
   - Open browser to: http://localhost:8001/docs
   - Should show interactive Swagger UI with all endpoints

3. **Verify Models Loaded**
   ```bash
   curl http://localhost:8001/api/v1/models/status
   # Expected: List of loaded models with status
   ```

4. **Test with Sample PDF**
   ```bash
   curl -X POST \
     -F "file=@sample.pdf" \
     http://localhost:8001/api/v1/process-pdf
   # Expected: {"job_id": "...", "status": "processing"}
   ```

5. **Check Neo4j Connection**
   ```bash
   docker compose logs neo4j | tail -20
   # Should show successful startup logs
   ```

---

## 🐛 Troubleshooting

### Issue: "Virtual environment not found"

**Solution:**
```bash
cd /media/sagesujal/DEV1/bytes/structured
bash run_all.sh
```

### Issue: "Docker not found"

**Solution:** Install Docker
- Ubuntu/Debian: `sudo apt-get install docker.io`
- Docker Desktop: https://www.docker.com/products/docker-desktop

### Issue: "Port 8001 already in use"

**Solution:**
```bash
# Find and kill process using port 8001
lsof -i :8001
kill -9 <PID>

# Or just use a different terminal window
```

### Issue: "Out of disk space" during model download

**Solution:**
```bash
# Check disk usage
du -sh /media/sagesujal/DEV1/bytes/structured

# Free up space, then try again
bash run_all.sh
```

### Issue: Models take too long to download

**Normal:** First run takes 5-30 minutes depending on internet speed
- Subsequent runs are **much faster** (models cached)
- Can interrupt and resume anytime

---

## 📊 Performance Expectations

### First Run
```
Setup time:      ~5-10 minutes (downloading models)
Models cached:   ✓ (all ~2.5GB)
Server startup:  ~1-2 minutes
```

### Subsequent Runs
```
Setup time:      ~30-60 seconds (skips downloads)
Models loaded:   ✓ (from cache)
Server startup:  ~30 seconds
```

### API Response Times
```
Health check:            10ms
Model status check:      ~50ms
PDF upload & processing: 30-300 seconds (PDF size dependent)
Results retrieval:       ~100ms
```

---

## 📁 Expected File Structure After Setup

```
/media/sagesujal/DEV1/bytes/structured/
├── .venv/                          ✓ Virtual environment
├── models/                         ✓ ~2.5GB of AI models
│   ├── yolov8n.pt (YOLOv12)
│   ├── GroundingDINO checkpoint
│   ├── sam_vit_b (SAM2)
│   ├── bge-m3 (embeddings)
│   ├── bge-reranker (ranking)
│   ├── gliner (entity extraction)
│   ├── glirel (relation extraction)
│   └── blink (entity linking)
├── data/
│   ├── jobs/                       ✓ Processing results (JSON)
│   └── uploads/                    ✓ Uploaded PDFs
├── app/
│   ├── main.py
│   ├── pipeline/
│   │   ├── model_helpers.py        ✓ All 8 models
│   │   ├── engine_v2.py            ✓ Pipeline orchestrator
│   │   ├── entity_extractor.py     ✓ GLiNER
│   │   ├── entity_linker.py        ✓ BLINK
│   │   ├── relation_extractor.py   ✓ GLiREL+REBEL
│   │   └── ...other components
│   └── ...
├── run_all.sh                      ✓ Main startup script
├── README.md                       ✓ Updated docs
├── QUICKSTART.md                   ✓ Quick start guide
├── SETUP.md                        ✓ Detailed setup
├── INTEGRATION_GUIDE.md            ✓ Model documentation
├── IMPLEMENTATION_SUMMARY.md       ✓ Technical overview
└── ...other files
```

---

## ✨ Next Steps After Verification

1. **Upload Your First PDF**
   - Use http://localhost:8001/docs
   - Or: `curl -X POST -F "file=@document.pdf" http://localhost:8001/api/v1/process-pdf`

2. **Monitor Processing**
   - Check job status: `curl http://localhost:8001/api/v1/jobs/{job_id}`

3. **Review Results**
   - Extracted entities (from 8 entity types)
   - Relations identified
   - Semantic embeddings generated
   - Knowledge graph created in Neo4j

4. **Customize for Your Domain**
   - Fine-tune GLiNER with your data: `python scripts/fine_tune_gliner.py --data ... --output ...`
   - Adjust entity types in `app/config.py`
   - Modify relation types in `app/pipeline/relation_extractor.py`

---

## 📞 Support Resources

- **Interactive API Docs:** http://localhost:8001/docs
- **Quick Commands:** `bash QUICK_COMMANDS.sh`
- **Validation:** `python scripts/validate_pipeline.py`
- **Example:** `python scripts/run_pipeline_example.py`
- **Full Docs:** See `INTEGRATION_GUIDE.md`, `SETUP.md`, `IMPLEMENTATION_SUMMARY.md`

---

## 📋 Completion Verification

Once setup completes successfully:

- [x] Virtual environment created and activated
- [x] All dependencies installed
- [x] All AI models downloaded (~2.5GB)
- [x] Neo4j container running
- [x] FastAPI server listening on port 8001
- [x] API responding to requests
- [x] All 8 models loaded and ready
- [x] Ready to process PDFs!

---

**Status:** ✅ **READY TO USE**

Run: `bash run_all.sh` and start processing PDFs immediately!

For detailed information, see **QUICKSTART.md** or **README.md**
