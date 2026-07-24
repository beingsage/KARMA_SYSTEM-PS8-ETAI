# 🚀 Quick Start Guide

## One Command Setup

Just run this one command and everything will be set up automatically:

```bash
bash run_all.sh
```

That's all you need! ✨

---

## What Happens

The script automatically:

1. ✅ **Creates virtual environment** - If it doesn't exist
2. ✅ **Installs dependencies** - From `requirements.txt`
3. ✅ **Creates directories** - `data/`, `models/`, etc.
4. ✅ **Downloads AI models** - ~2.5GB of pre-trained models
   - YOLOv12 (P&ID detection)
   - GroundingDINO (object detection)
   - SAM2 (segmentation)
   - GLiNER (entity extraction)
   - BGE-M3 (embeddings)
   - And more...
5. ✅ **Starts Neo4j** - Knowledge graph database
6. ✅ **Launches backend** - REST API server

---

## After Run

Once the script starts the server, you'll see:

```
================================================================================
✓ Pipeline Ready!
================================================================================

Backend running on:
  HTTP:  http://127.0.0.1:8001
  Docs:  http://127.0.0.1:8001/docs
  ReDoc: http://127.0.0.1:8001/redoc

API Endpoints:
  POST   /api/v1/process-pdf        - Upload and process PDF
  GET    /api/v1/jobs               - List all jobs
  GET    /api/v1/jobs/{job_id}      - Get job results
  GET    /api/v1/models/status      - Check model status
  GET    /health                    - Health check
```

---

## Using the API

### Via Browser (Interactive Docs)

Open: http://127.0.0.1:8001/docs

You can test all endpoints interactively!

### Via Command Line

```bash
# Upload and process a PDF
curl -X POST \
  -F "file=@your_document.pdf" \
  http://localhost:8001/api/v1/process-pdf

# Get job results
curl http://localhost:8001/api/v1/jobs/{job_id}

# List all jobs
curl http://localhost:8001/api/v1/jobs

# Check model status
curl http://localhost:8001/api/v1/models/status
```

### Via Python

```python
import requests

# Upload PDF
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/api/v1/process-pdf',
        files={'file': f}
    )

job_id = response.json()['job_id']
print(f"Processing job: {job_id}")

# Get results
results = requests.get(f'http://localhost:8001/api/v1/jobs/{job_id}')
print(results.json())
```

---

## Stopping the Server

Press `Ctrl+C` in the terminal where the script is running.

---

## Running Again

After the first run, models are cached, so the next time `bash run_all.sh` runs much faster!

```
First run:  ~5-10 minutes (downloads ~2.5GB)
Later runs: ~1-2 minutes (models cached)
```

---

## Troubleshooting

### "Virtual environment not found" Error

If you get this error, make sure you're in the project directory:

```bash
cd /media/sagesujal/DEV1/bytes/structured
bash run_all.sh
```

### "Docker not found" Error

Install Docker:
- Linux: `sudo apt-get install docker.io`
- Mac: Download Docker Desktop
- Windows: Download Docker Desktop

### "Port 8001 already in use"

The backend is already running. Either:
- Stop the previous instance with `Ctrl+C`
- Or use a different port in the script

### Models not downloading

Check your internet connection and disk space:

```bash
# Check disk space
df -h

# Check internet
ping google.com
```

---

## Project Structure

```
.
├── run_all.sh              ← Run this!
├── README.md               ← Overview
├── QUICKSTART.md           ← This file
├── SETUP.md                ← Detailed setup
├── INTEGRATION_GUIDE.md    ← Model details
├── app/
│   ├── main.py            ← FastAPI app
│   ├── pipeline/          ← AI models
│   └── ...
├── models/                ← Downloaded models (~2.5GB)
├── data/                  ← Jobs and uploads
├── requirements.txt       ← Dependencies
└── docker-compose.yml     ← Neo4j config
```

---

## Next Steps

1. **Upload a PDF** - Use the API to process your first document
2. **Check Results** - View extracted entities and relations
3. **Explore Models** - Read `INTEGRATION_GUIDE.md` for technical details
4. **Customize** - Fine-tune models for your domain
5. **Deploy** - Set up for production use

---

## More Information

- **API Documentation:** http://localhost:8001/docs (interactive)
- **Setup Guide:** See `SETUP.md`
- **Integration Details:** See `INTEGRATION_GUIDE.md`
- **Implementation Summary:** See `IMPLEMENTATION_SUMMARY.md`

---

That's it! Just `bash run_all.sh` and you're ready to go! 🎉
