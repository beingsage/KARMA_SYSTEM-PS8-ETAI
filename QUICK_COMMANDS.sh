#!/usr/bin/env bash
# Industrial PDF-to-Graph Pipeline - Quick Reference Card
# Save this as: quick_commands.sh

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Industrial PDF-to-Graph Pipeline - Quick Commands            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "📦 QUICK START"
echo "  1️⃣  One-command setup:      bash run_all.sh"
echo ""

echo "🌐 API ACCESS"
echo "  📊 Interactive Docs:        http://localhost:8001/docs"
echo "  📖 Alternative Docs:        http://localhost:8001/redoc"
echo "  🏥 Health Check:            curl http://localhost:8001/health"
echo ""

echo "📤 UPLOAD & PROCESS"
echo "  Upload PDF:                 curl -X POST -F 'file=@file.pdf' \\"
echo "                              http://localhost:8001/api/v1/process-pdf"
echo ""

echo "📋 RETRIEVE RESULTS"
echo "  Get job by ID:              curl http://localhost:8001/api/v1/jobs/{job_id}"
echo "  List all jobs:              curl http://localhost:8001/api/v1/jobs"
echo "  Check models:               curl http://localhost:8001/api/v1/models/status"
echo ""

echo "🔧 UTILITIES"
echo "  Validate setup:             python scripts/validate_pipeline.py"
echo "  Download models:            python scripts/download_models.py"
echo "  Run example:                python scripts/run_pipeline_example.py"
echo ""

echo "📚 DOCUMENTATION"
echo "  Quick Start:                cat QUICKSTART.md"
echo "  Full Setup:                 cat SETUP.md"
echo "  Model Details:              cat INTEGRATION_GUIDE.md"
echo "  Implementation:             cat IMPLEMENTATION_SUMMARY.md"
echo ""

echo "🐳 DOCKER COMMANDS"
echo "  View Neo4j logs:            docker compose logs -f neo4j"
echo "  Stop services:              docker compose down"
echo "  View running containers:    docker compose ps"
echo ""

echo "🛑 STOP SERVER"
echo "  Press Ctrl+C in the terminal running: bash run_all.sh"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Models Included (~2.5GB total)                               ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  ✓ YOLOv12          → P&ID Symbol Detection     (~100MB)       ║"
echo "║  ✓ GroundingDINO    → Zero-shot Objects        (~500MB)       ║"
echo "║  ✓ SAM2             → Instance Segmentation    (~400MB)       ║"
echo "║  ✓ GLiNER           → Entity Extraction        (~600MB)       ║"
echo "║  ✓ GLiREL+REBEL     → Relation Extraction      (~400MB)       ║"
echo "║  ✓ BLINK            → Entity Linking           (~500MB)       ║"
echo "║  ✓ BGE-M3           → Semantic Embeddings      (~450MB)       ║"
echo "║  ✓ BGE-Reranker     → Relevance Ranking        (~300MB)       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

echo "💡 TIPS"
echo "  • First run downloads models (~2.5GB), next runs are faster"
echo "  • All models cached locally in ./models/ directory"
echo "  • Jobs stored in ./data/jobs/ as JSON files"
echo "  • Neo4j runs in Docker, accessible at bolt://localhost:7687"
echo "  • Use http://localhost:8001/docs to test API interactively"
echo ""
