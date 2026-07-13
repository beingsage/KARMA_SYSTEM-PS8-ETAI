Right now almost all your issues happened because this occurred:

Docling
        \
GLiNER ----\
SentenceTransformers ----> transformers
BERTopic ----/
LangGraph --/

Torch
TorchVision
TorchAudio
TorchCodec
SAM2
YOLO
GroundingDINO

↓

everyone upgrades/downgrades everyone else's packages

↓

everything breaks

Instead, build it like a production ML platform.

Layer 0 — Core Infrastructure

This layer should almost never change.

fastapi
uvicorn
pydantic
redis
neo4j
qdrant-client
sqlalchemy
networkx
orjson
python-dotenv
numpy
pandas

requirements-base.txt

requirements/
    base.txt
Layer 1 — Vision Stack

Only computer vision.

YOLO

SAM2

GroundingDINO

OpenCV

Pillow

torch

torchvision

torchaudio

requirements-vision.txt

-r base.txt

ultralytics
opencv-python
pillow

torch
torchvision
torchaudio

segment-anything

groundingdino

Nothing NLP belongs here.

Layer 2 — Document Understanding

Separate all OCR/document parsing.

Docling

Nougat

Surya

OCR

pdfminer

layoutparser

pymupdf

camelot

tabula

requirements-document.txt

-r base.txt

docling
surya-ocr
nougat-ocr

layoutparser
pdfminer.six
camelot

If Docling breaks,

Vision keeps working.

Layer 3 — NLP

Everything language.

GLiNER

GLiREL

SentenceTransformers

BGE

spaCy

transformers

tokenizers

datasets

requirements-nlp.txt

-r base.txt

transformers==<pinned>

sentence-transformers

gliner

glirel

spacy

Notice

Only ONE package controls

transformers

Nobody else.

Layer 4 — Knowledge Graph
Neo4j

Cypher

PyVis

NetworkX

OWLReady

RDFLib

requirements-graph.txt

-r base.txt

neo4j

rdflib

owlready2

pyvis
Layer 5 — Analytics
BERTopic

UMAP

HDBSCAN

scikit-learn

TimesFM

Prophet

requirements-analytics.txt

-r base.txt

bertopic

umap-learn

hdbscan

scikit-learn
Layer 6 — Agents
LangGraph

LangChain

Qwen

GraphRAG

LlamaIndex

requirements-agent.txt

-r base.txt

langgraph

langchain

llama-index

graphrag

No OCR.

No Torch.

No Vision.

Final Structure
requirements/

    base.txt

    vision.txt

    document.txt

    nlp.txt

    graph.txt

    analytics.txt

    agent.txt
Install Order

Never install everything randomly.

Instead

pip install -r requirements/base.txt

↓

pip install -r requirements/vision.txt

↓

pip install -r requirements/document.txt

↓

pip install -r requirements/nlp.txt

↓

pip install -r requirements/graph.txt

↓

pip install -r requirements/analytics.txt

↓

pip install -r requirements/agent.txt
Pin Every Version

Never write

transformers

torch

numpy

Always pin.

Example

torch==2.7.1

torchvision==0.22.1

torchaudio==2.7.1

transformers==4.53.3

sentence-transformers==5.0.0

Otherwise

pip

↓

upgrades one package

↓

50 packages silently change

↓

everything breaks
Lock the Entire Environment

Generate a lock file after installation.

pip freeze > requirements-lock.txt

Then future deployments use

pip install -r requirements-lock.txt

This ensures every machine gets the exact same versions.

Add Compatibility Checks

Before startup, validate critical package versions.

import importlib.metadata as md

REQUIRED = {
    "torch": "2.7.1",
    "transformers": "4.53.3",
    "torchaudio": "2.7.1",
}

for pkg, expected in REQUIRED.items():
    actual = md.version(pkg)
    if actual != expected:
        raise RuntimeError(
            f"{pkg} version mismatch. Expected {expected}, found {actual}"
        )

This fails immediately instead of producing obscure runtime errors later.

Introduce a Model Manager

Right now your logs show the same models loading twice.

YOLO

↓

loaded

↓

FastAPI starts

↓

loaded again

Create a singleton manager.

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
        return cls._instance

    def get(self, name):
        return self.models.get(name)

    def load(self, name, loader):
        if name not in self.models:
            self.models[name] = loader()
        return self.models[name]

Every pipeline stage requests models from this manager instead of instantiating them directly.

Build a Plugin-Based Architecture

Each subsystem should declare its own dependencies and load independently.

plugins/

    vision/

    document/

    graph/

    nlp/

    analytics/

    agents/

Each plugin exposes:

initialize()

health()

process()

shutdown()

If GLiREL fails, only the NLP plugin is degraded. The rest of the platform continues normally.

Add Health Checks

Instead of printing logs, expose structured health information.

{
  "vision": "healthy",
  "ocr": "healthy",
  "docling": "failed",
  "graph": "healthy",
  "neo4j": "failed_auth",
  "gliner": "healthy",
  "glirel": "missing_dependency",
  "embeddings": "healthy"
}

Your frontend and monitoring systems can then determine exactly which capabilities are available.

Use a Dedicated Environment Builder

Avoid pip install at runtime. Build the environment once.

For example:

scripts/
    install_base.sh
    install_vision.sh
    install_document.sh
    install_nlp.sh
    install_graph.sh
    install_analytics.sh
    install_agents.sh
    verify_environment.py

Your deployment process becomes:

git clone

↓

./scripts/install_base.sh

↓

./scripts/install_vision.sh

↓

./scripts/install_document.sh

↓

...
↓
python verify_environment.py
↓
start server
This catches version conflicts before the application ever starts.