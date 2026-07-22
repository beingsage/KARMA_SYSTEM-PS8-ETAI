# Industrial Knowledge Intelligence - Agent Implementation Guide

> Goal: Build **5 specialized AI agents** on top of an existing **Neo4j Knowledge Graph** using a **single shared reasoning engine**. Keep the implementation simple, modular, and hackathon-friendly. Avoid heavy agent frameworks.

---

# Tech Stack

- Gemini API (Planner + Reasoning + Synthesis)
- Neo4j (Knowledge Graph)
- Vector DB (RAG for manuals, SOPs, PDFs)
- Tavily API (Real-time Internet Search)
- FastAPI Backend
- Copilot UI (Chat + Voice + TTS)

Do **NOT** use LangGraph, CrewAI, AutoGen, DSPy, etc.

---

# Overall Architecture

```
User
(Chat / Voice)

        │

Gemini Planner
(Intent Detection)

        │

Agent Router

        │

Shared Reasoning Engine

        │

 ┌──────────────┬──────────────┬──────────────┐
 │              │              │
Neo4j      Vector Search   Internet Search
 │              │              │
 └──────────────┴──────────────┘

        │

Evidence Fusion

        │

Gemini Reasoning

        │

Reflection

        │

Final Answer + Citations
```

---

# Project Structure

```
backend/

agents/
    base_agent.py
    rca.py
    maintenance.py
    quality.py
    regulation.py
    failure.py

reasoning/
    planner.py
    retriever.py
    graph_search.py
    web_search.py
    evidence_ranker.py
    reasoning.py
    reflection.py

neo4j/
    queries.py

llm/
    gemini.py

api/
    chat.py
```

Only **one reasoning engine**.

Only agent prompts change.

---

# Shared Agent Pipeline

Every agent follows exactly this pipeline.

```
User Question

↓

Planner

↓

Intent Detection

↓

Entity Extraction

↓

Cypher Generation

↓

Neo4j Retrieval

↓

Vector Retrieval

↓

Internet Search (if required)

↓

Evidence Merge

↓

Evidence Ranking

↓

Gemini Reasoning

↓

Reflection

↓

Final Answer
```

---

# Base Agent

Every agent inherits the same logic.

```python
class BaseAgent:

    def run(question):

        intent = Planner.plan(question)

        graph = GraphRetriever.retrieve(intent)

        docs = VectorRetriever.retrieve(question)

        web = WebRetriever.retrieve(intent)

        evidence = EvidenceRanker.merge(
            graph,
            docs,
            web
        )

        answer = Gemini.reason(question, evidence)

        return Reflection.review(answer)
```

---

# Five Agents

## 1. Root Cause Analysis Agent

Mission

Find why failures occurred.

Focus

- Maintenance
- Alarms
- Sensors
- Inspection
- Incidents
- Failure History

Output

- Symptoms
- Timeline
- Root Causes
- Confidence
- Corrective Actions
- Preventive Actions

---

## 2. Maintenance Agent

Mission

Recommend maintenance.

Focus

- Maintenance History
- OEM Manuals
- Spare Parts
- Equipment Health
- PM Scheduling

Output

- Health Score
- Maintenance Priority
- Suggested Schedule
- Spare Parts
- Estimated Downtime

---

## 3. Quality Agent

Mission

Detect quality deviations.

Focus

- Inspection Reports
- CAPA
- Batch History
- SOP
- Calibration

Output

- Deviations
- Root Cause
- CAPA
- Risk
- Supporting Evidence

---

## 4. Regulation Agent

Mission

Check compliance.

Focus

- Factory Act
- PESO
- OISD
- ISO
- API
- Internal SOP

Output

- Compliance Status
- Missing Evidence
- Violations
- Required Actions
- Audit Package

---

## 5. Failure Intelligence Agent

Mission

Learn from previous incidents.

Focus

- Incident Reports
- Near Misses
- Lessons Learned
- External Failures

Output

- Failure Pattern
- Trend
- Lessons Learned
- Preventive Recommendations

---

# Shared Services

Every agent uses these.

- Planner
- Entity Extractor
- Cypher Generator
- Graph Retriever
- Vector Retriever
- Internet Retriever
- Evidence Ranker
- Recommendation Engine
- Citation Generator
- Reflection Engine
- Memory Manager

---

# Planner

Planner decides

- intent
- equipment
- document type
- regulations
- whether internet search is required

Example

```json
{
  "intent": "rca",
  "equipment": "P101",
  "internet": true
}
```

---

# Neo4j Retrieval

Never hardcode queries.

Planner generates Cypher.

Example

```cypher
MATCH (e:Equipment {tag:$equipment})
OPTIONAL MATCH (e)-[*1..3]-(n)
RETURN e,n
```

Increase graph depth for multi-hop retrieval.

---

# Multi-Hop Reasoning

Don't ask Gemini to perform graph traversal.

Neo4j does it.

Example

```
Pump

↓

Maintenance

↓

Inspection

↓

Incident

↓

Procedure

↓

Root Cause
```

Gemini only receives the retrieved evidence.

---

# Graph of Thoughts (GoT)

Treat retrieved nodes as connected evidence.

```
Pump

├── Alarm

├── Sensor

├── Incident

├── Procedure

├── Maintenance

└── Quality
```

Gemini synthesizes across the graph.

---

# Tree of Thoughts (ToT)

Instead of producing one answer,

generate multiple hypotheses.

Example

- Bearing Failure
- Lubrication
- Misalignment
- Cavitation
- Seal Failure

Rank using evidence.

Select best.

---

# Chain of Reasoning

Never rely on hidden Chain-of-Thought.

Use explicit reasoning.

```
Extract Entity

↓

Retrieve History

↓

Retrieve Documents

↓

Retrieve Similar Cases

↓

Rank Evidence

↓

Generate Conclusion
```

---

# Reflection

Always review the first answer.

```
Answer

↓

Critic Prompt

↓

Missing Evidence?

↓

Improve

↓

Return
```

This reduces hallucinations.

---

# Evidence Ranking

Rank using

- Semantic Similarity
- Graph Distance
- Source Quality
- Confidence
- Recency
- Equipment Criticality

Only send top evidence to Gemini.

---

# Real-Time Internet Search

Internet is another retrieval source.

Never maintain your own regulation dataset.

Recommended

- Tavily API ⭐⭐⭐⭐⭐

Search only when needed.

Examples

- Latest PESO Circular
- Latest OISD Update
- Latest ISO Revision
- OEM Bulletins
- Vendor Documentation

Restrict search to trusted domains when possible.

Examples

```
site:peso.gov.in

site:oisd.gov.in

site:bis.gov.in

site:iso.org

site:api.org

site:asme.org
```

---

# Internet Retriever

```python
graph = Neo4j.retrieve()

docs = Vector.retrieve()

web = Tavily.search()

evidence = merge(
    graph,
    docs,
    web
)
```

---

# Agent Memory

Maintain lightweight session memory.

Store

- Current Equipment
- Current Plant
- Previous Queries
- Previous Cypher
- Current Incident
- Current Hypothesis

Do not repeatedly retrieve identical information.

---

# Standard Response Schema

```json
{
  "summary": "",
  "key_findings": [],
  "evidence": [],
  "recommendations": [],
  "confidence": 0.0,
  "risk_level": "",
  "related_assets": [],
  "related_documents": [],
  "next_actions": [],
  "citations": []
}
```

---

# Design Principles

- One backend.
- One reasoning engine.
- One planner.
- One retrieval pipeline.
- Five lightweight agents.
- Neo4j performs graph traversal.
- Vector DB retrieves documents.
- Tavily provides real-time knowledge.
- Gemini plans, reasons, compares evidence, and generates responses.
- Keep logic deterministic, modular, explainable, and evidence-backed.

> **Rule:** The Knowledge Graph is the memory, the Retrieval Layer gathers evidence, and Gemini is the reasoning engine—not the source of truth.