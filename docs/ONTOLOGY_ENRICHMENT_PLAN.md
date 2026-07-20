# Ontology Enrichment Plan for Structured Codebase

> This document is the implementation blueprint for turning the current pipeline from a flat entity/relation extractor into an ontology-aware industrial knowledge system.
> Every major change is marked by repository file, class, and function name so it can be executed incrementally and reviewed safely.

---

## 1. Purpose of this document

This repository already contains a strong extraction and reasoning skeleton, but the ontology is still shallow. The system can identify some entities and relations, persist them to a graph, and generate summaries, yet it does not yet enforce an industrial ontology end to end.

The purpose of this document is to provide a concrete plan for upgrading the codebase in the following order:

1. Define a typed ontology contract.
2. Normalize all extraction outputs against that contract.
3. Make persistence use ontology-aware node and edge types.
4. Make reasoning and retrieval consume typed graph facts.
5. Expose the richer ontology through the API and docs.
6. Add evaluation coverage so ontology quality can be measured and improved.

This document is intentionally implementation-oriented. It does not only say “add more labels”; it says exactly where the work belongs and how each module will change.

---

## 2. Working assumptions

The following assumptions are used throughout this plan:

- The repository is a document-processing and knowledge-graph pipeline.
- The main runtime entrypoint is app/main.py.
- The current public API uses Pydantic models from app/schemas.py.
- The pipeline stages are implemented under app/pipeline/.
- The current ontology layer is a thin schema plus a small component taxonomy.
- The final state should support typed entities, typed relations, alias resolution, canonical IDs, graph persistence, and ontology-aware reasoning.

---

## 3. Current repository inventory

### 3.1 Core application surface

- app/main.py
  - FastAPI entrypoint.
  - Contains endpoint handlers such as process_pdf, process_document, get_job, list_job_summaries, model_status, and copilot/advanced endpoints.

- app/schemas.py
  - Defines the API contract for entities, relations, and job results.
  - Current shape is relatively flat and generic.

### 3.2 Pipeline modules

- app/pipeline/entity_extractor.py
  - Class: GlinerEntityExtractor
  - Functions: extract, _extract_with_model
  - Current behavior: outputs coarse labels such as equipment, process, parameter, material, control_system, location, failure_mode, maintenance.

- app/pipeline/relation_extractor.py
  - Class: GLiRELRelationExtractor
  - Functions: _model_extract, _heuristic_extract, _normalize_entities
  - Current behavior: uses a limited relation vocabulary and often falls back to related_to.

- app/pipeline/component_detector.py
  - Class: ComponentDetector
  - Functions: detect_from_text, _build_keyword_index, detect_pid_components
  - Current behavior: uses a small taxonomy and limited matching logic.

- app/pipeline/neo4j_store.py
  - Class: Neo4jGraphStore
  - Functions: persist_entities, persist_relations, _entity_hash, persist_to_neo4j
  - Current behavior: stores generic nodes and edges with minimal typed semantics.

- app/pipeline/engine_v2.py
  - Class: PipelineEngineV2 or equivalent orchestration class.
  - Functions: _link_entities, _format_entities, _graphrag_analyze, _copilot_analyze
  - Current behavior: links entities via simple normalization when BLINK is not available.

- app/pipeline/advanced_pipeline.py
  - Functions: stage_semantic_indexing, stage_graph_reasoning, stage_llm_analysis
  - Current behavior: mixes schema names like text/type and name/entity_type.

- app/pipeline/copilot_agent.py
  - Class: IndustrialCopilotAgent
  - Functions: reason, root_cause_analysis, get_maintenance_plan, compliance_check, risk_assessment
  - Current behavior: reasons mostly from coarse keywords and generic equipment assumptions.

- app/pipeline/graphrag_summarizer.py
  - Class: GraphRAGSummarizer
  - Functions: generate_summary, _check_evidence_sufficiency, _validate_claims
  - Current behavior: grounds claims but does not enforce ontology semantics.

### 3.3 Supporting data

- app/data/component_taxonomy.json
  - Current state: a small taxonomy intended for components and PID artifacts.

- app/pipeline/models.py
  - Contains fallback vocabulary and compatibility functions.
  - Current state: mostly legacy keywords and normalization helpers.

---

## 4. What must change, at a high level

### 4.1 The core problem

The current repository is not wrong; it is simply not ontology-first. It uses strings where semantics should live in typed objects. That leads to:

- unstable entity identities
- ambiguous relation meaning
- weak graph semantics
- fuzzy reasoning and retrieval
- inconsistent API payloads

### 4.2 The target state

The target system should support:

- ontology classes such as equipment, process, location, component, failure_mode, control_system, material, maintenance_action
- typed relations such as controls, measures, feeds, contains, located_in, connected_to, causes, affects
- canonical IDs for entities and relations
- provenance metadata for every extracted fact
- schema versioning so downstream systems can reason about data freshness
- typed graph nodes and edges in Neo4j
- ontology-aware query and reasoning layers

---

### 4.3 Ontology growth contract for new data model files

The plan must not stop at the current schema and pipeline modules. Every time a new data model file is introduced under app/data/, app/pipeline/, or any other domain module, that file must be treated as a first-class ontology artifact and must be wired into the whole pipeline. In practice, this means the following contract must be satisfied before the system can be considered ontology-driven.

#### 4.3.1 Rule: new data model files must register ontology intent

Any new data model file must be evaluated as an ontology source, not as a standalone schema fragment. The implementation should require the following:

- A canonical ontology class or relation definition must be declared for every new entity or relation type.
- Every new model must define its allowed parent/child ontology classes, aliases, and provenance expectations.
- The model must be referenced by an ontology registry or normalization helper so the rest of the pipeline can consume it consistently.
- If a new file introduces a new concept, the extractor, relation extractor, persistence layer, linking layer, reasoning layer, and API layer must all be updated to understand it.

#### 4.3.2 Required cross-layer update checklist

When a new model file is added, the following functions and classes must be inspected and updated where needed:

- API contract and public payloads
  - app/schemas.py :: EntityRecord
  - app/schemas.py :: RelationRecord
  - app/schemas.py :: JobResult
  - Purpose: remove flat payload drift and expose ontology-aware fields such as ontology_class, ontology_path, canonical_id, schema_version, provenance, and relation_category.

- Entity extraction
  - app/pipeline/entity_extractor.py :: GlinerEntityExtractor
  - app/pipeline/entity_extractor.py :: extract
  - app/pipeline/entity_extractor.py :: _extract_with_model
  - Purpose: replace the current narrow label set with an ontology mapping that can grow when new model files introduce new classes.

- Relation extraction
  - app/pipeline/relation_extractor.py :: GLiRELRelationExtractor
  - app/pipeline/relation_extractor.py :: _model_extract
  - app/pipeline/relation_extractor.py :: _heuristic_extract
  - app/pipeline/relation_extractor.py :: _normalize_entities
  - Purpose: stop depending on a small relation vocabulary and make relation extraction emit typed edges that match the new ontology.

- Taxonomy and keyword fallback
  - app/pipeline/component_detector.py :: ComponentDetector
  - app/pipeline/component_detector.py :: detect_from_text
  - app/pipeline/component_detector.py :: _build_keyword_index
  - app/data/component_taxonomy.json
  - app/pipeline/models.py :: canonicalize_entity_name
  - app/pipeline/models.py :: normalize_entity_payload
  - app/pipeline/models.py :: normalize_relation_payload
  - Purpose: make the taxonomy grow with new data-model concepts instead of staying limited to a small set of generic component groups.

- Graph persistence
  - app/pipeline/neo4j_store.py :: Neo4jGraphStore
  - app/pipeline/neo4j_store.py :: persist_entities
  - app/pipeline/neo4j_store.py :: persist_relations
  - Purpose: store typed ontology nodes and typed ontology edges rather than generic Entity and Relation nodes.

- Linking and formatting
  - app/pipeline/engine_v2.py :: _link_entities
  - app/pipeline/engine_v2.py :: _format_entities
  - app/pipeline/engine_v2.py :: _graphrag_analyze
  - app/pipeline/engine_v2.py :: _copilot_analyze
  - Purpose: replace string-only canonicalization with ontology-aware linking, alias resolution, and canonical ID propagation.

- Advanced pipeline semantics
  - app/pipeline/advanced_pipeline.py :: stage_semantic_indexing
  - app/pipeline/advanced_pipeline.py :: stage_graph_reasoning
  - app/pipeline/advanced_pipeline.py :: stage_llm_analysis
  - Purpose: eliminate schema drift such as text/type versus name/entity_type and ensure every stage uses a shared ontology contract.

- Reasoning and copilot behavior
  - app/pipeline/copilot_agent.py :: IndustrialCopilotAgent
  - app/pipeline/copilot_agent.py :: reason
  - app/pipeline/copilot_agent.py :: root_cause_analysis
  - app/pipeline/copilot_agent.py :: get_maintenance_plan
  - app/pipeline/copilot_agent.py :: compliance_check
  - app/pipeline/copilot_agent.py :: risk_assessment
  - Purpose: move away from coarse equipment/failure assumptions and reason over the richer ontology contract.

- GraphRAG and evidence-grounded summarization
  - app/pipeline/graphrag_summarizer.py :: GraphRAGSummarizer
  - app/pipeline/graphrag_summarizer.py :: generate_summary
  - app/pipeline/graphrag_summarizer.py :: _check_evidence_sufficiency
  - app/pipeline/graphrag_summarizer.py :: _validate_claims
  - Purpose: ensure summarization uses ontology-validated entities and relations rather than whatever raw facts happen to be present.

#### 4.3.3 Current issues that this contract must resolve

The following issues are explicitly in scope and must be addressed as part of the ontology growth contract:

- The public API is flat and must be upgraded through app/schemas.py :: EntityRecord, RelationRecord, and JobResult.
- Entity extraction is narrow and must be expanded through app/pipeline/entity_extractor.py :: GlinerEntityExtractor.
- Relation extraction is narrow and must be expanded through app/pipeline/relation_extractor.py :: GLiRELRelationExtractor.
- The taxonomy is still small and must be expanded through app/pipeline/component_detector.py and app/data/component_taxonomy.json.
- The fallback vocabulary is too small and must be expanded through app/pipeline/models.py.
- Neo4j persistence is generic and must be upgraded through app/pipeline/neo4j_store.py.
- Entity linking is still mostly string-based and must be upgraded through app/pipeline/engine_v2.py :: _link_entities.
- The semantic indexing layer has schema drift and must be unified through app/pipeline/advanced_pipeline.py.
- Copilot reasoning is still coarse and must be upgraded through app/pipeline/copilot_agent.py.
- GraphRAG is not ontology-enforced and must be upgraded through app/pipeline/graphrag_summarizer.py.
- Runtime outputs should eventually reflect the richer ontology rather than a limited artifact such as data/pipeline/12.pid_component_detection_enhanced.json.

#### 4.3.4 Required acceptance criteria for a unified system

A new data model file is only considered fully integrated when all of the following are true:

- The API exposes the same ontology-aware fields that the graph store and reasoning layers use.
- Entity extraction, relation extraction, linking, persistence, and summarization all rely on the same canonical ontology contract.
- No stage depends exclusively on free-form string names when a typed ontology class can be provided instead.
- The system can explain, through provenance and canonical IDs, where every entity and relation came from.
- The pipeline remains coherent when new schema files are added and does not require manual patching in isolated modules.

#### 4.3.5 Implementation guardrail

Whenever a new data model file is introduced, the change should be reviewed against this checklist:

1. Did the new model add a new entity or relation class?
2. Was the ontology registry updated?
3. Was the extractor updated to emit the new ontology class?
4. Was relation extraction updated to emit typed relations?
5. Was the graph store updated to persist the new typed structure?
6. Was the API updated to expose the richer contract?
7. Was reasoning and summarization updated to consume the richer contract?
8. Were tests added to make the new ontology path regression-proof?

If any answer is no, the change is not yet fully integrated into the unified ontology system.

---

## 5. Implementation strategy by layer

### Layer 1 — Schema and contract layer

This is the foundational change. Everything else will depend on it.

#### Files to change

- app/schemas.py
- app/pipeline/models.py
- app/pipeline/ontology.py (new, recommended)

#### Classes and functions to change

- app/schemas.py :: EntityRecord
- app/schemas.py :: RelationRecord
- app/schemas.py :: JobResult
- app/pipeline/models.py :: canonicalize_entity_name
- app/pipeline/models.py :: normalize_entity_payload
- app/pipeline/models.py :: normalize_relation_payload

#### Required actions

1. Introduce ontology-aware entity and relation models.
2. Preserve legacy fields for compatibility during transition.
3. Add schema version fields.
4. Add provenance fields.
5. Add type counts in job outputs.

#### Example target structure

- Entity fields: id, canonical_id, ontology_class, ontology_path, name, entity_type, confidence, source_method, provenance, schema_version
- Relation fields: id, source_id, target_id, relation_class, relation_category, confidence, provenance, schema_version

#### Acceptance criteria

- Every extraction pipeline output can be normalized into the new contract.
- The API payload contains typed ontology metadata.
- The graph store can persist the same contract without lossy transformation.

---

### Layer 2 — Entity extraction layer

This is where entities first enter the system. The ontology contract must be enforced here.

#### Files to change

- app/pipeline/entity_extractor.py
- app/pipeline/models.py
- app/pipeline/ontology.py (new)

#### Classes and functions to change

- app/pipeline/entity_extractor.py :: GlinerEntityExtractor
- app/pipeline/entity_extractor.py :: extract
- app/pipeline/entity_extractor.py :: _extract_with_model

#### Required actions

1. Replace the coarse label vocabulary with an ontology mapping.
2. Map model labels to ontology classes.
3. Attach canonical IDs and provenance.
4. Ensure every returned entity includes ontology metadata.
5. Add fallback logic so unknown labels are not silently dropped.

#### Recommended ontology map

- equipment -> equipment
- equipment.pump -> equipment.pump
- equipment.valve -> equipment.valve
- control_system -> control_system
- control_system.plc -> control_system.plc
- process -> process
- process.heat_exchange -> process.heat_exchange
- location -> location
- location.vessel -> location.vessel
- material -> material
- material.fluid -> material.fluid
- failure_mode -> failure_mode
- maintenance_action -> maintenance_action

#### Suggested implementation pattern

- Add a mapping function such as label_to_ontology_class(label).
- Add a normalization function such as normalize_extracted_entity(raw, document_context).
- Ensure the return structure includes both human-readable name and machine-readable ontology class.

#### Acceptance criteria

- Every entity emitted by the extractor includes ontology_class.
- Known labels are mapped deterministically.
- Unknown labels are routed through a fallback ontology class instead of being dropped or misclassified.

---

### Layer 3 — Relation extraction layer

This layer must stop producing generic relations and start producing typed relations.

#### Files to change

- app/pipeline/relation_extractor.py
- app/pipeline/ontology.py (new)
- app/pipeline/models.py

#### Classes and functions to change

- app/pipeline/relation_extractor.py :: GLiRELRelationExtractor
- app/pipeline/relation_extractor.py :: _model_extract
- app/pipeline/relation_extractor.py :: _heuristic_extract
- app/pipeline/relation_extractor.py :: _normalize_entities

#### Required actions

1. Define a richer relation taxonomy.
2. Make relation extraction emit source_id and target_id, not just name strings.
3. Remove excessive usage of generic related_to unless it is a true fallback.
4. Keep evidence and confidence.
5. Preserve relation category values such as functional, spatial, causal, operational.

#### Recommended relation types

- controls
- measures
- feeds
- contains
- located_in
- connected_to
- causes
- affects
- depends_on
- part_of

#### Suggested implementation pattern

- Add a relation classifier based on lexical patterns and surrounding context.
- For each extracted pair, create a relation object with relation_class, relation_category, and confidence.
- If the source/target cannot be resolved, emit a weakly typed relation with a provenance note.

#### Acceptance criteria

- Relation outputs contain relation_class.
- Relation outputs are linked to resolved entity IDs.
- Generic fallback relations are clearly marked as fallback and not treated as fully typed facts.

---

### Layer 4 — Component taxonomy layer

The current component detector is already valuable. It should become the first ontology-aware detector rather than just a keyword matcher.

#### Files to change

- app/pipeline/component_detector.py
- app/data/component_taxonomy.json
- app/pipeline/models.py

#### Classes and functions to change

- app/pipeline/component_detector.py :: ComponentDetector
- app/pipeline/component_detector.py :: detect_from_text
- app/pipeline/component_detector.py :: _build_keyword_index
- app/pipeline/component_detector.py :: detect_pid_components

#### Required actions

1. Expand the taxonomy beyond a few generic entries.
2. Add hierarchy and parent-child relationships.
3. Add canonical IDs and ontology classes.
4. Support synonyms and aliases.
5. Make the detector return typed component records rather than flat text matches.

#### Recommended taxonomy fields

- id
- canonical_id
- ontology_class
- parent_class
- parent_id
- synonyms
- keywords
- category
- component_family
- expected_context

#### Acceptance criteria

- Component detection results can be linked to a canonical ID.
- The taxonomy can represent parent-child relationships.
- The detector emits ontology-aware metadata consistently.

---

### Layer 5 — Linking and normalization layer

This layer aligns canonical identities across mentions, aliases, and extracted entities.

#### Files to change

- app/pipeline/engine_v2.py
- app/pipeline/models.py
- app/pipeline/ontology.py (new)

#### Classes and functions to change

- app/pipeline/engine_v2.py :: _link_entities
- app/pipeline/engine_v2.py :: _format_entities
- app/pipeline/engine_v2.py :: _graphrag_analyze
- app/pipeline/models.py :: canonicalize_entity_name

#### Required actions

1. Replace simplistic string-based linking with alias resolution.
2. Create a canonical entity identifier for every entity.
3. Preserve ontology_class during linking.
4. Make linking deterministic and explainable.
5. Add a `linking_confidence` field where useful.

#### Recommended behavior

- If two mentions refer to the same physical asset, they should share a canonical ID.
- If a synonym is detected, the system should map it to the parent ontology entry.
- Linking should be explicit enough for debugging and audit.

#### Acceptance criteria

- The same entity appearing in two places is represented by the same canonical ID.
- The graph store receives consistent IDs.
- Linking results can be traced to evidence.

---

### Layer 6 — Persistence layer

The graph layer should store typed semantics, not only labels.

#### Files to change

- app/pipeline/neo4j_store.py
- app/pipeline/engine_v2.py

#### Classes and functions to change

- app/pipeline/neo4j_store.py :: Neo4jGraphStore
- app/pipeline/neo4j_store.py :: persist_entities
- app/pipeline/neo4j_store.py :: persist_relations
- app/pipeline/neo4j_store.py :: _entity_hash
- app/pipeline/neo4j_store.py :: persist_to_neo4j

#### Required actions

1. Use typed node labels based on ontology_class.
2. Use typed relationships based on relation_class.
3. Persist provenance and schema_version.
4. Prefer entity ID and canonical ID over string names for graph matching.
5. Add support for ontology version metadata.

#### Example persistence behavior

- Equipment entities become nodes with labels like :Entity:Equipment.
- Process entities become :Entity:Process.
- Relations become :CONTROLS, :MEASURES, :LOCATED_IN, and so on.

#### Acceptance criteria

- The persisted graph contains typed nodes and edges.
- The graph can answer ontology-aware queries.
- The graph can be rebuilt from the same typed contract without semantic loss.

---

### Layer 7 — Advanced semantic pipeline

The indexing and reasoning stages must consume the same ontology contract generated earlier in the pipeline.

#### Files to change

- app/pipeline/advanced_pipeline.py
- app/pipeline/engine_v2.py

#### Classes and functions to change

- app/pipeline/advanced_pipeline.py :: stage_semantic_indexing
- app/pipeline/advanced_pipeline.py :: stage_graph_reasoning
- app/pipeline/advanced_pipeline.py :: stage_llm_analysis
- app/pipeline/engine_v2.py :: _graphrag_analyze

#### Required actions

1. Stop treating metadata as a generic text/type pair.
2. Build metadata from ontology_class, canonical_id, and relation_class.
3. Use typed evidence when forming prompts for downstream reasoning.
4. Ensure semantic indexing is consistent with the contract used by the extractor and graph store.

#### Suggested metadata shape

- entity_id
- canonical_id
- ontology_class
- ontology_path
- component_family
- provenance
- confidence

#### Acceptance criteria

- Semantic indexing can be queried by ontology class.
- Graph reasoning uses typed facts rather than raw strings.
- Prompt construction uses entity IDs and relation classes rather than loose text descriptions.

---

### Layer 8 — Copilot agent and GraphRAG reasoning layer

The agent and summarizer should reason over typed facts, not just generic keywords.

#### Files to change

- app/pipeline/copilot_agent.py
- app/pipeline/graphrag_summarizer.py
- app/pipeline/engine_v2.py

#### Classes and functions to change

- app/pipeline/copilot_agent.py :: IndustrialCopilotAgent
- app/pipeline/copilot_agent.py :: reason
- app/pipeline/copilot_agent.py :: root_cause_analysis
- app/pipeline/copilot_agent.py :: get_maintenance_plan
- app/pipeline/graphrag_summarizer.py :: GraphRAGSummarizer
- app/pipeline/graphrag_summarizer.py :: generate_summary
- app/pipeline/graphrag_summarizer.py :: _check_evidence_sufficiency

#### Required actions

1. Replace coarse logic and keyword heuristics with typed reasoning.
2. Add ontology-aware evidence checks.
3. Use typed entities and relations when forming root-cause and maintenance suggestions.
4. Use typed graph evidence to justify claims.

#### Suggested behavior

- If a failure_mode entity is present and linked to a equipment node by a causes relation, the agent can propose a targeted maintenance step.
- If a location entity is connected to a process and a control system, the agent can explain the topology more precisely.

#### Acceptance criteria

- Reasoning outputs include ontology-backed evidence.
- GraphRAG claims have a typed provenance trail.
- The agent does not rely solely on generic failure keywords.

---

### Layer 9 — API and documentation layer

The public API must expose ontology-aware outputs rather than only flat text records.

#### Files to change

- app/main.py
- app/schemas.py
- README.md
- SETUP.md
- SETUP_COMPLETE.md
- QUICKSTART.md

#### Functions to change

- app/main.py :: process_pdf
- app/main.py :: process_document
- app/main.py :: get_job
- app/main.py :: list_job_summaries
- app/main.py :: model_status

#### Required actions

1. Add ontology-aware response payloads.
2. Add new endpoints for schema inspection and ontology browsing.
3. Keep backward compatibility for existing clients.
4. Document the Swagger and OpenAPI routes clearly.

#### Proposed endpoints

- /api/v1/ontology/schema
- /api/v1/ontology/entities
- /api/v1/ontology/relations
- /api/v1/ontology/graph

#### Acceptance criteria

- Swagger shows ontology-aware response models.
- Existing endpoints return richer data without breaking prior clients.
- Documentation mentions the ontology contract and its version.

---

### Layer 10 — Evaluation and validation layer

Evaluation is necessary to ensure the ontology becomes an operational asset rather than a decorative label.

#### Files to change

- app/pipeline/ontology_metrics.py (new)
- app/pipeline/engine_v2.py
- README.md

#### Functions to change

- app/pipeline/engine_v2.py :: _format_entities
- app/pipeline/engine_v2.py :: _format_relations

#### Required actions

1. Measure entity precision by ontology class.
2. Measure relation precision by relation class.
3. Measure linking accuracy.
4. Track schema version and provenance coverage.
5. Surface metrics in pipeline outputs.

#### Example metrics

- entity_precision_by_type
- entity_recall_by_type
- relation_precision_by_class
- relation_recall_by_class
- linking_accuracy
- provenance_coverage
- graph_completeness

#### Acceptance criteria

- A pipeline run can report ontology quality scorecards.
- New regressions are visible in CI or validation runs.
- The ontology is treated as a measurable product surface, not a passive schema.

---

## 6. Phase-by-phase implementation plan

### Phase 1 — Define the ontology contract

Goal: create a typed contract that every layer can target.

Tasks:

- Update app/schemas.py with ontology-aware entity and relation models.
- Add new helpers in app/pipeline/models.py.
- Create app/pipeline/ontology.py to centralize ontology mappings and validation rules.
- Add schema_version and provenance fields.

Deliverables:

- A stable contract the extractor, graph store, and API can all reuse.

---

### Phase 2 — Normalize extraction outputs

Goal: make the extraction layer emit typed records.

Tasks:

- Refactor GlinerEntityExtractor to emit ontology_class and canonical_id.
- Refactor GLiRELRelationExtractor to emit relation_class and source_id/target_id.
- Make the extractor call normalization helpers before returning data.

Deliverables:

- Typed entities and relations that can flow into the rest of the pipeline.

---

### Phase 3 — Expand taxonomy and component detection

Goal: convert component detection into an ontology-aware capability.

Tasks:

- Expand app/data/component_taxonomy.json.
- Add hierarchy and synonyms to the taxonomy.
- Update ComponentDetector to return ontology-aware component records.

Deliverables:

- A richer component ontology that can participate in the broader industrial graph.

---

### Phase 4 — Implement linking and alias resolution

Goal: ensure one physical asset maps to one canonical ID.

Tasks:

- Refactor _link_entities in app/pipeline/engine_v2.py.
- Add canonicalization and alias mapping helpers.
- Ensure all later stages use canonical IDs.

Deliverables:

- Stable entity identity across extraction, storage, and reasoning.

---

### Phase 5 — Upgrade graph persistence

Goal: make Neo4j store typed semantics.

Tasks:

- Use typed node labels.
- Use typed edge labels.
- Persist provenance and schema_version.
- Store canonical IDs.

Deliverables:

- A knowledge graph that can support ontology-aware queries and reasoning.

---

### Phase 6 — Upgrade reasoning and retrieval

Goal: make reasoning consume the ontology rather than raw strings.

Tasks:

- Refactor IndustrialCopilotAgent.
- Refactor GraphRAGSummarizer.
- Update advanced pipeline stages to use typed metadata.

Deliverables:

- Ontology-aware summaries, maintenance recommendations, and reasoning outputs.

---

### Phase 7 — Expose the ontology through the API

Goal: make the ontology visible to downstream clients and agents.

Tasks:

- Add new ontology endpoints to app/main.py.
- Update schema definitions in app/schemas.py.
- Update docs in README and setup files.

Deliverables:

- A documented, browsable ontology surface for API consumers.

---

### Phase 8 — Add evaluation and regression checks

Goal: make ontology quality measurable.

Tasks:

- Add app/pipeline/ontology_metrics.py.
- Integrate metrics into pipeline output.
- Add validation scripts or tests.

Deliverables:

- A repeatable way to detect ontology quality regressions.

---

## 7. File-by-file change inventory

### 7.1 app/schemas.py

Change target: public API contract.

- Replace or extend EntityRecord.
- Replace or extend RelationRecord.
- Extend JobResult.
- Add support for ontology metadata and counts.

Recommended changes:

- Add fields: id, canonical_id, ontology_class, ontology_path, provenance, schema_version
- Add fields: source_id, target_id, relation_class, relation_category, provenance, schema_version

### 7.2 app/pipeline/models.py

Change target: normalization helpers and compatibility layer.

- Keep backwards-compatible helper names where possible.
- Add normalize_entity_record and normalize_relation_record.
- Add ontology-aware canonicalization helpers.
- Preserve existing fallback behavior while layering new functionality over it.

### 7.3 app/pipeline/entity_extractor.py

Change target: model output structure.

- Add ontology mapping to GlinerEntityExtractor.
- Ensure extract returns typed entities.
- Normalize entities before returning them.

### 7.4 app/pipeline/relation_extractor.py

Change target: relation typing.

- Add a richer relation taxonomy.
- Ensure relations use IDs rather than only names.
- Remove overuse of generic related_to.

### 7.5 app/pipeline/component_detector.py

Change target: taxonomy mapping and metadata output.

- Expand taxonomy schema.
- Add hierarchy.
- Add component_family and ontology_class.
- Return typed component records.

### 7.6 app/data/component_taxonomy.json

Change target: source taxonomy data.

- Add richer ontology entries.
- Add parent, synonyms, keywords, and category.
- Ensure each entry can map to a canonical ID and ontology class.

### 7.7 app/pipeline/neo4j_store.py

Change target: graph persistence semantics.

- Persist typed labels.
- Persist typed relationship labels.
- Keep provenance and schema metadata.
- Use canonical IDs for matching.

### 7.8 app/pipeline/engine_v2.py

Change target: orchestration and entity linking.

- Refactor _link_entities.
- Refactor _format_entities.
- Ensure the pipeline passes ontology metadata through all stages.
- Add compatibility wrappers if needed.

### 7.9 app/pipeline/advanced_pipeline.py

Change target: semantic indexing and advanced reasoning inputs.

- Unify field names.
- Use ontology_class and canonical_id in vectors and prompts.
- Ensure staged outputs are consistent with upstream models.

### 7.10 app/pipeline/copilot_agent.py

Change target: reasoning quality.

- Use ontology-aware evidence.
- Add typed maintenance and risk logic.
- Stop relying only on coarse keywords.

### 7.11 app/pipeline/graphrag_summarizer.py

Change target: evidence-grounded summarization.

- Require typed entities and relations for claims.
- Validate that evidence references ontology-backed facts.
- Improve explanation quality.

### 7.12 app/main.py

Change target: API surface.

- Update the response model for processing endpoints.
- Add ontology endpoints.
- Make docs and swagger reflect the richer payloads.

---

## 8. Detailed function-level change map

### 8.1 app/schemas.py

- EntityRecord: expand to include ontology_class, ontology_path, canonical_id, provenance, schema_version
- RelationRecord: expand to include source_id, target_id, relation_class, relation_category, provenance, schema_version
- JobResult: add ontology_metadata, entity_counts_by_type, relation_counts_by_class

### 8.2 app/pipeline/models.py

- canonicalize_entity_name: evolve into canonicalize_entity_identity with ontology awareness
- normalize_entity_payload: convert old payloads into new entity contract
- normalize_relation_payload: convert old payloads into new relation contract

### 8.3 app/pipeline/entity_extractor.py

- GlinerEntityExtractor.extract: ensure each result is normalized and validated
- GlinerEntityExtractor._extract_with_model: emit ontology_class and provenance
- GlinerEntityExtractor._post_process_entities: add alias and canonical ID handling if present

### 8.4 app/pipeline/relation_extractor.py

- GLiRELRelationExtractor._model_extract: emit typed relations
- GLiRELRelationExtractor._heuristic_extract: produce relation_class rather than generic strings
- GLiRELRelationExtractor._normalize_entities: resolve entity IDs before creating relations

### 8.5 app/pipeline/component_detector.py

- ComponentDetector.detect_from_text: emit typed component entities
- ComponentDetector._build_keyword_index: use taxonomy metadata instead of flat words only
- ComponentDetector.detect_pid_components: preserve compatibility while using ontology-aware records

### 8.6 app/pipeline/neo4j_store.py

- Neo4jGraphStore.persist_entities: write typed node labels and use canonical IDs
- Neo4jGraphStore.persist_relations: write typed edges and store provenance
- Neo4jGraphStore.persist_to_neo4j: delegate to the typed persistence flow

### 8.7 app/pipeline/engine_v2.py

- _link_entities: replace with ontology-aware entity linking
- _format_entities: convert to new schema contract
- _graphrag_analyze: use ontology-aware facts
- _copilot_analyze: pass typed entities and relations downstream

### 8.8 app/pipeline/advanced_pipeline.py

- stage_semantic_indexing: build metadata from ontology_class and canonical_id
- stage_graph_reasoning: use typed graph facts and ontology family queries
- stage_llm_analysis: include ontology_class and relation_class in prompts

### 8.9 app/pipeline/copilot_agent.py

- IndustrialCopilotAgent.reason: route through ontology-aware reasoning branches
- root_cause_analysis: use typed failures and equipment relations
- get_maintenance_plan: use entity and relation types, not generic keywords

### 8.10 app/pipeline/graphrag_summarizer.py

- GraphRAGSummarizer.generate_summary: require typed evidence
- GraphRAGSummarizer._check_evidence_sufficiency: enforce typed entities/relations
- GraphRAGSummarizer._validate_claims: validate relationship type and provenance

### 8.11 app/main.py

- process_pdf: return typed entities and relations in payloads
- process_document: return ontology metadata counts
- get_job: return richer job results and counts
- list_job_summaries: include ontology summary if available

---

## 9. Suggested migration approach

### 9.1 Step 1 — Compatibility first

Do not break the current API immediately. Instead:

- keep old fields for existing clients
- add new ontology fields in parallel
- add a compatibility normalization layer

### 9.2 Step 2 — Introduce a shared contract

All downstream consumers should use the same contract.

This means:

- extractor returns typed objects
- graph store accepts typed objects
- API returns typed objects
- reasoning uses typed objects

### 9.3 Step 3 — Refactor incrementally

Refactor per module rather than trying to replace everything at once.

Recommended order:

1. schemas and models
2. entity extraction
3. relation extraction
4. component taxonomy
5. engine linking
6. Neo4j persistence
7. advanced pipeline
8. copilot and GraphRAG
9. API docs
10. evaluation

### 9.4 Step 4 — Add validation and tests

Each refactor step should be accompanied by validation checks such as:

- payload normalization tests
- entity typing tests
- relation typing tests
- canonical ID consistency tests
- graph persistence tests
- API response contract tests

---

## 10. Example before/after payloads

### 10.1 Before

{
  "name": "pump-01",
  "entity_type": "equipment",
  "confidence": 0.91,
  "canonical_name": "pump"
}

### 10.2 After

{
  "id": "entity:doc-001:pump-01",
  "canonical_id": "asset:pump:001",
  "name": "pump-01",
  "entity_type": "equipment",
  "ontology_class": "equipment.pump",
  "ontology_path": ["equipment", "equipment.pump"],
  "confidence": 0.91,
  "provenance": {
    "source": "gliner",
    "document_id": "doc-001",
    "page": 3,
    "span": [12, 20]
  },
  "schema_version": "v1.1"
}

### 10.3 Before relation

{
  "source": "pump-01",
  "target": "valve-02",
  "relation_type": "related_to",
  "confidence": 0.73
}

### 10.4 After relation

{
  "id": "relation:doc-001:001",
  "source_id": "entity:doc-001:pump-01",
  "target_id": "entity:doc-001:valve-02",
  "relation_class": "controls",
  "relation_category": "functional",
  "confidence": 0.73,
  "provenance": {
    "source": "glirel",
    "document_id": "doc-001",
    "page": 3
  },
  "schema_version": "v1.1"
}

---

## 11. Risks and mitigation notes

### 11.1 Risk: Breaking existing clients

Mitigation:

- keep legacy fields
- add new fields gradually
- expose versioned endpoints if needed

### 11.2 Risk: Overengineering the ontology too early

Mitigation:

- start with a modest but structured taxonomy
- only add more hierarchy once the first phase is validated
- keep a clear separation between ontology classes and extraction heuristics

### 11.3 Risk: Inconsistent field names across pipeline stages

Mitigation:

- create one normalization layer and one contract module
- use shared helper functions for all pipeline stages

### 11.4 Risk: Graph persistence becoming a second source of truth

Mitigation:

- make the graph store a projection of typed output objects
- store schema version metadata so the graph can be reconstructed

---

## 12. Acceptance checklist for the final system

- [ ] All extracted entities include ontology_class and canonical_id.
- [ ] All extracted relations include relation_class and source/target IDs.
- [ ] Every persisted graph node and edge carries provenance and schema version.
- [ ] The API returns ontology-aware payloads by default.
- [ ] The reasoning layers use typed facts rather than raw strings.
- [ ] The system can answer ontology-aware queries such as “show all pumps connected to a valve.”
- [ ] The system can report ontology quality metrics per class.
- [ ] Swagger/OpenAPI documentation exposes the new response contract clearly.

---

## 13. Recommended implementation order

1. app/schemas.py
2. app/pipeline/models.py
3. app/pipeline/ontology.py
4. app/pipeline/entity_extractor.py
5. app/pipeline/relation_extractor.py
6. app/pipeline/component_detector.py
7. app/pipeline/engine_v2.py
8. app/pipeline/neo4j_store.py
9. app/pipeline/advanced_pipeline.py
10. app/pipeline/copilot_agent.py
11. app/pipeline/graphrag_summarizer.py
12. app/main.py
13. README and setup docs
14. ontology_metrics.py

---

## 14. Additional implementation notes

This repository already has the foundation needed for a better ontology layer. The key gap is not “more data”; the key gap is enforcing a shared semantic contract that spans extraction, normalization, storage, reasoning, and API output.

The work should be treated as a cross-cutting architecture improvement rather than isolated feature work. Each module should preserve existing behavior while gradually adopting the ontology contract.

If implemented incrementally, the upgrade path is clear: contract first, extraction second, linking third, storage fourth, reasoning fifth, and API exposure last.

---

## 15. Appendix A — minimal change checklist

- [ ] app/schemas.py :: EntityRecord
- [ ] app/schemas.py :: RelationRecord
- [ ] app/schemas.py :: JobResult
- [ ] app/pipeline/models.py :: canonicalize_entity_name
- [ ] app/pipeline/models.py :: normalize_entity_payload
- [ ] app/pipeline/models.py :: normalize_relation_payload
- [ ] app/pipeline/entity_extractor.py :: GlinerEntityExtractor
- [ ] app/pipeline/entity_extractor.py :: extract
- [ ] app/pipeline/entity_extractor.py :: _extract_with_model
- [ ] app/pipeline/relation_extractor.py :: GLiRELRelationExtractor
- [ ] app/pipeline/relation_extractor.py :: _model_extract
- [ ] app/pipeline/relation_extractor.py :: _heuristic_extract
- [ ] app/pipeline/component_detector.py :: ComponentDetector
- [ ] app/pipeline/component_detector.py :: detect_from_text
- [ ] app/pipeline/neo4j_store.py :: persist_entities
- [ ] app/pipeline/neo4j_store.py :: persist_relations
- [ ] app/pipeline/engine_v2.py :: _link_entities
- [ ] app/pipeline/engine_v2.py :: _format_entities
- [ ] app/pipeline/advanced_pipeline.py :: stage_semantic_indexing
- [ ] app/pipeline/copilot_agent.py :: IndustrialCopilotAgent
- [ ] app/pipeline/graphrag_summarizer.py :: GraphRAGSummarizer
- [ ] app/main.py :: process_pdf
- [ ] app/main.py :: process_document
- [ ] app/main.py :: get_job

---

## 16. Appendix B — implementation notes for maintainers

When editing these files, follow this rule:

- If a function currently returns strings, make it return a structured object.
- If a class currently stores only a name, add an ID and ontology class.
- If a persistence layer currently stores only generic nodes, add typed labels and provenance.
- If an API currently returns a flat record, add ontology metadata while preserving backward-compatible fields.

This rule should apply consistently across the codebase.

---

## 17. Appendix C — review questions for each PR

Every PR that touches this system should answer the following questions:

1. Does the change introduce a typed ontology field where previously only a string existed?
2. Does the change preserve backward compatibility?
3. Does the change carry provenance and version metadata?
4. Does the change make the graph or reasoning layer more ontology-aware?
5. Does the change improve the API contract in a way clients can leverage?

If the answer is “no” to several of these, the change likely needs more work.

---

## 18. Final note

This repository has enough structure to support a high-quality ontology layer. The main task now is not to invent a new system from scratch. The task is to make the existing extraction, graph, reasoning, and API layers converge on one shared semantic contract.

The implementation plan above gives that path in concrete, file- and function-level terms.
