from pathlib import Path

from app.pipeline.engine_v2 import IndustrialGraphPipeline
from app.pipeline.models import normalize_entity_payload, normalize_relation_payload
from app.pipeline.ontology import build_domain_pack_from_markdown, load_ontology_registry


def test_normalize_entity_payload_adds_ontology_metadata():
    payload = normalize_entity_payload({
        "name": "Centrifugal Pump",
        "entity_type": "equipment",
        "confidence": 0.82,
    })

    assert payload["stable_id"] == "centrifugal_pump"
    assert payload["ontology_status"] in {"active", "proposed"}
    assert payload["ontology_type_id"]
    assert payload["ontology"]["type_id"]


def test_normalize_relation_payload_adds_stable_ids_and_status():
    payload = normalize_relation_payload({
        "source": "Pump",
        "target": "Motor",
        "relation_type": "related_to",
        "confidence": 0.63,
    })

    assert payload["source_stable_id"] == "pump"
    assert payload["target_stable_id"] == "motor"
    assert payload["stable_id"]
    assert payload["ontology_status"] in {"active", "proposed"}


def test_markdown_domain_pack_loader_builds_registry_entries(tmp_path: Path):
    sample_md = tmp_path / "sample_pack.md"
    sample_md.write_text(
        "# Sample Pack\n"
        "## Types\n"
        "- Pump\n"
        "- Valve\n"
        "## Relations\n"
        "- controls\n",
        encoding="utf-8",
    )

    pack = build_domain_pack_from_markdown(sample_md)

    assert pack["name"] == "sample_pack"
    assert "pump" in pack["types"]
    assert "valve" in pack["types"]
    assert "controls" in pack["relations"]


def test_registry_loads_markdown_backed_domain_packs():
    registry = load_ontology_registry()

    assert registry.get_type("asset.component.pump") is not None or registry.get_type("asset") is not None
    assert registry.get_relation("controls") is not None


def test_schema_evolution_stage_emits_proposals_for_unknowns():
    pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
    entities = [{
        "name": "Mystery Unit",
        "entity_type": "unknown",
        "confidence": 0.72,
        "unknown_candidate": {
            "candidate_label": "Mystery Unit",
            "candidate_type": "unknown",
            "parent_type_id": "asset",
            "reason": "uncertain mapping",
            "aliases": ["mystery"],
        },
    }]
    relations = [{
        "source": "Pump",
        "target": "Mystery Unit",
        "relation_type": "related_to",
        "confidence": 0.6,
    }]

    result = pipeline._evolve_schema(entities, relations, text="Pump and Mystery Unit")

    assert result["schema_proposals"]
    assert any(proposal["kind"] == "entity" for proposal in result["schema_proposals"])
    assert result["entities"][0]["status"] == "proposed"
    assert result["relations"][0]["status"] == "proposed"


def test_schema_governance_promotes_repeated_proposals(tmp_path: Path):
    pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)
    pipeline.ontology_enricher = None
    entities = [{
        "name": "XQ-47",
        "entity_type": "Widget",
        "confidence": 0.9,
        "status": "proposed",
        "ontology_status": "proposed",
        "ontology_type_id": "entity.xq_47",
        "ontology_label": "XQ-47",
        "ontology_parent_type_id": "asset",
        "ontology_confidence": 0.9,
    }]

    result = pipeline._govern_proposed_schema(
        entities=entities,
        relations=[],
        schema_proposals=[{
            "proposal_id": "entity-xq_47",
            "kind": "entity",
            "candidate_id": "entity.xq_47",
            "label": "XQ-47",
            "parent_type_id": "asset",
            "status": "proposed",
            "confidence": 0.9,
            "source": "zero_shot",
            "evidence": "XQ-47",
            "aliases": ["xq-47"],
            "examples": [],
            "source_docs": ["sample.md"],
        }],
        state_path=tmp_path / "ontology_state.json",
        source_document="sample.md",
    )

    assert result["governance_report"]["promoted_entities"] == 1
    assert result["entities"][0]["status"] == "active"
    assert result["entities"][0]["ontology_status"] == "active"
    assert result["entities"][0]["ontology_type_id"] == "entity.xq_47"
