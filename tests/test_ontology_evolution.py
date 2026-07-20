import json
import tempfile
import unittest
from pathlib import Path

from app.pipeline.engine_v2 import IndustrialGraphPipeline
from app.pipeline.models import MODEL_STAGE_SEQUENCE, get_model_stage_manifest
from app.pipeline.neo4j_store import Neo4jGraphStore
from app.pipeline.ontology import OntologyEnricher, OntologyStateStore, load_ontology_registry


class OntologyEvolutionTests(unittest.TestCase):
    def test_stage_manifest_includes_ontology_enrichment(self) -> None:
        manifest_names = [item["name"] for item in get_model_stage_manifest()]
        self.assertIn("ontology_enrichment", MODEL_STAGE_SEQUENCE)
        self.assertIn("ontology_enrichment", manifest_names)

    def test_pipeline_formatters_preserve_ontology_metadata(self) -> None:
        pipeline = IndustrialGraphPipeline.__new__(IndustrialGraphPipeline)

        entities = [
            {
                "name": "Pump-7",
                "entity_type": "Pump",
                "confidence": 0.94,
                "canonical_name": "pump_7",
                "stable_id": "pump_7",
                "ontology_type_id": "asset.pump",
                "ontology_label": "Pump",
                "ontology_parent_type_id": "asset",
                "ontology_status": "active",
                "ontology_confidence": 0.98,
                "ontology": {"type_id": "asset.pump", "label": "Pump"},
            }
        ]
        relations = [
            {
                "source": "Pump-7",
                "target": "Motor B",
                "relation_type": "controls",
                "confidence": 0.87,
                "stable_id": "pump_7__controls__motor_b",
                "source_stable_id": "pump_7",
                "target_stable_id": "motor_b",
                "ontology_relation_id": "controls",
                "ontology_label": "controls",
                "ontology_status": "active",
                "ontology_confidence": 0.91,
                "ontology": {"relation_id": "controls", "label": "controls"},
            }
        ]

        formatted_entities = pipeline._format_entities(entities)
        formatted_relations = pipeline._format_relations(relations)

        self.assertEqual(formatted_entities[0]["stable_id"], "pump_7")
        self.assertEqual(formatted_entities[0]["ontology_type_id"], "asset.pump")
        self.assertEqual(formatted_relations[0]["stable_id"], "pump_7__controls__motor_b")
        self.assertEqual(formatted_relations[0]["source_stable_id"], "pump_7")
        self.assertEqual(formatted_relations[0]["ontology_relation_id"], "controls")

    def test_enricher_promotes_repeated_unknown_entity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "ontology_state.json"
            state_store = OntologyStateStore(state_path=state_path)
            enricher = OntologyEnricher(state_store=state_store, auto_promote=True)
            candidate_id = "entity.xq_47"

            result = {}
            for _ in range(3):
                result = enricher.enrich(
                    entities=[
                        {
                            "name": "XQ-47",
                            "entity_type": "Widget",
                            "confidence": 0.92,
                        }
                    ],
                    relations=[],
                    text="XQ-47",
                    source_document="sample.md",
                )

            report = result["ontology_report"]
            self.assertEqual(report["status"], "completed")
            self.assertIn(candidate_id, report["promotions"]["types"])
            self.assertIn(candidate_id, report["active_extensions"]["types"])
            self.assertIsNotNone(enricher.registry.get_type(candidate_id))
            self.assertTrue(state_path.exists())

            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["type_observations"][candidate_id]["count"], 3)
            self.assertGreaterEqual(state["recent_observations"][-1]["entity_count"], 1)

    def test_registry_resolves_core_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "ontology_state.json"
            registry = load_ontology_registry(state_path=state_path)

            entity_match = registry.resolve_entity("Pump")
            relation_match = registry.resolve_relation("controls", context="PLC controls Valve")

            self.assertIsNotNone(entity_match)
            self.assertEqual(entity_match.type_id, "asset.pump")
            self.assertIsNotNone(relation_match)
            self.assertEqual(relation_match.type_id, "controls")

    def test_registry_exposes_domain_packs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "ontology_state.json"
            registry = load_ontology_registry(state_path=state_path)
            packs = registry.list_packs()
            pack_names = {pack["pack"] for pack in packs}

            self.assertIn("core", pack_names)
            self.assertTrue(any(name.endswith("_pack") for name in pack_names if name != "core"))

    def test_backfill_metadata_resolver_builds_stable_identity(self) -> None:
        registry = load_ontology_registry()
        entity = {
            "name": "Pump-7",
            "entity_type": "Pump",
            "confidence": 0.9,
        }
        metadata = Neo4jGraphStore._resolve_entity_ontology_metadata(entity, registry)

        self.assertEqual(metadata["stable_id"], "pump_7")
        self.assertEqual(metadata["canonical_name"], "pump_7")
        self.assertIn("ontology_type_id", metadata)
        self.assertIn("ontology", metadata)


if __name__ == "__main__":
    unittest.main()
