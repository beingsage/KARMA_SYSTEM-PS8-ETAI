#!/usr/bin/env python3
"""Backfill stable IDs and ontology metadata into legacy Neo4j graph data."""

from __future__ import annotations

import argparse
import json
from typing import Any

from app.pipeline.neo4j_store import Neo4jGraphStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill ontology metadata in Neo4j")
    parser.add_argument("--dry-run", action="store_true", help="Compute updates without writing to Neo4j")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of entities/relations processed")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    store = Neo4jGraphStore()
    result: dict[str, Any] = store.backfill_ontology_metadata(dry_run=args.dry_run, limit=args.limit)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"status: {result.get('status')}")
        print(f"dry_run: {result.get('dry_run')}")
        print(f"entity_updates: {result.get('entity_updates', 0)}")
        print(f"relation_updates: {result.get('relation_updates', 0)}")
        print(f"proposed_entities: {result.get('proposed_entities', 0)}")
        print(f"proposed_relations: {result.get('proposed_relations', 0)}")
        if result.get("error"):
            print(f"error: {result['error']}")

    return 0 if result.get("status") in {"completed", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
