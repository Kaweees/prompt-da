"""
Qdrant collection setup — run this once to initialize collections on kaweees-dgx-spark.

Based on commahackopenpilot/infrastructure/qdrant/collections_init.py

Usage:
    uv run strag-setup
    uv run strag-setup --url http://kaweees-dgx-spark.local:6333
    uv run strag-setup --reset  # drop and recreate
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


COLLECTIONS = {
    "spatial_memory": {
        "vector_size": 512,          # CLIP ViT-B/32
        "distance": "Cosine",
        "payload_indices": [
            ("pos_x",    "float"),
            ("pos_y",    "float"),
            ("pos_z",    "float"),
            ("timestamp", "float"),
        ],
        "description": "CLIP-embedded camera frames with world pose payload",
    },
    "spatial_locations": {
        "vector_size": 512,
        "distance": "Cosine",
        "payload_indices": [
            ("name", "keyword"),
        ],
        "description": "Named locations tagged by the agent (kitchen, door, etc.)",
    },
}


def init_collections(
    qdrant_url: str = "http://kaweees-dgx-spark.local:6333",
    reset: bool = False,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

    distance_map = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot":    Distance.DOT,
    }
    type_map = {
        "float":   PayloadSchemaType.FLOAT,
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
        "bool":    PayloadSchemaType.BOOL,
        "text":    PayloadSchemaType.TEXT,
    }

    client = QdrantClient(url=qdrant_url, timeout=30)
    logger.info(f"Connected to Qdrant at {qdrant_url}")

    existing = {c.name for c in client.get_collections().collections}
    logger.info(f"Existing collections: {existing}")

    for name, cfg in COLLECTIONS.items():
        if name in existing:
            if reset:
                client.delete_collection(name)
                logger.info(f"Dropped: {name}")
                existing.discard(name)
            else:
                logger.info(f"Already exists (skipping): {name}")
                continue

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=cfg["vector_size"],
                distance=distance_map[cfg["distance"]],
            ),
        )
        logger.info(f"Created: {name} ({cfg['description']})")

        for field_name, field_type in cfg.get("payload_indices", []):
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=type_map[field_type],
            )
            logger.info(f"  Index: {field_name} ({field_type})")

    # Verify
    final = {c.name for c in client.get_collections().collections}
    logger.info(f"Collections after setup: {final}")
    print(f"\nQdrant setup complete at {qdrant_url}")
    print(f"Collections: {', '.join(sorted(final))}")


def status(qdrant_url: str = "http://kaweees-dgx-spark.local:6333") -> None:
    """Print collection stats."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url, timeout=10)
    cols = client.get_collections().collections
    print(f"\nQdrant at {qdrant_url}")
    print(f"{'Collection':<25} {'Vectors':>10} {'Status':<10}")
    print("-" * 50)
    for col in cols:
        info = client.get_collection(col.name)
        count = info.vectors_count or 0
        status = info.status.value if hasattr(info.status, "value") else str(info.status)
        print(f"{col.name:<25} {count:>10} {status:<10}")


def main_cli():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Initialize Qdrant collections on kaweees-dgx-spark")
    parser.add_argument("--url", default="http://kaweees-dgx-spark.local:6333",
                        help="Qdrant URL (default: http://kaweees-dgx-spark.local:6333)")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate existing collections")
    parser.add_argument("--status", action="store_true",
                        help="Print collection stats and exit")
    args = parser.parse_args()

    if args.status:
        status(args.url)
    else:
        init_collections(args.url, reset=args.reset)


if __name__ == "__main__":
    main_cli()
