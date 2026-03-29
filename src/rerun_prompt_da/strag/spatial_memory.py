"""
Spatial Memory — CLIP-embeds video frames at robot poses and stores them in
Qdrant on kaweees-dgx-spark for semantic/location-based retrieval.

HOW DIMOS DOES SPATIAL REASONING:
  1. Every time the robot moves (min_distance_m) or enough time passes (min_time_s),
     the current camera frame is embedded with CLIP (512-dim ViT-B/32).
  2. The embedding + 6DOF pose (pos_x, pos_y, pos_z, rot_x/y/z) are stored as a
     Qdrant point with a unique frame_id.
  3. At query time:
     - query_by_location(): uses Qdrant payload range filters on pos_x/pos_y
       to find frames within a bounding box radius — no semantic search needed.
     - query_by_text(): CLIP text encoder converts query text to a 512-dim vector,
       then cosine similarity search finds visually matching frames.
     - query_by_image(): same as text but with an image query vector.
  4. Named locations (e.g. "kitchen") are tagged and stored as a separate
     text-searchable collection. query_tagged_location() does semantic lookup.

This lets the agent answer "where is the kitchen?" by finding the stored frame
closest to that description, then returning its pose as a navigation goal.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION = "spatial_memory"
LOCATIONS_COLLECTION = "spatial_locations"
VECTOR_DIM = 512


@dataclass
class SpatialFrame:
    frame_id: str
    pos_x: float
    pos_y: float
    pos_z: float
    rot_x: float
    rot_y: float
    rot_z: float
    timestamp: float
    score: float = 0.0


@dataclass
class NamedLocation:
    name: str
    description: str
    pos_x: float
    pos_y: float
    pos_z: float
    rot_z: float  # yaw


class SpatialMemory:
    """
    Stores and retrieves camera frames by semantic content or world position.

    Args:
        qdrant_url: URL to Qdrant server (e.g. http://kaweees-dgx-spark.local:6333)
        embedder: ClipEmbedder instance (from clip_embedder.py)
        min_distance_m: Don't store a frame unless robot moved this far
        min_time_s: Don't store a frame more often than this interval
    """

    def __init__(
        self,
        qdrant_url: str,
        embedder,
        min_distance_m: float = 0.5,
        min_time_s: float = 1.0,
        collection: str = COLLECTION,
        new_memory: bool = False,
    ):
        from qdrant_client import QdrantClient

        self._q = QdrantClient(url=qdrant_url)
        self._embedder = embedder
        self._min_dist = min_distance_m
        self._min_time = min_time_s
        self._collection = collection

        self._last_pos: Optional[tuple[float, float, float]] = None
        self._last_store_t: float = 0.0
        self._frame_count = 0
        self._stored_count = 0

        if new_memory:
            self._drop_collections()
        self._ensure_collections()

    def _drop_collections(self) -> None:
        for col in [self._collection, LOCATIONS_COLLECTION]:
            try:
                self._q.delete_collection(col)
                logger.info(f"Dropped collection: {col}")
            except Exception:
                pass

    def _ensure_collections(self) -> None:
        from qdrant_client.models import VectorParams, Distance

        existing = {c.name for c in self._q.get_collections().collections}

        if self._collection not in existing:
            self._q.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            # Payload indices for efficient range queries
            from qdrant_client.models import PayloadSchemaType
            for field in ["pos_x", "pos_y", "pos_z"]:
                self._q.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.FLOAT,
                )
            logger.info(f"Created Qdrant collection: {self._collection}")

        if LOCATIONS_COLLECTION not in existing:
            self._q.create_collection(
                collection_name=LOCATIONS_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {LOCATIONS_COLLECTION}")

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def store_frame(
        self,
        frame: np.ndarray,
        pos_x: float,
        pos_y: float,
        pos_z: float = 0.0,
        rot_x: float = 0.0,
        rot_y: float = 0.0,
        rot_z: float = 0.0,
    ) -> Optional[str]:
        """
        Embed and store a frame if the robot has moved enough / enough time passed.
        Returns the frame_id if stored, None if skipped.
        """
        self._frame_count += 1
        now = time.time()

        # Distance gate
        if self._last_pos is not None:
            dist = np.linalg.norm([
                pos_x - self._last_pos[0],
                pos_y - self._last_pos[1],
                pos_z - self._last_pos[2],
            ])
            if dist < self._min_dist:
                return None

        # Time gate
        if now - self._last_store_t < self._min_time:
            return None

        try:
            embedding = self._embedder.embed_image(frame)
        except Exception as e:
            logger.warning(f"CLIP embed failed: {e}")
            return None

        frame_id = str(uuid.uuid4())

        from qdrant_client.models import PointStruct
        self._q.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=frame_id,
                vector=embedding,
                payload={
                    "frame_id": frame_id,
                    "pos_x": pos_x,
                    "pos_y": pos_y,
                    "pos_z": pos_z,
                    "rot_x": rot_x,
                    "rot_y": rot_y,
                    "rot_z": rot_z,
                    "timestamp": now,
                },
            )],
        )

        self._last_pos = (pos_x, pos_y, pos_z)
        self._last_store_t = now
        self._stored_count += 1
        logger.debug(
            f"Stored frame {frame_id} at ({pos_x:.2f},{pos_y:.2f},{pos_z:.2f}) "
            f"[{self._stored_count}/{self._frame_count}]"
        )
        return frame_id

    def tag_location(
        self,
        name: str,
        description: str,
        pos_x: float,
        pos_y: float,
        pos_z: float = 0.0,
        rot_z: float = 0.0,
    ) -> None:
        """Store a named location (e.g. 'kitchen') with a text description embedding."""
        embedding = self._embedder.embed_text(f"{name}: {description}")
        loc_id = str(uuid.uuid4())
        from qdrant_client.models import PointStruct
        self._q.upsert(
            collection_name=LOCATIONS_COLLECTION,
            points=[PointStruct(
                id=loc_id,
                vector=embedding,
                payload={
                    "name": name,
                    "description": description,
                    "pos_x": pos_x,
                    "pos_y": pos_y,
                    "pos_z": pos_z,
                    "rot_z": rot_z,
                },
            )],
        )
        logger.info(f"Tagged location '{name}' at ({pos_x:.2f},{pos_y:.2f})")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query_by_location(
        self,
        x: float,
        y: float,
        radius_m: float = 3.0,
        limit: int = 10,
    ) -> list[SpatialFrame]:
        """Find stored frames within a bounding box around (x, y)."""
        from qdrant_client.models import Filter, FieldCondition, Range

        filt = Filter(must=[
            FieldCondition(key="pos_x", range=Range(gte=x - radius_m, lte=x + radius_m)),
            FieldCondition(key="pos_y", range=Range(gte=y - radius_m, lte=y + radius_m)),
        ])
        results, _ = self._q.scroll(
            collection_name=self._collection,
            scroll_filter=filt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [_point_to_frame(p) for p in results]

    def query_by_text(
        self,
        text: str,
        near_xy: Optional[tuple[float, float]] = None,
        radius_m: float = 10.0,
        limit: int = 8,
    ) -> list[SpatialFrame]:
        """
        Semantic search — CLIP text embedding → cosine similarity in Qdrant.
        Optionally restrict to frames within radius_m of near_xy.
        """
        try:
            text_emb = self._embedder.embed_text(text)
        except Exception as e:
            logger.warning(f"CLIP text embed failed: {e}")
            return []

        filt = None
        if near_xy is not None:
            from qdrant_client.models import Filter, FieldCondition, Range
            x, y = near_xy
            filt = Filter(must=[
                FieldCondition(key="pos_x", range=Range(gte=x - radius_m, lte=x + radius_m)),
                FieldCondition(key="pos_y", range=Range(gte=y - radius_m, lte=y + radius_m)),
            ])

        results = self._q.search(
            collection_name=self._collection,
            query_vector=text_emb,
            query_filter=filt,
            limit=limit,
            with_payload=True,
        )
        return [_point_to_frame(p) for p in results]

    def query_by_image(
        self,
        frame: np.ndarray,
        near_xy: Optional[tuple[float, float]] = None,
        radius_m: float = 10.0,
        limit: int = 8,
    ) -> list[SpatialFrame]:
        """Find visually similar frames."""
        embedding = self._embedder.embed_image(frame)

        filt = None
        if near_xy is not None:
            from qdrant_client.models import Filter, FieldCondition, Range
            x, y = near_xy
            filt = Filter(must=[
                FieldCondition(key="pos_x", range=Range(gte=x - radius_m, lte=x + radius_m)),
                FieldCondition(key="pos_y", range=Range(gte=y - radius_m, lte=y + radius_m)),
            ])

        results = self._q.search(
            collection_name=self._collection,
            query_vector=embedding,
            query_filter=filt,
            limit=limit,
            with_payload=True,
        )
        return [_point_to_frame(p) for p in results]

    def query_tagged_location(self, query: str, threshold: float = 0.3) -> Optional[NamedLocation]:
        """Find a named location by semantic similarity to the query text."""
        try:
            query_emb = self._embedder.embed_text(query)
        except Exception as e:
            logger.warning(f"CLIP text embed failed: {e}")
            return None

        results = self._q.search(
            collection_name=LOCATIONS_COLLECTION,
            query_vector=query_emb,
            limit=1,
            with_payload=True,
        )
        if not results:
            return None

        best = results[0]
        # Cosine similarity: 1 - distance. For cosine distance, score is similarity.
        if best.score < threshold:
            return None

        p = best.payload or {}
        return NamedLocation(
            name=p.get("name", ""),
            description=p.get("description", ""),
            pos_x=p.get("pos_x", 0.0),
            pos_y=p.get("pos_y", 0.0),
            pos_z=p.get("pos_z", 0.0),
            rot_z=p.get("rot_z", 0.0),
        )

    def stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "frames_stored": self._stored_count,
            "collection": self._collection,
        }


def _point_to_frame(point) -> SpatialFrame:
    p = point.payload or {}
    return SpatialFrame(
        frame_id=p.get("frame_id", str(getattr(point, "id", ""))),
        pos_x=p.get("pos_x", 0.0),
        pos_y=p.get("pos_y", 0.0),
        pos_z=p.get("pos_z", 0.0),
        rot_x=p.get("rot_x", 0.0),
        rot_y=p.get("rot_y", 0.0),
        rot_z=p.get("rot_z", 0.0),
        timestamp=p.get("timestamp", 0.0),
        score=float(getattr(point, "score", 0.0)),
    )
