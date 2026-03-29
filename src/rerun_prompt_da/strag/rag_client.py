"""
Spatio-Temporal RAG Client — the core fusion layer.

HOW DIMENSIONAL DOES SPATIO-TEMPORAL RAG:
============================================

Two memory streams are maintained in parallel:

1. SPATIAL MEMORY (Qdrant)
   - Each camera frame is CLIP-embedded (512-dim) and stored with the robot's
     world pose (x, y, z, yaw).
   - At query time, the agent can search by:
       a) Bounding box location → find frames within radius_m of current pos
       b) Text semantics → "where is the kitchen" → CLIP text→image cosine search
       c) Image similarity → find frames visually similar to current camera view
   - This answers WHERE questions: "What does the area at (5,3) look like?"

2. TEMPORAL MEMORY (SQLite EntityGraphDB)
   - Every `stride_s` seconds, a VLM analyzes a window of frames and extracts:
       - Entities (person_1, chair_2, door_3) with types and descriptors
       - Relations between entities (person holds coffee, person near desk)
       - Spatial distances (person ~2m from door)
   - Entities are stored with their world position (robot pos at time of sighting).
   - This answers WHO/WHAT/HOW questions: "Where is the person?" "What is near the chair?"

3. RAG FUSION at query time:
   - build_context(x, y) → combines both:
       * Nearby Qdrant frames (visual context of current area)
       * Entity graph context (structured knowledge about objects/people)
   - The combined context is injected as a [CONTEXT] block into the agent's prompt
   - The agent then reasons over this context to make navigation decisions

EXPLORE SKILL — HOW IT WORKS:
==============================
"Explore" is NOT a single function. It's an emergent behavior from:

1. OTA Loop (Observe-Think-Act every N seconds):
   - Timer fires every `ota_interval_s`
   - Injects an OTA prompt asking the agent to maximize an exploration score
   - Score: +5 new area, +2 goal reached, +1 field note, -3 obstacle, -2 stagnation

2. The agent's decision process:
   a) Observes: latest camera frame auto-injected into context
   b) Builds spatial context: query_by_location(current_pos) → what's nearby in memory
   c) Builds temporal context: entity graph around current position
   d) Calls think(reasoning) to reason before acting
   e) Calls navigate_with_text("unexplored area") or specific waypoints

3. Frontier Detection (wavefront algorithm):
   - Occupancy grid (2D map from SLAM) is analyzed for frontiers:
     * A frontier cell = a FREE cell adjacent to UNKNOWN space
   - BFS from robot position finds all frontier regions
   - Frontier with best information gain (size, distance) selected as next goal
   - This ensures systematic coverage of unknown space

4. Memory prevents revisiting:
   - SpatialMemory records frames at poses → agent sees what's already mapped
   - NarrativeLedger tracks visited areas → agent penalized (-2) for revisiting
   - Together they guide the agent toward unmapped territories
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .spatial_memory import SpatialMemory, SpatialFrame
    from .entity_graph_db import EntityGraphDB
    from .temporal_memory import TemporalMemory

logger = logging.getLogger(__name__)


@dataclass
class RagContext:
    """Combined spatio-temporal context for agent prompting."""
    # Nearby visual frames from Qdrant
    nearby_frames: list
    # Entities from the graph near current position
    nearby_entities: list
    # Graph context string (entity relations, distances)
    graph_context: str
    # Temporal rolling summary
    temporal_summary: str
    # Currently visible entities (from last VLM window)
    currently_present: list
    # Text-formatted context ready for LLM injection
    formatted: str


class RagClient:
    """
    Unified RAG interface combining SpatialMemory (Qdrant) + EntityGraphDB (SQLite).

    Usage:
        ctx = rag.build_context(x=5.0, y=3.0)
        prompt = f"[CONTEXT]\\n{ctx.formatted}\\n[/CONTEXT]\\n{user_query}"
    """

    def __init__(
        self,
        spatial_memory: "SpatialMemory",
        entity_db: "EntityGraphDB",
        temporal_memory: Optional["TemporalMemory"] = None,
        spatial_radius_m: float = 6.0,
        entity_radius_m: float = 4.0,
    ) -> None:
        self._spatial = spatial_memory
        self._entity_db = entity_db
        self._temporal = temporal_memory
        self._spatial_radius = spatial_radius_m
        self._entity_radius = entity_radius_m

    def build_context(
        self,
        x: float,
        y: float,
        spatial_radius: Optional[float] = None,
        entity_radius: Optional[float] = None,
        k_spatial: int = 8,
    ) -> RagContext:
        """
        Build full spatio-temporal context for position (x, y).

        1. Qdrant location query → nearby visual frames
        2. EntityGraphDB spatial query → nearby entities
        3. EntityGraphDB graph context → entity relations/distances
        4. TemporalMemory state → rolling summary + currently visible
        """
        sr = spatial_radius or self._spatial_radius
        er = entity_radius or self._entity_radius

        # 1. Spatial: nearby frames from Qdrant
        nearby_frames = self._spatial.query_by_location(x, y, sr, limit=k_spatial)

        # 2. Temporal: entities near position
        nearby_entities = self._entity_db.query_near(x, y, er)

        # 3. Graph context string
        entity_ids = [e["entity_id"] for e in nearby_entities]
        # Also include recently-seen entities from temporal memory
        temporal_summary = ""
        currently_present: list = []
        if self._temporal:
            state = self._temporal.get_state()
            temporal_summary = state.get("rolling_summary", "")
            currently_present = state.get("currently_present", [])
            for e in state.get("entities", []):
                eid = e.get("id")
                if eid and eid not in entity_ids:
                    entity_ids.append(eid)

        graph_ctx = self._entity_db.build_graph_context(entity_ids[:30]) if entity_ids else "(none)"

        formatted = _format_context(
            x, y, nearby_frames, nearby_entities, graph_ctx,
            temporal_summary, currently_present
        )

        return RagContext(
            nearby_frames=nearby_frames,
            nearby_entities=nearby_entities,
            graph_context=graph_ctx,
            temporal_summary=temporal_summary,
            currently_present=currently_present,
            formatted=formatted,
        )

    def text_search(
        self,
        query: str,
        near_xy: Optional[tuple[float, float]] = None,
        k: int = 6,
    ) -> list:
        """Semantic text search over stored frames."""
        return self._spatial.query_by_text(query, near_xy=near_xy, limit=k)

    def find_location(self, description: str) -> Optional[dict]:
        """
        Find a named location or semantic location by text description.
        Returns {pos_x, pos_y, pos_z, rot_z, name} or None.
        """
        # Try named locations first (tagged by agent)
        loc = self._spatial.query_tagged_location(description)
        if loc:
            return {
                "pos_x": loc.pos_x, "pos_y": loc.pos_y,
                "pos_z": loc.pos_z, "rot_z": loc.rot_z,
                "name": loc.name, "source": "tagged_location",
            }
        # Fall back to visual semantic search
        results = self._spatial.query_by_text(description, limit=1)
        if results:
            best = results[0]
            return {
                "pos_x": best.pos_x, "pos_y": best.pos_y,
                "pos_z": best.pos_z, "rot_z": best.rot_z,
                "name": description, "source": "visual_search",
                "score": best.score,
            }
        return None

    def answer_question(self, question: str) -> str:
        """Route a question to temporal memory VLM query."""
        if self._temporal:
            return self._temporal.query(question)
        return "Temporal memory not available."


def _format_context(
    x: float,
    y: float,
    nearby_frames: list,
    nearby_entities: list,
    graph_context: str,
    temporal_summary: str,
    currently_present: list,
) -> str:
    lines: list[str] = []
    lines.append(f"## Current position: ({x:.2f}, {y:.2f})")

    if temporal_summary:
        lines.append(f"\n## Scene summary\n{temporal_summary}")

    if currently_present:
        visible = [
            e.get("id", "?") if isinstance(e, dict) else str(e)
            for e in currently_present[:10]
        ]
        lines.append(f"\n## Currently visible: {', '.join(visible)}")

    if nearby_entities:
        lines.append("\n## Entities near current position")
        for e in nearby_entities[:8]:
            meta = e.get("metadata") or {}
            wx, wy = meta.get("world_x", "?"), meta.get("world_y", "?")
            lines.append(
                f"  [{e.get('entity_type','?')}] {e.get('entity_id','?')}: "
                f"{e.get('descriptor','?')} @ ({wx},{wy})"
            )

    if nearby_frames:
        lines.append("\n## Nearby visual observations")
        for f in nearby_frames[:5]:
            lines.append(
                f"  Frame @ ({f.pos_x:.1f},{f.pos_y:.1f}) "
                f"yaw={f.rot_z:.2f}rad ts={f.timestamp:.0f}"
            )

    if graph_context and graph_context != "(none)":
        lines.append(f"\n## Entity graph\n{graph_context}")

    return "\n".join(lines) if lines else "(no context available)"
