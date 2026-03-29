"""
Entity Graph Database — SQLite-backed store for entities, relations, and spatial distances.

Extracted from dimos/perception/experimental/temporal_memory/entity_graph_db.py.
Thread-safe via connection-per-thread pattern.

Schema:
  entities(entity_id, entity_type, descriptor, first_seen_ts, last_seen_ts, metadata JSON)
  relations(id, relation_type, subject_id, object_id, confidence, timestamp_s, evidence, notes)
  distances(id, entity_a_id, entity_b_id, distance_meters, distance_category, confidence, ts, method)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

import logging
logger = logging.getLogger(__name__)


class EntityGraphDB:
    """SQLite entity/relation graph. Thread-safe (connection-per-thread)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"EntityGraphDB at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        c = self._conn()
        cur = c.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                descriptor TEXT,
                first_seen_ts REAL NOT NULL,
                last_seen_ts REAL NOT NULL,
                metadata TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_e_first ON entities(first_seen_ts);
            CREATE INDEX IF NOT EXISTS idx_e_last  ON entities(last_seen_ts);

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                timestamp_s REAL NOT NULL,
                evidence TEXT,
                notes TEXT,
                FOREIGN KEY (subject_id) REFERENCES entities(entity_id),
                FOREIGN KEY (object_id)  REFERENCES entities(entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_r_subject ON relations(subject_id);
            CREATE INDEX IF NOT EXISTS idx_r_object  ON relations(object_id);
            CREATE INDEX IF NOT EXISTS idx_r_type    ON relations(relation_type);
            CREATE INDEX IF NOT EXISTS idx_r_ts      ON relations(timestamp_s);

            CREATE TABLE IF NOT EXISTS distances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a_id TEXT NOT NULL,
                entity_b_id TEXT NOT NULL,
                distance_meters REAL,
                distance_category TEXT,
                confidence REAL DEFAULT 1.0,
                timestamp_s REAL NOT NULL,
                method TEXT,
                FOREIGN KEY (entity_a_id) REFERENCES entities(entity_id),
                FOREIGN KEY (entity_b_id) REFERENCES entities(entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_d_pair ON distances(entity_a_id, entity_b_id);
            CREATE INDEX IF NOT EXISTS idx_d_ts   ON distances(timestamp_s);
        """)
        c.commit()

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        descriptor: str,
        timestamp_s: float,
        metadata: Optional[dict] = None,
    ) -> None:
        meta_json = json.dumps(metadata) if metadata else None
        self._conn().execute(
            """
            INSERT INTO entities(entity_id, entity_type, descriptor,
                                 first_seen_ts, last_seen_ts, metadata)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(entity_id) DO UPDATE SET
                last_seen_ts = excluded.last_seen_ts,
                descriptor   = COALESCE(excluded.descriptor, descriptor),
                metadata     = COALESCE(metadata, excluded.metadata)
            """,
            (entity_id, entity_type, descriptor, timestamp_s, timestamp_s, meta_json),
        )
        self._conn().commit()

    def get_entity(self, entity_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM entities WHERE entity_id=?", (entity_id,)
        ).fetchone()
        return _row_to_entity(row) if row else None

    def get_all_entities(self, entity_type: Optional[str] = None) -> list[dict]:
        if entity_type:
            rows = self._conn().execute(
                "SELECT * FROM entities WHERE entity_type=? ORDER BY last_seen_ts DESC",
                (entity_type,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM entities ORDER BY last_seen_ts DESC"
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def query_near(
        self, world_x: float, world_y: float, radius_m: float = 4.0
    ) -> list[dict]:
        """Return entities whose world_x/world_y metadata is within radius_m."""
        rows = self._conn().execute(
            "SELECT * FROM entities ORDER BY last_seen_ts DESC"
        ).fetchall()
        results = []
        for row in rows:
            e = _row_to_entity(row)
            meta = e.get("metadata") or {}
            ex, ey = meta.get("world_x"), meta.get("world_y")
            if ex is not None and ey is not None:
                if ((ex - world_x) ** 2 + (ey - world_y) ** 2) ** 0.5 <= radius_m:
                    results.append(e)
        return results

    def get_entities_by_time(
        self, time_window: tuple[float, float], first_seen: bool = True
    ) -> list[dict]:
        field = "first_seen_ts" if first_seen else "last_seen_ts"
        rows = self._conn().execute(
            f"SELECT * FROM entities WHERE {field} BETWEEN ? AND ? ORDER BY {field} DESC",
            time_window,
        ).fetchall()
        return [_row_to_entity(r) for r in rows]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        relation_type: str,
        subject_id: str,
        object_id: str,
        confidence: float,
        timestamp_s: float,
        evidence: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> None:
        self._conn().execute(
            """
            INSERT INTO relations(relation_type, subject_id, object_id,
                                  confidence, timestamp_s, evidence, notes)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                relation_type, subject_id, object_id, confidence, timestamp_s,
                json.dumps(evidence) if evidence else None, notes,
            ),
        )
        self._conn().commit()

    def get_relations_for_entity(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        time_window: Optional[tuple[float, float]] = None,
    ) -> list[dict]:
        q = "SELECT * FROM relations WHERE (subject_id=? OR object_id=?)"
        params: list = [entity_id, entity_id]
        if relation_type:
            q += " AND relation_type=?"
            params.append(relation_type)
        if time_window:
            q += " AND timestamp_s BETWEEN ? AND ?"
            params.extend(time_window)
        q += " ORDER BY timestamp_s DESC"
        rows = self._conn().execute(q, params).fetchall()
        return [_row_to_relation(r) for r in rows]

    def get_recent_relations(self, limit: int = 50) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM relations ORDER BY timestamp_s DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_relation(r) for r in rows]

    # ------------------------------------------------------------------
    # Distances
    # ------------------------------------------------------------------

    def add_distance(
        self,
        entity_a_id: str,
        entity_b_id: str,
        distance_meters: Optional[float],
        distance_category: Optional[str],
        confidence: float,
        timestamp_s: float,
        method: str = "vlm",
    ) -> None:
        a, b = sorted([entity_a_id, entity_b_id])
        self._conn().execute(
            """
            INSERT INTO distances(entity_a_id, entity_b_id, distance_meters,
                                  distance_category, confidence, timestamp_s, method)
            VALUES (?,?,?,?,?,?,?)
            """,
            (a, b, distance_meters, distance_category, confidence, timestamp_s, method),
        )
        self._conn().commit()

    def get_distance(self, entity_a_id: str, entity_b_id: str) -> Optional[dict]:
        a, b = sorted([entity_a_id, entity_b_id])
        row = self._conn().execute(
            "SELECT * FROM distances WHERE entity_a_id=? AND entity_b_id=? "
            "ORDER BY timestamp_s DESC LIMIT 1",
            (a, b),
        ).fetchone()
        return _row_to_distance(row) if row else None

    def get_nearby_entities(
        self, entity_id: str, max_distance: float = 5.0
    ) -> list[dict]:
        rows = self._conn().execute(
            """
            SELECT d.*, e.entity_type, e.descriptor
            FROM distances d
            JOIN entities e ON (
                CASE WHEN d.entity_a_id=? THEN e.entity_id=d.entity_b_id
                     WHEN d.entity_b_id=? THEN e.entity_id=d.entity_a_id END
            )
            WHERE (d.entity_a_id=? OR d.entity_b_id=?)
              AND d.distance_meters IS NOT NULL
              AND d.distance_meters <= ?
              AND d.id IN (
                  SELECT MAX(id) FROM distances
                  WHERE (entity_a_id=d.entity_a_id AND entity_b_id=d.entity_b_id)
                  GROUP BY entity_a_id, entity_b_id
              )
            ORDER BY d.distance_meters ASC
            """,
            (entity_id, entity_id, entity_id, entity_id, max_distance),
        ).fetchall()
        return [
            {
                "entity_id": r["entity_b_id"] if r["entity_a_id"] == entity_id else r["entity_a_id"],
                "entity_type": r["entity_type"],
                "descriptor": r["descriptor"],
                "distance_meters": r["distance_meters"],
                "distance_category": r["distance_category"],
                "confidence": r["confidence"],
                "timestamp_s": r["timestamp_s"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Bulk save (from VLM window parse)
    # ------------------------------------------------------------------

    def save_window_data(
        self,
        parsed: dict,
        timestamp_s: float,
        metadata: Optional[dict] = None,
    ) -> None:
        """Persist entities + relations from a VLM window parse result."""
        try:
            for entity in parsed.get("new_entities", []):
                self.upsert_entity(
                    entity_id=entity["id"],
                    entity_type=entity.get("type", "object"),
                    descriptor=entity.get("descriptor", "unknown"),
                    timestamp_s=timestamp_s,
                    metadata=metadata,
                )
            for entity in parsed.get("entities_present", []):
                if not isinstance(entity, dict) or "id" not in entity:
                    continue
                if entity.get("descriptor"):
                    self.upsert_entity(
                        entity_id=entity["id"],
                        entity_type=entity.get("type", "unknown"),
                        descriptor=entity["descriptor"],
                        timestamp_s=timestamp_s,
                        metadata=metadata,
                    )
                else:
                    existing = self.get_entity(entity["id"])
                    if existing:
                        self.upsert_entity(
                            entity_id=entity["id"],
                            entity_type=existing["entity_type"],
                            descriptor=existing["descriptor"] or "unknown",
                            timestamp_s=timestamp_s,
                            metadata=metadata,
                        )
            for rel in parsed.get("relations", []):
                subj = rel["subject"].split("|")[0] if "|" in rel.get("subject", "") else rel.get("subject", "")
                obj  = rel["object"].split("|")[0]  if "|" in rel.get("object",  "") else rel.get("object",  "")
                if subj and obj:
                    self.add_relation(
                        relation_type=rel.get("type", "related"),
                        subject_id=subj,
                        object_id=obj,
                        confidence=rel.get("confidence", 1.0),
                        timestamp_s=timestamp_s,
                        evidence=rel.get("evidence"),
                        notes=rel.get("notes"),
                    )
        except Exception as e:
            logger.error(f"save_window_data failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Graph context builder (for RAG injection into LLM)
    # ------------------------------------------------------------------

    def build_graph_context(
        self,
        entity_ids: list[str],
        max_relations: int = 10,
        nearby_distance_m: float = 5.0,
    ) -> str:
        """
        Build a natural-language context string for a set of entity IDs.
        Used as RAG context injected into the agent's prompt.

        For each entity:
          - its descriptor + type
          - relations it participates in
          - nearby entities (by distance graph)
        """
        lines: list[str] = []

        seen_entities: set[str] = set()
        all_relations: list[dict] = []

        for eid in entity_ids:
            entity = self.get_entity(eid)
            if not entity:
                continue
            if eid not in seen_entities:
                seen_entities.add(eid)
                lines.append(
                    f"[{entity['entity_type']}] {eid}: {entity.get('descriptor', '?')}"
                )

            rels = self.get_relations_for_entity(eid)[:max_relations]
            for r in rels:
                all_relations.append(r)
                other = r["object_id"] if r["subject_id"] == eid else r["subject_id"]
                if other not in seen_entities:
                    other_e = self.get_entity(other)
                    if other_e:
                        seen_entities.add(other)
                        lines.append(
                            f"  [related:{other_e['entity_type']}] {other}: {other_e.get('descriptor','?')}"
                        )

            nearby = self.get_nearby_entities(eid, max_distance=nearby_distance_m)
            for n in nearby:
                nid = n["entity_id"]
                if nid not in seen_entities:
                    seen_entities.add(nid)
                    lines.append(
                        f"  [nearby ~{n['distance_meters']:.1f}m] {nid}: {n.get('descriptor','?')}"
                    )

        if all_relations:
            lines.append("\nRelations:")
            for r in all_relations[:max_relations]:
                lines.append(
                    f"  {r['subject_id']} --[{r['relation_type']}]--> {r['object_id']} "
                    f"(conf={r['confidence']:.2f})"
                )

        return "\n".join(lines) if lines else "(no entities)"

    # ------------------------------------------------------------------
    # Stats & lifecycle
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        cur = self._conn().cursor()
        entity_count  = cur.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relation_count = cur.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        distance_count = cur.execute("SELECT COUNT(*) FROM distances").fetchone()[0]
        return {"entities": entity_count, "relations": relation_count, "distances": distance_count}

    def commit(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.commit()
            try:
                self._local.conn.execute("PRAGMA wal_checkpoint(FULL)")
            except Exception:
                pass

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# ------------------------------------------------------------------
# Row helpers
# ------------------------------------------------------------------

def _row_to_entity(row) -> dict:
    return {
        "entity_id":    row["entity_id"],
        "entity_type":  row["entity_type"],
        "descriptor":   row["descriptor"],
        "first_seen_ts": row["first_seen_ts"],
        "last_seen_ts":  row["last_seen_ts"],
        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
    }


def _row_to_relation(row) -> dict:
    return {
        "id":            row["id"],
        "relation_type": row["relation_type"],
        "subject_id":    row["subject_id"],
        "object_id":     row["object_id"],
        "confidence":    row["confidence"],
        "timestamp_s":   row["timestamp_s"],
        "evidence": json.loads(row["evidence"]) if row["evidence"] else None,
        "notes":         row["notes"],
    }


def _row_to_distance(row) -> dict:
    return {
        "entity_a_id":       row["entity_a_id"],
        "entity_b_id":       row["entity_b_id"],
        "distance_meters":   row["distance_meters"],
        "distance_category": row["distance_category"],
        "confidence":        row["confidence"],
        "timestamp_s":       row["timestamp_s"],
        "method":            row["method"],
    }
