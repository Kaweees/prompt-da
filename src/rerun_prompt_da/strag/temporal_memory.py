"""
Temporal Memory — sliding window VLM analysis for entity/relation extraction.

HOW IT WORKS (from dimos):
  1. Frames are buffered at `fps` Hz. Every `stride_s` seconds, a window of
     `window_s` worth of frames is extracted.
  2. Up to `max_frames_per_window` keyframes are selected (diversity via CLIP
     cosine dissimilarity, or fallback: sharpness/Laplacian variance).
  3. A VLM (GPT-4o or Ollama Qwen2-VL) receives the keyframes + current state
     and returns JSON with:
       - caption: scene description
       - new_entities: [{id, type, descriptor}]
       - entities_present: [{id, type}]
       - relations: [{subject, type, object, confidence}]
  4. Parsed data is persisted to EntityGraphDB (SQLite).
  5. A second async VLM call estimates distances between entity pairs.
  6. A rolling summary is maintained (updated every `summary_interval_s`).

USAGE:
    from rerun_prompt_da.strag.temporal_memory import TemporalMemory, TemporalMemoryConfig
    from rerun_prompt_da.strag.entity_graph_db import EntityGraphDB

    db = EntityGraphDB("~/.local/state/prompt_da/entity_graph.db")
    tm = TemporalMemory(TemporalMemoryConfig(), db)
    tm.start()

    # In your camera loop:
    tm.add_frame(frame_bgr, timestamp=time.time())

    # Query:
    answer = tm.query("What objects are near the person?")
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .entity_graph_db import EntityGraphDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TemporalMemoryConfig:
    fps: float = 1.0
    window_s: float = 5.0
    stride_s: float = 5.0
    max_frames_per_window: int = 3
    max_buffer_frames: int = 100
    summary_interval_s: float = 30.0
    enable_distance_estimation: bool = True
    max_distance_pairs: int = 5
    max_tokens: int = 900
    temperature: float = 0.2
    # VLM backend: "openai" | "ollama"
    vlm_backend: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    ollama_url: str = "http://kaweees-dgx-spark.local:11434"
    ollama_model: str = "qwen2-vl:7b"


# ---------------------------------------------------------------------------
# Frame buffer
# ---------------------------------------------------------------------------

@dataclass
class BufferedFrame:
    image: np.ndarray
    timestamp_s: float


class FrameBuffer:
    """Fixed-size circular buffer of timestamped frames."""

    def __init__(self, max_frames: int, fps: float) -> None:
        self._buf: deque[BufferedFrame] = deque(maxlen=max_frames)
        self._fps = fps
        self._last_add_t: float = 0.0
        self._start_t: float = 0.0

    def set_start(self, t: float) -> None:
        self._start_t = t

    def add(self, frame: np.ndarray, ts: float) -> bool:
        """Add frame; returns True if accepted (rate-limited by fps)."""
        interval = 1.0 / max(self._fps, 0.01)
        if ts - self._last_add_t < interval:
            return False
        self._last_add_t = ts
        self._buf.append(BufferedFrame(image=frame.copy(), timestamp_s=ts - self._start_t))
        return True

    def extract_window(self, window_s: float) -> Optional[list[BufferedFrame]]:
        """Return frames spanning the last window_s seconds, or None if not enough."""
        if not self._buf:
            return None
        latest = self._buf[-1].timestamp_s
        frames = [f for f in self._buf if f.timestamp_s >= latest - window_s]
        return frames if len(frames) >= 2 else None

    def latest(self) -> Optional[BufferedFrame]:
        return self._buf[-1] if self._buf else None

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


def select_keyframes(frames: list[BufferedFrame], max_n: int) -> list[BufferedFrame]:
    """Select up to max_n diverse keyframes by sharpness (Laplacian variance)."""
    if len(frames) <= max_n:
        return frames
    # Score by sharpness
    import cv2
    scored = []
    for f in frames:
        gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scored.append((score, f))
    scored.sort(key=lambda x: -x[0])
    selected = [f for _, f in scored[:max_n]]
    selected.sort(key=lambda f: f.timestamp_s)
    return selected


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------

WINDOW_ANALYSIS_PROMPT = """You are analyzing a short video clip from a robot's camera.
The clip spans from {t_start:.1f}s to {t_end:.1f}s.

Current scene context:
{context}

Analyze the provided frame(s) and return ONLY valid JSON:
{{
  "caption": "<one sentence scene description>",
  "new_entities": [
    {{"id": "<short_id>", "type": "<person|object|location|surface|vehicle>", "descriptor": "<brief description>"}}
  ],
  "entities_present": [
    {{"id": "<entity_id>", "type": "<type>"}}
  ],
  "relations": [
    {{"subject": "<id>", "type": "<holds|near|on_top_of|blocks|looks_at|talks_to>",
       "object": "<id>", "confidence": 0.9}}
  ]
}}

Rules:
- Use short snake_case ids (person_1, chair_2, door_3)
- Only output JSON, no other text
- new_entities: only entities NOT in context
- entities_present: ALL entities visible (including existing ones)
"""

DISTANCE_ESTIMATION_PROMPT = """Looking at this image, estimate distances between entity pairs.
Return ONLY valid JSON:
{{"pairs": [{{"a": "<id_a>", "b": "<id_b>",
              "distance_m": <float_or_null>,
              "category": "<near|medium|far>",
              "confidence": <0.0-1.0>}}]}}

Entity pairs to estimate:
{pairs}
"""

QUERY_PROMPT = """You are answering a question about a video stream using memory and graph data.

Context:
{context}

Question: {question}

Answer concisely based on the context. If you don't know, say so."""


def _encode_frames(frames: list[BufferedFrame]) -> list[dict]:
    """Convert frames to base64 image_url content blocks."""
    import base64
    import cv2
    blocks = []
    for f in frames:
        _, buf = cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        b64 = base64.b64encode(buf.tobytes()).decode()
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
        })
    return blocks


def _call_openai_vlm(
    frames: list[BufferedFrame],
    prompt: str,
    api_key: str,
    model: str = "gpt-4o",
    max_tokens: int = 900,
    temperature: float = 0.2,
) -> Optional[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        content = [{"type": "text", "text": prompt}] + _encode_frames(frames)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI VLM error: {e}")
        return None


def _call_ollama_vlm(
    frames: list[BufferedFrame],
    prompt: str,
    url: str,
    model: str = "qwen2-vl:7b",
    max_tokens: int = 900,
) -> Optional[str]:
    try:
        import base64
        import cv2
        import requests

        images_b64 = []
        for f in frames:
            _, buf = cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            images_b64.append(base64.b64encode(buf.tobytes()).decode())

        payload = {
            "model": model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = requests.post(f"{url}/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        logger.error(f"Ollama VLM error: {e}")
        return None


def _parse_vlm_json(raw: str) -> dict:
    if not raw:
        return {"_error": "empty response"}
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"_error": str(e), "_raw": raw[:500]}


# ---------------------------------------------------------------------------
# TemporalMemory
# ---------------------------------------------------------------------------

class TemporalMemory:
    """
    Orchestrates frame buffering → keyframe selection → VLM analysis →
    EntityGraphDB persistence.

    Runs its analysis loop in a background thread (call start()/stop()).
    Can also be used synchronously: call add_frame() + tick() manually.
    """

    def __init__(
        self,
        config: TemporalMemoryConfig,
        db: EntityGraphDB,
        jsonl_path: Optional[str | Path] = None,
    ) -> None:
        self._cfg = config
        self._db = db
        self._buf = FrameBuffer(config.max_buffer_frames, config.fps)
        self._jsonl = Path(jsonl_path) if jsonl_path else None

        self._entity_roster: list[dict] = []
        self._rolling_summary: str = ""
        self._last_present: list[dict] = []
        self._recent_windows: deque[dict] = deque(maxlen=50)
        self._next_summary_at: float = config.summary_interval_s
        self._robot_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._dist_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stopped = False
        self._buf.set_start(time.time())
        self._thread = threading.Thread(target=self._loop, daemon=True, name="temporal-memory")
        self._thread.start()
        logger.info("TemporalMemory started")

    def stop(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=15.0)
        for t in self._dist_threads:
            t.join(timeout=5.0)
        self._db.commit()
        logger.info("TemporalMemory stopped")

    def _loop(self) -> None:
        while not self._stopped:
            time.sleep(self._cfg.stride_s)
            if not self._stopped:
                try:
                    self.tick()
                except Exception as e:
                    logger.error(f"TemporalMemory tick error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        self._buf.add(frame, timestamp or time.time())

    def update_pose(self, x: float, y: float, z: float = 0.0) -> None:
        self._robot_pos = (x, y, z)

    # ------------------------------------------------------------------
    # Analysis tick (called periodically or manually)
    # ------------------------------------------------------------------

    def tick(self) -> Optional[dict]:
        """Run one analysis window. Returns parsed VLM result or None."""
        window = self._buf.extract_window(self._cfg.window_s)
        if window is None:
            return None

        w_start = window[0].timestamp_s
        w_end   = window[-1].timestamp_s
        keyframes = select_keyframes(window, self._cfg.max_frames_per_window)
        logger.info(f"[temporal] analyzing [{w_start:.1f}-{w_end:.1f}s] ({len(keyframes)} frames)")

        context_summary = (
            f"Rolling summary: {self._rolling_summary or 'none'}\n"
            f"Known entities: {', '.join(e.get('id','?') for e in self._entity_roster[-20:]) or 'none'}"
        )
        prompt = WINDOW_ANALYSIS_PROMPT.format(
            t_start=w_start, t_end=w_end, context=context_summary
        )

        raw = self._vlm_call(keyframes, prompt)
        if not raw:
            return None

        parsed = _parse_vlm_json(raw)
        self._log_jsonl({"ts": time.time(), "type": "window_analysis",
                         "window": [w_start, w_end], "raw": raw, "parsed": parsed})

        if "_error" not in parsed:
            # Update in-memory state
            for e in parsed.get("new_entities", []):
                if not any(r.get("id") == e.get("id") for r in self._entity_roster):
                    self._entity_roster.append(e)
            self._last_present = parsed.get("entities_present", [])
            self._recent_windows.append(parsed)

            # Persist to EntityGraphDB
            self._db.save_window_data(
                parsed, w_end,
                metadata={"world_x": self._robot_pos[0],
                           "world_y": self._robot_pos[1],
                           "world_z": self._robot_pos[2]},
            )

            # Rolling summary update
            if w_end >= self._next_summary_at:
                self._update_summary(w_end)

            # Distance estimation (background)
            if self._cfg.enable_distance_estimation and keyframes:
                mid = keyframes[len(keyframes) // 2]
                t = threading.Thread(
                    target=self._estimate_distances,
                    args=(parsed, mid, w_end),
                    daemon=True,
                )
                t.start()
                self._dist_threads = [dt for dt in self._dist_threads if dt.is_alive()]
                self._dist_threads.append(t)

        return parsed

    def _update_summary(self, w_end: float) -> None:
        latest = self._buf.latest()
        if not latest or not self._recent_windows:
            return
        chunks = " | ".join(
            w.get("caption", "") for w in list(self._recent_windows)[-10:] if w.get("caption")
        )
        if not chunks:
            return
        prompt = (
            f"Previous summary: {self._rolling_summary or 'none'}\n"
            f"Recent events: {chunks}\n\n"
            "Write a concise updated summary (2-3 sentences) of everything observed so far."
        )
        raw = self._vlm_call([latest], prompt)
        if raw:
            self._rolling_summary = raw.strip()
            self._next_summary_at = w_end + self._cfg.summary_interval_s
            self._log_jsonl({"ts": time.time(), "type": "rolling_summary", "summary": self._rolling_summary})
            logger.info(f"[temporal] summary: {self._rolling_summary[:200]}")

    def _estimate_distances(self, parsed: dict, frame: BufferedFrame, ts: float) -> None:
        entities: list[dict] = []
        for e in parsed.get("new_entities", []) + parsed.get("entities_present", []):
            if isinstance(e, dict) and "id" in e:
                db_e = self._db.get_entity(e["id"])
                entities.append({"id": e["id"], "descriptor": (db_e or {}).get("descriptor", e.get("descriptor", "?"))})

        if len(entities) < 2:
            return

        pairs = [
            (entities[i], entities[j])
            for i in range(len(entities))
            for j in range(i + 1, len(entities))
            if not self._db.get_distance(entities[i]["id"], entities[j]["id"])
        ][:self._cfg.max_distance_pairs]

        if not pairs:
            return

        pairs_str = "\n".join(f"  {a['id']} ({a['descriptor']}) vs {b['id']} ({b['descriptor']})" for a, b in pairs)
        prompt = DISTANCE_ESTIMATION_PROMPT.format(pairs=pairs_str)
        raw = self._vlm_call([frame], prompt)
        if not raw:
            return

        parsed_dist = _parse_vlm_json(raw)
        for item in parsed_dist.get("pairs", []):
            cat = item.get("category", "")
            if cat in ("near", "medium", "far"):
                self._db.add_distance(
                    entity_a_id=item["a"],
                    entity_b_id=item["b"],
                    distance_meters=item.get("distance_m"),
                    distance_category=cat,
                    confidence=item.get("confidence", 0.5),
                    timestamp_s=ts,
                    method="vlm",
                )

    # ------------------------------------------------------------------
    # Query (RAG)
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Answer a question using temporal memory + entity graph context."""
        latest = self._buf.latest()
        if not latest:
            return "No frames available yet."

        currently_present: set[str] = set()
        for e in self._last_present:
            if isinstance(e, dict) and "id" in e:
                currently_present.add(e["id"])
        for w in list(self._recent_windows)[-3:]:
            for e in w.get("entities_present", []) + w.get("new_entities", []):
                if isinstance(e, dict) and "id" in e:
                    currently_present.add(e["id"])

        entity_ids = [e.get("id") for e in self._entity_roster if isinstance(e, dict) and "id" in e]
        graph_ctx = self._db.build_graph_context(entity_ids) if entity_ids else "(none)"

        context = (
            f"Rolling summary: {self._rolling_summary or 'none'}\n"
            f"Entities ever seen: {', '.join(e.get('id','?') for e in self._entity_roster)}\n"
            f"Currently visible: {', '.join(sorted(currently_present))}\n"
            f"Entity graph:\n{graph_ctx}"
        )
        prompt = QUERY_PROMPT.format(context=context, question=question)
        raw = self._vlm_call([latest], prompt)
        return raw.strip() if raw else "VLM query failed."

    # ------------------------------------------------------------------
    # VLM dispatch
    # ------------------------------------------------------------------

    def _vlm_call(self, frames: list[BufferedFrame], prompt: str) -> Optional[str]:
        if self._cfg.vlm_backend == "openai" and self._cfg.openai_api_key:
            return _call_openai_vlm(
                frames, prompt,
                api_key=self._cfg.openai_api_key,
                model=self._cfg.openai_model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
            )
        else:
            return _call_ollama_vlm(
                frames, prompt,
                url=self._cfg.ollama_url,
                model=self._cfg.ollama_model,
                max_tokens=self._cfg.max_tokens,
            )

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        return {
            "entity_count": len(self._entity_roster),
            "entities": self._entity_roster,
            "rolling_summary": self._rolling_summary,
            "buffer_size": len(self._buf),
            "recent_windows": len(self._recent_windows),
            "currently_present": self._last_present,
        }

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _log_jsonl(self, record: dict) -> None:
        if not self._jsonl:
            return
        try:
            with open(self._jsonl, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"JSONL log failed: {e}")
