"""
OTA (Observe-Think-Act) Agent — autonomous self-driving loop.

Extracted from dimos McpClient + commahackopenpilot robot_agent.

HOW THE OTA LOOP WORKS:
========================
1. A background timer fires every `interval_s` seconds.
2. Only fires if the agent is IDLE (not currently processing a request).
3. Injects an OTA prompt into the message queue.
4. The prompt contains: current camera frame + spatial-temporal RAG context.
5. Agent calls think(reasoning) first, then ONE action tool.
6. This cycle repeats indefinitely → autonomous exploration.

The key insight: you don't need a special "explore" command. Just give the agent:
  - A scoring system (+5 new area, -2 stagnation)
  - Access to current visual context (camera frame)
  - Access to what's been seen before (SpatialMemory via RAG)
  - Navigation tools (navigate_to, stop)
  - Memory tools (tag_location, update_ledger)

The agent NATURALLY explores because exploring = max score.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Narrative Ledger (agent's internal memory / score tracker)
# ---------------------------------------------------------------------------

@dataclass
class LedgerState:
    objective: str = "Explore the environment and maximize your score."
    subgoal: str = "Find unexplored areas."
    recent_findings: str = "None yet."
    current_curiosity: str = "What's in the next room?"
    last_failed_action: str = "None."
    score: int = 0
    visited_areas: list[str] = field(default_factory=list)
    field_notes: list[str] = field(default_factory=list)


class NarrativeLedger:
    """
    Agent's running chronicle — tracks score, goals, and discoveries.
    Exposed to the agent as tools: get_ledger / update_ledger.
    """

    def __init__(self) -> None:
        self._state = LedgerState()

    def get_ledger(self) -> str:
        s = self._state
        notes = "\n  ".join(s.field_notes[-5:]) if s.field_notes else "none"
        areas = ", ".join(s.visited_areas[-10:]) if s.visited_areas else "none"
        return (
            f"[CONTEXT]\n"
            f"Score: {s.score}\n"
            f"Objective: {s.objective}\n"
            f"Current subgoal: {s.subgoal}\n"
            f"Recent findings: {s.recent_findings}\n"
            f"Curiosity: {s.current_curiosity}\n"
            f"Last failure: {s.last_failed_action}\n"
            f"Visited areas: {areas}\n"
            f"Field notes:\n  {notes}\n"
            f"[/CONTEXT]"
        )

    def update_ledger(
        self,
        objective: str = "",
        subgoal: str = "",
        recent_findings: str = "",
        current_curiosity: str = "",
        last_failed_action: str = "",
        experience: str = "",
        score_delta: int = 0,
    ) -> str:
        if objective:
            self._state.objective = objective
        if subgoal:
            self._state.subgoal = subgoal
        if recent_findings:
            self._state.recent_findings = recent_findings
        if current_curiosity:
            self._state.current_curiosity = current_curiosity
        if last_failed_action:
            self._state.last_failed_action = last_failed_action
        if experience:
            self._state.field_notes.append(experience)
        if score_delta:
            self._state.score += score_delta
        return "Ledger updated."

    def add_area(self, area_name: str) -> None:
        if area_name not in self._state.visited_areas:
            self._state.visited_areas.append(area_name)
            self._state.score += 5  # +5 for new area


# ---------------------------------------------------------------------------
# OTA Loop config
# ---------------------------------------------------------------------------

@dataclass
class OTAConfig:
    interval_s: float = 1.0
    model: str = "claude-sonnet-4-6"
    # OpenAI-compatible API settings
    api_key: str = ""
    base_url: str = ""      # e.g. http://kaweees-dgx-spark.local:8000/v1
    system_prompt: str = _OTA_SYSTEM_PROMPT()
    ota_prompt: str = _OTA_LOOP_PROMPT()
    max_history_messages: int = 8
    include_camera: bool = True
    max_image_width: int = 512
    max_image_height: int = 288
    image_quality: int = 60


def _OTA_SYSTEM_PROMPT() -> str:
    return """\
You are an autonomous robot agent. You have a live camera and can navigate freely.

# VISION
A recent camera frame is automatically attached to every message.
Use it to observe your surroundings before deciding what to do. You are not blind.

# THINKING BEFORE ACTING
Before calling any navigation or action tool, call `think` first with your reasoning:
  1. What do you see?
  2. What is your current score and strategy?
  3. Which direction maximizes new discoveries?

# SCORING
You earn:  +5 new area, +2 goal reached, +1 field note written
You lose:  -3 obstacle/stall, -2 stagnation, -1 revisiting known area

# MEMORY
Use update_ledger(experience=...) to record discoveries.
Use get_ledger() to recall your context when lost.
Use navigate_to_location("kitchen") to go to semantic destinations.

# RULES
- Call exactly ONE action tool per cycle (after think)
- Do not ask questions — just act
- Never just describe what you see without calling a tool
"""


def _OTA_LOOP_PROMPT() -> str:
    return """\
Your context and score are below. A live camera frame is attached.
GOAL: MAXIMIZE YOUR SCORE by exploring new areas.

Call `think` with: (1) what you see, (2) your score/strategy, \
(3) which direction maximizes discoveries.
Then act. Push into unexplored territory. Do not answer in plain English.\
"""


# ---------------------------------------------------------------------------
# Agent tools (what the LLM can call)
# ---------------------------------------------------------------------------

class AgentTools:
    """
    Tool registry for the OTA agent.
    Each method here becomes an LLM-callable tool.
    Wire these to your actual robot actuators.
    """

    def __init__(
        self,
        ledger: NarrativeLedger,
        rag_client=None,               # RagClient from rag_client.py
        navigate_fn: Optional[Callable] = None,   # fn(x, y) → None
        stop_fn: Optional[Callable] = None,
        get_pos_fn: Optional[Callable] = None,    # fn() → (x, y)
    ) -> None:
        self._ledger = ledger
        self._rag = rag_client
        self._navigate = navigate_fn
        self._stop = stop_fn
        self._get_pos = get_pos_fn

    def think(self, reasoning: str) -> str:
        """Record reasoning before acting. Call this BEFORE any action tool.

        Args:
            reasoning: Your observations, score analysis, and chosen strategy.
        """
        logger.info(f"[think] {reasoning[:300]}")
        return "[thought recorded]"

    def navigate_to(self, x: float, y: float) -> str:
        """Navigate to world coordinates (x, y).

        Args:
            x: Target X coordinate in meters
            y: Target Y coordinate in meters
        """
        if self._navigate:
            try:
                self._navigate(x, y)
                return f"Navigating to ({x:.2f}, {y:.2f})"
            except Exception as e:
                return f"Navigation failed: {e}"
        return f"[stub] navigate_to({x:.2f}, {y:.2f})"

    def navigate_to_location(self, description: str) -> str:
        """Navigate to a semantically described location (e.g. 'kitchen', 'doorway').

        Args:
            description: Text description of the destination
        """
        if self._rag:
            loc = self._rag.find_location(description)
            if loc:
                return self.navigate_to(loc["pos_x"], loc["pos_y"])
            return f"Location '{description}' not found in memory."
        return f"[stub] navigate_to_location({description!r})"

    def stop(self) -> str:
        """Stop all movement."""
        if self._stop:
            self._stop()
        return "Stopped."

    def tag_location(self, name: str, description: str = "") -> str:
        """Tag the current location with a name for future navigation.

        Args:
            name: Short name for this location (e.g. 'kitchen', 'charging_station')
            description: Optional description
        """
        if self._rag and self._get_pos:
            pos = self._get_pos()
            if pos:
                self._rag._spatial.tag_location(
                    name=name,
                    description=description or name,
                    pos_x=pos[0], pos_y=pos[1],
                )
                self._ledger.add_area(name)
                return f"Tagged '{name}' at ({pos[0]:.2f},{pos[1]:.2f}). +5 score!"
        return f"[stub] tag_location({name!r})"

    def query_memory(self, question: str) -> str:
        """Ask a question about what has been observed (temporal memory).

        Args:
            question: e.g. "Where is the person?", "What objects are near the desk?"
        """
        if self._rag:
            return self._rag.answer_question(question)
        return "[stub] No temporal memory available."

    def get_ledger(self) -> str:
        """Get your narrative ledger (current score, goals, field notes)."""
        return self._ledger.get_ledger()

    def update_ledger(
        self,
        subgoal: str = "",
        recent_findings: str = "",
        current_curiosity: str = "",
        experience: str = "",
        last_failed_action: str = "",
    ) -> str:
        """Update your narrative ledger with current status.

        Args:
            subgoal: What you're pursuing right now
            recent_findings: What you just learned
            current_curiosity: What you want to investigate next
            experience: Something worth remembering (adds to field notes, +1 score)
            last_failed_action: Most recent failure or block
        """
        delta = 1 if experience else 0
        return self._ledger.update_ledger(
            subgoal=subgoal,
            recent_findings=recent_findings,
            current_curiosity=current_curiosity,
            experience=experience,
            last_failed_action=last_failed_action,
            score_delta=delta,
        )

    def get_spatial_context(self) -> str:
        """Get RAG context about current surroundings (nearby frames + entities)."""
        if self._rag and self._get_pos:
            pos = self._get_pos()
            if pos:
                ctx = self._rag.build_context(pos[0], pos[1])
                return ctx.formatted
        return "(no spatial context)"

    def as_langchain_tools(self) -> list:
        """Convert all tools to LangChain StructuredTool objects."""
        from langchain_core.tools import StructuredTool
        import inspect

        tools = []
        for name in [
            "think", "navigate_to", "navigate_to_location", "stop",
            "tag_location", "query_memory", "get_ledger", "update_ledger",
            "get_spatial_context",
        ]:
            method = getattr(self, name)
            sig = inspect.signature(method)
            params = {
                k: v for k, v in sig.parameters.items() if k != "self"
            }
            tools.append(StructuredTool.from_function(func=method, name=name))
        return tools


# ---------------------------------------------------------------------------
# OTA Agent runner
# ---------------------------------------------------------------------------

class OTAAgent:
    """
    Self-driving Observe-Think-Act agent loop.

    Wires together:
      - LangGraph ReAct agent (LangChain)
      - AgentTools (robot actions)
      - NarrativeLedger (score/memory)
      - Camera frame injection
      - RAG context injection

    Usage:
        agent = OTAAgent(config, tools)
        agent.start()

        # Inject camera frames:
        agent.set_camera_frame(bgr_frame)

        # Human override:
        agent.send_message("Go to the kitchen")

        # Stop:
        agent.stop()
    """

    def __init__(self, config: OTAConfig, tools: AgentTools) -> None:
        self._cfg = config
        self._tools = tools
        self._queue: Queue = Queue()
        self._stop_event = Event()
        self._processing = False
        self._latest_frame = None
        self._history: list = []
        self._agent_graph = None
        self._thread: Optional[Thread] = None
        self._ota_thread: Optional[Thread] = None

    def start(self) -> None:
        self._agent_graph = self._build_agent()
        self._thread = Thread(target=self._process_loop, daemon=True, name="ota-agent")
        self._thread.start()
        self._ota_thread = Thread(target=self._ota_loop, daemon=True, name="ota-timer")
        self._ota_thread.start()
        logger.info("OTAAgent started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._ota_thread:
            self._ota_thread.join(timeout=5.0)
        logger.info("OTAAgent stopped")

    def set_camera_frame(self, frame) -> None:
        """Feed latest camera frame (BGR numpy array)."""
        self._latest_frame = frame

    def send_message(self, text: str) -> None:
        """Inject a human message (overrides OTA tick)."""
        from langchain_core.messages import HumanMessage
        self._queue.put(HumanMessage(content=text))

    # ------------------------------------------------------------------
    # OTA timer loop
    # ------------------------------------------------------------------

    def _ota_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._cfg.interval_s)
            if self._stop_event.is_set():
                break
            if not self._processing and self._queue.empty():
                from langchain_core.messages import HumanMessage
                ledger = self._tools.get_ledger()
                prompt = f"{ledger}\n\n{self._cfg.ota_prompt}"
                self._queue.put(HumanMessage(content=prompt))

    # ------------------------------------------------------------------
    # Message processing loop
    # ------------------------------------------------------------------

    def _process_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._queue.get(timeout=0.5)
            except Empty:
                continue
            self._processing = True
            try:
                self._process_message(msg)
            except Exception as e:
                logger.error(f"OTA agent error: {e}", exc_info=True)
                self._history = []  # Reset history on error
            finally:
                self._processing = False

    def _process_message(self, message) -> None:
        if not self._agent_graph:
            return

        self._history.append(message)
        # Trim history
        if len(self._history) > self._cfg.max_history_messages:
            self._history = self._history[-self._cfg.max_history_messages:]
            # Advance to first HumanMessage
            while self._history and getattr(self._history[0], "type", "") != "human":
                self._history = self._history[1:]

        context_msgs = self._build_context_messages()
        all_msgs = self._history + context_msgs

        for update in self._agent_graph.stream(
            {"messages": all_msgs},
            stream_mode="updates",
        ):
            for node_output in update.values():
                for msg in node_output.get("messages", []):
                    if _has_content(msg):
                        self._history.append(msg)
                        _log_message(msg)

    def _build_context_messages(self) -> list:
        """Build context messages to append: camera frame + spatial RAG."""
        from langchain_core.messages import HumanMessage

        content: list[dict] = []

        # Camera frame
        if self._cfg.include_camera and self._latest_frame is not None:
            try:
                import base64, cv2
                frame = self._latest_frame
                h, w = frame.shape[:2]
                max_h, max_w = self._cfg.max_image_height, self._cfg.max_image_width
                if h > max_h or w > max_w:
                    scale = min(max_w / w, max_h / h)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.image_quality])
                b64 = base64.b64encode(buf.tobytes()).decode()
                content.append({"type": "text", "text": "[VISION] Live camera frame:"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as e:
                logger.debug(f"Camera inject failed: {e}")

        if not content:
            return []

        return [HumanMessage(
            additional_kwargs={"internal_message_type": "vision_context"},
            content=content,
        )]

    def _build_agent(self):
        """Build LangGraph ReAct agent with tools."""
        from langchain.agents import create_react_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langgraph.prebuilt import create_react_agent as lg_create_agent

        lc_tools = self._tools.as_langchain_tools()

        if self._cfg.base_url:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(
                model=self._cfg.model,
                base_url=self._cfg.base_url,
                api_key=self._cfg.api_key or "EMPTY",
            )
        else:
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(
                model=self._cfg.model,
                api_key=self._cfg.api_key,
            )

        return lg_create_agent(
            model=model,
            tools=lc_tools,
            prompt=self._cfg.system_prompt,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_content(msg) -> bool:
    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return True
    if isinstance(content, list) and content:
        return True
    return bool(getattr(msg, "tool_calls", []))


def _log_message(msg) -> None:
    msg_type = getattr(msg, "type", "?")
    content = getattr(msg, "content", "")
    tool_calls = getattr(msg, "tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            logger.info(f"[{msg_type}] → {tc.get('name', '?')}({tc.get('args', {})})")
    elif isinstance(content, str) and content.strip():
        logger.info(f"[{msg_type}] {content[:200]}")


# ---------------------------------------------------------------------------
# Quick start helper
# ---------------------------------------------------------------------------

def make_ota_agent(
    qdrant_url: str = "http://kaweees-dgx-spark.local:6333",
    clip_host: str = "kaweees-dgx-spark.local",
    db_path: str = "~/.local/state/prompt_da/entity_graph.db",
    api_key: str = "",
    model: str = "claude-sonnet-4-6",
    base_url: str = "",
    navigate_fn=None,
    stop_fn=None,
    get_pos_fn=None,
) -> OTAAgent:
    """
    Convenience factory: wire up all components and return a ready OTAAgent.

    Example:
        agent = make_ota_agent(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            navigate_fn=lambda x, y: robot.nav_to(x, y),
            stop_fn=robot.stop,
            get_pos_fn=lambda: robot.get_pose()[:2],
        )
        agent.start()
    """
    import os
    from pathlib import Path
    from .clip_embedder import make_embedder
    from .spatial_memory import SpatialMemory
    from .entity_graph_db import EntityGraphDB
    from .temporal_memory import TemporalMemory, TemporalMemoryConfig
    from .rag_client import RagClient

    embedder = make_embedder(clip_host=clip_host)
    spatial = SpatialMemory(qdrant_url=qdrant_url, embedder=embedder)

    db_resolved = Path(db_path).expanduser()
    entity_db = EntityGraphDB(db_path=db_resolved)

    tm_cfg = TemporalMemoryConfig(
        ollama_url=f"http://{clip_host}:11434",
        open_api_key=os.environ.get("OPENAI_API_KEY", ""),
        vlm_backend="openai" if os.environ.get("OPENAI_API_KEY") else "ollama",
    )
    temporal = TemporalMemory(tm_cfg, entity_db)
    temporal.start()

    rag = RagClient(spatial, entity_db, temporal)
    ledger = NarrativeLedger()
    tools = AgentTools(ledger, rag, navigate_fn=navigate_fn, stop_fn=stop_fn, get_pos_fn=get_pos_fn)

    cfg = OTAConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    return OTAAgent(cfg, tools)
