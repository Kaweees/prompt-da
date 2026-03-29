"""
strag — extracted spatio-temporal reasoning, RAG, frontier exploration, and OTA agent.

Files:
  config.py          — centralized env-var config (Qdrant on kaweees-dgx-spark)
  clip_embedder.py   — CLIP 512-dim embeddings (gRPC to DGX or local torch)
  spatial_memory.py  — Qdrant-backed visual/location memory + CLIP text search
  entity_graph_db.py — SQLite entity/relation/distance graph
  temporal_memory.py — VLM sliding-window entity extraction from video
  rag_client.py      — fuses spatial + temporal into agent-ready context strings
  exploration.py     — wavefront BFS frontier detection for autonomous exploration
  ota_agent.py       — OTA (Observe-Think-Act) autonomous agent loop
  mcp_server.py      — FastAPI JSON-RPC tool server (dimos MCP protocol)
"""
from .config import QDRANT_URL, CLIP_HOST, OLLAMA_URL, DB_DIR
from .clip_embedder import make_embedder
from .spatial_memory import SpatialMemory
from .entity_graph_db import EntityGraphDB
from .temporal_memory import TemporalMemory, TemporalMemoryConfig
from .rag_client import RagClient, RagContext
from .exploration import FrontierExplorer, FrontierGoal
from .ota_agent import OTAAgent, OTAConfig, AgentTools, NarrativeLedger, make_ota_agent
from .mcp_server import MCPServer, MCPClient, make_mcp_server

__all__ = [
    "QDRANT_URL", "CLIP_HOST", "OLLAMA_URL", "DB_DIR",
    "make_embedder",
    "SpatialMemory",
    "EntityGraphDB",
    "TemporalMemory", "TemporalMemoryConfig",
    "RagClient", "RagContext",
    "FrontierExplorer", "FrontierGoal",
    "OTAAgent", "OTAConfig", "AgentTools", "NarrativeLedger", "make_ota_agent",
    "MCPServer", "MCPClient", "make_mcp_server",
]
