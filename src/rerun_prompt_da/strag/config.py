"""
Centralized config — reads from env vars with defaults pointing at kaweees-dgx-spark.
Set env vars to override.
"""
import os

# --- kaweees-dgx-spark (local network) ---
DGX_HOST: str = os.environ.get("DGX_HOST", "kaweees-dgx-spark.local")

# Qdrant on kaweees-dgx-spark (from commahackopenpilot project)
QDRANT_URL: str = os.environ.get("QDRANT_URL", f"http://{DGX_HOST}:6333")
QDRANT_COLLECTION: str = os.environ.get("QDRANT_COLLECTION", "spatial_memory")
VECTOR_DIM: int = 512  # CLIP ViT-B/32

# CLIP gRPC server on kaweees-dgx-spark (optional — falls back to local torch)
CLIP_HOST: str = os.environ.get("CLIP_HOST", DGX_HOST)
CLIP_PORT: int = int(os.environ.get("CLIP_PORT", "50051"))

# Ollama for VLM (temporal memory window analysis)
OLLAMA_URL: str = os.environ.get("OLLAMA_URL", f"http://{DGX_HOST}:11434")
OLLAMA_VLM_MODEL: str = os.environ.get("OLLAMA_VLM_MODEL", "qwen2-vl:7b")

# OpenAI fallback for VLM (set OPENAI_API_KEY to enable)
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_VLM_MODEL: str = os.environ.get("OPENAI_VLM_MODEL", "gpt-4o")

# Agent LLM (OTA loop)
AGENT_MODEL: str = os.environ.get("AGENT_MODEL", "claude-sonnet-4-6")
AGENT_CYCLE_HZ: float = float(os.environ.get("AGENT_CYCLE_HZ", "1.0"))

# Spatial memory thresholds
MIN_DISTANCE_M: float = float(os.environ.get("MIN_DISTANCE_M", "0.5"))
MIN_TIME_S: float = float(os.environ.get("MIN_TIME_S", "1.0"))

# Temporal memory
TEMPORAL_FPS: float = float(os.environ.get("TEMPORAL_FPS", "1.0"))
TEMPORAL_WINDOW_S: float = float(os.environ.get("TEMPORAL_WINDOW_S", "5.0"))
TEMPORAL_STRIDE_S: float = float(os.environ.get("TEMPORAL_STRIDE_S", "5.0"))
TEMPORAL_MAX_FRAMES: int = int(os.environ.get("TEMPORAL_MAX_FRAMES", "3"))

# Storage
DB_DIR: str = os.environ.get(
    "DB_DIR",
    os.path.join(os.path.expanduser("~"), ".local", "state", "prompt_da", "memory"),
)
