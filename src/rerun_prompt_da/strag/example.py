"""
Quick-start example — wire up the full stack against kaweees-dgx-spark.

Run:
    cd dimos/
    OPENAI_API_KEY=sk-... uv run strag-example

Or with a local robot mock:
    uv run strag-example --mock
"""
import argparse
import os
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")


def run(mock: bool = False):
    from rerun_prompt_da.strag.config import QDRANT_URL, CLIP_HOST, DB_DIR, OLLAMA_URL
    from rerun_prompt_da.strag.clip_embedder import make_embedder
    from rerun_prompt_da.strag.spatial_memory import SpatialMemory
    from rerun_prompt_da.strag.entity_graph_db import EntityGraphDB
    from rerun_prompt_da.strag.temporal_memory import TemporalMemory, TemporalMemoryConfig
    from rerun_prompt_da.strag.rag_client import RagClient
    from rerun_prompt_da.strag.exploration import FrontierExplorer
    from rerun_prompt_da.strag.ota_agent import AgentTools, NarrativeLedger, OTAAgent, OTAConfig
    from rerun_prompt_da.strag.mcp_server import make_mcp_server
    from pathlib import Path
    import cv2

    logger.info(f"Qdrant: {QDRANT_URL}")
    logger.info(f"CLIP:   {CLIP_HOST}")
    logger.info(f"Ollama: {OLLAMA_URL}")

    # --- Embedder (gRPC to kaweees-dgx-spark or local torch) ---
    embedder = make_embedder(
        clip_host=CLIP_HOST,
        force_local=mock,
    )

    # --- Spatial Memory (Qdrant) ---
    spatial = SpatialMemory(
        qdrant_url=QDRANT_URL,
        embedder=embedder,
        min_distance_m=0.5,
        min_time_s=1.0,
        new_memory=False,
    )

    # --- Entity Graph DB (SQLite) ---
    db_path = Path(DB_DIR) / "entity_graph.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    entity_db = EntityGraphDB(db_path=db_path)

    # --- Temporal Memory (VLM) ---
    tm_cfg = TemporalMemoryConfig(
        fps=1.0,
        window_s=5.0,
        stride_s=5.0,
        vlm_backend="openai" if os.environ.get("OPENAI_API_KEY") else "ollama",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        ollama_url=OLLAMA_URL,
    )
    temporal = TemporalMemory(
        config=tm_cfg,
        db=entity_db,
        jsonl_path=Path(DB_DIR) / "temporal.jsonl",
    )
    temporal.start()

    # --- RAG Client ---
    rag = RagClient(spatial, entity_db, temporal)

    # --- Frontier Explorer (needs occupancy grid from SLAM) ---
    frontier_explorer = FrontierExplorer(
        min_frontier_size=3,
        safe_distance_m=0.5,
        max_distance_m=10.0,
    )

    # --- Robot pose / navigation stubs ---
    robot_pose = [0.0, 0.0, 0.0]  # x, y, yaw

    def navigate_to(x, y):
        logger.info(f"[robot] navigate_to({x:.2f}, {y:.2f})")
        robot_pose[0], robot_pose[1] = x, y  # mock movement

    def stop():
        logger.info("[robot] stop()")

    def get_pos():
        return tuple(robot_pose[:2])

    # --- Agent Tools ---
    ledger = NarrativeLedger()
    tools = AgentTools(
        ledger=ledger,
        rag_client=rag,
        navigate_fn=navigate_to,
        stop_fn=stop,
        get_pos_fn=get_pos,
    )

    # --- MCP Server (exposes tools over HTTP JSON-RPC) ---
    server = make_mcp_server(tools, port=9990)
    server.run_in_background()
    logger.info(f"MCP server: {server.url()}")

    # --- OTA Agent ---
    cfg = OTAConfig(
        interval_s=2.0,
        model=os.environ.get("AGENT_MODEL", "gpt-4o"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("AGENT_BASE_URL", ""),
    )
    agent = OTAAgent(cfg, tools)
    agent.start()

    # --- Simulate camera + exploration loop ---
    logger.info("Starting camera + exploration loop. Ctrl+C to stop.")

    # Mock occupancy grid for demo
    grid = np.full((100, 100), -1, dtype=np.int8)  # all unknown
    grid[40:60, 40:60] = 0   # free space around robot
    origin_x, origin_y, resolution = -5.0, -5.0, 0.1

    try:
        frame_num = 0
        while True:
            # Simulate camera frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {frame_num}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame_num += 1

            # Feed to temporal memory
            temporal.add_frame(frame)
            temporal.update_pose(*robot_pose[:3] if len(robot_pose) >= 3 else (robot_pose[0], robot_pose[1], 0.0))

            # Store in spatial memory at current pose
            spatial.store_frame(
                frame,
                pos_x=robot_pose[0],
                pos_y=robot_pose[1],
            )

            # Inject camera into OTA agent
            agent.set_camera_frame(frame)

            # Frontier exploration (every 5s)
            if frame_num % 5 == 0:
                goal = frontier_explorer.next_goal(
                    grid=grid,
                    robot_x=robot_pose[0],
                    robot_y=robot_pose[1],
                    origin_x=origin_x,
                    origin_y=origin_y,
                    resolution=resolution,
                )
                if goal:
                    logger.info(f"Frontier goal: ({goal.world_x:.2f}, {goal.world_y:.2f}) gain={goal.info_gain}")

            # Query RAG context
            if frame_num % 10 == 0:
                ctx = rag.build_context(robot_pose[0], robot_pose[1])
                logger.info(f"RAG context:\n{ctx.formatted[:500]}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        agent.stop()
        temporal.stop()
        entity_db.commit()
        entity_db.close()


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use local torch CLIP (no DGX)")
    args = parser.parse_args()
    run(mock=args.mock)


if __name__ == "__main__":
    main_cli()
