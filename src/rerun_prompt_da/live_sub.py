"""
Live Prompt-DA Subscriber — two-thread architecture:

  Mapping thread:  frames → depth → cost grid → costmap + overlay → Rerun
  Planning thread: manages waypoints, runs A* on latest cost grid,
                   sends path + waypoint overlays back to the mapping thread.

Topics subscribed:
    body/camera/wide   — wide camera frames
    body/camera/road   — road camera frames
    nav/waypoints      — JSON waypoint commands (add / remove / clear)

Topics published:
    slam/pose          — 4x4 camera-to-world pose
    slam/depth         — dense uint16 depth map (mm)

Usage:
    uv run prompt-da-sub                          # depth only, Rerun on :9090
    uv run prompt-da-sub --depth-every 3          # run depth every 3rd frame
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zenoh

from rerun_prompt_da.hardware import (
    NATIVE_FX,
    NATIVE_W,
    scaled_intrinsics,
    k_matrix,
    distortion_coeffs,
)
from rerun_prompt_da.motor_controller import motor_control_thread
from rerun_prompt_da.path_planning import (
    Waypoint,
    astar,
    path_to_world_coords,
)
from rerun_prompt_da.zenoh_codec import (
    CAMERA_TOPICS,
    POSE_TOPIC,
    DEPTH_TOPIC,
    decode_frame,
    encode_pose,
    encode_depth,
)

# Costmap defaults — sized to cover the full depth range (20m)
DEFAULT_COSTMAP_SIZE = 800
DEFAULT_COSTMAP_RESOLUTION = 0.05
DEFAULT_COSTMAP_RADIUS = 4

WAYPOINT_TOPIC = "nav/waypoints"


# ── Shared state between mapping and planning threads ─────────────────────

class SharedState:
    """Thread-safe container for data exchanged between mapping & planning."""

    def __init__(self, grid_size: int, cell_res: float, cam_z_frac: float):
        self.lock = threading.Lock()

        # Written by mapping, read by planning
        self.cost_grid: np.ndarray | None = None  # float32 [H,W] in [0,1]

        # Written by planning, read by mapping
        self.path_cells: list[tuple[int, int]] = []
        self.waypoint_cells: list[tuple[int, int]] = []

        # Grid geometry (immutable after init)
        self.grid_size = grid_size
        self.cell_res = cell_res
        self.half_x = grid_size // 2
        self.cam_z_row = int(grid_size * cam_z_frac)

    # -- mapping → planning --------------------------------------------------
    def update_cost_grid(self, cost: np.ndarray):
        with self.lock:
            self.cost_grid = cost.copy()

    def get_cost_grid(self) -> np.ndarray | None:
        with self.lock:
            return self.cost_grid.copy() if self.cost_grid is not None else None

    # -- planning → mapping --------------------------------------------------
    def update_overlay(
        self,
        path_cells: list[tuple[int, int]],
        wp_cells: list[tuple[int, int]],
    ):
        with self.lock:
            self.path_cells = list(path_cells)
            self.waypoint_cells = list(wp_cells)

    def get_overlay(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        with self.lock:
            return list(self.path_cells), list(self.waypoint_cells)


# ── Costmap builder (pure function, no waypoints / A*) ────────────────────

def build_costmap(
    depth_mm: np.ndarray,
    K: np.ndarray,
    pose_wc: np.ndarray | None = None,
    *,
    grid_size: int = DEFAULT_COSTMAP_SIZE,
    cell_res: float = DEFAULT_COSTMAP_RESOLUTION,
    inflate_radius: int = DEFAULT_COSTMAP_RADIUS,
    y_min: float = -0.5,
    y_max: float = 2.0,
    cam_z_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Project depth into a BEV costmap.

    Returns (grid_rgb, cost_norm) where cost_norm is float32 [0,1] for A*.
    """
    half_x = grid_size // 2
    cam_z_row = int(grid_size * cam_z_frac)
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    cost_norm = np.zeros((grid_size, grid_size), dtype=np.float32)

    h, w = depth_mm.shape[:2]
    valid = depth_mm > 0
    if not np.any(valid):
        return grid, cost_norm

    ys_px, xs_px = np.where(valid)
    depths = depth_mm[valid].astype(np.float32) / 1000.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x3d = (xs_px - cx) * depths / fx
    y3d = (ys_px - cy) * depths / fy
    z3d = depths
    pts_cam = np.stack([x3d, y3d, z3d], axis=-1)

    if pose_wc is not None:
        R_wc = pose_wc[:3, :3]
        t_wc = pose_wc[:3, 3]
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        origin_x, origin_z = t_wc[0], t_wc[2]
    else:
        pts_world = pts_cam
        origin_x, origin_z = 0.0, 0.0

    y_rel = pts_world[:, 1] - (pose_wc[1, 3] if pose_wc is not None else 0.0)
    height_mask = (y_rel >= y_min) & (y_rel <= y_max)
    pts_world = pts_world[height_mask]

    if len(pts_world) == 0:
        return grid, cost_norm

    gx = ((pts_world[:, 0] - origin_x) / cell_res + half_x).astype(np.int32)
    gz = ((pts_world[:, 2] - origin_z) / cell_res + cam_z_row).astype(np.int32)

    mask = (gx >= 0) & (gx < grid_size) & (gz >= 0) & (gz < grid_size)
    gx, gz = gx[mask], gz[mask]

    # Accumulate point counts per cell for a density heatmap
    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    np.add.at(density, (gz, gx), 1)

    if inflate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflate_radius + 1, 2 * inflate_radius + 1),
        )
        density = cv2.dilate(density, kernel)

    # Normalize to 0-255 using log scale for better sensitivity
    density_nz = density[density > 0]
    if len(density_nz) > 0:
        log_density = np.zeros_like(density)
        log_density[density > 0] = np.log1p(density[density > 0])
        log_max = np.percentile(log_density[log_density > 0], 95)
        if log_max > 0:
            heat = np.clip(log_density / log_max * 255, 0, 255).astype(np.uint8)
        else:
            heat = np.zeros((grid_size, grid_size), dtype=np.uint8)
    else:
        heat = np.zeros((grid_size, grid_size), dtype=np.uint8)

    # Apply colormap: blue (far/sparse) -> red (close/dense)
    grid = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    grid[heat == 0] = 0

    cost_norm = heat.astype(np.float32) / 255.0

    cv2.circle(grid, (half_x, cam_z_row), 3, (0, 255, 0), -1)

    return grid, cost_norm


def draw_overlay(
    grid: np.ndarray,
    path_cells: list[tuple[int, int]],
    wp_cells: list[tuple[int, int]],
):
    """Draw path and waypoints onto a costmap image (in-place)."""
    grid_size = grid.shape[0]
    for r, c in path_cells:
        if 0 <= r < grid_size and 0 <= c < grid_size:
            cv2.circle(grid, (c, r), 2, (255, 200, 0), -1)
    for r, c in wp_cells:
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                rr_, cc_ = r + dr, c + dc
                if 0 <= rr_ < grid_size and 0 <= cc_ < grid_size:
                    grid[rr_, cc_] = [0, 255, 255]


# ── Planning thread ───────────────────────────────────────────────────────

def planning_thread(
    shared: SharedState,
    session: zenoh.Session,
    initial_waypoints: list[Waypoint],
    obstacle_threshold: float,
    stop_event: threading.Event,
):
    """Manage waypoints and run A* whenever the cost grid or waypoints change.

    Listens for waypoint commands on WAYPOINT_TOPIC:
        {"action": "add",    "x": float, "z": float, "label": str}
        {"action": "remove", "index": int}
        {"action": "clear"}
    """
    waypoints: list[Waypoint] = list(initial_waypoints)
    wp_version = 0  # bumped on every waypoint change
    last_planned_version = -1
    last_cost_id = -1
    cost_gen = 0

    cmd_queue: queue.Queue = queue.Queue()

    def _on_waypoint_cmd(sample):
        try:
            msg = json.loads(sample.payload.to_bytes().decode())
            cmd_queue.put(msg)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    wp_sub = session.declare_subscriber(WAYPOINT_TOPIC, _on_waypoint_cmd)
    print(f"Planning thread: listening on '{WAYPOINT_TOPIC}' for waypoint commands")
    if waypoints:
        print(f"Planning thread: {len(waypoints)} initial waypoints")

    def _replan():
        nonlocal last_planned_version, last_cost_id
        cost = shared.get_cost_grid()
        if cost is None:
            shared.update_overlay([], [])
            return

        start_cell = (shared.cam_z_row, shared.half_x)

        wp_cells: list[tuple[int, int]] = []
        for wp in waypoints:
            gc = int(wp.x / shared.cell_res + shared.half_x)
            gr = int(wp.z / shared.cell_res + shared.cam_z_row)
            wp_cells.append((gr, gc))

        if not wp_cells:
            shared.update_overlay([], [])
            return

        all_cells = [start_cell] + wp_cells
        full_path: list[tuple[int, int]] = []
        for i in range(len(all_cells) - 1):
            segment = astar(
                cost, all_cells[i], all_cells[i + 1],
                obstacle_threshold=obstacle_threshold,
            )
            if segment is not None:
                if full_path:
                    segment = segment[1:]
                full_path.extend(segment)

        shared.update_overlay(full_path, wp_cells)

        # Log path and waypoints to Rerun for 3D visualization
        if full_path:
            path_xz = np.array(
                [((c - shared.half_x) * shared.cell_res, 0.0,
                  (r - shared.cam_z_row) * shared.cell_res)
                 for r, c in full_path], dtype=np.float32,
            )
            rr.log("world/nav/path", rr.LineStrips3D([path_xz], colors=[(255, 200, 0)]))
        else:
            rr.log("world/nav/path", rr.Clear(recursive=False))

        if wp_cells:
            wp_pts = np.array(
                [((c - shared.half_x) * shared.cell_res, 0.0,
                  (r - shared.cam_z_row) * shared.cell_res)
                 for r, c in wp_cells], dtype=np.float32,
            )
            rr.log("world/nav/waypoints", rr.Points3D(wp_pts, radii=0.15, colors=[(0, 255, 255)]))
        else:
            rr.log("world/nav/waypoints", rr.Clear(recursive=False))

        last_planned_version = wp_version
        last_cost_id = cost_gen

    while not stop_event.is_set():
        # Process any pending waypoint commands
        changed = False
        while True:
            try:
                msg = cmd_queue.get_nowait()
            except queue.Empty:
                break
            action = msg.get("action", "")
            if action == "add":
                wp = Waypoint(
                    x=float(msg["x"]),
                    z=float(msg["z"]),
                    label=msg.get("label", f"WP{len(waypoints)}"),
                )
                waypoints.append(wp)
                wp_version += 1
                changed = True
                print(f"Planning: added {wp}")
            elif action == "remove":
                idx = int(msg["index"])
                if 0 <= idx < len(waypoints):
                    removed = waypoints.pop(idx)
                    wp_version += 1
                    changed = True
                    print(f"Planning: removed {removed}")
            elif action == "clear":
                waypoints.clear()
                wp_version += 1
                changed = True
                print("Planning: cleared all waypoints")

        # Replan if waypoints changed or cost grid was updated
        cost = shared.get_cost_grid()
        new_cost = cost is not None
        if changed or (new_cost and (last_planned_version != wp_version or last_cost_id != cost_gen)):
            cost_gen += 1
            _replan()

        stop_event.wait(timeout=0.1)

    wp_sub.undeclare()
    print("Planning thread stopped")


# ── Main (mapping thread) ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prompt-DA live subscriber (Zenoh + Rerun)")
    parser.add_argument("--depth-model", default="large", choices=["large"],
                        help="Prompt-DA model size (default: large)")
    parser.add_argument("--depth-every", type=int, default=5,
                        help="Run depth completion every N frames (default: 5)")
    parser.add_argument("--max-image-size", type=int, default=1008,
                        help="Max image size for PromptDA inference (default: 1008)")
    parser.add_argument("--max-depth-range", type=float, default=20.0,
                        help="Maximum depth range in meters (default: 20.0)")
    parser.add_argument("--focal", type=float, default=None,
                        help="Override focal length (default: OS04C10 intrinsics scaled to frame)")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/localhost:7447)")
    parser.add_argument("--web-port", type=int, default=9090,
                        help="Rerun web viewer port (default: 9090)")
    parser.add_argument("--grpc-port", type=int, default=9876,
                        help="Rerun gRPC server port (default: 9876)")
    parser.add_argument("--rerun-connect-host", type=str, default="127.0.0.1",
                        help="Host for rerun+http:// URI (default: 127.0.0.1)")
    parser.add_argument("--drop-nonmonotonic", action="store_true",
                        help="Drop frames whose timestamps go backwards or repeat")
    parser.add_argument("--costmap-size", type=int, default=DEFAULT_COSTMAP_SIZE,
                        help=f"Costmap grid cells per side (default: {DEFAULT_COSTMAP_SIZE})")
    parser.add_argument("--costmap-resolution", type=float, default=DEFAULT_COSTMAP_RESOLUTION,
                        help=f"Costmap cell size in meters (default: {DEFAULT_COSTMAP_RESOLUTION})")
    parser.add_argument("--costmap-radius", type=int, default=DEFAULT_COSTMAP_RADIUS,
                        help=f"Costmap obstacle inflation radius in cells (default: {DEFAULT_COSTMAP_RADIUS})")
    parser.add_argument("--waypoints", type=float, nargs="+", default=None,
                        help="Initial waypoints as x1 z1 x2 z2 ... in world meters")
    parser.add_argument("--obstacle-threshold", type=float, default=0.8,
                        help="Cost threshold for impassable cells (default: 0.8)")
    parser.add_argument("--enable-rag", action="store_true",
                        help="Enable spatio-temporal RAG (requires [rag] extras)")
    parser.add_argument("--qdrant-url", type=str, default="http://kaweees-dgx-spark.local:6333",
                        help="Qdrant server URL (default: http://kaweees-dgx-spark.local:6333)")
    parser.add_argument("--clip-host", type=str, default=None,
                        help="gRPC CLIP server host (default: derive from qdrant-url, or local)")
    parser.add_argument("--force-local-clip", action="store_true",
                        help="Force local CLIP instead of gRPC")
    parser.add_argument("--new-memory", action="store_true",
                        help="Drop and recreate spatial memory collections on startup")
    args = parser.parse_args()

    # Parse initial waypoints from flat list: x1 z1 x2 z2 ...
    initial_waypoints: list[Waypoint] = []
    if args.waypoints:
        coords = args.waypoints
        if len(coords) % 2 != 0:
            parser.error("--waypoints requires pairs of x z values")
        for i in range(0, len(coords), 2):
            initial_waypoints.append(Waypoint(x=coords[i], z=coords[i + 1], label=f"WP{i // 2}"))

    # ---- Rerun setup ----
    rr.init("prompt_da_live", spawn=False)
    server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
    connect_uri = f"rerun+http://{args.rerun_connect_host}:{args.grpc_port}/proxy"
    rr.serve_web_viewer(open_browser=False, web_port=args.web_port)
    viewer_url = f"http://0.0.0.0:{args.web_port}/?url={connect_uri}"
    print(f"Rerun web viewer at {viewer_url}")
    print(f"Rerun gRPC server at {server_uri}")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world"),
                rrb.Vertical(
                    rrb.Spatial2DView(origin="world/wide/image"),
                    rrb.Spatial2DView(origin="world/wide/depth"),
                    rrb.Spatial2DView(origin="costmap/wide"),
                ),
                column_shares=[20, 9],
            ),
            collapse_panels=True,
        )
    )

    # ---- Zenoh setup ----
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    frame_queue: queue.Queue = queue.Queue()
    pose_pub = session.declare_publisher(POSE_TOPIC)
    depth_pub = session.declare_publisher(DEPTH_TOPIC)

    def _make_frame_cb(cam_name):
        def _on_frame(sample):
            payload = sample.payload.to_bytes()
            timestamp, gray, seq = decode_frame(payload)
            frame_queue.put((cam_name, timestamp, gray, seq))
        return _on_frame

    frame_subs = []
    for cam_name, topic in CAMERA_TOPICS.items():
        sub = session.declare_subscriber(topic, _make_frame_cb(cam_name))
        frame_subs.append(sub)
        print(f"Subscribing to '{topic}' ({cam_name})")
    print(f"Publishing poses on '{POSE_TOPIC}'")
    print(f"Publishing depth on '{DEPTH_TOPIC}'")
    print("Waiting for frames...")

    # ---- Shared state & planning thread ----
    shared = SharedState(
        grid_size=args.costmap_size,
        cell_res=args.costmap_resolution,
        cam_z_frac=0.15,
    )
    stop_event = threading.Event()
    planner = threading.Thread(
        target=planning_thread,
        args=(shared, session, initial_waypoints, args.obstacle_threshold, stop_event),
        daemon=True,
    )
    planner.start()

    motor = threading.Thread(
        target=motor_control_thread,
        args=(shared, session, stop_event),
        daemon=True,
    )
    motor.start()

    # ---- Spatio-temporal RAG (optional) ----
    spatial_mem = None
    temporal_mem = None
    rag_robot_pose = [0.0, 0.0, 0.0]

    if args.enable_rag:
        try:
            from rerun_prompt_da.strag.clip_embedder import make_embedder
            from rerun_prompt_da.strag.spatial_memory import SpatialMemory
            from rerun_prompt_da.strag.entity_graph_db import EntityGraphDB
            from rerun_prompt_da.strag.temporal_memory import TemporalMemory, TemporalMemoryConfig
            from rerun_prompt_da.strag.config import DB_DIR, OLLAMA_URL
            import os
            from pathlib import Path

            clip_host = args.clip_host
            embedder = make_embedder(clip_host=clip_host, force_local=args.force_local_clip)

            spatial_mem = SpatialMemory(
                qdrant_url=args.qdrant_url,
                embedder=embedder,
                new_memory=args.new_memory,
            )
            print(f"RAG: SpatialMemory connected to {args.qdrant_url}")

            db_path = Path(DB_DIR) / "entity_graph.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            entity_db = EntityGraphDB(db_path=db_path)

            tm_cfg = TemporalMemoryConfig(
                vlm_backend="openai" if os.environ.get("OPENAI_API_KEY") else "ollama",
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                ollama_url=OLLAMA_URL,
            )
            temporal_mem = TemporalMemory(
                config=tm_cfg,
                db=entity_db,
                jsonl_path=Path(DB_DIR) / "temporal.jsonl",
            )
            temporal_mem.start()
            print("RAG: TemporalMemory started")
        except ImportError as e:
            print(f"RAG: Failed to import strag modules ({e}). Install with: uv pip install -e '.[rag]'")
            spatial_mem = None
            temporal_mem = None
        except Exception as e:
            print(f"RAG: Initialization failed ({e}). Continuing without RAG.")
            spatial_mem = None
            temporal_mem = None

    # ---- Per-camera state ----
    class CameraState:
        def __init__(self, name: str):
            self.name = name
            self.model = None
            self.K = None
            self.focal = None
            self.cx = self.cy = 0.0
            self.w = self.h = 0
            self.frame_count = 0
            self.last_timestamp = None
            self.dropped_nonmonotonic = 0
            self.last_depth_mm = None
            self.last_pose_wc = None
            self.undistort_map1 = None
            self.undistort_map2 = None

    cam_states: dict[str, CameraState] = {}
    shared_model = None

    try:
        while True:
            try:
                cam_name, timestamp, gray, seq = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if cam_name not in cam_states:
                cam_states[cam_name] = CameraState(cam_name)
            cs = cam_states[cam_name]

            # Non-monotonic timestamp check
            if args.drop_nonmonotonic and cs.last_timestamp is not None and timestamp <= cs.last_timestamp:
                cs.dropped_nonmonotonic += 1
                if cs.dropped_nonmonotonic <= 5 or cs.dropped_nonmonotonic % 30 == 0:
                    print(
                        f"[{cam_name}] Dropping non-monotonic frame: "
                        f"ts={timestamp:.6f} <= last={cs.last_timestamp:.6f} "
                        f"(dropped={cs.dropped_nonmonotonic})"
                    )
                continue
            cs.last_timestamp = timestamp

            # Lazy-init on first frame for this camera
            if cs.model is None:
                cs.h, cs.w = gray.shape[:2]
                print(f"[{cam_name}] First frame: {cs.w}x{cs.h}, initializing...")

                if shared_model is None:
                    print(f"Loading Prompt-DA ({args.depth_model})...")
                    from monopriors.depth_completion_models.prompt_da import PromptDAPredictor
                    shared_model = PromptDAPredictor(
                        device="cuda",
                        model_type=args.depth_model,
                        max_size=args.max_image_size,
                    )
                cs.model = shared_model

                K_fisheye = k_matrix(cs.w)
                D = distortion_coeffs().reshape(4, 1)

                if args.focal:
                    cs.focal = args.focal
                    cs.cx, cs.cy = cs.w / 2.0, cs.h / 2.0
                else:
                    fx, fy, cs.cx, cs.cy = scaled_intrinsics(cs.w)
                    cs.focal = fx

                K_undistorted = np.array([
                    [cs.focal, 0.0, cs.cx],
                    [0.0,  cs.focal, cs.cy],
                    [0.0,  0.0,  1.0],
                ], dtype=np.float64)

                cs.undistort_map1, cs.undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                    K_fisheye, D, np.eye(3), K_undistorted, (cs.w, cs.h), cv2.CV_16SC2,
                )
                cs.K = K_undistorted

                print(f"[{cam_name}] Camera: focal={cs.focal:.1f} cx={cs.cx:.1f} cy={cs.cy:.1f}")
                print(f"[{cam_name}] Fisheye undistortion enabled")

            # ---- Undistort fisheye frame ----
            gray = cv2.remap(gray, cs.undistort_map1, cs.undistort_map2,
                             interpolation=cv2.INTER_LINEAR)

            cs.frame_count += 1
            rr.set_time("frame", sequence=cs.frame_count)
            rr.set_time("timestamp", timestamp=timestamp)

            # ---- Run depth completion every N frames ----
            run_depth = (cs.frame_count % args.depth_every) == 1 or args.depth_every == 1

            if run_depth:
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                h_in, w_in = rgb.shape[:2]
                _model_scale = args.max_image_size / max(h_in, w_in)
                _model_h = int(h_in * _model_scale) // 14 * 14
                _model_w = int(w_in * _model_scale) // 14 * 14
                if (_model_w, _model_h) != (w_in, h_in):
                    rgb_model = cv2.resize(rgb, (_model_w, _model_h),
                                           interpolation=cv2.INTER_LINEAR)
                else:
                    rgb_model = rgb

                PROMPT_H, PROMPT_W = 192, 256
                if cs.last_depth_mm is not None:
                    prompt_depth = cv2.resize(cs.last_depth_mm, (PROMPT_W, PROMPT_H), interpolation=cv2.INTER_NEAREST)
                else:
                    prompt_depth = np.full((PROMPT_H, PROMPT_W), 3000, dtype=np.uint16)
                    prompt_depth[0, 0] = 200
                    prompt_depth[-1, -1] = 20000

                depth_pred = cs.model(rgb=rgb_model, prompt_depth=prompt_depth)
                depth_mm = depth_pred.depth_mm

                if depth_mm.shape[:2] != (cs.h, cs.w):
                    depth_mm = cv2.resize(
                        depth_mm, (cs.w, cs.h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                max_mm = int(args.max_depth_range * 1000)
                depth_mm[depth_mm > max_mm] = 0
                cs.last_depth_mm = depth_mm

                depth_pub.put(encode_depth(timestamp, depth_mm))

                rr.log(f"world/{cam_name}/depth", rr.DepthImage(depth_mm, meter=1000))
                rr.log("prompt_da/state", rr.TextLog(
                    f"[{cam_name}] frame={cs.frame_count} depth_completed "
                    f"valid_px={int(np.count_nonzero(depth_mm))} "
                    f"max={float(depth_mm.max()) / 1000:.2f}m"
                ))

                # ---- Feed to RAG spatial memory ----
                if spatial_mem is not None:
                    spatial_mem.store_frame(
                        rgb,
                        pos_x=rag_robot_pose[0],
                        pos_y=rag_robot_pose[1],
                        pos_z=rag_robot_pose[2],
                    )

            # ---- Feed to RAG temporal memory ----
            if temporal_mem is not None:
                temporal_mem.add_frame(gray, timestamp=timestamp)
                temporal_mem.update_pose(*rag_robot_pose)

            # ---- Log camera image ----
            rr.log(f"world/{cam_name}", rr.Pinhole(
                focal_length=[cs.focal, cs.focal],
                principal_point=[cs.cx, cs.cy],
                resolution=[cs.w, cs.h],
                camera_xyz=rr.ViewCoordinates.RDF,
            ), static=True)
            rr.log(f"world/{cam_name}/image", rr.Image(gray))

            # ---- Costmap: build + overlay from planning thread ----
            if cs.last_depth_mm is not None:
                costmap, cost_norm = build_costmap(
                    cs.last_depth_mm, cs.K, cs.last_pose_wc,
                    grid_size=args.costmap_size,
                    cell_res=args.costmap_resolution,
                    inflate_radius=args.costmap_radius,
                )

                # Push cost grid to planning thread
                shared.update_cost_grid(cost_norm)

                # Pull path + waypoints from planning thread and draw
                path_cells, wp_cells = shared.get_overlay()
                draw_overlay(costmap, path_cells, wp_cells)

                rr.log(f"costmap/{cam_name}", rr.Image(costmap))

            # ---- Status ----
            if cs.frame_count % 30 == 0:
                if cs.last_depth_mm is not None:
                    valid_mask = cs.last_depth_mm > 0
                    n_valid = int(np.count_nonzero(valid_mask))
                    if n_valid > 0:
                        valid_vals = cs.last_depth_mm[valid_mask].astype(np.float32)
                        d_min = float(valid_vals.min()) / 1000.0
                        d_max = float(valid_vals.max()) / 1000.0
                        d_mean = float(valid_vals.mean()) / 1000.0
                    else:
                        d_min = d_max = d_mean = 0.0
                    print(
                        f"[{cam_name}:{cs.frame_count}] "
                        f"depth: {n_valid}px min={d_min:.2f}m max={d_max:.2f}m mean={d_mean:.2f}m"
                    )
                else:
                    print(f"[{cam_name}:{cs.frame_count}] depth: none")

    except KeyboardInterrupt:
        print("\nStopping subscriber")
    finally:
        if temporal_mem is not None:
            temporal_mem.stop()
            print("RAG: TemporalMemory stopped")
        stop_event.set()
        planner.join(timeout=2.0)
        motor.join(timeout=2.0)
        for sub in frame_subs:
            sub.undeclare()
        pose_pub.undeclare()
        depth_pub.undeclare()
        session.close()
        for cs in cam_states.values():
            if cs.dropped_nonmonotonic:
                print(f"[{cs.name}] Dropped {cs.dropped_nonmonotonic} non-monotonic frames")
            print(f"[{cs.name}] Processed {cs.frame_count} frames")


if __name__ == "__main__":
    main()
