"""
Live ORB-SLAM3 Subscriber — visual SLAM for comma body navigation.

Subscribes to camera frames (and optionally IMU data) over Zenoh, runs
ORB-SLAM3 monocular SLAM, generates traversability cost maps from sparse
map points, plans paths through waypoints using A*, and streams
everything to Rerun.

Topics subscribed:
    body/camera/wide   — wide camera frames
    slam/imu           — IMU samples (optional)

Topics published:
    slam/pose          — 4x4 camera-to-world pose
    slam/path          — planned path waypoints

Usage:
    uv run prompt-da-sub --vocab ORBvoc.txt
    uv run prompt-da-sub --vocab ORBvoc.txt --waypoints "1.0,2.0;3.0,5.0"
    uv run prompt-da-sub --vocab ORBvoc.txt --imu --use-viewer
"""

from __future__ import annotations

import argparse
import heapq
import queue
import struct
import tempfile
import threading
import time

import cv2
import numpy as np
import orbslam3
import rerun as rr
import rerun.blueprint as rrb
import zenoh

from rerun_prompt_da.hardware import (
    scaled_intrinsics,
    k_matrix,
    distortion_coeffs,
    generate_orbslam_settings,
)
from rerun_prompt_da.zenoh_codec import (
    CAMERA_TOPICS,
    IMU_TOPIC,
    POSE_TOPIC,
    decode_frame,
    decode_imu,
    encode_pose,
)

# Costmap defaults
DEFAULT_COSTMAP_SIZE = 400
DEFAULT_COSTMAP_RESOLUTION = 0.05
DEFAULT_COSTMAP_RADIUS = 6

# Path topic
PATH_TOPIC = "slam/path"

# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------

def world_to_grid(
    x: float, z: float, camera_pos: np.ndarray,
    grid_size: int, cell_res: float, cam_z_frac: float = 0.15,
) -> tuple[int, int]:
    """Convert world XZ to grid (row, col)."""
    half_x = grid_size // 2
    cam_z_row = int(grid_size * cam_z_frac)
    col = int((x - camera_pos[0]) / cell_res + half_x)
    row = int((z - camera_pos[2]) / cell_res + cam_z_row)
    return row, col


def grid_to_world(
    row: int, col: int, camera_pos: np.ndarray,
    grid_size: int, cell_res: float, cam_z_frac: float = 0.15,
) -> tuple[float, float]:
    """Convert grid (row, col) back to world (x, z)."""
    half_x = grid_size // 2
    cam_z_row = int(grid_size * cam_z_frac)
    x = (col - half_x) * cell_res + camera_pos[0]
    z = (row - cam_z_row) * cell_res + camera_pos[2]
    return x, z


# ---------------------------------------------------------------------------
# A* path planner
# ---------------------------------------------------------------------------

_SQRT2 = 1.4142135623730951
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
]


def plan_path(
    costmap: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """A* on the costmap grid. Returns path as [(row, col), ...] or []."""
    grid_size = costmap.shape[0]
    occ = costmap[:, :, 0]  # red channel = obstacles

    sr, sc = start
    gr, gc = goal

    # Clamp to grid bounds
    sr, sc = max(0, min(sr, grid_size - 1)), max(0, min(sc, grid_size - 1))
    gr, gc = max(0, min(gr, grid_size - 1)), max(0, min(gc, grid_size - 1))

    if occ[gr, gc] > 0:
        # Goal is inside an obstacle — find nearest free cell
        best_r, best_c, best_d = gr, gc, float("inf")
        search_r = 20
        for dr in range(-search_r, search_r + 1):
            for dc in range(-search_r, search_r + 1):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size and occ[nr, nc] == 0:
                    d = dr * dr + dc * dc
                    if d < best_d:
                        best_r, best_c, best_d = nr, nc, d
        if best_d == float("inf"):
            return []
        gr, gc = best_r, best_c

    if occ[sr, sc] > 0:
        return []

    def heuristic(r, c):
        dr, dc = abs(r - gr), abs(c - gc)
        return max(dr, dc) + (_SQRT2 - 1) * min(dr, dc)

    open_set = [(heuristic(sr, sc), 0.0, sr, sc)]
    g_score = np.full((grid_size, grid_size), np.inf, dtype=np.float32)
    g_score[sr, sc] = 0.0
    came_from = {}
    visited = np.zeros((grid_size, grid_size), dtype=bool)

    while open_set:
        _, g, r, c = heapq.heappop(open_set)

        if visited[r, c]:
            continue
        visited[r, c] = True

        if r == gr and c == gc:
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        for dr, dc, cost in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size and not visited[nr, nc] and occ[nr, nc] == 0:
                ng = g + cost
                if ng < g_score[nr, nc]:
                    g_score[nr, nc] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (ng + heuristic(nr, nc), ng, nr, nc))

    return []


# ---------------------------------------------------------------------------
# Path encoding for Zenoh
# ---------------------------------------------------------------------------

def encode_path(timestamp: float, path_xz: list[tuple[float, float]]) -> bytes:
    """Encode a planned path as [timestamp, num_points, x0, z0, x1, z1, ...]."""
    header = struct.pack("<di", timestamp, len(path_xz))
    body = b"".join(struct.pack("<dd", x, z) for x, z in path_xz)
    return header + body


# ---------------------------------------------------------------------------
# Costmap builder
# ---------------------------------------------------------------------------

def build_costmap(
    points_3d: np.ndarray,
    camera_pos: np.ndarray,
    *,
    grid_size: int = DEFAULT_COSTMAP_SIZE,
    cell_res: float = DEFAULT_COSTMAP_RESOLUTION,
    inflate_radius: int = DEFAULT_COSTMAP_RADIUS,
    y_min: float = -0.5,
    y_max: float = 2.0,
    cam_z_frac: float = 0.15,
) -> np.ndarray:
    """Project sparse 3D map points into a bird's-eye 2D costmap.

    Points are already in world coordinates. Filter by height relative to
    the camera, then bin onto an XZ grid.
    """
    half_x = grid_size // 2
    cam_z_row = int(grid_size * cam_z_frac)
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    if len(points_3d) == 0:
        cv2.circle(grid, (half_x, cam_z_row), 3, (0, 255, 0), -1)
        return grid

    y_rel = points_3d[:, 1] - camera_pos[1]
    height_mask = (y_rel >= y_min) & (y_rel <= y_max)
    pts = points_3d[height_mask]

    if len(pts) == 0:
        cv2.circle(grid, (half_x, cam_z_row), 3, (0, 255, 0), -1)
        return grid

    gx = ((pts[:, 0] - camera_pos[0]) / cell_res + half_x).astype(np.int32)
    gz = ((pts[:, 2] - camera_pos[2]) / cell_res + cam_z_row).astype(np.int32)

    mask = (gx >= 0) & (gx < grid_size) & (gz >= 0) & (gz < grid_size)
    gx, gz = gx[mask], gz[mask]

    occ = np.zeros((grid_size, grid_size), dtype=np.uint8)
    occ[gz, gx] = 255
    if inflate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflate_radius + 1, 2 * inflate_radius + 1),
        )
        occ = cv2.dilate(occ, kernel)

    grid[:, :, 0] = occ
    cv2.circle(grid, (half_x, cam_z_row), 3, (0, 255, 0), -1)

    return grid


def draw_path_on_costmap(
    costmap: np.ndarray,
    path: list[tuple[int, int]],
    waypoint_cells: list[tuple[int, int]],
) -> None:
    """Draw the planned path (blue) and waypoints (yellow) on the costmap image."""
    grid_size = costmap.shape[0]

    # Draw path as blue polyline
    if len(path) >= 2:
        pts = np.array([(c, r) for r, c in path], dtype=np.int32)
        cv2.polylines(costmap, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    # Draw waypoints as yellow circles
    for wr, wc in waypoint_cells:
        if 0 <= wr < grid_size and 0 <= wc < grid_size:
            cv2.circle(costmap, (wc, wr), 5, (0, 255, 255), -1)


# ---------------------------------------------------------------------------
# Waypoint parser
# ---------------------------------------------------------------------------

def parse_waypoints(s: str) -> list[tuple[float, float]]:
    """Parse 'x1,z1;x2,z2;...' into [(x1,z1), (x2,z2), ...]."""
    waypoints = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(",")
        if len(parts) != 2:
            raise ValueError(f"Waypoint must be 'x,z', got '{pair}'")
        waypoints.append((float(parts[0]), float(parts[1])))
    return waypoints


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ORB-SLAM3 live subscriber (Zenoh + Rerun)")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to ORB vocabulary file (ORBvoc.txt)")
    parser.add_argument("--use-viewer", action="store_true",
                        help="Enable ORB-SLAM3 Pangolin viewer")
    parser.add_argument("--imu", action="store_true",
                        help="Subscribe to IMU data on slam/imu")
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
    parser.add_argument("--waypoints", type=str, default=None,
                        help="Navigation waypoints as 'x1,z1;x2,z2;...' in world XZ coords")
    args = parser.parse_args()

    # Parse waypoints
    nav_waypoints: list[tuple[float, float]] = []
    if args.waypoints:
        nav_waypoints = parse_waypoints(args.waypoints)
        print(f"Navigation waypoints: {nav_waypoints}")

    # ---- Rerun setup ----
    rr.init("orbslam3_live", spawn=False)
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
                    rrb.Spatial2DView(origin="costmap/wide"),
                    rrb.TextLogView(origin="slam/log"),
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
    path_pub = session.declare_publisher(PATH_TOPIC)

    def _make_frame_cb(cam_name):
        def _on_frame(sample):
            payload = sample.payload.to_bytes()
            timestamp, gray, seq = decode_frame(payload)
            frame_queue.put((cam_name, timestamp, gray, seq))
        return _on_frame

    imu_buffer: list = []
    imu_lock = threading.Lock()

    def _on_imu(sample):
        payload = sample.payload.to_bytes()
        samples = decode_imu(payload)
        with imu_lock:
            imu_buffer.extend(samples)

    frame_subs = []
    for cam_name, topic in CAMERA_TOPICS.items():
        sub = session.declare_subscriber(topic, _make_frame_cb(cam_name))
        frame_subs.append(sub)
        print(f"Subscribing to '{topic}' ({cam_name})")
    if args.imu:
        print(f"Subscribing to '{IMU_TOPIC}' for IMU data")
    print(f"Publishing poses on '{POSE_TOPIC}'")
    if nav_waypoints:
        print(f"Publishing path on '{PATH_TOPIC}'")
    print("Waiting for frames...")

    imu_sub = session.declare_subscriber(IMU_TOPIC, _on_imu) if args.imu else None

    # ---- Per-camera state ----
    class CameraState:
        def __init__(self, name: str):
            self.name = name
            self.slam = None
            self.K = None
            self.focal = None
            self.cx = self.cy = 0.0
            self.w = self.h = 0
            self.frame_count = 0
            self.last_timestamp = None
            self.dropped_nonmonotonic = 0
            self.last_pose_wc = None
            self.last_mappoints = np.empty((0, 3), dtype=np.float32)
            self.undistort_map1 = None
            self.undistort_map2 = None
            self.settings_tmpfile = None

    cam_states: dict[str, CameraState] = {}
    shared_slam = None

    try:
        while True:
            try:
                cam_name, timestamp, gray, seq = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Get or create per-camera state
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

            # Drain IMU samples
            imu_samples = None
            if args.imu:
                with imu_lock:
                    if imu_buffer:
                        imu_samples = list(imu_buffer)
                        imu_buffer.clear()

            # Lazy-init on first frame
            if cs.slam is None:
                cs.h, cs.w = gray.shape[:2]
                print(f"[{cam_name}] First frame: {cs.w}x{cs.h}, initializing...")

                # Camera intrinsics scaled to frame width
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

                # Generate ORB-SLAM3 settings and write to temp file
                settings_yaml = generate_orbslam_settings(
                    cs.focal, cs.focal, cs.cx, cs.cy, cs.w, cs.h,
                )
                cs.settings_tmpfile = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False,
                )
                cs.settings_tmpfile.write(settings_yaml)
                cs.settings_tmpfile.flush()

                if shared_slam is None:
                    print(f"Initializing ORB-SLAM3 (vocab: {args.vocab})...")
                    shared_slam = orbslam3.System(
                        args.vocab,
                        cs.settings_tmpfile.name,
                        orbslam3.Sensor.MONOCULAR,
                    )
                    shared_slam.set_use_viewer(args.use_viewer)
                    shared_slam.initialize()
                    print("ORB-SLAM3 initialized")
                cs.slam = shared_slam

                print(f"[{cam_name}] Camera: focal={cs.focal:.1f} cx={cs.cx:.1f} cy={cs.cy:.1f}")
                print(f"[{cam_name}] Fisheye undistortion enabled")

            # ---- Undistort fisheye frame ----
            gray = cv2.remap(gray, cs.undistort_map1, cs.undistort_map2,
                             interpolation=cv2.INTER_LINEAR)

            cs.frame_count += 1
            rr.set_time("frame", sequence=cs.frame_count)
            rr.set_time("timestamp", timestamp=timestamp)

            # ---- Process frame through ORB-SLAM3 ----
            cs.slam.process_image_mono(gray, timestamp)

            # ---- Get pose and map points ----
            pose = cs.slam.get_frame_pose()
            tracking_state = cs.slam.get_tracking_state()
            mappoints = cs.slam.get_tracked_mappoints()
            n_features = cs.slam.get_num_features()
            n_matched = cs.slam.get_num_matched_features()

            # Parse tracking state
            state_names = {
                -1: "NOT_INITIALIZED",
                0: "NO_IMAGES_YET",
                1: "NOT_INITIALIZED",
                2: "OK",
                3: "RECENTLY_LOST",
                4: "LOST",
                5: "OK",
            }
            state_str = state_names.get(tracking_state, f"UNKNOWN({tracking_state})")

            if pose is not None and len(pose) > 0:
                # pose from get_frame_pose() is a 4x4 matrix
                pose_wc = np.array(pose, dtype=np.float64).reshape(4, 4)
                cs.last_pose_wc = pose_wc

                # Publish pose on Zenoh
                pose_pub.put(encode_pose(timestamp, pose_wc, state_str))

                # Log camera transform
                rr.log(f"world/{cam_name}", rr.Transform3D(
                    mat3x3=pose_wc[:3, :3],
                    translation=pose_wc[:3, 3],
                ))

            # Parse map points
            if mappoints is not None and len(mappoints) > 0:
                pts = np.array(mappoints, dtype=np.float32)
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 3)
                cs.last_mappoints = pts

                rr.log("world/map_points", rr.Points3D(
                    positions=pts,
                    radii=0.01,
                ))

            # ---- Log camera image ----
            rr.log(f"world/{cam_name}", rr.Pinhole(
                focal_length=[cs.focal, cs.focal],
                principal_point=[cs.cx, cs.cy],
                resolution=[cs.w, cs.h],
                camera_xyz=rr.ViewCoordinates.RDF,
            ), static=True)
            rr.log(f"world/{cam_name}/image", rr.Image(gray))

            # ---- Costmap + path planning ----
            if len(cs.last_mappoints) > 0 and cs.last_pose_wc is not None:
                camera_pos = cs.last_pose_wc[:3, 3]
                costmap = build_costmap(
                    cs.last_mappoints, camera_pos,
                    grid_size=args.costmap_size,
                    cell_res=args.costmap_resolution,
                    inflate_radius=args.costmap_radius,
                )

                # Path planning through waypoints
                if nav_waypoints and state_str == "OK":
                    half_x = args.costmap_size // 2
                    cam_z_row = int(args.costmap_size * 0.15)
                    start_rc = (cam_z_row, half_x)  # camera position on grid

                    # Convert waypoints to grid coords
                    wp_cells = []
                    for wx, wz in nav_waypoints:
                        wr, wc = world_to_grid(
                            wx, wz, camera_pos,
                            args.costmap_size, args.costmap_resolution,
                        )
                        wp_cells.append((wr, wc))

                    # Plan path: camera → wp1 → wp2 → ...
                    full_path: list[tuple[int, int]] = []
                    current = start_rc
                    for goal_rc in wp_cells:
                        segment = plan_path(costmap, current, goal_rc)
                        if segment:
                            if full_path:
                                segment = segment[1:]  # skip duplicate junction
                            full_path.extend(segment)
                            current = goal_rc
                        else:
                            rr.log("slam/log", rr.TextLog(
                                f"Path planning failed to waypoint ({goal_rc[1]},{goal_rc[0]})"
                            ))

                    # Draw on costmap
                    draw_path_on_costmap(costmap, full_path, wp_cells)

                    # Log path in 3D + publish on Zenoh
                    if full_path:
                        path_world = []
                        for pr, pc in full_path:
                            wx, wz = grid_to_world(
                                pr, pc, camera_pos,
                                args.costmap_size, args.costmap_resolution,
                            )
                            path_world.append((wx, wz))

                        path_3d = np.array(
                            [[wx, camera_pos[1], wz] for wx, wz in path_world],
                            dtype=np.float32,
                        )
                        rr.log("world/planned_path", rr.LineStrips3D(
                            strips=[path_3d],
                            colors=[[0, 0, 255]],
                            radii=[0.02],
                        ))

                        # Log waypoints in 3D
                        wp_3d = np.array(
                            [[wx, camera_pos[1], wz] for wx, wz in nav_waypoints],
                            dtype=np.float32,
                        )
                        rr.log("world/waypoints", rr.Points3D(
                            positions=wp_3d,
                            colors=[[255, 255, 0]] * len(wp_3d),
                            radii=0.05,
                        ))

                        path_pub.put(encode_path(timestamp, path_world))

                rr.log(f"costmap/{cam_name}", rr.Image(costmap))

            # ---- Log tracking state ----
            n_points = len(cs.last_mappoints)
            rr.log("slam/log", rr.TextLog(
                f"[{cam_name}] frame={cs.frame_count} state={state_str} "
                f"map_points={n_points}"
            ))

            # ---- Status ----
            if cs.frame_count % 30 == 0 or cs.frame_count <= 5:
                pose_info = "none"
                if pose is not None:
                    try:
                        pose_info = f"shape={np.array(pose).shape} type={type(pose)}"
                    except Exception:
                        pose_info = f"len={len(pose)} type={type(pose)}"
                mp_info = "none"
                if mappoints is not None:
                    try:
                        mp_info = f"len={len(mappoints)} type={type(mappoints)}"
                        if len(mappoints) > 0:
                            mp_info += f" first={mappoints[0]}"
                    except Exception:
                        mp_info = f"type={type(mappoints)}"
                print(
                    f"[{cam_name}:{cs.frame_count}] "
                    f"state={state_str} features={n_features} matched={n_matched} "
                    f"points={n_points} pose={pose_info} mp={mp_info}"
                )

    except KeyboardInterrupt:
        print("\nStopping subscriber")
    finally:
        if shared_slam is not None:
            shared_slam.shutdown()
        for sub in frame_subs:
            sub.undeclare()
        if imu_sub:
            imu_sub.undeclare()
        pose_pub.undeclare()
        path_pub.undeclare()
        session.close()
        for cs in cam_states.values():
            if cs.settings_tmpfile:
                import os
                os.unlink(cs.settings_tmpfile.name)
            if cs.dropped_nonmonotonic:
                print(f"[{cs.name}] Dropped {cs.dropped_nonmonotonic} non-monotonic frames")
            print(f"[{cs.name}] Processed {cs.frame_count} frames")


if __name__ == "__main__":
    main()
