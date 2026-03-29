"""
Live Prompt-DA Subscriber — replaces the SLAM subscriber for comma body
navigation.

Subscribes to camera frames (and optionally IMU data) over Zenoh, runs
Prompt Depth Anything depth completion, generates traversability cost maps,
publishes poses and dense depth, and streams everything to Rerun.

Topics subscribed:
    body/camera/wide   — wide camera frames
    slam/imu           — IMU samples (optional)

Topics published:
    slam/pose          — 4x4 camera-to-world pose
    slam/depth         — dense uint16 depth map (mm)

Usage:
    uv run prompt-da-sub                          # depth only, Rerun on :9090
    uv run prompt-da-sub --imu                    # with IMU data
    uv run prompt-da-sub --depth-every 3          # run depth every 3rd frame
"""

from __future__ import annotations

import argparse
import queue
import struct
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
    IMU_DEFAULTS,
)
from rerun_prompt_da.zenoh_codec import (
    CAMERA_TOPICS,
    IMU_TOPIC,
    POSE_TOPIC,
    DEPTH_TOPIC,
    decode_frame,
    decode_imu,
    encode_pose,
    encode_depth,
)

# Costmap defaults
DEFAULT_COSTMAP_SIZE = 400
DEFAULT_COSTMAP_RESOLUTION = 0.025
DEFAULT_COSTMAP_RADIUS = 6


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
) -> np.ndarray:
    """Project a depth map into a bird's-eye 2D costmap on the XZ ground plane.

    Uses the camera intrinsics to back-project depth pixels into 3D, then
    bins them onto a grid.  The camera is placed near the top of the image
    (at *cam_z_frac* from the top) so that the forward-facing area fills
    most of the grid.
    """
    half_x = grid_size // 2
    cam_z_row = int(grid_size * cam_z_frac)
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

    h, w = depth_mm.shape[:2]
    valid = depth_mm > 0
    if not np.any(valid):
        return grid

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
        return grid

    gx = ((pts_world[:, 0] - origin_x) / cell_res + half_x).astype(np.int32)
    gz = ((pts_world[:, 2] - origin_z) / cell_res + cam_z_row).astype(np.int32)

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


def main():
    parser = argparse.ArgumentParser(
        description="Prompt-DA live subscriber (Zenoh + Rerun)")
    parser.add_argument("--depth-model", default="large", choices=["large"],
                        help="Prompt-DA model size (default: large)")
    parser.add_argument("--depth-every", type=int, default=5,
                        help="Run depth completion every N frames (default: 5)")
    parser.add_argument("--max-image-size", type=int, default=1008,
                        help="Max image size for PromptDA inference (default: 1008)")
    parser.add_argument("--max-depth-range", type=float, default=4.0,
                        help="Maximum depth range in meters (default: 4.0)")
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
    args = parser.parse_args()

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
    print(f"Publishing depth on '{DEPTH_TOPIC}'")
    print("Waiting for frames...")

    imu_sub = session.declare_subscriber(IMU_TOPIC, _on_imu) if args.imu else None

    # ---- Per-camera state ----
    # Each camera gets its own model init, intrinsics, undistortion maps,
    # depth state, and frame counter.
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

    # Shared model (loaded once, used for all cameras)
    shared_model = None

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

                PROMPT_H, PROMPT_W = 192, 256
                if cs.last_depth_mm is not None:
                    prompt_depth = cv2.resize(cs.last_depth_mm, (PROMPT_W, PROMPT_H), interpolation=cv2.INTER_NEAREST)
                else:
                    prompt_depth = np.zeros((PROMPT_H, PROMPT_W), dtype=np.uint16)

                depth_pred = cs.model(rgb=rgb, prompt_depth=prompt_depth)
                depth_mm = depth_pred.depth_mm

                # Resize depth to match camera resolution so the Rerun point
                # cloud aligns with the pinhole intrinsics.
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

            # ---- Log camera image ----
            rr.log(f"world/{cam_name}", rr.Pinhole(
                focal_length=[cs.focal, cs.focal],
                principal_point=[cs.cx, cs.cy],
                resolution=[cs.w, cs.h],
                camera_xyz=rr.ViewCoordinates.RDF,
            ), static=True)
            rr.log(f"world/{cam_name}/image", rr.Image(gray))

            # ---- Costmap from latest depth ----
            if cs.last_depth_mm is not None:
                costmap = build_costmap(
                    cs.last_depth_mm, cs.K, cs.last_pose_wc,
                    grid_size=args.costmap_size,
                    cell_res=args.costmap_resolution,
                    inflate_radius=args.costmap_radius,
                )
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
        for sub in frame_subs:
            sub.undeclare()
        if imu_sub:
            imu_sub.undeclare()
        pose_pub.undeclare()
        depth_pub.undeclare()
        session.close()
        for cs in cam_states.values():
            if cs.dropped_nonmonotonic:
                print(f"[{cs.name}] Dropped {cs.dropped_nonmonotonic} non-monotonic frames")
            print(f"[{cs.name}] Processed {cs.frame_count} frames")


if __name__ == "__main__":
    main()
