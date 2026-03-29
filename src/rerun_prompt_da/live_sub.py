"""
Live Depth Anything 3 Subscriber — multi-camera depth estimation for
comma body navigation.

Subscribes to camera frames (and optionally IMU data) over Zenoh, runs
Depth Anything 3 joint multi-view inference across wide + road cameras,
generates traversability cost maps, publishes poses and dense depth,
and streams everything to Rerun.

Topics subscribed:
    body/camera/wide   — wide-angle road camera frames
    body/camera/road   — front road camera frames
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
import threading
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zenoh

from rerun_prompt_da.hardware import (
    NATIVE_W,
    scaled_intrinsics,
    k_matrix,
    distortion_coeffs,
    is_fisheye,
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

# Costmap parameters (same as SLAM system for compatibility)
COSTMAP_SIZE = 200         # grid cells per side
COSTMAP_RESOLUTION = 0.05  # meters per cell (5 cm)
COSTMAP_RADIUS = 3         # inflation radius in cells
COSTMAP_HALF = COSTMAP_SIZE // 2
COSTMAP_Y_MIN = -0.5
COSTMAP_Y_MAX = 2.0


CAMERA_HEIGHT = 0.30  # comma body camera height above ground (meters)
OBSTACLE_MIN_HEIGHT = 0.10  # obstacles must be at least this tall above ground
OBSTACLE_MAX_HEIGHT = 1.50  # ignore stuff above this (ceilings, sky)


def build_costmap(depth_mm: np.ndarray, K: np.ndarray,
                  pose_wc: np.ndarray | None = None) -> np.ndarray:
    """Project depth into a bird's-eye costmap. Green=free, Red=obstacle.

    Works in camera frame (RDF: X=right, Y=down, Z=forward).
    Ground plane is at Y ≈ +CAMERA_HEIGHT in camera frame.
    """
    grid = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE, 3), dtype=np.uint8)

    h, w = depth_mm.shape[:2]
    valid = depth_mm > 0
    if not np.any(valid):
        return grid

    # Back-project to 3D camera-frame points (RDF)
    ys_px, xs_px = np.where(valid)
    depths = depth_mm[valid].astype(np.float32) / 1000.0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x3d = (xs_px - cx) * depths / fx  # right
    y3d = (ys_px - cy) * depths / fy  # down
    z3d = depths                       # forward

    # In camera RDF frame, ground is at y ≈ +CAMERA_HEIGHT
    # Height above ground = CAMERA_HEIGHT - y3d (since y points down)
    height_above_ground = CAMERA_HEIGHT - y3d

    # Classify: ground vs obstacle
    is_ground = (height_above_ground >= -0.05) & (height_above_ground < OBSTACLE_MIN_HEIGHT)
    is_obstacle = (height_above_ground >= OBSTACLE_MIN_HEIGHT) & (height_above_ground < OBSTACLE_MAX_HEIGHT)

    # Project onto XZ (bird's-eye): X=right, Z=forward
    # Grid center = camera position
    def to_grid(x, z):
        gx = (x / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)
        gz = (-z / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)  # flip Z so forward=up
        in_bounds = (gx >= 0) & (gx < COSTMAP_SIZE) & (gz >= 0) & (gz < COSTMAP_SIZE)
        return gx, gz, in_bounds

    # Mark ground cells as free (green)
    if np.any(is_ground):
        gx_g, gz_g, mask_g = to_grid(x3d[is_ground], z3d[is_ground])
        gx_g, gz_g = gx_g[mask_g], gz_g[mask_g]
        grid[gz_g, gx_g, 1] = 180  # green = free

    # Mark obstacle cells (red, overwrites green)
    if np.any(is_obstacle):
        gx_o, gz_o, mask_o = to_grid(x3d[is_obstacle], z3d[is_obstacle])
        gx_o, gz_o = gx_o[mask_o], gz_o[mask_o]
        occ = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE), dtype=np.uint8)
        occ[gz_o, gx_o] = 255
        if COSTMAP_RADIUS > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * COSTMAP_RADIUS + 1, 2 * COSTMAP_RADIUS + 1),
            )
            occ = cv2.dilate(occ, kernel)
        grid[:, :, 0] = np.maximum(grid[:, :, 0], occ)
        # Clear green where obstacles are
        grid[:, :, 1] = np.where(occ > 0, 0, grid[:, :, 1])

    # Camera position marker (white dot)
    cv2.circle(grid, (COSTMAP_HALF, COSTMAP_HALF), 3, (255, 255, 255), -1)

    return grid



def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything 3 live subscriber (Zenoh + Rerun)")
    parser.add_argument("--depth-model", default="da3-large",
                        choices=["da3-small", "da3-base", "da3-large",
                                 "da3metric-large", "da3mono-large"],
                        help="DA3 model preset (default: da3-large)")
    parser.add_argument("--depth-every", type=int, default=5,
                        help="Run depth inference every N frames (default: 5)")
    parser.add_argument("--process-res", type=int, default=504,
                        help="DA3 processing resolution (default: 504)")
    parser.add_argument("--max-depth-range", type=float, default=10.0,
                        help="Maximum depth range in meters (default: 10.0)")
    parser.add_argument("--imu", action="store_true",
                        help="Subscribe to IMU data on slam/imu")
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
                # Left: 3D view + costmaps
                rrb.Vertical(
                    rrb.Spatial3DView(origin="world"),
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="costmap/wide", name="Wide Costmap"),
                        rrb.Spatial2DView(origin="costmap/road", name="Road Costmap"),
                    ),
                    row_shares=[3, 1],
                ),
                # Right: per-camera RGB + depth (colorized)
                rrb.Vertical(
                    rrb.Spatial2DView(origin="world/wide/image", name="Wide RGB"),
                    rrb.Spatial2DView(origin="depth/wide", name="Wide Depth"),
                    rrb.Spatial2DView(origin="world/road/image", name="Road RGB"),
                    rrb.Spatial2DView(origin="depth/road", name="Road Depth"),
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
    class CameraState:
        def __init__(self, name: str):
            self.name = name
            self.K = None                    # 3x3 intrinsic (after undistortion for fisheye)
            self.w = self.h = 0
            self.frame_count = 0
            self.last_timestamp = None
            self.dropped_nonmonotonic = 0
            self.last_depth_mm = None
            self.last_pose_wc = None
            self.undistort_map1 = None       # None for pinhole (road)
            self.undistort_map2 = None
            self.latest_rgb = None           # most recent undistorted RGB frame

    cam_states: dict[str, CameraState] = {}
    da3_model = None
    total_frames = 0

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
            if cs.K is None:
                cs.h, cs.w = gray.shape[:2]
                print(f"[{cam_name}] First frame: {cs.w}x{cs.h}")

                # Per-camera intrinsics
                fx, fy, cx, cy = scaled_intrinsics(cs.w, cam_name)

                if is_fisheye(cam_name):
                    # Fisheye cameras need undistortion
                    K_fisheye = k_matrix(cs.w, cam_name)
                    D = distortion_coeffs().reshape(4, 1)
                    K_undistorted = np.array([
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    cs.undistort_map1, cs.undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                        K_fisheye, D, np.eye(3), K_undistorted, (cs.w, cs.h), cv2.CV_16SC2,
                    )
                    cs.K = K_undistorted
                    print(f"[{cam_name}] Fisheye: focal={fx:.1f} cx={cx:.1f} cy={cy:.1f} (undistorted)")
                else:
                    # Road camera is pinhole — no undistortion needed
                    cs.K = np.array([
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ], dtype=np.float64)
                    print(f"[{cam_name}] Pinhole: focal={fx:.1f} cx={cx:.1f} cy={cy:.1f}")

            # ---- Undistort fisheye cameras only ----
            if cs.undistort_map1 is not None:
                gray = cv2.remap(gray, cs.undistort_map1, cs.undistort_map2,
                                 interpolation=cv2.INTER_LINEAR)

            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cs.latest_rgb = rgb

            cs.frame_count += 1
            total_frames += 1
            rr.set_time("frame", sequence=total_frames)
            rr.set_time("timestamp", timestamp=timestamp)

            # ---- Log camera image ----
            rr.log(f"world/{cam_name}", rr.Pinhole(
                focal_length=[cs.K[0, 0], cs.K[1, 1]],
                principal_point=[cs.K[0, 2], cs.K[1, 2]],
                resolution=[cs.w, cs.h],
                camera_xyz=rr.ViewCoordinates.RDF,
            ), static=True)
            rr.log(f"world/{cam_name}/image", rr.Image(gray))

            # ---- Run DA3 multi-view depth every N frames ----
            run_depth = (total_frames % args.depth_every) == 1 or args.depth_every == 1
            # Only run when we have frames from all active cameras
            all_cams_ready = all(
                cs2.latest_rgb is not None for cs2 in cam_states.values()
            )

            if run_depth and all_cams_ready and len(cam_states) > 0:
                # Lazy-load DA3 model
                if da3_model is None:
                    print(f"Loading Depth Anything 3 ({args.depth_model})...")
                    from depth_anything_3.api import DepthAnything3
                    da3_model = DepthAnything3(model_name=args.depth_model)
                    da3_model = da3_model.to("cuda")
                    da3_model.eval()
                    print("DA3 model loaded.")

                # Collect frames and intrinsics in consistent order
                cam_order = sorted(cam_states.keys())
                images = [cam_states[cn].latest_rgb for cn in cam_order]
                intrinsics = np.stack([cam_states[cn].K for cn in cam_order], axis=0)

                # Joint multi-view inference
                prediction = da3_model.inference(
                    image=images,
                    intrinsics=intrinsics,
                    process_res=args.process_res,
                )

                # Process per-camera depth outputs
                for i, cn in enumerate(cam_order):
                    cs2 = cam_states[cn]
                    depth_m = prediction.depth[i]  # (H, W) float32 meters

                    # Clip to [0, max_depth_range]: keeps valid gradient for
                    # visualization; negative/NaN values collapse to 0 (no-data).
                    depth_m = np.clip(depth_m, 0.0, args.max_depth_range)

                    # Convert to uint16 millimeters
                    depth_mm = (depth_m * 1000).astype(np.uint16)

                    # Resize depth back to original frame size if needed
                    dh, dw = depth_mm.shape[:2]
                    if dh != cs2.h or dw != cs2.w:
                        depth_mm = cv2.resize(depth_mm, (cs2.w, cs2.h),
                                              interpolation=cv2.INTER_NEAREST)

                    cs2.last_depth_mm = depth_mm

                    # Publish depth over Zenoh (use wide as primary)
                    if cn == "wide":
                        depth_pub.put(encode_depth(timestamp, depth_mm))

                    # Log depth under the camera Pinhole (projects into 3D view)
                    rr.log(f"world/{cn}/depth", rr.DepthImage(depth_mm, meter=1000))

                # Log DA3 estimated poses if available
                if prediction.extrinsics is not None:
                    for i, cn in enumerate(cam_order):
                        ext = prediction.extrinsics[i]  # (3, 4) world-to-cam
                        # Convert to 4x4
                        T_cw = np.eye(4, dtype=np.float32)
                        T_cw[:3, :] = ext
                        T_wc = np.linalg.inv(T_cw)
                        cam_states[cn].last_pose_wc = T_wc

                rr.log("prompt_da/state", rr.TextLog(
                    f"DA3 multi-view: {len(cam_order)} cameras, "
                    f"cams={cam_order}"
                ))

            # ---- Costmap from latest depth ----
            if cs.last_depth_mm is not None:
                costmap = build_costmap(cs.last_depth_mm, cs.K)
                rr.log(f"costmap/{cam_name}", rr.Image(costmap))

            # ---- Status ----
            if cs.frame_count % 30 == 0:
                has_depth = "+" if cs.last_depth_mm is not None else "-"
                print(
                    f"[{cam_name}:{cs.frame_count}] depth={has_depth} "
                    f"imu_samples={'n/a' if imu_samples is None else len(imu_samples)}"
                )

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
