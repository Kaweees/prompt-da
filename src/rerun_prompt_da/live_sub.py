"""
Live Prompt-DA Subscriber — replaces the SLAM subscriber for comma body
navigation.

Subscribes to camera frames (and optionally IMU data) over Zenoh, runs
Prompt Depth Anything depth completion, generates traversability cost maps,
publishes poses and dense depth, and streams everything to Rerun.

Topics subscribed:
    slam/camera/frame  — grayscale video frames
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
    FRAME_TOPIC,
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


def build_costmap(depth_mm: np.ndarray, K: np.ndarray,
                  pose_wc: np.ndarray | None = None) -> np.ndarray:
    """Project a depth map into a bird's-eye 2D costmap on the XZ ground plane.

    Uses the camera intrinsics to back-project depth pixels into 3D, then
    bins them onto a grid centered on the camera position.
    """
    grid = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE, 3), dtype=np.uint8)

    h, w = depth_mm.shape[:2]
    valid = depth_mm > 0
    if not np.any(valid):
        return grid

    # Back-project valid depth pixels to 3D camera-frame points
    ys_px, xs_px = np.where(valid)
    depths = depth_mm[valid].astype(np.float32) / 1000.0  # mm -> meters

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x3d = (xs_px - cx) * depths / fx
    y3d = (ys_px - cy) * depths / fy
    z3d = depths
    pts_cam = np.stack([x3d, y3d, z3d], axis=-1)  # (N, 3)

    # Transform to world frame if pose available
    if pose_wc is not None:
        R_wc = pose_wc[:3, :3]
        t_wc = pose_wc[:3, 3]
        pts_world = (R_wc @ pts_cam.T).T + t_wc
        origin_x, origin_z = t_wc[0], t_wc[2]
    else:
        pts_world = pts_cam
        origin_x, origin_z = 0.0, 0.0

    # Height filter
    y_rel = pts_world[:, 1] - (pose_wc[1, 3] if pose_wc is not None else 0.0)
    height_mask = (y_rel >= COSTMAP_Y_MIN) & (y_rel <= COSTMAP_Y_MAX)
    pts_world = pts_world[height_mask]

    if len(pts_world) == 0:
        return grid

    gx = ((pts_world[:, 0] - origin_x) / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)
    gz = ((pts_world[:, 2] - origin_z) / COSTMAP_RESOLUTION + COSTMAP_HALF).astype(np.int32)

    mask = (gx >= 0) & (gx < COSTMAP_SIZE) & (gz >= 0) & (gz < COSTMAP_SIZE)
    gx, gz = gx[mask], gz[mask]

    occ = np.zeros((COSTMAP_SIZE, COSTMAP_SIZE), dtype=np.uint8)
    occ[gz, gx] = 255
    if COSTMAP_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * COSTMAP_RADIUS + 1, 2 * COSTMAP_RADIUS + 1),
        )
        occ = cv2.dilate(occ, kernel)

    grid[:, :, 0] = occ  # red = obstacle
    cv2.circle(grid, (COSTMAP_HALF, COSTMAP_HALF), 3, (0, 255, 0), -1)

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

    # ---- Zenoh setup ----
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    frame_queue: queue.Queue = queue.Queue()
    pose_pub = session.declare_publisher(POSE_TOPIC)
    depth_pub = session.declare_publisher(DEPTH_TOPIC)

    def _on_frame(sample):
        payload = sample.payload.to_bytes()
        timestamp, gray, seq = decode_frame(payload)
        frame_queue.put((timestamp, gray, seq))

    imu_buffer: list = []
    imu_lock = threading.Lock()

    def _on_imu(sample):
        payload = sample.payload.to_bytes()
        samples = decode_imu(payload)
        with imu_lock:
            imu_buffer.extend(samples)

    print(f"Subscribing to '{FRAME_TOPIC}' -- waiting for frames...")
    if args.imu:
        print(f"Subscribing to '{IMU_TOPIC}' for IMU data")
    print(f"Publishing poses on '{POSE_TOPIC}'")
    print(f"Publishing depth on '{DEPTH_TOPIC}'")

    frame_sub = session.declare_subscriber(FRAME_TOPIC, _on_frame)
    imu_sub = session.declare_subscriber(IMU_TOPIC, _on_imu) if args.imu else None

    # ---- Lazy-init model and camera params on first frame ----
    model = None
    K = None
    focal = None
    cx = cy = 0.0
    w = h = 0
    frame_count = 0
    last_timestamp = None
    dropped_nonmonotonic = 0
    last_depth_mm = None
    last_pose_wc = None
    undistort_map1 = None
    undistort_map2 = None

    try:
        while True:
            try:
                timestamp, gray, seq = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Non-monotonic timestamp check
            if args.drop_nonmonotonic and last_timestamp is not None and timestamp <= last_timestamp:
                dropped_nonmonotonic += 1
                if dropped_nonmonotonic <= 5 or dropped_nonmonotonic % 30 == 0:
                    print(
                        f"Dropping non-monotonic frame: "
                        f"ts={timestamp:.6f} <= last={last_timestamp:.6f} "
                        f"(dropped={dropped_nonmonotonic})"
                    )
                continue
            last_timestamp = timestamp

            # Drain IMU samples
            imu_samples = None
            if args.imu:
                with imu_lock:
                    if imu_buffer:
                        imu_samples = list(imu_buffer)
                        imu_buffer.clear()

            # Lazy-init on first frame
            if model is None:
                h, w = gray.shape[:2]
                print(f"First frame: {w}x{h}, initializing Prompt-DA ({args.depth_model})...")

                from monopriors.depth_completion_models.prompt_da import PromptDAPredictor
                model = PromptDAPredictor(
                    device="cuda",
                    model_type=args.depth_model,
                    max_size=args.max_image_size,
                )

                # Camera intrinsics scaled to frame width
                K_fisheye = k_matrix(w)
                D = distortion_coeffs().reshape(4, 1)

                if args.focal:
                    focal = args.focal
                    cx, cy = w / 2.0, h / 2.0
                else:
                    fx, fy, cx, cy = scaled_intrinsics(w)
                    focal = fx

                # Build the undistorted (pinhole) intrinsic matrix.
                # After undistortion the image is rectilinear so we use
                # the same focal length but re-center the principal point.
                K_undistorted = np.array([
                    [focal, 0.0, cx],
                    [0.0,  focal, cy],
                    [0.0,  0.0,  1.0],
                ], dtype=np.float64)

                # Pre-compute fisheye undistortion remap tables (done once)
                undistort_map1, undistort_map2 = cv2.fisheye.initUndistortRectifyMap(
                    K_fisheye, D, np.eye(3), K_undistorted, (w, h), cv2.CV_16SC2,
                )

                # Use the undistorted K for all downstream geometry
                K = K_undistorted

                print(f"Camera: focal={focal:.1f} cx={cx:.1f} cy={cy:.1f}")
                print(f"Distortion (KannalaBrandt8 K1-K4): {D.ravel()}")
                print(f"Fisheye undistortion enabled (cv2.fisheye.initUndistortRectifyMap)")
                print(f"IMU freq: {IMU_DEFAULTS['Frequency']} Hz")

            # ---- Undistort fisheye frame using OS04C10 distortion coefficients ----
            gray = cv2.remap(gray, undistort_map1, undistort_map2,
                             interpolation=cv2.INTER_LINEAR)

            frame_count += 1
            rr.set_time("frame", sequence=frame_count)
            rr.set_time("timestamp", timestamp=timestamp)

            # ---- Run depth completion every N frames ----
            run_depth = (frame_count % args.depth_every) == 1 or args.depth_every == 1

            if run_depth:
                # PromptDA expects RGB; replicate grayscale to 3-channel
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

                # Use previous depth as prompt if available, else zeros
                prompt_depth = last_depth_mm if last_depth_mm is not None else np.zeros((h, w), dtype=np.uint16)

                depth_pred = model(rgb=rgb, prompt_depth=prompt_depth)
                depth_mm = depth_pred.depth_mm  # uint16, millimeters

                # Clamp to max range
                max_mm = int(args.max_depth_range * 1000)
                depth_mm[depth_mm > max_mm] = 0

                last_depth_mm = depth_mm

                # Publish depth over Zenoh
                depth_pub.put(encode_depth(timestamp, depth_mm))

                # Log depth to Rerun
                rr.log("world/camera/depth", rr.DepthImage(depth_mm, meter=1000))

                rr.log("prompt_da/state", rr.TextLog(
                    f"frame={frame_count} depth_completed "
                    f"valid_px={int(np.count_nonzero(depth_mm))} "
                    f"max={float(depth_mm.max()) / 1000:.2f}m"
                ))

            # ---- Log camera image ----
            rr.log("world/camera/image", rr.Pinhole(
                focal_length=[focal, focal],
                principal_point=[cx, cy],
                resolution=[w, h],
                camera_xyz=rr.ViewCoordinates.RDF,
            ))
            rr.log("world/camera/image", rr.Image(gray))

            # ---- Costmap from latest depth ----
            if last_depth_mm is not None:
                costmap = build_costmap(last_depth_mm, K, last_pose_wc)
                rr.log("costmap", rr.Image(costmap))

            # ---- Status ----
            if frame_count % 30 == 0:
                has_depth = "+" if last_depth_mm is not None else "-"
                print(
                    f"[{frame_count}] depth={has_depth} "
                    f"imu_samples={'n/a' if imu_samples is None else len(imu_samples)}"
                )

    except KeyboardInterrupt:
        print("\nStopping subscriber")
    finally:
        frame_sub.undeclare()
        if imu_sub:
            imu_sub.undeclare()
        pose_pub.undeclare()
        depth_pub.undeclare()
        session.close()
        if dropped_nonmonotonic:
            print(f"Dropped {dropped_nonmonotonic} non-monotonic frames")
        print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    main()
