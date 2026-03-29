"""
Camera (+ IMU) Publisher — captures frames from webcam, video file, or comma
device and publishes them over Zenoh for the prompt-da subscriber.

Topics:
    slam/camera/frame  — grayscale video frames
    slam/imu           — IMU samples (accel + gyro) [comma mode only]

Usage:
    python -m rerun_prompt_da.camera_pub                           # webcam
    python -m rerun_prompt_da.camera_pub video.mp4                 # video file
    python -m rerun_prompt_da.camera_pub --comma 192.168.1.10      # comma device
    python -m rerun_prompt_da.camera_pub --comma 192.168.1.10 --imu  # comma + IMU
"""

from __future__ import annotations

import argparse
import struct
import threading
import time

import cv2
import numpy as np
import zenoh

from rerun_prompt_da.hardware import COMMA_STREAM_MAP, DEFAULT_CAMERA_STREAM
from rerun_prompt_da.zenoh_codec import (
    FRAME_TOPIC,
    IMU_TOPIC,
    encode_frame,
    encode_imu,
)


def _compute_scale(src_w: int, src_h: int, target_w: int):
    """Return (scale, w, h) for resizing."""
    if src_w > target_w:
        scale = target_w / src_w
        return scale, target_w, int(src_h * scale)
    return 1.0, src_w, src_h


def _run_comma(args, frame_pub, imu_pub):
    """Capture loop for comma device via VisionIPC + optional cereal IMU."""
    from cereal.visionipc import VisionIpcClient, VisionStreamType

    stream_name = COMMA_STREAM_MAP[args.comma_camera]
    stream_type = getattr(VisionStreamType, stream_name)

    print(f"Connecting to comma device at {args.comma} "
          f"(stream: {args.comma_camera})...")
    vipc = VisionIpcClient("camerad", stream_type, True, addr=args.comma)

    while not vipc.connect(True):
        time.sleep(0.1)
    vipc.recv()  # first frame to get dimensions

    src_w, src_h = vipc.width, vipc.height
    scale, w, h = _compute_scale(src_w, src_h, args.width)

    # --- Optional IMU forwarding via cereal ---
    imu_buffer = []
    imu_lock = threading.Lock()
    imu_thread = None

    if args.imu and imu_pub is not None:
        import cereal.messaging as messaging

        def _imu_loop():
            sm = messaging.SubMaster(
                ["accelerometer", "gyroscope"], addr=args.comma)
            last_accel = (0.0, 0.0, 0.0)
            last_gyro = (0.0, 0.0, 0.0)
            while True:
                sm.update(100)  # 100ms timeout
                if sm.updated["accelerometer"]:
                    a = sm["accelerometer"].acceleration
                    last_accel = (a.v[0], a.v[1], a.v[2])
                if sm.updated["gyroscope"]:
                    g = sm["gyroscope"].gyroscope
                    last_gyro = (g.v[0], g.v[1], g.v[2])
                    t = sm.logMonoTime["gyroscope"] / 1e9
                    sample = (*last_accel, *last_gyro, t)
                    with imu_lock:
                        imu_buffer.append(sample)

        imu_thread = threading.Thread(target=_imu_loop, daemon=True)
        imu_thread.start()
        print("IMU forwarding enabled (accelerometer + gyroscope)")

    dt = 1.0 / args.fps
    seq = 0
    print(f"Publishing frames on '{FRAME_TOPIC}' -- {w}x{h} @ {args.fps} fps")
    print(f"Source: comma device {args.comma} ({args.comma_camera} camera)")
    print("Press Ctrl+C to stop")

    try:
        while True:
            t0 = time.monotonic()
            buf = vipc.recv()
            if buf is None:
                continue

            frame = np.frombuffer(buf.data, dtype=np.uint8).reshape(
                (src_h, src_w, 3))

            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = buf.timestamp_eof / 1e9

            frame_pub.put(encode_frame(timestamp, gray, seq))

            # Flush accumulated IMU samples
            if args.imu and imu_pub is not None:
                with imu_lock:
                    if imu_buffer:
                        imu_pub.put(encode_imu(imu_buffer))
                        imu_buffer.clear()

            seq += 1

            if args.show:
                cv2.imshow("Camera Publisher", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping publisher")
    finally:
        cv2.destroyAllWindows()


def _run_opencv(args, frame_pub):
    """Capture loop for webcam / video file via OpenCV."""
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale, w, h = _compute_scale(src_w, src_h, args.width)

    dt = 1.0 / args.fps
    seq = 0
    print(f"Publishing frames on '{FRAME_TOPIC}' -- {w}x{h} @ {args.fps} fps")
    print(f"Source: {'webcam' if isinstance(source, int) else source}")
    print("Press Ctrl+C to stop")

    try:
        while cap.isOpened():
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    continue
                if args.loop and not isinstance(source, int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if scale != 1.0:
                frame = cv2.resize(frame, (w, h))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            timestamp = time.time()

            frame_pub.put(encode_frame(timestamp, gray, seq))
            seq += 1

            if args.show:
                cv2.imshow("Camera Publisher", gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping publisher")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Camera (+IMU) publisher for prompt-da (Zenoh)")
    parser.add_argument("source", nargs="?", default="0",
                        help="Video source: device index (0) or file path "
                             "(ignored when --comma is set)")
    parser.add_argument("--width", type=int, default=640,
                        help="Target frame width")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Max publish rate (fps)")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/localhost:7447)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop video file continuously")
    parser.add_argument("--show", action="store_true",
                        help="Show local preview window")
    parser.add_argument("--comma", type=str, default=None, metavar="ADDR",
                        help="Connect to comma device at this IP/hostname")
    parser.add_argument("--comma-camera", type=str, default=DEFAULT_CAMERA_STREAM,
                        choices=list(COMMA_STREAM_MAP),
                        help=f"Which comma camera to use (default: {DEFAULT_CAMERA_STREAM})")
    parser.add_argument("--imu", action="store_true",
                        help="Also publish IMU data from comma device "
                             "(requires --comma)")
    args = parser.parse_args()

    if args.imu and not args.comma:
        parser.error("--imu requires --comma")

    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)
    frame_pub = session.declare_publisher(FRAME_TOPIC)
    imu_pub = session.declare_publisher(IMU_TOPIC) if args.imu else None

    try:
        if args.comma:
            _run_comma(args, frame_pub, imu_pub)
        else:
            _run_opencv(args, frame_pub)
    finally:
        frame_pub.undeclare()
        if imu_pub:
            imu_pub.undeclare()
        session.close()


if __name__ == "__main__":
    main()
