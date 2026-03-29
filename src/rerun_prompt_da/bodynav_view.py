"""
Bodynav Camera Viewer — subscribes to all three comma body cameras over Zenoh
and displays them live in Rerun.

Topics subscribed:
    slam/camera/road        — road-facing camera
    slam/camera/wide_road   — wide road-facing camera
    slam/camera/driver      — driver-facing camera

Usage:
    uv run bodynav-view                                    # local Zenoh peer
    uv run bodynav-view --connect tcp/100.94.67.9:7447     # connect to DGX
    uv run bodynav-view --web-port 9090 --grpc-port 9876
"""

from __future__ import annotations

import argparse
import queue
import threading

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import zenoh

from rerun_prompt_da.zenoh_codec import decode_frame

# ---------------------------------------------------------------------------
# Camera topic definitions
# ---------------------------------------------------------------------------
CAMERAS: dict[str, str] = {
    "road":       "slam/camera/road",
    "wide_road":  "slam/camera/wide_road",
    "driver":     "slam/camera/driver",
}

# Rerun entity paths (one per camera)
ENTITY: dict[str, str] = {
    name: f"cameras/{name}/image" for name in CAMERAS
}


def _decode_image(payload: bytes) -> tuple[float, np.ndarray] | None:
    """Decode a camera payload.

    Tries the custom binary frame format used by zenoh_codec first, then
    falls back to JPEG decoding so the viewer works regardless of which
    publisher the body is running.
    """
    try:
        timestamp, img, _seq = decode_frame(payload)
        return timestamp, img
    except Exception:
        pass

    # JPEG / compressed-image fallback
    try:
        arr = np.frombuffer(payload, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            import time
            return time.time(), img
    except Exception:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Bodynav camera viewer (Zenoh → Rerun)")
    parser.add_argument("--connect", type=str, default=None,
                        help="Zenoh router endpoint (e.g. tcp/100.94.67.9:7447)")
    parser.add_argument("--web-port", type=int, default=9090,
                        help="Rerun web viewer port (default: 9090)")
    parser.add_argument("--grpc-port", type=int, default=9876,
                        help="Rerun gRPC server port (default: 9876)")
    parser.add_argument("--rerun-connect-host", type=str, default="127.0.0.1",
                        help="Host for rerun+http:// URI (default: 127.0.0.1)")
    args = parser.parse_args()

    # ---- Rerun setup ----
    rr.init("bodynav_camera_view", spawn=False)
    server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
    connect_uri = f"rerun+http://{args.rerun_connect_host}:{args.grpc_port}/proxy"
    rr.serve_web_viewer(open_browser=False, web_port=args.web_port)
    viewer_url = f"http://0.0.0.0:{args.web_port}/?url={connect_uri}"
    print(f"Rerun web viewer at {viewer_url}")
    print(f"Rerun gRPC server at {server_uri}")

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=ENTITY["road"],       name="Road"),
                rrb.Spatial2DView(origin=ENTITY["wide_road"],  name="Wide Road"),
                rrb.Spatial2DView(origin=ENTITY["driver"],     name="Driver"),
            ),
            collapse_panels=True,
        )
    )

    # ---- Zenoh setup ----
    conf = zenoh.Config()
    if args.connect:
        conf.insert_json5("connect/endpoints", f'["{args.connect}"]')
    session = zenoh.open(conf)

    # One queue per camera so callbacks stay non-blocking
    queues: dict[str, queue.Queue] = {name: queue.Queue(maxsize=4) for name in CAMERAS}
    frame_counts: dict[str, int] = {name: 0 for name in CAMERAS}

    def _make_callback(name: str):
        def _on_sample(sample):
            result = _decode_image(sample.payload.to_bytes())
            if result is None:
                return
            try:
                queues[name].put_nowait(result)
            except queue.Full:
                pass  # drop oldest if the display loop falls behind

        return _on_sample

    subs = []
    for name, topic in CAMERAS.items():
        sub = session.declare_subscriber(topic, _make_callback(name))
        subs.append(sub)
        print(f"  Subscribed: {topic}")

    print("Waiting for camera frames — press Ctrl+C to stop")

    # ---- Drain queues and log to Rerun ----
    try:
        while True:
            got_any = False
            for name in CAMERAS:
                try:
                    timestamp, img = queues[name].get_nowait()
                except queue.Empty:
                    continue

                got_any = True
                frame_counts[name] += 1

                rr.set_time("frame", sequence=frame_counts[name])
                rr.set_time("timestamp", timestamp=timestamp)
                rr.log(ENTITY[name], rr.Image(img))

            if not got_any:
                # Nothing arrived — yield to avoid busy-spinning
                import time
                time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping viewer")
    finally:
        for sub in subs:
            sub.undeclare()
        session.close()
        for name, count in frame_counts.items():
            print(f"  {name}: {count} frames received")


if __name__ == "__main__":
    main()
