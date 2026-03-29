"""
Zenoh pub/sub message encoding and decoding.

Binary wire formats are compatible with the SLAM system (mono_slam)
so that prompt-da can subscribe to the same camera/IMU topics and
publish poses and depth on matching topics.

Topics
------
body/camera/wide    — wide-angle road camera   (subscribed)
body/camera/road    — front road camera         (subscribed)
slam/imu            — IMU samples               (subscribed)
slam/pose           — 4x4 camera-to-world pose  (published)
slam/depth          — dense depth map            (published)
"""

import struct

import numpy as np

# ---------------------------------------------------------------------------
# Topic names — must match the comma body publisher (navigate.py --stream)
# ---------------------------------------------------------------------------
CAMERA_TOPICS = {
    "wide": "body/camera/wide",
    "road": "body/camera/road",
}
IMU_TOPIC = "slam/imu"
POSE_TOPIC = "slam/pose"
DEPTH_TOPIC = "slam/depth"

# Legacy single-topic alias (kept for backward compat with camera_pub.py)
FRAME_TOPIC = "slam/camera/frame"

# ---------------------------------------------------------------------------
# Frame codec
# Format: [8B float64 timestamp | 4B int32 height | 4B int32 width |
#          8B int64 sequence | raw grayscale pixels]
# ---------------------------------------------------------------------------
_FRAME_HEADER = struct.Struct("<diiq")


def encode_frame(timestamp: float, frame: np.ndarray, seq: int) -> bytes:
    """Encode a grayscale frame into a compact binary message."""
    h, w = frame.shape[:2]
    return _FRAME_HEADER.pack(timestamp, h, w, int(seq)) + frame.tobytes()


def decode_frame(payload: bytes) -> tuple[float, np.ndarray, int]:
    """Decode a binary frame message into (timestamp, grayscale_image, seq)."""
    timestamp, h, w, seq = _FRAME_HEADER.unpack_from(payload)
    pixels = np.frombuffer(
        payload[_FRAME_HEADER.size:], dtype=np.uint8
    ).reshape(h, w)
    return timestamp, pixels, seq


# ---------------------------------------------------------------------------
# IMU codec
# Format: N x [7 x 8B float64] = N x 56B per sample
#   Each sample: (ax, ay, az, gx, gy, gz, timestamp)
#   ax/ay/az in m/s^2, gx/gy/gz in rad/s, timestamp in seconds.
# ---------------------------------------------------------------------------
_IMU_SAMPLE = struct.Struct("<7d")


def encode_imu(samples: list[tuple]) -> bytes:
    """Encode a batch of IMU samples into bytes."""
    return b"".join(_IMU_SAMPLE.pack(*s) for s in samples)


def decode_imu(
    payload: bytes,
) -> list[tuple[float, float, float, float, float, float, float]]:
    """Decode a binary IMU message into [(ax, ay, az, gx, gy, gz, t), ...]."""
    n = len(payload) // _IMU_SAMPLE.size
    return [
        _IMU_SAMPLE.unpack_from(payload, i * _IMU_SAMPLE.size)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Pose codec
# Format: [8B float64 timestamp | 128B float64[16] row-major 4x4 T_wc |
#          UTF-8 tracking state string]
# ---------------------------------------------------------------------------

def encode_pose(timestamp: float, pose_wc: np.ndarray, state: str) -> bytes:
    """Encode a camera-to-world 4x4 pose into a binary message."""
    header = struct.pack("<d", timestamp)
    return header + pose_wc.astype(np.float64).tobytes() + state.encode("utf-8")


def decode_pose(payload: bytes) -> tuple[float, np.ndarray, str]:
    """Decode a binary pose message into (timestamp, 4x4 T_wc, state)."""
    timestamp = struct.unpack_from("<d", payload)[0]
    pose = np.frombuffer(payload[8:136], dtype=np.float64).reshape(4, 4).copy()
    state = payload[136:].decode("utf-8")
    return timestamp, pose, state


# ---------------------------------------------------------------------------
# Depth codec
# Format: [8B float64 timestamp | 4B int32 height | 4B int32 width |
#          H*W*2 bytes uint16 depth in millimeters]
# ---------------------------------------------------------------------------
_DEPTH_HEADER = struct.Struct("<dii")


def encode_depth(timestamp: float, depth_mm: np.ndarray) -> bytes:
    """Encode a uint16 depth map (millimeters) into a binary message."""
    h, w = depth_mm.shape[:2]
    return _DEPTH_HEADER.pack(timestamp, h, w) + depth_mm.astype(np.uint16).tobytes()


def decode_depth(payload: bytes) -> tuple[float, np.ndarray]:
    """Decode a binary depth message into (timestamp, uint16 depth_mm)."""
    timestamp, h, w = _DEPTH_HEADER.unpack_from(payload)
    depth = np.frombuffer(
        payload[_DEPTH_HEADER.size:], dtype=np.uint16
    ).reshape(h, w).copy()
    return timestamp, depth
