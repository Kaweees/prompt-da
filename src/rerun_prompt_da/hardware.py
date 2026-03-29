"""
Hardware constants for the comma device (mici variant).

Camera intrinsics, distortion coefficients, and IMU noise parameters
migrated from the SLAM system (mono_slam.slam).

Camera: OS04C10 wide fisheye
IMU: BMI088
Calibration: cv2.fisheye (KannalaBrandt8 distortion model)
"""

import numpy as np

# ---------------------------------------------------------------------------
# OS04C10 wide fisheye calibration at native 1344x760
# ---------------------------------------------------------------------------
NATIVE_W, NATIVE_H = 1344, 760
NATIVE_FX, NATIVE_FY = 598.0, 598.0
NATIVE_CX, NATIVE_CY = 672.0, 380.0

# KannalaBrandt8 fisheye distortion coefficients (resolution-independent)
NATIVE_K1 = -0.0143559
NATIVE_K2 = -0.00558797
NATIVE_K3 = 0.00237681
NATIVE_K4 = -0.00077131

# ---------------------------------------------------------------------------
# BMI088 IMU noise parameters (conservative starting values)
# ---------------------------------------------------------------------------
IMU_DEFAULTS = {
    "NoiseGyro": 1.7e-4,       # gyroscope noise density   (rad/s/sqrt(Hz))
    "NoiseAcc": 2.0e-3,        # accelerometer noise density (m/s^2/sqrt(Hz))
    "GyroWalk": 1.9e-5,        # gyroscope random walk     (rad/s^2/sqrt(Hz))
    "AccWalk": 3.0e-3,         # accelerometer random walk (m/s^3/sqrt(Hz))
    "Frequency": 100,          # IMU sample rate (Hz)
}

# Camera-to-body (IMU) extrinsic for comma body (mici).
# wideFromDeviceEuler ~ [0, 0, 0] => identity rotation, small translation
# for physical offset between IMU and camera chip on PCB.
TBC_DEFAULT = np.eye(4, dtype=np.float64)

# ---------------------------------------------------------------------------
# VisionIPC stream mapping (comma device camera streams)
# ---------------------------------------------------------------------------
COMMA_STREAM_MAP = {
    "road": "VISION_STREAM_ROAD",
    "wide": "VISION_STREAM_WIDE_ROAD",
    "driver": "VISION_STREAM_DRIVER",
}

# Default camera stream used for navigation
DEFAULT_CAMERA_STREAM = "wide"


def scaled_intrinsics(target_w: int) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) scaled from native resolution to target width.

    Intrinsics scale linearly with resolution. Distortion coefficients
    (K1-K4) are resolution-independent and do not need scaling.
    """
    scale = float(target_w) / NATIVE_W
    return (
        NATIVE_FX * scale,
        NATIVE_FY * scale,
        NATIVE_CX * scale,
        NATIVE_CY * scale,
    )


def k_matrix(target_w: int | float) -> np.ndarray:
    """Return a 3x3 camera intrinsic matrix scaled to *target_w*."""
    target_w = int(target_w)
    fx, fy, cx, cy = scaled_intrinsics(target_w)
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def distortion_coeffs() -> np.ndarray:
    """Return the KannalaBrandt8 distortion vector [K1, K2, K3, K4]."""
    return np.array([NATIVE_K1, NATIVE_K2, NATIVE_K3, NATIVE_K4],
                    dtype=np.float64)
