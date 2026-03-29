"""Pure pursuit path follower for differential-drive robots.

Runs as a thread inside live_sub.  Reads the latest A* path from
SharedState, computes velocity commands via pure pursuit, and publishes
them on Zenoh for the body driver.

Zenoh topics
------------
nav/command          — incoming commands: {"action":"go","waypoint_index":0}
                       or {"action":"stop"}
body/control/velocity — outgoing: {"linear":float,"angular":float,
                        "left":float,"right":float}
nav/status           — outgoing: {"state":str,"target":str,"distance":float}
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time

import numpy as np
import zenoh

# ── Comma body physical constants ────────────────────────────────────────
WHEEL_BASE = 0.235          # metres between left and right wheels
MAX_LINEAR_VEL = 0.9        # m/s  (3x speed)
MAX_ANGULAR_VEL = 3.6       # rad/s
LOOKAHEAD_DIST = 0.35       # metres ahead on the path
GOAL_TOLERANCE = 0.20       # metres — "arrived" threshold
SLOWDOWN_DIST = 0.6         # start decelerating within this range

# Zenoh topics
VELOCITY_TOPIC = "body/control/velocity"
NAV_COMMAND_TOPIC = "nav/command"
NAV_STATUS_TOPIC = "nav/status"


# ── Pure pursuit ─────────────────────────────────────────────────────────

def pure_pursuit(
    path_xz: np.ndarray,
    lookahead: float = LOOKAHEAD_DIST,
) -> tuple[float, float] | None:
    """Compute (linear_vel, angular_vel) to follow *path_xz*.

    Parameters
    ----------
    path_xz : (N, 2) float array of (x, z) in **ego frame**.
        Robot is at the origin, facing +z.
    lookahead : target distance along the path.

    Returns None when no valid steering can be computed.
    """
    if len(path_xz) < 2:
        return None

    # Walk along the path to find the lookahead point.
    cum = 0.0
    target = path_xz[-1]
    for i in range(1, len(path_xz)):
        dx = path_xz[i, 0] - path_xz[i - 1, 0]
        dz = path_xz[i, 1] - path_xz[i - 1, 1]
        cum += math.hypot(dx, dz)
        if cum >= lookahead:
            target = path_xz[i]
            break

    xt, zt = float(target[0]), float(target[1])
    dist = math.hypot(xt, zt)
    if dist < 1e-3:
        return 0.0, 0.0

    # Angle from forward (+z) to the target point.
    alpha = math.atan2(xt, zt)

    # Curvature (pure pursuit formula).
    kappa = 2.0 * math.sin(alpha) / max(dist, 0.05)

    # Scale speed down for sharp turns and when near the goal.
    remaining = _path_length(path_xz)
    speed_curv = MAX_LINEAR_VEL * max(0.15, 1.0 - abs(kappa) * 0.4)
    speed_goal = MAX_LINEAR_VEL * min(1.0, remaining / SLOWDOWN_DIST)
    linear = min(speed_curv, speed_goal)

    angular = linear * kappa
    angular = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular))

    return linear, angular


def differential_drive(
    linear: float,
    angular: float,
    wheel_base: float = WHEEL_BASE,
) -> tuple[float, float]:
    """Convert (linear, angular) velocities to (left, right) wheel speeds."""
    left = linear - angular * wheel_base / 2.0
    right = linear + angular * wheel_base / 2.0
    return left, right


# ── Helpers ──────────────────────────────────────────────────────────────

def _path_length(path_xz: np.ndarray) -> float:
    if len(path_xz) < 2:
        return 0.0
    diffs = np.diff(path_xz, axis=0)
    return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))


def _path_cells_to_ego_xz(
    path_cells: list[tuple[int, int]],
    half_x: int,
    cam_z_row: int,
    cell_res: float,
) -> np.ndarray:
    """Convert grid-cell path to ego-frame (x, z) in metres.

    The robot sits at grid position (cam_z_row, half_x).
    """
    if not path_cells:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(path_cells, dtype=np.float32)       # (N, 2) of (row, col)
    x = (arr[:, 1] - half_x) * cell_res                # col → x  (right +)
    z = (arr[:, 0] - cam_z_row) * cell_res              # row → z  (forward +)
    return np.column_stack([x, z])


# ── Navigation state machine ────────────────────────────────────────────

class NavState:
    IDLE = "idle"
    NAVIGATING = "navigating"
    ARRIVED = "arrived"
    STUCK = "stuck"


# ── Motor-control thread ────────────────────────────────────────────────

def motor_control_thread(
    shared,                       # SharedState from live_sub
    session: zenoh.Session,
    stop_event: threading.Event,
    control_hz: float = 10.0,
):
    """Run the pure-pursuit loop, publishing velocity commands.

    Parameters
    ----------
    shared : live_sub.SharedState
        Thread-safe container with cost grid, path, and waypoint overlays.
    session : zenoh.Session
        Used to subscribe/publish Zenoh topics.
    stop_event : threading.Event
        Set to request a clean shutdown.
    control_hz : float
        Control loop frequency.
    """
    vel_pub = session.declare_publisher(VELOCITY_TOPIC)
    status_pub = session.declare_publisher(NAV_STATUS_TOPIC)
    cmd_queue: queue.Queue = queue.Queue()

    # Current navigation goal (waypoint index or None).
    nav_target: int | None = None
    nav_state = NavState.IDLE
    stuck_counter = 0

    def _on_nav_cmd(sample):
        try:
            msg = json.loads(sample.payload.to_bytes().decode())
            cmd_queue.put(msg)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    cmd_sub = session.declare_subscriber(NAV_COMMAND_TOPIC, _on_nav_cmd)
    print(f"Motor control: listening on '{NAV_COMMAND_TOPIC}'")

    def _publish_vel(linear: float, angular: float):
        left, right = differential_drive(linear, angular)
        vel_pub.put(json.dumps({
            "linear": round(linear, 4),
            "angular": round(angular, 4),
            "left": round(left, 4),
            "right": round(right, 4),
        }).encode())

    def _publish_status():
        status_pub.put(json.dumps({
            "state": nav_state,
            "target": nav_target if nav_target is not None else -1,
        }).encode())

    def _stop_motors():
        _publish_vel(0.0, 0.0)

    dt = 1.0 / control_hz

    try:
        while not stop_event.is_set():
            t0 = time.monotonic()

            # ── Process pending commands ──────────────────────────────
            while True:
                try:
                    msg = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                action = msg.get("action", "")
                if action == "go":
                    nav_target = int(msg.get("waypoint_index", 0))
                    nav_state = NavState.NAVIGATING
                    stuck_counter = 0
                    print(f"Motor control: navigating to waypoint {nav_target}")
                    _publish_status()
                elif action == "stop":
                    nav_target = None
                    nav_state = NavState.IDLE
                    _stop_motors()
                    print("Motor control: stopped")
                    _publish_status()

            # ── Run pure pursuit if navigating ────────────────────────
            if nav_state == NavState.NAVIGATING:
                path_cells, wp_cells = shared.get_overlay()

                if not path_cells:
                    # No path available — might be stuck or waypoint unreachable
                    stuck_counter += 1
                    if stuck_counter > int(3.0 * control_hz):
                        nav_state = NavState.STUCK
                        _stop_motors()
                        print("Motor control: stuck — no path available")
                        _publish_status()
                    else:
                        _stop_motors()
                    stop_event.wait(timeout=dt)
                    continue

                stuck_counter = 0
                ego_path = _path_cells_to_ego_xz(
                    path_cells,
                    shared.half_x,
                    shared.cam_z_row,
                    shared.cell_res,
                )

                # Check if we've arrived (path end is within tolerance).
                remaining = _path_length(ego_path)
                if remaining < GOAL_TOLERANCE:
                    nav_state = NavState.ARRIVED
                    _stop_motors()
                    print(f"Motor control: arrived at waypoint {nav_target}")
                    _publish_status()
                    stop_event.wait(timeout=dt)
                    continue

                result = pure_pursuit(ego_path)
                if result is None:
                    _stop_motors()
                else:
                    linear, angular = result
                    _publish_vel(linear, angular)
            else:
                # Not navigating — make sure motors are off.
                _stop_motors()

            elapsed = time.monotonic() - t0
            remaining_dt = dt - elapsed
            if remaining_dt > 0:
                stop_event.wait(timeout=remaining_dt)

    finally:
        _stop_motors()
        cmd_sub.undeclare()
        vel_pub.undeclare()
        status_pub.undeclare()
        print("Motor control thread stopped")
