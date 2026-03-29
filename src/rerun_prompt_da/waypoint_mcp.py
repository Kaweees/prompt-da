"""MCP server for managing navigation waypoints, direct motor control,
objectives, and scene understanding for the comma body robot.

Publishes waypoint and navigation commands via Zenoh.  Supports direct
motor control via the bodynav (accel, steer) model, which the comma-side
body_driver converts to testJoystick cereal messages.

Run as:  prompt-da-mcp                (stdio transport for Claude Code)

Tools:
    -- Waypoint management --
    add_waypoint(x, z, label?)   -- add a waypoint at world coords (meters)
    remove_waypoint(index)       -- remove waypoint by index
    clear_waypoints()            -- remove all waypoints
    list_waypoints()             -- show current waypoint list

    -- Scene understanding --
    capture_scene()              -- grab camera frame + depth, save to disk

    -- A* Navigation (DGX pure-pursuit) --
    navigate_to_waypoint(index)  -- drive the robot to a waypoint
    navigate_all_waypoints()     -- drive through all waypoints in order
    stop_navigation()            -- halt the pure-pursuit controller
    get_nav_status()             -- check navigation progress

    -- Direct motor control (bodynav) --
    drive_forward(speed, steer)  -- drive forward
    drive_reverse(speed, steer)  -- drive in reverse
    turn_left(steer_magnitude)   -- turn left in place
    turn_right(steer_magnitude)  -- turn right in place
    stop_motors()                -- zero all motor output

    -- Objectives (bodynav agent) --
    set_explore_objective(instructions?)  -- free exploration
    set_prompt_objective(prompt)          -- freeform goal
    clear_objective()                     -- clear objective, robot stops
    get_objective_status()               -- read current objective

    -- Persistent waypoints (bodynav) --
    save_waypoint(name, x, y, theta)     -- save named waypoint to disk
    delete_saved_waypoint(name)          -- remove saved waypoint
    list_saved_waypoints()               -- list all saved waypoints
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import cv2
import numpy as np
import zenoh
from mcp.server.fastmcp import Context, FastMCP

from rerun_prompt_da.hardware import k_matrix
from rerun_prompt_da.live_sub import (
    build_costmap,
    DEFAULT_COSTMAP_RESOLUTION,
)
from rerun_prompt_da.motor_controller import (
    NAV_COMMAND_TOPIC,
    NAV_STATUS_TOPIC,
    VELOCITY_TOPIC,
)
from rerun_prompt_da.path_planning import astar
from rerun_prompt_da.zenoh_codec import (
    CAMERA_TOPICS,
    DEPTH_TOPIC,
    decode_depth,
    decode_frame,
)

WAYPOINT_TOPIC = "nav/waypoints"

# Directory where scene captures are saved for Claude to read.
SCENE_DIR = os.path.join(tempfile.gettempdir(), "prompt_da_scenes")

# Bodynav objective + waypoint persistence paths.
OBJECTIVE_FILE = os.path.expanduser("~/.bodynav/objective.json")
WAYPOINT_STORE_FILE = os.path.expanduser("~/.bodynav/waypoints.json")

# ---------------------------------------------------------------------------
# Bodynav conversion constants (must match openpilot/tools/bodynav/body_driver.py)
# ---------------------------------------------------------------------------
_BODY_MAX_LINEAR = 0.3   # m/s from pure pursuit
_BODY_MAX_ACCEL = 0.4    # joystick accel range
_BODY_MAX_STEER = 1.0
_BODY_ANGULAR_SCALE = 1.2
_BODY_MAX_SPEED = 0.6


def _accel_steer_to_velocity(accel: float, steer: float) -> tuple[float, float]:
    """Convert bodynav (accel, steer) to (linear, angular) velocity.

    Inverse of bodynav body_driver.py velocity_to_joystick().
    """
    linear = (accel / _BODY_MAX_ACCEL) * _BODY_MAX_LINEAR if _BODY_MAX_ACCEL > 0 else 0.0
    angular = -(steer / _BODY_MAX_STEER) * _BODY_ANGULAR_SCALE if _BODY_MAX_STEER > 0 else 0.0
    return linear, angular


# ---------------------------------------------------------------------------
# Objective file helpers (inlined from openpilot/tools/bodynav/objectives.py)
# ---------------------------------------------------------------------------

def _write_objective(obj: dict | None):
    """Atomically write an objective to ~/.bodynav/objective.json."""
    os.makedirs(os.path.dirname(OBJECTIVE_FILE), exist_ok=True)
    if obj is None:
        try:
            os.unlink(OBJECTIVE_FILE)
        except OSError:
            pass
        return
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(OBJECTIVE_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, OBJECTIVE_FILE)
    except BaseException:
        os.unlink(tmp)
        raise


def _read_objective() -> dict | None:
    if not os.path.exists(OBJECTIVE_FILE):
        return None
    try:
        with open(OBJECTIVE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Persistent waypoint store (inlined from openpilot/tools/bodynav/waypoints.py)
# ---------------------------------------------------------------------------

def _wp_store_load() -> dict:
    if not os.path.exists(WAYPOINT_STORE_FILE):
        return {"version": 1, "waypoints": {}}
    try:
        with open(WAYPOINT_STORE_FILE, "r") as f:
            data = json.load(f)
        if "waypoints" not in data:
            return {"version": 1, "waypoints": {}}
        return data
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "waypoints": {}}


def _wp_store_save(data: dict):
    os.makedirs(os.path.dirname(WAYPOINT_STORE_FILE), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(WAYPOINT_STORE_FILE), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, WAYPOINT_STORE_FILE)
    except BaseException:
        os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# AppState
# ---------------------------------------------------------------------------

@dataclass
class WaypointEntry:
    x: float
    z: float
    label: str


class AppState:
    """Tracks waypoints locally and publishes changes over Zenoh."""

    def __init__(
        self,
        session: zenoh.Session,
        publisher: zenoh.Publisher,
        vel_publisher: zenoh.Publisher,
    ):
        self.session = session
        self.publisher = publisher
        self.vel_publisher = vel_publisher
        self.waypoints: list[WaypointEntry] = []

        # Latest camera frame + depth (populated by background subscribers).
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_ts: float = 0.0
        self._latest_depth_mm: np.ndarray | None = None
        self._latest_depth_ts: float = 0.0
        self._frame_lock = threading.Lock()

        # Latest nav status from the motor controller.
        self._nav_status: dict = {"state": "idle", "target": -1}
        self._status_lock = threading.Lock()

        # Direct motor control state (repeater thread publishes at 10Hz).
        self._direct_cmd: tuple[float, float] | None = None  # (linear, angular)
        self._direct_lock = threading.Lock()

        # Zenoh subscribers (kept alive for the server lifetime).
        self._subs: list = []

    # -- waypoint management --------------------------------------------------

    def _publish(self, msg: dict):
        self.publisher.put(json.dumps(msg).encode())

    def add(self, x: float, z: float, label: str) -> WaypointEntry:
        wp = WaypointEntry(x=x, z=z, label=label or f"WP{len(self.waypoints)}")
        self.waypoints.append(wp)
        self._publish({"action": "add", "x": x, "z": z, "label": wp.label})
        return wp

    def remove(self, index: int) -> WaypointEntry | None:
        if 0 <= index < len(self.waypoints):
            wp = self.waypoints.pop(index)
            self._publish({"action": "remove", "index": index})
            return wp
        return None

    def clear(self):
        self.waypoints.clear()
        self._publish({"action": "clear"})

    # -- scene capture helpers ------------------------------------------------

    def update_frame(self, ts: float, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_frame_ts = ts

    def update_depth(self, ts: float, depth_mm: np.ndarray):
        with self._frame_lock:
            self._latest_depth_mm = depth_mm
            self._latest_depth_ts = ts

    def get_latest_frame(self) -> tuple[float, np.ndarray | None]:
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame_ts, self._latest_frame.copy()
            return 0.0, None

    def get_latest_depth(self) -> tuple[float, np.ndarray | None]:
        with self._frame_lock:
            if self._latest_depth_mm is not None:
                return self._latest_depth_ts, self._latest_depth_mm.copy()
            return 0.0, None

    # -- nav status helpers ---------------------------------------------------

    def update_nav_status(self, status: dict):
        with self._status_lock:
            self._nav_status = dict(status)

    def get_nav_status(self) -> dict:
        with self._status_lock:
            return dict(self._nav_status)

    # -- direct motor control -------------------------------------------------

    def set_direct_cmd(self, linear: float, angular: float):
        """Set direct velocity command (repeater thread publishes at 10Hz)."""
        # Stop pure-pursuit first to avoid conflicts.
        self.session.put(NAV_COMMAND_TOPIC, json.dumps({"action": "stop"}).encode())
        with self._direct_lock:
            self._direct_cmd = (linear, angular)

    def clear_direct_cmd(self):
        """Stop direct control and zero motors."""
        with self._direct_lock:
            self._direct_cmd = None
        self._publish_velocity(0.0, 0.0)

    def get_direct_cmd(self) -> tuple[float, float] | None:
        with self._direct_lock:
            return self._direct_cmd

    def _publish_velocity(self, linear: float, angular: float):
        """Publish a velocity command on the Zenoh velocity topic."""
        wheel_base = 0.235
        left = linear - angular * wheel_base / 2.0
        right = linear + angular * wheel_base / 2.0
        self.vel_publisher.put(json.dumps({
            "linear": round(linear, 4),
            "angular": round(angular, 4),
            "left": round(left, 4),
            "right": round(right, 4),
        }).encode())

    # -- A* path planning from depth ------------------------------------------

    def plan_path_to(
        self, target_x: float, target_z: float,
    ) -> tuple[bool, str]:
        """Build a costmap from latest depth and run A* to (target_x, target_z).

        Returns (success, message).
        """
        _, depth_mm = self.get_latest_depth()
        if depth_mm is None:
            return False, "No depth data available -- cannot plan a path."

        K = k_matrix(depth_mm.shape[1])
        _, cost_norm = build_costmap(depth_mm, K)

        grid_size = cost_norm.shape[0]
        cell_res = DEFAULT_COSTMAP_RESOLUTION
        half_x = grid_size // 2
        cam_z_row = int(grid_size * 0.15)

        start = (cam_z_row, half_x)
        goal_col = int(target_x / cell_res + half_x)
        goal_row = int(target_z / cell_res + cam_z_row)
        goal = (
            max(0, min(grid_size - 1, goal_row)),
            max(0, min(grid_size - 1, goal_col)),
        )

        path = astar(cost_norm, start, goal)
        if path is None:
            return False, (
                f"A* found no path from robot to ({target_x:.2f}, {target_z:.2f}). "
                "The target may be blocked by obstacles."
            )

        return True, f"A* path planned: {len(path)} cells from robot to target."


# ---------------------------------------------------------------------------
# Direct control repeater thread
# ---------------------------------------------------------------------------

def _direct_control_thread(
    state: AppState,
    stop_event: threading.Event,
    hz: float = 10.0,
):
    """Publish active direct control command at a fixed rate."""
    dt = 1.0 / hz
    while not stop_event.is_set():
        cmd = state.get_direct_cmd()
        if cmd is not None:
            linear, angular = cmd
            state._publish_velocity(linear, angular)
        stop_event.wait(timeout=dt)


# ---------------------------------------------------------------------------
# Lifespan (Zenoh + background subscribers + repeater thread)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    os.makedirs(SCENE_DIR, exist_ok=True)

    conf = zenoh.Config()
    session = zenoh.open(conf)
    publisher = session.declare_publisher(WAYPOINT_TOPIC)
    vel_publisher = session.declare_publisher(VELOCITY_TOPIC)
    print(f"Waypoint MCP: publishing on '{WAYPOINT_TOPIC}', '{VELOCITY_TOPIC}'")

    state = AppState(session=session, publisher=publisher, vel_publisher=vel_publisher)

    # Subscribe to camera frames.
    def _on_frame(sample):
        try:
            ts, gray, _seq = decode_frame(sample.payload.to_bytes())
            state.update_frame(ts, gray)
        except Exception:
            pass

    for _cam, topic in CAMERA_TOPICS.items():
        sub = session.declare_subscriber(topic, _on_frame)
        state._subs.append(sub)
        print(f"Waypoint MCP: subscribed to '{topic}'")

    # Subscribe to depth.
    def _on_depth(sample):
        try:
            ts, depth_mm = decode_depth(sample.payload.to_bytes())
            state.update_depth(ts, depth_mm)
        except Exception:
            pass

    depth_sub = session.declare_subscriber(DEPTH_TOPIC, _on_depth)
    state._subs.append(depth_sub)
    print(f"Waypoint MCP: subscribed to '{DEPTH_TOPIC}'")

    # Subscribe to nav status.
    def _on_status(sample):
        try:
            msg = json.loads(sample.payload.to_bytes().decode())
            state.update_nav_status(msg)
        except Exception:
            pass

    status_sub = session.declare_subscriber(NAV_STATUS_TOPIC, _on_status)
    state._subs.append(status_sub)

    # Start direct control repeater thread.
    stop_event = threading.Event()
    repeater = threading.Thread(
        target=_direct_control_thread, args=(state, stop_event), daemon=True,
    )
    repeater.start()

    try:
        yield state
    finally:
        stop_event.set()
        repeater.join(timeout=2.0)
        state.clear_direct_cmd()
        for sub in state._subs:
            sub.undeclare()
        vel_publisher.undeclare()
        publisher.undeclare()
        session.close()


mcp = FastMCP(
    "prompt-da-waypoints",
    lifespan=app_lifespan,
)


# =========================================================================
# Waypoint tools
# =========================================================================

@mcp.tool()
def add_waypoint(x: float, z: float, ctx: Context, label: str = "") -> str:
    """Add a navigation waypoint at world coordinates (meters).

    x: lateral position (positive = right)
    z: forward distance (positive = ahead of camera)
    label: optional human-readable name
    """
    state: AppState = ctx.request_context.lifespan_context
    wp = state.add(x, z, label)
    return f"Added {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Total: {len(state.waypoints)}"


@mcp.tool()
def remove_waypoint(index: int, ctx: Context) -> str:
    """Remove a waypoint by its index (0-based)."""
    state: AppState = ctx.request_context.lifespan_context
    wp = state.remove(index)
    if wp is None:
        return f"Invalid index {index}. Have {len(state.waypoints)} waypoints."
    return f"Removed {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Remaining: {len(state.waypoints)}"


@mcp.tool()
def clear_waypoints(ctx: Context) -> str:
    """Remove all waypoints."""
    state: AppState = ctx.request_context.lifespan_context
    n = len(state.waypoints)
    state.clear()
    return f"Cleared {n} waypoints."


@mcp.tool()
def list_waypoints(ctx: Context) -> str:
    """List all current waypoints with their indices."""
    state: AppState = ctx.request_context.lifespan_context
    if not state.waypoints:
        return "No waypoints set."
    lines = []
    for i, wp in enumerate(state.waypoints):
        lines.append(f"  [{i}] {wp.label}: ({wp.x:.2f}, {wp.z:.2f})")
    return f"{len(state.waypoints)} waypoints:\n" + "\n".join(lines)


# =========================================================================
# Scene understanding tools
# =========================================================================

@mcp.tool()
def capture_scene(ctx: Context) -> str:
    """Capture the current camera view and depth map.

    Saves the latest camera frame and depth visualization to disk and
    returns their file paths so you can read and analyze the images.
    Use this to see what the robot sees before deciding where to place
    waypoints.
    """
    state: AppState = ctx.request_context.lifespan_context

    ts_frame, frame = state.get_latest_frame()
    ts_depth, depth_mm = state.get_latest_depth()

    if frame is None:
        return (
            "No camera frame available yet. Make sure prompt-da-sub and "
            "prompt-da-pub are running."
        )

    results = []

    # Save camera frame.
    frame_path = os.path.join(SCENE_DIR, "scene_camera.jpg")
    cv2.imwrite(frame_path, frame)
    results.append(f"Camera frame saved to: {frame_path}")
    results.append(f"  Timestamp: {ts_frame:.3f}")
    results.append(f"  Size: {frame.shape[1]}x{frame.shape[0]}")

    # Save depth visualization if available.
    if depth_mm is not None:
        valid = depth_mm > 0
        depth_vis = np.zeros_like(depth_mm, dtype=np.uint8)
        if np.any(valid):
            d_min = float(depth_mm[valid].min())
            d_max = float(depth_mm[valid].max())
            if d_max > d_min:
                norm = ((depth_mm.astype(np.float32) - d_min) / (d_max - d_min) * 255)
                depth_vis = np.clip(norm, 0, 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            depth_color[~valid] = 0
        else:
            depth_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        depth_path = os.path.join(SCENE_DIR, "scene_depth.jpg")
        cv2.imwrite(depth_path, depth_color)
        results.append(f"Depth map saved to: {depth_path}")
        results.append(f"  Depth timestamp: {ts_depth:.3f}")
        if np.any(valid):
            results.append(
                f"  Range: {d_min / 1000:.2f}m -- {d_max / 1000:.2f}m"
            )

        # Provide a text summary of depth zones for spatial reasoning.
        h, w = depth_mm.shape
        zones = {
            "left": depth_mm[:, : w // 3],
            "center": depth_mm[:, w // 3 : 2 * w // 3],
            "right": depth_mm[:, 2 * w // 3 :],
        }
        results.append("Depth summary (median distance per zone):")
        for name, zone in zones.items():
            zv = zone[zone > 0]
            if len(zv) > 0:
                results.append(f"  {name}: {float(np.median(zv)) / 1000:.2f}m")
            else:
                results.append(f"  {name}: no depth data")
    else:
        results.append("No depth data available yet.")

    results.append(
        "\nRead the image files above to see what the robot sees, then "
        "use add_waypoint to mark interesting locations."
    )
    return "\n".join(results)


# =========================================================================
# A* Navigation tools (DGX pure-pursuit)
# =========================================================================

@mcp.tool()
def navigate_to_waypoint(index: int, ctx: Context) -> str:
    """Plan an A* path from the robot to the waypoint and start driving.

    Builds a costmap from the latest depth, runs A* to find a collision-free
    path, then commands the motor controller to follow it via pure pursuit.
    Use get_nav_status() to monitor progress.
    """
    state: AppState = ctx.request_context.lifespan_context

    # Clear any active direct control to avoid conflicts.
    state.clear_direct_cmd()

    if index < 0 or index >= len(state.waypoints):
        return (
            f"Invalid waypoint index {index}. "
            f"Have {len(state.waypoints)} waypoints. Use list_waypoints() to see them."
        )

    wp = state.waypoints[index]

    ok, plan_msg = state.plan_path_to(wp.x, wp.z)
    if not ok:
        return f"Cannot navigate to {wp.label}: {plan_msg}"

    cmd = {"action": "go", "waypoint_index": index}
    state.session.put(NAV_COMMAND_TOPIC, json.dumps(cmd).encode())

    return (
        f"Navigation started: driving to {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). "
        f"{plan_msg} Use get_nav_status() to check progress."
    )


@mcp.tool()
def navigate_all_waypoints(ctx: Context) -> str:
    """Plan an A* path through all waypoints in order and start driving.

    Verifies that a path exists from the robot through each waypoint
    sequentially, then starts navigation from the first waypoint.
    The robot will follow the full path via pure pursuit.
    """
    state: AppState = ctx.request_context.lifespan_context

    # Clear any active direct control.
    state.clear_direct_cmd()

    if not state.waypoints:
        return "No waypoints set. Use add_waypoint() first."

    _, depth_mm = state.get_latest_depth()
    if depth_mm is None:
        return "No depth data available -- cannot plan a path."

    K = k_matrix(depth_mm.shape[1])
    _, cost_norm = build_costmap(depth_mm, K)

    grid_size = cost_norm.shape[0]
    cell_res = DEFAULT_COSTMAP_RESOLUTION
    half_x = grid_size // 2
    cam_z_row = int(grid_size * 0.15)

    all_points = [(cam_z_row, half_x)]
    for wp in state.waypoints:
        col = max(0, min(grid_size - 1, int(wp.x / cell_res + half_x)))
        row = max(0, min(grid_size - 1, int(wp.z / cell_res + cam_z_row)))
        all_points.append((row, col))

    total_cells = 0
    for i in range(len(all_points) - 1):
        segment = astar(cost_norm, all_points[i], all_points[i + 1])
        if segment is None:
            wp_label = state.waypoints[i].label if i > 0 else "robot"
            next_label = state.waypoints[min(i, len(state.waypoints) - 1)].label
            return (
                f"A* failed between {wp_label} and {next_label}. "
                "Some waypoints may be blocked by obstacles."
            )
        total_cells += len(segment)

    cmd = {"action": "go", "waypoint_index": 0}
    state.session.put(NAV_COMMAND_TOPIC, json.dumps(cmd).encode())

    wp_names = ", ".join(wp.label for wp in state.waypoints)
    return (
        f"Full route planned through {len(state.waypoints)} waypoints "
        f"({wp_names}): {total_cells} total A* cells. "
        f"Navigation started. Use get_nav_status() to monitor."
    )


@mcp.tool()
def stop_navigation(ctx: Context) -> str:
    """Immediately stop the pure-pursuit navigation controller."""
    state: AppState = ctx.request_context.lifespan_context
    cmd = {"action": "stop"}
    state.session.put(NAV_COMMAND_TOPIC, json.dumps(cmd).encode())
    return "Stop command sent. Pure-pursuit controller halted."


@mcp.tool()
def get_nav_status(ctx: Context) -> str:
    """Check the current navigation status.

    Returns the state (idle, navigating, arrived, stuck) and target waypoint.
    """
    state: AppState = ctx.request_context.lifespan_context
    status = state.get_nav_status()
    nav_state = status.get("state", "unknown")
    target = status.get("target", -1)

    if nav_state == "idle":
        return "Robot is idle. Use navigate_to_waypoint(index) to start moving."
    elif nav_state == "navigating":
        if 0 <= target < len(state.waypoints):
            wp = state.waypoints[target]
            return f"Navigating to {wp.label} at ({wp.x:.2f}, {wp.z:.2f})..."
        return f"Navigating to waypoint {target}..."
    elif nav_state == "arrived":
        if 0 <= target < len(state.waypoints):
            wp = state.waypoints[target]
            return f"Arrived at {wp.label} ({wp.x:.2f}, {wp.z:.2f})!"
        return f"Arrived at waypoint {target}."
    elif nav_state == "stuck":
        return (
            "Robot is stuck -- no viable path to the target. "
            "Try clearing obstacles or choosing a different waypoint."
        )
    return f"Navigation state: {nav_state}"


# =========================================================================
# Direct motor control tools (bodynav)
# =========================================================================

@mcp.tool()
def drive_forward(ctx: Context, speed: float = 0.3, steer: float = 0.0) -> str:
    """Drive forward at a given speed with optional steering bias.

    The robot will keep driving until stop_motors() is called or another
    movement command is issued.

    speed: forward speed in m/s (0.0 to 0.6, recommended 0.2-0.3 indoors)
    steer: steering bias (-1.0 = full left, 0.0 = straight, 1.0 = full right)
    """
    state: AppState = ctx.request_context.lifespan_context
    accel = min(abs(speed), _BODY_MAX_SPEED)
    steer = max(-_BODY_MAX_STEER, min(_BODY_MAX_STEER, steer))
    linear, angular = _accel_steer_to_velocity(accel, steer)
    state.set_direct_cmd(linear, angular)
    return f"Driving forward: speed={accel:.2f} m/s, steer={steer:.2f}. Call stop_motors() to halt."


@mcp.tool()
def drive_reverse(ctx: Context, speed: float = 0.2, steer: float = 0.0) -> str:
    """Drive in reverse at a given speed with optional steering bias.

    speed: reverse speed in m/s (0.0 to 0.6)
    steer: steering bias (-1.0 = full left, 0.0 = straight, 1.0 = full right)
    """
    state: AppState = ctx.request_context.lifespan_context
    accel = -min(abs(speed), _BODY_MAX_SPEED)
    steer = max(-_BODY_MAX_STEER, min(_BODY_MAX_STEER, steer))
    linear, angular = _accel_steer_to_velocity(accel, steer)
    state.set_direct_cmd(linear, angular)
    return f"Driving reverse: speed={abs(accel):.2f} m/s, steer={steer:.2f}. Call stop_motors() to halt."


@mcp.tool()
def turn_left(ctx: Context, steer_magnitude: float = 0.8) -> str:
    """Turn left in place.

    steer_magnitude: how hard to turn (0.1 to 1.0)
    """
    state: AppState = ctx.request_context.lifespan_context
    mag = max(0.1, min(1.0, abs(steer_magnitude)))
    linear, angular = _accel_steer_to_velocity(0.0, -mag)
    state.set_direct_cmd(linear, angular)
    return f"Turning left: magnitude={mag:.2f}. Call stop_motors() to halt."


@mcp.tool()
def turn_right(ctx: Context, steer_magnitude: float = 0.8) -> str:
    """Turn right in place.

    steer_magnitude: how hard to turn (0.1 to 1.0)
    """
    state: AppState = ctx.request_context.lifespan_context
    mag = max(0.1, min(1.0, abs(steer_magnitude)))
    linear, angular = _accel_steer_to_velocity(0.0, mag)
    state.set_direct_cmd(linear, angular)
    return f"Turning right: magnitude={mag:.2f}. Call stop_motors() to halt."


@mcp.tool()
def stop_motors(ctx: Context) -> str:
    """Immediately stop all motor output.

    Halts both direct motor control and pure-pursuit navigation.
    """
    state: AppState = ctx.request_context.lifespan_context
    state.clear_direct_cmd()
    state.session.put(NAV_COMMAND_TOPIC, json.dumps({"action": "stop"}).encode())
    return "All motors stopped."


# =========================================================================
# Objective tools (bodynav agent)
# =========================================================================

@mcp.tool()
def set_explore_objective(ctx: Context, instructions: str = "") -> str:
    """Set a free exploration objective for the bodynav agent.

    The comma-side Claude agent will autonomously explore the environment,
    avoiding obstacles and building spatial understanding.

    instructions: optional guidance (e.g. "find the door", "stay in this room")
    """
    obj = {
        "type": "explore",
        "instructions": instructions or "Roam around freely, avoid obstacles, explore the environment.",
        "ts": time.monotonic(),
    }
    _write_objective(obj)
    return f"Explore objective set: {obj['instructions']}"


@mcp.tool()
def set_prompt_objective(prompt: str, ctx: Context) -> str:
    """Set a freeform natural language objective for the bodynav agent.

    The comma-side Claude agent will interpret and pursue this goal.

    prompt: natural language instruction (e.g. "go to the kitchen and look around")
    """
    obj = {
        "type": "prompt",
        "prompt": prompt,
        "ts": time.monotonic(),
    }
    _write_objective(obj)
    return f"Prompt objective set: {prompt}"


@mcp.tool()
def clear_objective(ctx: Context) -> str:
    """Clear the current objective. The bodynav agent will stop."""
    _write_objective(None)
    return "Objective cleared. Bodynav agent will stop."


@mcp.tool()
def get_objective_status(ctx: Context) -> str:
    """Read the current bodynav objective."""
    obj = _read_objective()
    if obj is None:
        return "No objective set."
    obj_type = obj.get("type", "unknown")
    if obj_type == "explore":
        return f"Explore: {obj.get('instructions', '')}"
    elif obj_type == "prompt":
        return f"Prompt: {obj.get('prompt', '')}"
    elif obj_type == "waypoint":
        name = obj.get("name", "?")
        return f"Waypoint: {name} ({obj.get('x', 0):.2f}, {obj.get('y', 0):.2f})"
    elif obj_type == "patrol":
        names = [w.get("name", "?") for w in obj.get("waypoints", [])]
        return f"Patrol: {' -> '.join(names)}"
    return f"Objective: {json.dumps(obj)}"


# =========================================================================
# Persistent waypoint tools (bodynav)
# =========================================================================

@mcp.tool()
def save_waypoint(name: str, x: float, y: float, ctx: Context, theta: float = 0.0) -> str:
    """Save a named waypoint to persistent storage (~/.bodynav/waypoints.json).

    These are reusable across sessions, unlike the ephemeral costmap waypoints.

    name: human-readable name (e.g. "kitchen", "front_door")
    x: lateral position in meters
    y: forward position in meters
    theta: heading in radians (default 0.0)
    """
    data = _wp_store_load()
    data["waypoints"][name] = {
        "x": round(x, 4),
        "y": round(y, 4),
        "theta": round(theta, 4),
        "ts": time.time(),
    }
    _wp_store_save(data)
    return f"Saved waypoint '{name}' at ({x:.2f}, {y:.2f}, theta={theta:.2f})"


@mcp.tool()
def delete_saved_waypoint(name: str, ctx: Context) -> str:
    """Remove a named waypoint from persistent storage."""
    data = _wp_store_load()
    if name not in data["waypoints"]:
        available = sorted(data["waypoints"].keys())
        return f"Waypoint '{name}' not found. Available: {', '.join(available) or 'none'}"
    del data["waypoints"][name]
    _wp_store_save(data)
    return f"Deleted waypoint '{name}'."


@mcp.tool()
def list_saved_waypoints(ctx: Context) -> str:
    """List all persistently saved waypoints from ~/.bodynav/waypoints.json."""
    data = _wp_store_load()
    wps = data["waypoints"]
    if not wps:
        return "No saved waypoints."
    lines = []
    for name, wp in sorted(wps.items()):
        lines.append(f"  {name}: ({wp['x']:.2f}, {wp['y']:.2f}, theta={wp.get('theta', 0):.2f})")
    return f"{len(wps)} saved waypoints:\n" + "\n".join(lines)


# =========================================================================
# Entry point
# =========================================================================

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
