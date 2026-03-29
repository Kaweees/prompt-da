"""MCP server for managing navigation waypoints and scene understanding.

Publishes waypoint and navigation commands via Zenoh.
Run as:  prompt-da-mcp                (stdio transport for Claude Code)

Tools:
    add_waypoint(x, z, label?)   — add a waypoint at world coords (meters)
    remove_waypoint(index)       — remove waypoint by index
    clear_waypoints()            — remove all waypoints
    list_waypoints()             — show current waypoint list
    capture_scene()              — grab camera frame + depth, save to disk
    navigate_to_waypoint(index)  — drive the robot to a waypoint
    stop_navigation()            — halt the robot
    get_nav_status()             — check navigation progress
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import zenoh
from mcp.server.fastmcp import Context, FastMCP

from rerun_prompt_da.motor_controller import NAV_COMMAND_TOPIC, NAV_STATUS_TOPIC
from rerun_prompt_da.zenoh_codec import (
    CAMERA_TOPICS,
    DEPTH_TOPIC,
    decode_depth,
    decode_frame,
)

WAYPOINT_TOPIC = "nav/waypoints"

# Directory where scene captures are saved for Claude to read.
SCENE_DIR = os.path.join(tempfile.gettempdir(), "prompt_da_scenes")


@dataclass
class WaypointEntry:
    x: float
    z: float
    label: str


class AppState:
    """Tracks waypoints locally and publishes changes over Zenoh."""

    def __init__(self, session: zenoh.Session, publisher: zenoh.Publisher):
        self.session = session
        self.publisher = publisher
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

        # Zenoh subscribers (kept alive for the server lifetime).
        self._subs: list = []

    # ── waypoint management ──────────────────────────────────────────

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

    # ── scene capture helpers ────────────────────────────────────────

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

    # ── nav status helpers ───────────────────────────────────────────

    def update_nav_status(self, status: dict):
        with self._status_lock:
            self._nav_status = dict(status)

    def get_nav_status(self) -> dict:
        with self._status_lock:
            return dict(self._nav_status)


# ── Lifespan (Zenoh + background subscribers) ────────────────────────────

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    os.makedirs(SCENE_DIR, exist_ok=True)

    conf = zenoh.Config()
    session = zenoh.open(conf)
    publisher = session.declare_publisher(WAYPOINT_TOPIC)
    print(f"Waypoint MCP: publishing on '{WAYPOINT_TOPIC}'")

    state = AppState(session=session, publisher=publisher)

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

    try:
        yield state
    finally:
        for sub in state._subs:
            sub.undeclare()
        publisher.undeclare()
        session.close()


mcp = FastMCP(
    "prompt-da-waypoints",
    lifespan=app_lifespan,
)


# ── Waypoint tools ───────────────────────────────────────────────────────

@mcp.tool()
def add_waypoint(x: float, z: float, ctx: Context, label: str = "") -> str:
    """Add a navigation waypoint at world coordinates (meters).

    x: lateral position (positive = right)
    z: forward distance (positive = ahead of camera)
    label: optional human-readable name
    """
    state = ctx.request_context.lifespan_context
    wp = state.add(x, z, label)
    return f"Added {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Total: {len(state.waypoints)}"


@mcp.tool()
def remove_waypoint(index: int, ctx: Context) -> str:
    """Remove a waypoint by its index (0-based)."""
    state = ctx.request_context.lifespan_context
    wp = state.remove(index)
    if wp is None:
        return f"Invalid index {index}. Have {len(state.waypoints)} waypoints."
    return f"Removed {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Remaining: {len(state.waypoints)}"


@mcp.tool()
def clear_waypoints(ctx: Context) -> str:
    """Remove all waypoints."""
    state = ctx.request_context.lifespan_context
    n = len(state.waypoints)
    state.clear()
    return f"Cleared {n} waypoints."


@mcp.tool()
def list_waypoints(ctx: Context) -> str:
    """List all current waypoints with their indices."""
    state = ctx.request_context.lifespan_context
    if not state.waypoints:
        return "No waypoints set."
    lines = []
    for i, wp in enumerate(state.waypoints):
        lines.append(f"  [{i}] {wp.label}: ({wp.x:.2f}, {wp.z:.2f})")
    return f"{len(state.waypoints)} waypoints:\n" + "\n".join(lines)


# ── Scene understanding tools ────────────────────────────────────────────

@mcp.tool()
def capture_scene(ctx: Context) -> str:
    """Capture the current camera view and depth map.

    Saves the latest camera frame and depth visualization to disk and
    returns their file paths so you can read and analyze the images.
    Use this to see what the robot sees before deciding where to place
    waypoints.
    """
    state = ctx.request_context.lifespan_context

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
        # Normalize depth to 0-255 for visualization.
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
                f"  Range: {d_min / 1000:.2f}m — {d_max / 1000:.2f}m"
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


# ── Navigation tools ─────────────────────────────────────────────────────

@mcp.tool()
def navigate_to_waypoint(index: int, ctx: Context) -> str:
    """Drive the robot to the waypoint at the given index.

    The robot will follow the A* planned path using pure pursuit control.
    Use get_nav_status() to monitor progress.
    """
    state = ctx.request_context.lifespan_context

    if index < 0 or index >= len(state.waypoints):
        return (
            f"Invalid waypoint index {index}. "
            f"Have {len(state.waypoints)} waypoints. Use list_waypoints() to see them."
        )

    wp = state.waypoints[index]

    # Publish navigation command over Zenoh.
    cmd = {"action": "go", "waypoint_index": index}
    state.session.put(NAV_COMMAND_TOPIC, json.dumps(cmd).encode())

    return (
        f"Navigation started: driving to {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). "
        f"Use get_nav_status() to check progress."
    )


@mcp.tool()
def stop_navigation(ctx: Context) -> str:
    """Immediately stop the robot."""
    state = ctx.request_context.lifespan_context
    cmd = {"action": "stop"}
    state.session.put(NAV_COMMAND_TOPIC, json.dumps(cmd).encode())
    return "Stop command sent. Robot should halt."


@mcp.tool()
def get_nav_status(ctx: Context) -> str:
    """Check the current navigation status.

    Returns the state (idle, navigating, arrived, stuck) and target waypoint.
    """
    state = ctx.request_context.lifespan_context
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
            "Robot is stuck — no viable path to the target. "
            "Try clearing obstacles or choosing a different waypoint."
        )
    return f"Navigation state: {nav_state}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
