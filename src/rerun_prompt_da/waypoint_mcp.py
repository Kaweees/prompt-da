"""MCP server for managing navigation waypoints.

Publishes waypoint commands to the planning thread in live_sub via Zenoh.
Run as:  prompt-da-mcp                (stdio transport for Claude Code)

Tools:
    add_waypoint(x, z, label?)  — add a waypoint at world coords (meters)
    remove_waypoint(index)      — remove waypoint by index
    clear_waypoints()           — remove all waypoints
    list_waypoints()            — show current waypoint list
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import zenoh
from mcp.server.fastmcp import Context, FastMCP

WAYPOINT_TOPIC = "nav/waypoints"


@dataclass
class WaypointEntry:
    x: float
    z: float
    label: str


@dataclass
class AppState:
    """Tracks waypoints locally and publishes changes over Zenoh."""

    session: zenoh.Session
    publisher: zenoh.Publisher
    waypoints: list[WaypointEntry] = field(default_factory=list)

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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppState]:
    conf = zenoh.Config()
    session = zenoh.open(conf)
    publisher = session.declare_publisher(WAYPOINT_TOPIC)
    print(f"Waypoint MCP: publishing on '{WAYPOINT_TOPIC}'")
    try:
        yield AppState(session=session, publisher=publisher)
    finally:
        publisher.undeclare()
        session.close()


mcp = FastMCP(
    "prompt-da-waypoints",
    lifespan=app_lifespan,
)


@mcp.tool()
def add_waypoint(x: float, z: float, label: str = "", ctx: Context[AppState] = None) -> str:
    """Add a navigation waypoint at world coordinates (meters).

    x: lateral position (positive = right)
    z: forward distance (positive = ahead of camera)
    label: optional human-readable name
    """
    state = ctx.request_context.lifespan_context
    wp = state.add(x, z, label)
    return f"Added {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Total: {len(state.waypoints)}"


@mcp.tool()
def remove_waypoint(index: int, ctx: Context[AppState] = None) -> str:
    """Remove a waypoint by its index (0-based)."""
    state = ctx.request_context.lifespan_context
    wp = state.remove(index)
    if wp is None:
        return f"Invalid index {index}. Have {len(state.waypoints)} waypoints."
    return f"Removed {wp.label} at ({wp.x:.2f}, {wp.z:.2f}). Remaining: {len(state.waypoints)}"


@mcp.tool()
def clear_waypoints(ctx: Context[AppState] = None) -> str:
    """Remove all waypoints."""
    state = ctx.request_context.lifespan_context
    n = len(state.waypoints)
    state.clear()
    return f"Cleared {n} waypoints."


@mcp.tool()
def list_waypoints(ctx: Context[AppState] = None) -> str:
    """List all current waypoints with their indices."""
    state = ctx.request_context.lifespan_context
    if not state.waypoints:
        return "No waypoints set."
    lines = []
    for i, wp in enumerate(state.waypoints):
        lines.append(f"  [{i}] {wp.label}: ({wp.x:.2f}, {wp.z:.2f})")
    return f"{len(state.waypoints)} waypoints:\n" + "\n".join(lines)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
