"""A* path planning over a 2D traversability cost grid with waypoint support."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    """A navigation waypoint in world coordinates (XZ ground plane, Y-up)."""

    x: float
    z: float
    label: str = ""


def world_to_grid(
    wx: float,
    wz: float,
    origin_x: float,
    origin_z: float,
    cell_size: float,
) -> tuple[int, int]:
    """Convert world XZ coordinates to grid (row, col) indices."""
    col = int((wx - origin_x) / cell_size)
    row = int((wz - origin_z) / cell_size)
    return row, col


def grid_to_world(
    row: int,
    col: int,
    origin_x: float,
    origin_z: float,
    cell_size: float,
) -> tuple[float, float]:
    """Convert grid (row, col) indices to world XZ coordinates (cell center)."""
    wx = origin_x + (col + 0.5) * cell_size
    wz = origin_z + (row + 0.5) * cell_size
    return wx, wz


def astar(
    cost_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    obstacle_threshold: float = 0.8,
) -> list[tuple[int, int]] | None:
    """Find the lowest-cost path on *cost_grid* from *start* to *goal*.

    Parameters
    ----------
    cost_grid : float32 [H, W] with values in [0, 1].
        0 = free, 1 = obstacle.
    start : (row, col) grid cell of the start position.
    goal : (row, col) grid cell of the goal position.
    obstacle_threshold : cells with cost >= this value are impassable.

    Returns
    -------
    A list of (row, col) grid cells from start to goal inclusive,
    or ``None`` if no path exists.
    """
    rows, cols = cost_grid.shape

    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not _in_bounds(*start) or not _in_bounds(*goal):
        return None
    if cost_grid[start] >= obstacle_threshold or cost_grid[goal] >= obstacle_threshold:
        return None

    # 8-connected neighbors: (dr, dc, move_cost_multiplier)
    _NEIGHBORS = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.414),
        (-1, 1, 1.414),
        (1, -1, 1.414),
        (1, 1, 1.414),
    ]

    # Octile distance heuristic (admissible for 8-connected grid)
    def _heuristic(r: int, c: int) -> float:
        dr = abs(r - goal[0])
        dc = abs(c - goal[1])
        return max(dr, dc) + 0.414 * min(dr, dc)

    # g-scores
    g = np.full((rows, cols), np.inf, dtype=np.float64)
    g[start] = 0.0

    # Open set: (f, tiebreak_counter, row, col)
    counter = 0
    open_set: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_set, (_heuristic(*start), counter, start[0], start[1]))

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros((rows, cols), dtype=bool)

    while open_set:
        _f, _cnt, r, c = heapq.heappop(open_set)

        if (r, c) == goal:
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path

        if closed[r, c]:
            continue
        closed[r, c] = True

        for dr, dc, move_mult in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc) or closed[nr, nc]:
                continue
            if cost_grid[nr, nc] >= obstacle_threshold:
                continue

            # Traversal cost: base move distance * (1 + cell cost)
            # This makes the planner prefer low-cost cells.
            move_cost = move_mult * (1.0 + cost_grid[nr, nc])
            tentative_g = g[r, c] + move_cost

            if tentative_g < g[nr, nc]:
                g[nr, nc] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f = tentative_g + _heuristic(nr, nc)
                counter += 1
                heapq.heappush(open_set, (f, counter, nr, nc))

    return None  # No path found


def plan_waypoint_route(
    cost_grid: np.ndarray,
    waypoints: list[Waypoint],
    origin_x: float,
    origin_z: float,
    cell_size: float,
    obstacle_threshold: float = 0.8,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Plan a route through an ordered list of waypoints using A*.

    Runs A* between each consecutive pair of waypoints and concatenates
    the segments into a single path.

    Returns
    -------
    full_path : list of (row, col) grid cells for the complete route.
        Empty if any segment is unreachable.
    waypoint_cells : list of (row, col) for each waypoint on the grid.
    """
    if len(waypoints) < 2:
        wp_cells = [
            world_to_grid(wp.x, wp.z, origin_x, origin_z, cell_size)
            for wp in waypoints
        ]
        return [], wp_cells

    waypoint_cells = [
        world_to_grid(wp.x, wp.z, origin_x, origin_z, cell_size)
        for wp in waypoints
    ]

    full_path: list[tuple[int, int]] = []
    for i in range(len(waypoint_cells) - 1):
        segment = astar(
            cost_grid,
            waypoint_cells[i],
            waypoint_cells[i + 1],
            obstacle_threshold=obstacle_threshold,
        )
        if segment is None:
            return [], waypoint_cells

        # Avoid duplicating the junction point between segments
        if full_path:
            segment = segment[1:]
        full_path.extend(segment)

    return full_path, waypoint_cells


def path_to_world_coords(
    path: list[tuple[int, int]],
    origin_x: float,
    origin_z: float,
    cell_size: float,
) -> np.ndarray:
    """Convert a grid path to world XZ coordinates.

    Returns an (N, 2) float array of [x, z] world positions.
    """
    if not path:
        return np.empty((0, 2), dtype=np.float32)
    coords = np.array(
        [grid_to_world(r, c, origin_x, origin_z, cell_size) for r, c in path],
        dtype=np.float32,
    )
    return coords
