"""
Frontier Exploration — Wavefront BFS algorithm for autonomous environment exploration.

HOW DIMOS EXPLORES AN ENVIRONMENT (deep dive):
===============================================

1. OCCUPANCY GRID
   The robot maintains a 2D occupancy grid (from SLAM / Nav2):
   - FREE cells (0):    navigable space
   - OCCUPIED cells (100): obstacles/walls
   - UNKNOWN cells (-1): unexplored space

2. FRONTIER DETECTION (Wavefront BFS)
   A "frontier" = a FREE cell that is adjacent to at least one UNKNOWN cell.
   These are the boundaries between what's known and what hasn't been explored.

   Algorithm (BFS from robot position):
   a) Start BFS from robot's current cell
   b) For each FREE cell popped from queue:
      - If any of its 8 neighbors is UNKNOWN → it's a frontier cell
      - Add non-UNKNOWN neighbors to the "map BFS" queue
      - Group adjacent frontier cells into frontier REGIONS
   c) Each frontier region is a candidate exploration goal

3. GOAL SELECTION
   For each frontier region:
   - info_gain = region size (more cells = more unknown space exposed)
   - distance  = Euclidean distance from robot to region centroid
   - score     = info_gain / (distance + epsilon)  [greedy nearest-new-area]

   The highest-scoring frontier that:
   - Is far enough from the robot (>safe_distance)
   - Is within reachable range (< max_explored_distance)
   - Has info_gain > threshold (not a tiny crack)
   ... is selected as the next navigation goal.

4. GOAL EXECUTION
   NavigationInterface.set_goal(pose) sends the frontier centroid as a
   Nav2 goal pose. The robot drives there via the global/local planners.

5. LOOP UNTIL DONE
   When no frontiers remain → map is fully explored.
   Track stagnation: if robot hasn't moved after `goal_timeout` seconds,
   mark that frontier as failed and try the next one.

6. INTEGRATION WITH OTA AGENT (how they work together):
   The OTA agent doesn't directly call frontier exploration. Instead:
   - navigate_with_text("unexplored area") → SpatialMemory CLIP search for
     frames with low coverage → navigates toward them
   - The frontier explorer runs as a SEPARATE module providing goals
   - The agent can OVERRIDE frontier goals with semantic navigation
   - NarrativeLedger tracks score, so agent naturally prefers frontiers (+5)

7. SCORING SYSTEM (OTA loop prompt):
   +5 for reaching a new area (frontier)
   +2 for completing a named goal
   +1 for recording a field note (update_task_ledger)
   -3 for hitting an obstacle or stalling
   -2 for stagnating (not moving for too long)
   -1 for revisiting known areas (already in SpatialMemory)

   The agent maximizes this score → emergent exploration behavior.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CellClass(IntFlag):
    """BFS classification flags for each grid cell."""
    NoInfo        = 0
    MapOpen       = 1   # Queued in map BFS
    MapClosed     = 2   # Visited in map BFS
    FrontierOpen  = 4   # Queued in frontier BFS
    FrontierClosed = 8  # Visited in frontier BFS


@dataclass
class FrontierRegion:
    cells: list[tuple[int, int]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return float(np.mean(xs)), float(np.mean(ys))

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass
class FrontierGoal:
    """A selected frontier goal in world coordinates."""
    world_x: float
    world_y: float
    info_gain: float   # Number of frontier cells (proxy for new area)
    distance: float    # Distance from robot


# ---------------------------------------------------------------------------
# Occupancy grid helpers
# ---------------------------------------------------------------------------

FREE     = 0
OCCUPIED = 100
UNKNOWN  = -1

DIRECTIONS_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIRECTIONS_8 = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]


def _is_frontier_cell(grid: np.ndarray, r: int, c: int) -> bool:
    """A cell is a frontier if it's FREE and has at least one UNKNOWN neighbor."""
    if grid[r, c] != FREE:
        return False
    rows, cols = grid.shape
    for dr, dc in DIRECTIONS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == UNKNOWN:
            return True
    return False


def _is_navigable(grid: np.ndarray, r: int, c: int) -> bool:
    return grid[r, c] == FREE


# ---------------------------------------------------------------------------
# Wavefront frontier detector
# ---------------------------------------------------------------------------

class WavefrontFrontierDetector:
    """
    Detects frontier regions in an occupancy grid using wavefront BFS.

    Based on: Yamauchi (1997) "A frontier-based approach for autonomous exploration"
    and the dimos WavefrontFrontierExplorer implementation.
    """

    def __init__(
        self,
        min_frontier_size: int = 3,
        occupancy_threshold: int = 65,
    ) -> None:
        self._min_size = min_frontier_size
        self._occ_threshold = occupancy_threshold

    def detect(
        self,
        grid: np.ndarray,          # shape (rows, cols), values: FREE=0, OCCUPIED=100, UNKNOWN=-1
        robot_row: int,
        robot_col: int,
    ) -> list[FrontierRegion]:
        """
        Run wavefront BFS from robot position and return frontier regions.

        Args:
            grid: 2D occupancy grid
            robot_row, robot_col: Robot's current cell coordinates

        Returns:
            List of FrontierRegion objects sorted by size (descending)
        """
        rows, cols = grid.shape
        classification = np.zeros((rows, cols), dtype=np.int32)
        frontiers: list[FrontierRegion] = []

        # Outer BFS: expand from robot through FREE cells
        map_queue: deque[tuple[int, int]] = deque()
        map_queue.append((robot_row, robot_col))
        classification[robot_row, robot_col] |= CellClass.MapOpen

        while map_queue:
            r, c = map_queue.popleft()
            classification[r, c] |= CellClass.MapClosed

            if _is_frontier_cell(grid, r, c):
                # Found a frontier cell — run inner BFS to collect the full region
                if not (classification[r, c] & (CellClass.FrontierOpen | CellClass.FrontierClosed)):
                    region = self._collect_frontier_region(
                        grid, classification, r, c, rows, cols
                    )
                    if region.size >= self._min_size:
                        frontiers.append(region)

            # Expand map BFS to free neighbors
            for dr, dc in DIRECTIONS_8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if classification[nr, nc] & (CellClass.MapOpen | CellClass.MapClosed):
                    continue
                if not _is_navigable(grid, nr, nc):
                    continue
                classification[nr, nc] |= CellClass.MapOpen
                map_queue.append((nr, nc))

        frontiers.sort(key=lambda f: -f.size)
        return frontiers

    def _collect_frontier_region(
        self,
        grid: np.ndarray,
        classification: np.ndarray,
        start_r: int,
        start_c: int,
        rows: int,
        cols: int,
    ) -> FrontierRegion:
        """Inner BFS to collect all cells in a contiguous frontier region."""
        region = FrontierRegion()
        queue: deque[tuple[int, int]] = deque()
        queue.append((start_r, start_c))
        classification[start_r, start_c] |= CellClass.FrontierOpen

        while queue:
            r, c = queue.popleft()
            classification[r, c] |= CellClass.FrontierClosed

            if _is_frontier_cell(grid, r, c):
                region.cells.append((r, c))

                for dr, dc in DIRECTIONS_8:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if classification[nr, nc] & (CellClass.FrontierOpen | CellClass.FrontierClosed):
                        continue
                    if grid[nr, nc] == UNKNOWN:
                        continue
                    classification[nr, nc] |= CellClass.FrontierOpen
                    queue.append((nr, nc))

        return region


# ---------------------------------------------------------------------------
# Goal selector
# ---------------------------------------------------------------------------

class FrontierGoalSelector:
    """
    Selects the best frontier goal from detected regions.

    Scoring: info_gain / (distance + 1) — prefers large nearby frontiers.
    """

    def __init__(
        self,
        safe_distance_cells: float = 3.0,
        max_distance_cells: float = 100.0,
        info_gain_threshold: float = 3.0,
    ) -> None:
        self._safe_dist = safe_distance_cells
        self._max_dist = max_distance_cells
        self._info_gain_threshold = info_gain_threshold

    def select(
        self,
        frontiers: list[FrontierRegion],
        robot_row: float,
        robot_col: float,
    ) -> Optional[tuple[float, float]]:
        """
        Select best frontier centroid in grid coordinates.
        Returns (row, col) of best frontier, or None if no valid frontier.
        """
        best_score = -1.0
        best_centroid: Optional[tuple[float, float]] = None

        for region in frontiers:
            if region.size < self._info_gain_threshold:
                continue
            cr, cc = region.centroid
            dist = ((cr - robot_row) ** 2 + (cc - robot_col) ** 2) ** 0.5
            if dist < self._safe_dist or dist > self._max_dist:
                continue
            score = region.size / (dist + 1.0)
            if score > best_score:
                best_score = score
                best_centroid = (cr, cc)

        return best_centroid


# ---------------------------------------------------------------------------
# Coordinate transform helpers
# ---------------------------------------------------------------------------

def grid_to_world(
    row: float,
    col: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> tuple[float, float]:
    """Convert grid cell (row, col) to world coordinates (x, y)."""
    world_x = origin_x + col * resolution
    world_y = origin_y + row * resolution
    return world_x, world_y


def world_to_grid(
    world_x: float,
    world_y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    rows: int,
    cols: int,
) -> Optional[tuple[int, int]]:
    """Convert world (x, y) to grid (row, col). Returns None if out of bounds."""
    col = int((world_x - origin_x) / resolution)
    row = int((world_y - origin_y) / resolution)
    if 0 <= row < rows and 0 <= col < cols:
        return row, col
    return None


# ---------------------------------------------------------------------------
# High-level Explorer
# ---------------------------------------------------------------------------

class FrontierExplorer:
    """
    Full pipeline: occupancy grid → frontier detection → goal selection → world pose.

    Usage:
        explorer = FrontierExplorer()
        goal = explorer.next_goal(
            grid=occupancy_grid_2d,  # numpy array, values: 0=free, 100=occ, -1=unknown
            robot_x=5.0, robot_y=3.0,
            origin_x=-10.0, origin_y=-10.0,
            resolution=0.05,  # meters per cell
        )
        if goal:
            navigate_to(goal.world_x, goal.world_y)
    """

    def __init__(
        self,
        min_frontier_size: int = 3,
        safe_distance_m: float = 0.5,
        max_distance_m: float = 10.0,
        info_gain_threshold: float = 3.0,
    ) -> None:
        self._detector = WavefrontFrontierDetector(min_frontier_size=min_frontier_size)
        self._selector = FrontierGoalSelector(
            info_gain_threshold=info_gain_threshold,
        )
        self._safe_dist_m = safe_distance_m
        self._max_dist_m = max_distance_m

    def next_goal(
        self,
        grid: np.ndarray,
        robot_x: float,
        robot_y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Optional[FrontierGoal]:
        """
        Find the next best exploration frontier goal.

        Returns FrontierGoal with world coordinates, or None if fully explored.
        """
        rows, cols = grid.shape

        robot_cell = world_to_grid(robot_x, robot_y, origin_x, origin_y, resolution, rows, cols)
        if robot_cell is None:
            logger.warning("Robot position out of map bounds")
            return None

        robot_row, robot_col = robot_cell
        safe_cells = self._safe_dist_m / resolution
        max_cells = self._max_dist_m / resolution

        self._selector._safe_dist = safe_cells
        self._selector._max_dist = max_cells

        frontiers = self._detector.detect(grid, robot_row, robot_col)
        logger.info(f"Found {len(frontiers)} frontier regions")

        if not frontiers:
            logger.info("No frontiers — map may be fully explored")
            return None

        best = self._selector.select(frontiers, float(robot_row), float(robot_col))
        if best is None:
            logger.warning("No valid frontier goal found (all too close/far/small)")
            return None

        brow, bcol = best
        wx, wy = grid_to_world(brow, bcol, origin_x, origin_y, resolution)
        dist = ((robot_x - wx) ** 2 + (robot_y - wy) ** 2) ** 0.5

        # Use best frontier's size for info_gain
        best_region = max(frontiers, key=lambda r: (
            r.size / (((r.centroid[0] - robot_row)**2 + (r.centroid[1] - robot_col)**2)**0.5 + 1)
        ))

        logger.info(f"Next frontier goal: world=({wx:.2f},{wy:.2f}) dist={dist:.2f}m info_gain={best_region.size}")
        return FrontierGoal(world_x=wx, world_y=wy, info_gain=best_region.size, distance=dist)

    def get_all_frontiers_world(
        self,
        grid: np.ndarray,
        robot_x: float,
        robot_y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> list[FrontierGoal]:
        """Return ALL frontier goals ranked by score (for visualization / planning)."""
        rows, cols = grid.shape
        robot_cell = world_to_grid(robot_x, robot_y, origin_x, origin_y, resolution, rows, cols)
        if robot_cell is None:
            return []
        robot_row, robot_col = robot_cell
        frontiers = self._detector.detect(grid, robot_row, robot_col)
        goals = []
        for region in frontiers:
            cr, cc = region.centroid
            wx, wy = grid_to_world(cr, cc, origin_x, origin_y, resolution)
            dist = ((robot_x - wx) ** 2 + (robot_y - wy) ** 2) ** 0.5
            goals.append(FrontierGoal(world_x=wx, world_y=wy, info_gain=region.size, distance=dist))
        goals.sort(key=lambda g: -(g.info_gain / (g.distance + 1)))
        return goals
