from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import UInt8, UInt16
from monopriors.depth_completion_models.base_completion_depth import (
    CompletionDepthPrediction,
)
from monopriors.depth_completion_models.prompt_da import PromptDAPredictor
from numpy import ndarray
from simplecv.camera_parameters import Intrinsics, rescale_intri
from simplecv.data.polycam import (
    DepthConfidenceLevel,
    PolycamData,
    PolycamDataset,
    load_polycam_data,
)
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from tqdm import tqdm


@dataclass
class PDAPolycamConfig:
    polycam_zip_path: Path
    rr_config: RerunTyroConfig
    max_image_size: int = 1008
    max_depth_range_meter: float = 4.0
    depth_fusion_resolution: float = 0.04
    log_incremental_mesh: bool = True
    cost_map_resolution: float = 0.05
    """Grid cell size in meters for the 2D cost map."""
    max_step_height: float = 0.15
    """Maximum traversable height difference (meters) within a cell before it becomes an obstacle."""
    robot_height: float = 0.5
    """Vertical clearance required by the robot (meters). Cells with height range exceeding this are obstacles."""


def log_polycam_data(
    parent_path: Path,
    polycam_data: PolycamData,
    depth_pred: UInt16[ndarray, "h w"],
    rescale_factor: int = 1,
) -> None:
    cam_path: Path = parent_path / "cam"
    pinhole_path: Path = cam_path / "pinhole"

    rgb: UInt8[np.ndarray, "h w 3"] = polycam_data.rgb_hw3
    depth: UInt16[np.ndarray, "h w"] = polycam_data.depth_hw
    confidence: UInt8[np.ndarray, "h w"] = polycam_data.confidence_hw

    # resize images to be half the size
    target_height: int = rgb.shape[0] // rescale_factor
    target_width: int = rgb.shape[1] // rescale_factor
    rgb_resized = cv2.resize(rgb, (target_width, target_height))
    depth_resized = cv2.resize(depth, (target_width, target_height))
    confidence_resized = cv2.resize(confidence, (target_width, target_height))
    depth_pred_resized = cv2.resize(depth_pred, (target_width, target_height))

    # rescale intrinsics to match the image size
    rescaled_intrinsics: Intrinsics = rescale_intri(
        camera_intrinsics=polycam_data.pinhole_params.intrinsics,
        target_height=target_height,
        target_width=target_width,
    )

    polycam_data.pinhole_params.intrinsics = rescaled_intrinsics

    log_pinhole(camera=polycam_data.pinhole_params, cam_log_path=cam_path)
    rr.log(f"{pinhole_path}/image", rr.Image(rgb_resized).compress(jpeg_quality=75))
    rr.log(f"{pinhole_path}/confidence", rr.SegmentationImage(confidence_resized))
    rr.log(f"{pinhole_path}/arkit_depth", rr.DepthImage(depth_resized, meter=1000))
    rr.log(f"{pinhole_path}/pred_depth", rr.DepthImage(depth_pred_resized, meter=1000))


def compute_cost_map_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    cell_size: float = 0.05,
    max_step_height: float = 0.15,
    robot_height: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build a 2D traversability cost grid from a 3D mesh.

    Projects mesh vertices onto the XZ ground plane (Y-up / RUB convention).
    For each grid cell, cost is derived from:
      - Height variance: rough terrain increases cost
      - Step height: max - min height in the cell; exceeding max_step_height = obstacle
      - Surface normals: steep faces increase cost

    Returns:
        cost_grid: float32 [H, W] with values in [0, 1]. 0 = free, 1 = obstacle.
        height_grid: float32 [H, W] minimum height per cell (for visualization).
        origin_x: world X coordinate of grid column 0.
        origin_z: world Z coordinate of grid row 0.
    """
    vertices = np.asarray(mesh.vertices)  # (N, 3) — X right, Y up, Z back
    if len(vertices) == 0:
        return np.ones((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32), 0.0, 0.0

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)  # (N, 3)

    xs, ys, zs = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Grid extents
    x_min, x_max = xs.min(), xs.max()
    z_min, z_max = zs.min(), zs.max()

    cols = max(1, int(np.ceil((x_max - x_min) / cell_size)) + 1)
    rows = max(1, int(np.ceil((z_max - z_min) / cell_size)) + 1)

    # Bin each vertex into a grid cell
    ci = np.clip(((xs - x_min) / cell_size).astype(int), 0, cols - 1)
    ri = np.clip(((zs - z_min) / cell_size).astype(int), 0, rows - 1)

    # Per-cell accumulators
    height_min = np.full((rows, cols), np.inf, dtype=np.float64)
    height_max = np.full((rows, cols), -np.inf, dtype=np.float64)
    height_sum = np.zeros((rows, cols), dtype=np.float64)
    height_sq_sum = np.zeros((rows, cols), dtype=np.float64)
    normal_y_sum = np.zeros((rows, cols), dtype=np.float64)
    count = np.zeros((rows, cols), dtype=np.int64)

    # Use np.add.at for unbuffered accumulation
    np.minimum.at(height_min, (ri, ci), ys)
    np.maximum.at(height_max, (ri, ci), ys)
    np.add.at(height_sum, (ri, ci), ys)
    np.add.at(height_sq_sum, (ri, ci), ys ** 2)
    np.add.at(normal_y_sum, (ri, ci), np.abs(normals[:, 1]))
    np.add.at(count, (ri, ci), 1)

    observed = count > 0

    # --- Cost components ---

    # 1. Step height cost: height range in cell vs max_step_height
    height_range = np.where(observed, height_max - height_min, 0.0)
    step_cost = np.clip(height_range / max_step_height, 0.0, 1.0)

    # 2. Roughness cost: height standard deviation normalized
    mean_h = np.where(observed, height_sum / count, 0.0)
    var_h = np.where(observed, height_sq_sum / count - mean_h ** 2, 0.0)
    var_h = np.maximum(var_h, 0.0)  # numerical safety
    std_h = np.sqrt(var_h)
    roughness_cost = np.clip(std_h / (max_step_height * 0.5), 0.0, 1.0)

    # 3. Slope cost: average surface normal deviation from vertical
    #    normal_y = 1 means flat ground, 0 means vertical wall
    avg_normal_y = np.where(observed, normal_y_sum / count, 0.0)
    slope_cost = np.clip(1.0 - avg_normal_y, 0.0, 1.0)

    # Combined cost (weighted blend)
    cost = np.where(
        observed,
        0.4 * step_cost + 0.3 * roughness_cost + 0.3 * slope_cost,
        1.0,  # unobserved = obstacle
    ).astype(np.float32)

    # Hard obstacle: cells where height range exceeds robot clearance
    cost[height_range > robot_height] = 1.0

    height_grid = np.where(observed, height_min, 0.0).astype(np.float32)

    return cost, height_grid, float(x_min), float(z_min)


def filter_depth(
    depth_mm: UInt16[np.ndarray, "h w"],
    confidence: UInt8[np.ndarray, "h w"],
    confidence_threshold: DepthConfidenceLevel,
    max_depth_meter: float,
) -> UInt16[np.ndarray, "h w"]:
    filtered_depth_mm: UInt16[np.ndarray, "h w"] = depth_mm.copy()
    filtered_depth_mm[confidence < confidence_threshold] = 0
    filtered_depth_mm[depth_mm > max_depth_meter * 1000] = 0

    return filtered_depth_mm


def create_blueprint(parent_log_path: Path) -> rrb.Blueprint:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.Spatial2DView(origin=parent_log_path / "cam" / "pinhole" / "pred_depth"),
                rrb.Spatial2DView(origin=f"{parent_log_path}/cost_map"),
            ),
            column_shares=[20, 9],
        ),
        collapse_panels=True,
    )
    return blueprint


def log_cost_map(
    entity_path: str | Path,
    cost_rgb: np.ndarray,
    *,
    origin_x: float,
    origin_z: float,
    cell_size: float,
) -> None:
    rr.log(
        str(entity_path),
        rr.Transform3D(
            translation=[origin_x, 0.0, origin_z],
            scale=[cell_size, cell_size, 1.0],
        ),
        rr.Image(cost_rgb),
    )


def pda_polycam_inference(
    config: PDAPolycamConfig,
) -> None:
    parent_log_path: Path = Path("world")
    rr.log("/", rr.ViewCoordinates.RUB, static=True)

    blueprint: rrb.Blueprint = create_blueprint(parent_log_path)
    rr.send_blueprint(blueprint=blueprint)
    polycam_zip_path: Path = config.polycam_zip_path
    polycam_dataset: PolycamDataset = load_polycam_data(polycam_zip_or_directory_path=polycam_zip_path)

    pred_fuser = Open3DFuser(
        fusion_resolution=config.depth_fusion_resolution,
        max_fusion_depth=config.max_depth_range_meter,
    )

    model = PromptDAPredictor(device="cuda", model_type="large", max_size=config.max_image_size)
    pbar = tqdm(polycam_dataset, desc="Inferring", total=len(polycam_dataset))
    polycam_data: PolycamData
    for frame_idx, polycam_data in enumerate(pbar):
        rr.set_time("frame_idx", sequence=frame_idx)
        # convert image data to tensor
        depth_pred: CompletionDepthPrediction = model(
            rgb=polycam_data.rgb_hw3, prompt_depth=polycam_data.original_depth_hw
        )

        # filter depthmaps based on confidence, only keep with max confidence
        pred_filtered_depth_mm: UInt16[np.ndarray, "h w"] = filter_depth(
            depth_mm=depth_pred.depth_mm,
            confidence=polycam_data.confidence_hw,
            confidence_threshold=DepthConfidenceLevel.MEDIUM,
            max_depth_meter=config.max_depth_range_meter,
        )

        # fuse the predicted depth and the ground truth depth
        pred_fuser.fuse_frames(
            depth_hw=pred_filtered_depth_mm,
            K_33=polycam_data.pinhole_params.intrinsics.k_matrix,
            cam_T_world_44=polycam_data.pinhole_params.extrinsics.cam_T_world,
            rgb_hw3=polycam_data.rgb_hw3,
        )

        log_polycam_data(
            parent_path=parent_log_path,
            polycam_data=polycam_data,
            depth_pred=depth_pred.depth_mm,
            rescale_factor=1,
        )

        if config.log_incremental_mesh:
            pred_mesh = pred_fuser.get_mesh()
            pred_mesh.compute_vertex_normals()

            rr.log(
                f"{parent_log_path}/pred_mesh",
                rr.Mesh3D(
                    vertex_positions=pred_mesh.vertices,
                    triangle_indices=pred_mesh.triangles,
                    vertex_normals=pred_mesh.vertex_normals,
                    vertex_colors=pred_mesh.vertex_colors,
                ),
            )

            # Update cost map from current mesh state
            cost_grid, _height_grid, origin_x, origin_z = compute_cost_map_from_mesh(
                mesh=pred_mesh,
                cell_size=config.cost_map_resolution,
                max_step_height=config.max_step_height,
                robot_height=config.robot_height,
            )
            ch, cw = cost_grid.shape
            cost_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
            cost_rgb[:, :, 0] = (cost_grid * 255).astype(np.uint8)
            cost_rgb[:, :, 1] = ((1 - cost_grid) * 255).astype(np.uint8)
            log_cost_map(
                f"{parent_log_path}/cost_map",
                cost_rgb,
                origin_x=origin_x,
                origin_z=origin_z,
                cell_size=config.cost_map_resolution,
            )

    # Final mesh and cost map
    pred_mesh = pred_fuser.get_mesh()
    pred_mesh.compute_vertex_normals()

    rr.log(
        f"{parent_log_path}/pred_mesh",
        rr.Mesh3D(
            vertex_positions=pred_mesh.vertices,
            triangle_indices=pred_mesh.triangles,
            vertex_normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.vertex_colors,
        ),
    )

    cost_grid, _height_grid, origin_x, origin_z = compute_cost_map_from_mesh(
        mesh=pred_mesh,
        cell_size=config.cost_map_resolution,
        max_step_height=config.max_step_height,
        robot_height=config.robot_height,
    )
    ch, cw = cost_grid.shape
    cost_rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
    cost_rgb[:, :, 0] = (cost_grid * 255).astype(np.uint8)
    cost_rgb[:, :, 1] = ((1 - cost_grid) * 255).astype(np.uint8)
    log_cost_map(
        f"{parent_log_path}/cost_map",
        cost_rgb,
        origin_x=origin_x,
        origin_z=origin_z,
        cell_size=config.cost_map_resolution,
    )
    rr.log(
        f"{parent_log_path}/cost_map/metadata",
        rr.TextLog(
            f"origin=({origin_x:.2f}, {origin_z:.2f}) "
            f"cell_size={config.cost_map_resolution}m "
            f"grid={cw}x{ch} cells"
        ),
    )
