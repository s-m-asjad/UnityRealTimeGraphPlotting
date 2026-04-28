# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn an Intel RealSense D455 in Isaac Lab, point it at a cube, and capture RGB-D.

The script reproduces the RealSense D455 sensor described in the Isaac Sim docs
(https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_camera_depth_sensors.html):

* The D455 USD model is loaded from Nucleus as a visual mount.
* A Camera sensor is attached as a sibling under the same parent Xform so the body
  and the camera share a single world pose (no positional drift between them).
* That parent Xform is moved to a look-at pose targeting a cube on the ground, so
  the visible D455 model and the camera rays are co-located.
* A second viewport window is opened that is pinned to the camera so its live RGB
  feed is visible alongside the perspective view in the Isaac Sim editor.
* Only the *last* RGB + depth frame is written to disk on exit.

Usage:

.. code-block:: bash

    # Just run it. (Cameras are enabled automatically.)
    python camera.py

    # Headless still works, in which case the only output is the saved RGB-D frame.
    python camera.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RealSense D455 RGB-D capture demo.")
parser.add_argument(
    "--num_frames",
    type=int,
    default=0,
    help="Number of simulation steps to run before exiting. 0 means run until the window is closed.",
)
parser.add_argument(
    "--no_d455_model",
    action="store_true",
    help="Skip spawning the RealSense D455 USD visual model (use only the camera sensor).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera rendering must be enabled before AppLauncher boots Kit; force it on so the
# script works with a plain `python camera.py` invocation.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# RealSense D455 specification (from the Isaac Sim camera/depth sensor docs).
##

# Path to the D455 USD on Nucleus. If your Isaac Sim version stores it elsewhere,
# update this constant (e.g. f"{ISAAC_NUCLEUS_DIR}/Sensors/Realsense/D455/rsd455.usd").
RSD455_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd"

# Intrinsics (color stream). Values are taken from the D455 table in the docs.
D455_FOCAL_LENGTH = 1.93  # mm
D455_HORIZONTAL_APERTURE = 3.896  # mm
D455_VERTICAL_APERTURE = 2.453  # mm
D455_FOCUS_DISTANCE = 0.6  # m
D455_F_STOP = 2.0
D455_CLIPPING_RANGE = (0.01, 1.0e6)

# Output resolution. The native max for the D455 color stream is 1280x800; we use
# 1280x720 here (the depth-stream resolution) so RGB and depth share the same shape.
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Scene placement.
CUBE_POSITION = (0.0, 0.0, 0.05)  # 10 cm cube sitting on the ground.
D455_MOUNT_POSITION = (0.6, 0.0, 0.4)  # Where to put the D455 mount in world coords.

# Prim paths.
D455_MOUNT_PATH = "/World/D455"
D455_BODY_PATH = f"{D455_MOUNT_PATH}/Body"
D455_CAMERA_PATH = f"{D455_MOUNT_PATH}/CameraSensor"


def look_at_quat_world(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Return a `(w, x, y, z)` quaternion that aligns the local +X axis with the look direction.

    Matches Isaac Sim's "world" camera convention: +X forward, +Y left, +Z up. With this
    orientation applied to a parent Xform, both the D455 body USD (whose native lens
    direction is +X) and a Camera child configured with ``convention="world"`` and an
    identity offset will face the target.
    """
    eye_arr = np.asarray(eye, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    up_arr = np.asarray(world_up, dtype=np.float64)

    forward = target_arr - eye_arr
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    forward /= norm

    left = np.cross(up_arr, forward)
    left_norm = np.linalg.norm(left)
    if left_norm < 1e-6:
        # Forward is parallel to world-up — pick any perpendicular vector.
        left = np.array([0.0, 1.0, 0.0])
    else:
        left /= left_norm
    up = np.cross(forward, left)

    # Columns are local axes (X, Y, Z) expressed in world coordinates.
    rot = np.column_stack([forward, left, up])

    # Convert rotation matrix -> quaternion (w, x, y, z). Standard Shepperd's method.
    trace = rot.trace()
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return (float(w), float(x), float(y), float(z))


def design_scene() -> dict:
    """Populate the stage with a ground plane, light, cube, and the D455 camera."""
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.95), metallic=0.1
            ),
            semantic_tags=[("class", "cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POSITION),
    )
    cube = RigidObject(cfg=cube_cfg)

    # Compute the look-at orientation (world convention: +X is forward) so that both
    # the body USD and the camera point at the cube once parented under D455_MOUNT_PATH.
    mount_orientation = look_at_quat_world(D455_MOUNT_POSITION, CUBE_POSITION)
    sim_utils.create_prim(
        D455_MOUNT_PATH,
        "Xform",
        translation=D455_MOUNT_POSITION,
        orientation=mount_orientation,
    )

    if not args_cli.no_d455_model:
        d455_model = sim_utils.UsdFileCfg(
            usd_path=RSD455_USD_PATH,
            # The D455 USD ships with rigid-body / collision schemas baked in. Disabling
            # them keeps the asset purely cosmetic so the camera doesn't get pushed around.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
        )
        try:
            d455_model.func(D455_BODY_PATH, d455_model)
        except Exception as exc:
            print(f"[WARN] Failed to load RealSense D455 USD model from '{RSD455_USD_PATH}': {exc}")
            print("[WARN] Continuing with just the camera sensor (use --no_d455_model to silence).")

    # Camera as a child of the mount Xform. With convention="world" and an identity
    # offset, the camera's forward axis follows the mount's +X — exactly the same
    # direction the body USD lens points — so they cannot drift apart.
    camera_cfg = CameraCfg(
        prim_path=D455_CAMERA_PATH,
        update_period=0.0,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=D455_FOCAL_LENGTH,
            focus_distance=D455_FOCUS_DISTANCE,
            f_stop=D455_F_STOP,
            horizontal_aperture=D455_HORIZONTAL_APERTURE,
            vertical_aperture=D455_VERTICAL_APERTURE,
            clipping_range=D455_CLIPPING_RANGE,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    camera = Camera(cfg=camera_cfg)

    return {"cube": cube, "camera": camera}


def open_camera_viewport(camera_prim_path: str) -> None:
    """Open a second viewport window pinned to the camera so it streams live in the editor.

    No-op (with a warning) if running headless or if the viewport extension isn't loaded.
    """
    if args_cli.headless:
        return
    try:
        from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
    except ImportError as exc:
        print(f"[WARN] Could not import omni.kit.viewport.utility ({exc}); skipping viewport setup.")
        return

    try:
        vp_window = create_viewport_window(
            window_name="RealSense D455", width=720, height=1080
        )
        vp_window.viewport_api.camera_path = camera_prim_path
        print(f"[INFO] Opened 'RealSense D455' viewport pinned to {camera_prim_path}.")
    except Exception as exc:
        print(f"[WARN] Failed to create extra viewport ({exc}); falling back to active viewport.")
        active_vp = get_active_viewport()
        if active_vp is not None:
            active_vp.camera_path = camera_prim_path


def save_rgbd(rgb: np.ndarray, depth: np.ndarray, output_dir: str) -> None:
    """Write the final RGB image, raw depth array, and a colorized depth preview to disk."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    plt.imsave(os.path.join(output_dir, "rgb.png"), rgb)

    np.save(os.path.join(output_dir, "depth.npy"), depth)
    depth_vis = np.where(np.isfinite(depth), depth, 0.0)
    depth_vis = np.clip(depth_vis, 0.0, 5.0)
    plt.imsave(os.path.join(output_dir, "depth.png"), depth_vis, cmap="turbo")

    print(f"[INFO] Saved last RGB-D frame to: {output_dir}")


def run_simulator(sim: sim_utils.SimulationContext, scene: dict) -> None:
    """Step the simulator and remember the latest RGB-D frame."""
    camera: Camera = scene["camera"]

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "d455")

    # Warm-up steps — the renderer needs a few frames before textures are valid.
    for _ in range(5):
        sim.step()
        camera.update(dt=sim.get_physics_dt())

    latest_rgb: np.ndarray | None = None
    latest_depth: np.ndarray | None = None
    step = 0
    try:
        while simulation_app.is_running():
            sim.step()
            camera.update(dt=sim.get_physics_dt())

            if "rgb" in camera.data.output and "distance_to_image_plane" in camera.data.output:
                latest_rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                latest_depth = camera.data.output["distance_to_image_plane"][0].detach().cpu().numpy()

            step += 1
            if args_cli.num_frames and step >= args_cli.num_frames:
                break
    finally:
        if latest_rgb is not None and latest_depth is not None:
            save_rgbd(latest_rgb, latest_depth, output_dir)
        else:
            print("[WARN] No camera frames were captured; nothing to save.")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.0])

    scene = design_scene()

    sim.reset()

    open_camera_viewport(D455_CAMERA_PATH)

    print("[INFO] Setup complete. Starting simulation...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
