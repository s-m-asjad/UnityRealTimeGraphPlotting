#!/usr/bin/env python3
"""Geometric gripper–bin check for ``grasp_sim`` YAML grasps (object inside a fixed box).

Grasp root poses in the file are **object-relative** (same convention as
``compute_relative_pos_and_rot_kernel``). The scene uses **zero gravity**, a
**kinematic** mug with **collision disabled** (so only gripper–bin contacts matter),
and a normal **articulated** gripper (dynamic links, zero gravity).

Bin walls are **kinematic** cuboids under ``/Box/``. If USD enables CCD on those prims,
PhysX logs that CCD is not supported on kinematic bodies and **ignores CCD** (harmless).
We still attempt to turn off per-prim CCD flags on the bin after spawn when the schema exposes them.

By default, failure is detected **geometrically** without trusting USD stage bounds after
PhysX (Fabric often leaves ``BBoxCache`` at bind pose). Each collision **mesh** in the
gripper USD is processed once: an axis-aligned box in **mesh local space** (from vertices)
has its eight corners transformed into the **parent link body frame** using rest-pose USD
transforms. At runtime, corners are mapped to **world** with ``ArticulationData.body_link_pose_w``
from Isaac Lab (same FK PhysX uses), then world AABB vs the five bin cuboids is tested.

Optional ``--bin_hit_method contact`` restores the older **filtered contact-sensor** path
(``force_matrix_w`` vs ``/Box/*``), including ``--bin_contact_settle_substeps`` for that mode.

By default, joints are set to the stored **closed** grasp only (``cspace_position``).
Use ``--joint_motion_samples N`` with **N ≥ 2** to also linearly interpolate in joint space
from ``pregrasp_cspace_position`` to ``cspace_position`` (finger motion while closing).

After a run, results are written as **two** YAML files next to ``--grasp_file``:
``{same_stem}_bin_clear.yaml`` (``in_bin_no_wall_collision: true`` only) and
``{same_stem}_bin_wall_hit.yaml`` (false only). No combined output file.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import warp as wp
import yaml

from graspgen_utils import (
    add_arg_to_group,
    add_isaac_lab_args_if_needed,
    print_blue,
    print_green,
    print_purple,
    print_red,
    print_yellow,
    register_argument_group,
    save_yaml,
    start_isaac_lab_if_needed,
    str_to_bool,
)
from gripper import add_gripper_args, apply_gripper_configuration, collect_gripper_args
from object import ObjectConfig, add_object_args, collect_object_args

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ContactSensor filter patterns (one per bin piece) — must match ``_spawn_bin_walls_for_env`` prim names.
_BOX_CONTACT_FILTER_PRIM_PATHS: Tuple[str, ...] = (
    "{ENV_REGEX_NS}/Box/Base",
    "{ENV_REGEX_NS}/Box/W1",
    "{ENV_REGEX_NS}/Box/W2",
    "{ENV_REGEX_NS}/Box/W3",
    "{ENV_REGEX_NS}/Box/W4",
)

# Newtons (filtered contact force magnitude); above this ⇒ some gripper link touches the bin.
_DEFAULT_BIN_CONTACT_FORCE_EPS = 1e-6
# PhysX substeps after each pose write (contact mode only). Geometry mode defaults to 1 so
# the gripper is not pushed through the bin floor by extra dynamics solves.
_DEFAULT_BIN_CONTACT_SETTLE_SUBSTEPS = 1
_DEFAULT_BIN_HIT_METHOD = "geometry"


def _physics_step_scene(sim, scene, sim_dt: float, *, render: bool, substeps: int = 1) -> None:
    """Run PhysX ``substeps`` times and refresh Isaac Lab scene buffers (contacts, articulations, sensors).

    When ``render`` is True, only the **last** substep requests a render (avoids redundant viewport work).
    """
    n = max(1, int(substeps))
    for i in range(n):
        sim.step(render=bool(render and i == n - 1))
        scene.update(sim_dt)


def _apply_bin_wall_preview_colors(env_path: str, rgb: Tuple[float, float, float]) -> None:
    """Set UsdPreviewSurface ``inputs:diffuseColor`` on each bin cuboid (Isaac Lab ``.../geometry/material/Shader``)."""
    try:
        import omni.usd
        from pxr import Gf
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    col = Gf.Vec3f(r, g, b)
    for name in ("Base", "W1", "W2", "W3", "W4"):
        for shader_path in (
            f"{env_path}/Box/{name}/geometry/material/Shader",
            f"{env_path}/Box/{name}/geometry/material/shader",
        ):
            prim = stage.GetPrimAtPath(shader_path)
            if not prim.IsValid():
                continue
            attr = prim.GetAttribute("inputs:diffuseColor")
            if attr and attr.IsValid():
                attr.Set(col)
                break


def _try_disable_ccd_on_bin_prims(scene, n_env: int) -> None:
    """Best-effort: clear CCD flags on spawned bin rigid bodies to reduce PhysX kinematic+CCD log spam."""
    try:
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    for i in range(n_env):
        env_path = scene.env_prim_paths[i] if hasattr(scene, "env_prim_paths") and i < len(scene.env_prim_paths) else f"/World/envs/env_{i}"
        for name in ("Base", "W1", "W2", "W3", "W4"):
            prim = stage.GetPrimAtPath(f"{env_path}/Box/{name}")
            if not prim.IsValid() or not prim.IsA(UsdGeom.Cube):
                continue
            try:
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    api = PhysxSchema.PhysxRigidBodyAPI(prim)
                    attr = api.GetEnableCCDAttr()
                    if attr.IsValid():
                        attr.Set(False)
                for p in Usd.PrimRange(prim):
                    if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                        capi = PhysxSchema.PhysxCollisionAPI(p)
                        getter = getattr(capi, "GetEnableCCDAttr", None)
                        if getter is None:
                            continue
                        cattr = getter()
                        if cattr.IsValid():
                            cattr.Set(False)
            except Exception:
                continue


def _aligned_world_aabb6(prim_path: str) -> Tuple[float, float, float, float, float, float]:
    """World-space axis-aligned bounds (min xyz, max xyz) for a USD prim."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for AABB: {prim_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bbox = cache.ComputeWorldBound(prim)
    box_range = world_bbox.ComputeAlignedRange()
    mn = box_range.GetMin()
    mx = box_range.GetMax()
    return (
        float(mn[0]),
        float(mn[1]),
        float(mn[2]),
        float(mx[0]),
        float(mx[1]),
        float(mx[2]),
    )


def _aabb_overlap_6(a: Tuple[float, float, float, float, float, float], b: Tuple[float, float, float, float, float, float]) -> bool:
    """True if two world AABBs (min_xyz, max_xyz each) intersect with positive volume overlap."""
    ax0, ay0, az0, ax1, ay1, az1 = a
    bx0, by0, bz0, bx1, by1, bz1 = b
    return ax1 >= bx0 and bx1 >= ax0 and ay1 >= by0 and by1 >= ay0 and az1 >= bz0 and bz1 >= az0


def _load_gripper_collision_mesh_corners_body_frame(gripper_usd_path: str) -> List[Tuple[str, np.ndarray]]:
    """For each collision-enabled Mesh under the articulation root, return (link_name, (8,3) corners in link frame).

    Corners are the eight vertices of the mesh-local axis-aligned box (min/max of mesh points).
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(gripper_usd_path, Usd.Stage.LoadAll)
    if not stage:
        return []
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        return []
    root_path = root.GetPath().pathString
    prefix = root_path + "/"
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out: List[Tuple[str, np.ndarray]] = []

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        pth = prim.GetPath().pathString
        if not pth.startswith(prefix):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        capi = UsdPhysics.CollisionAPI(prim)
        ce = capi.GetCollisionEnabledAttr()
        if ce.IsValid() and not ce.Get():
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts or len(pts) < 1:
            continue
        arr = np.asarray([(float(p[0]), float(p[1]), float(p[2])) for p in pts], dtype=np.float64)
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        rel = pth[len(prefix) :]
        link_name = rel.split("/")[0]
        body_prim = stage.GetPrimAtPath(f"{root_path}/{link_name}")
        if not body_prim.IsValid():
            continue
        tw_m = xf_cache.GetLocalToWorldTransform(prim)
        tw_b = xf_cache.GetLocalToWorldTransform(body_prim)
        t_bm = tw_b.GetInverse() * tw_m
        corners_b: List[Tuple[float, float, float]] = []
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    pb = t_bm.Transform(Gf.Vec3d(float(cx), float(cy), float(cz)))
                    corners_b.append((float(pb[0]), float(pb[1]), float(pb[2])))
        out.append((link_name, np.asarray(corners_b, dtype=np.float64)))
    return out


def _world_aabb_from_body_corners(
    body_pose_w: torch.Tensor,
    body_name_to_idx: dict,
    mesh_specs: List[Tuple[str, torch.Tensor]],
    bin_aabbs: List[Tuple[float, float, float, float, float, float]],
) -> bool:
    """True if any collision-mesh world AABB (via link pose × rest mesh→link corners) hits any bin AABB."""
    for link_name, corners_b in mesh_specs:
        bi = body_name_to_idx.get(link_name)
        if bi is None:
            continue
        pos = body_pose_w[bi, :3]
        quat = body_pose_w[bi, 3:7]
        qexp = quat.unsqueeze(0).expand(8, -1)
        cw = _quat_rotate_vec_wxyz(qexp, corners_b) + pos.unsqueeze(0)
        amin = cw.amin(dim=0)
        amax = cw.amax(dim=0)
        a6 = (float(amin[0]), float(amin[1]), float(amin[2]), float(amax[0]), float(amax[1]), float(amax[2]))
        for b6 in bin_aabbs:
            if _aabb_overlap_6(a6, b6):
                return True
    return False


def _env_gripper_bin_contact_from_filtered_sensors(
    scene, env_i: int, eps: float, contact_sensor_keys: List[str]
) -> bool:
    """True if any monitored Robot link's filtered ``force_matrix_w`` (vs ``/Box/*``) exceeds ``eps``."""
    for key in contact_sensor_keys:
        fm = scene[key].data.force_matrix_w
        if fm is None:
            continue
        # L2 norm per (body, filter); also max |component| so tiny one-axis normals are not missed.
        v = fm[env_i]
        t = torch.maximum(torch.norm(v, dim=-1), torch.amax(torch.abs(v), dim=-1))
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        if bool((t > eps).any().item()):
            return True
    return False


def resolve_usd_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for base in (os.getcwd(), _REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "graspgen")):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return os.path.normpath(cand)
    return path


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, both (..., 4) in w,x,y,z order."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate_vec_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors v (..., 3) by unit quaternion q (..., 4) wxyz."""
    w, x, y, z = q_wxyz.unbind(-1)
    uv = torch.stack(
        (
            y * v[..., 2] - z * v[..., 1],
            z * v[..., 0] - x * v[..., 2],
            x * v[..., 1] - y * v[..., 0],
        ),
        dim=-1,
    )
    uuv = torch.stack(
        (
            y * uv[..., 2] - z * uv[..., 1],
            z * uv[..., 0] - x * uv[..., 2],
            x * uv[..., 1] - y * uv[..., 0],
        ),
        dim=-1,
    )
    return v + 2.0 * (w.unsqueeze(-1) * uv + uuv)


def compose_object_relative_grasps_world(
    env_origins: torch.Tensor,
    grasp_pos_obj: torch.Tensor,
    grasp_quat_xyzw: torch.Tensor,
) -> torch.Tensor:
    """World gripper root pose for Isaac Lab ``write_root_pose_to_sim``: (N,7) = pos_xyz + quat_wxyz.

    Object pose per env: translation ``env_origins``, identity rotation.
    Stored grasp: translation and quaternion (xyzw) of gripper in the object frame.
    """
    device = env_origins.device
    dtype = env_origins.dtype
    n = grasp_pos_obj.shape[0]
    p_obj = env_origins.to(device=device, dtype=dtype)
    q_obj = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).expand(n, 4)
    q_rel = torch.cat(
        (
            grasp_quat_xyzw[:, 3:4],
            grasp_quat_xyzw[:, 0:1],
            grasp_quat_xyzw[:, 1:2],
            grasp_quat_xyzw[:, 2:3],
        ),
        dim=-1,
    ).to(device=device, dtype=dtype)
    q_w = _quat_mul_wxyz(q_obj, q_rel)
    p_w = _quat_rotate_vec_wxyz(q_obj, grasp_pos_obj.to(device=device, dtype=dtype)) + p_obj
    return torch.cat((p_w, q_w), dim=-1)


def collect_collision_check_args(input_dict):
    keys = [
        "default_grasp_file",
        "default_max_num_envs",
        "default_max_num_grasps",
        "default_env_spacing",
        "default_box_wall_thickness",
        "default_box_horizontal_padding",
        "default_box_vertical_padding",
        "default_start_with_pregrasp_cspace_position",
        "default_skip_missing_collision_ids",
        "default_joint_motion_samples",
        "default_bin_contact_force_eps",
        "default_bin_contact_settle_substeps",
        "default_bin_hit_method",
    ]
    return {k: input_dict.get(k, globals()[k]) for k in keys}


default_grasp_file = os.path.join(
    os.environ.get("GRASP_DATASET_DIR", ""),
    "grasp_sim_data/robotiq_2f_85/mug.yaml",
)
default_max_num_envs = 1024
default_max_num_grasps = 0
default_env_spacing = 1.0
default_box_wall_thickness = 0.01
default_box_horizontal_padding = 0.15
default_box_vertical_padding = 0.06
default_start_with_pregrasp_cspace_position = False
default_skip_missing_collision_ids = False
default_joint_motion_samples = 1
default_bin_contact_force_eps = _DEFAULT_BIN_CONTACT_FORCE_EPS
default_bin_contact_settle_substeps = _DEFAULT_BIN_CONTACT_SETTLE_SUBSTEPS
default_bin_hit_method = _DEFAULT_BIN_HIT_METHOD


def add_collision_check_args(parser, param_dict, **kwargs):
    register_argument_group(parser, "collision_check", "collision_check", "In-box collision re-check (grasp_sim output)")
    add_gripper_args(parser, param_dict, **collect_gripper_args(param_dict))
    add_object_args(parser, param_dict, **collect_object_args(param_dict))

    add_arg_to_group(
        "collision_check",
        parser,
        "--grasp_file",
        type=str,
        default=kwargs.get("default_grasp_file", default_grasp_file),
        help="YAML from grasp_sim (or compatible isaac_grasp) with grasps + object/gripper metadata.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--max_num_envs",
        type=int,
        default=kwargs.get("default_max_num_envs", default_max_num_envs),
        help="Parallel envs per PhysX batch.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--max_num_grasps",
        type=int,
        default=kwargs.get("default_max_num_grasps", default_max_num_grasps),
        help="Cap number of grasps (0 = all in file).",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--env_spacing",
        type=float,
        default=kwargs.get("default_env_spacing", default_env_spacing),
        help="Spacing between Isaac Lab env copies.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--box_wall_thickness",
        type=float,
        default=kwargs.get("default_box_wall_thickness", default_box_wall_thickness),
        help="Thickness (m) of fixed box walls / floor.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--box_horizontal_padding",
        type=float,
        default=kwargs.get("default_box_horizontal_padding", default_box_horizontal_padding),
        help="Extra half-width/depth added around the object axis-aligned bounds.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--box_vertical_padding",
        type=float,
        default=kwargs.get("default_box_vertical_padding", default_box_vertical_padding),
        help="Extra height above the object top for the open bin.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--start_with_pregrasp_cspace_position",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=kwargs.get(
            "default_start_with_pregrasp_cspace_position",
            default_start_with_pregrasp_cspace_position,
        ),
        help="If true, use pregrasp_cspace_position / pregrasp pose like grasp_sim can.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--skip_missing_collision_ids",
        action="store_true",
        default=kwargs.get("default_skip_missing_collision_ids", default_skip_missing_collision_ids),
        help="If a PhysX collision shape id cannot be resolved, treat as no hit instead of failing.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--joint_motion_samples",
        type=int,
        default=kwargs.get("default_joint_motion_samples", default_joint_motion_samples),
        help="1 (default): check only the closed grasp ``cspace_position``. "
        "N≥2: check N poses along a joint-space lerp from ``pregrasp_cspace_position`` (α=0) "
        "to ``cspace_position`` (α=1); if pregrasp cspace is missing, start equals goal.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--bin_contact_force_eps",
        type=float,
        default=kwargs.get("default_bin_contact_force_eps", default_bin_contact_force_eps),
        help="``contact`` mode only: min filtered |force| on any gripper–/Box/ pair to count as a wall hit.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--bin_contact_settle_substeps",
        type=int,
        default=kwargs.get("default_bin_contact_settle_substeps", default_bin_contact_settle_substeps),
        help="``contact`` mode only: PhysX steps after each pose write before reading contact sensors.",
    )
    add_arg_to_group(
        "collision_check",
        parser,
        "--bin_hit_method",
        type=str,
        choices=("geometry", "contact"),
        default=kwargs.get("default_bin_hit_method", default_bin_hit_method),
        help="``geometry`` (default): mesh vertex AABBs in link frames × PhysX link poses vs bin AABBs. "
        "``contact``: filtered PhysX contact-sensor forces vs /Box/*.",
    )
    add_isaac_lab_args_if_needed(parser)


class GraspInBinCollisionChecker:
    """Loads grasp_sim YAML, spawns Isaac Lab scene + fixed box; bin hits via geometry or contact sensors."""

    def __init__(
        self,
        grasp_file: str,
        max_num_envs: int,
        max_num_grasps: int,
        env_spacing: float,
        box_wall_thickness: float,
        box_horizontal_padding: float,
        box_vertical_padding: float,
        start_with_pregrasp_cspace_position: bool,
        skip_missing_collision_ids: bool,
        joint_motion_samples: int,
        bin_contact_force_eps: float,
        bin_contact_settle_substeps: int,
        bin_hit_method: str,
        device: str,
        object_args_override: Optional[argparse.Namespace] = None,
    ):
        self.grasp_file = grasp_file
        self.max_num_envs = max_num_envs
        self.max_num_grasps = max_num_grasps
        self.env_spacing = env_spacing
        self.box_wall_thickness = box_wall_thickness
        self.box_horizontal_padding = box_horizontal_padding
        self.box_vertical_padding = box_vertical_padding
        self.start_with_pregrasp_cspace_position = start_with_pregrasp_cspace_position
        self.skip_missing_collision_ids = skip_missing_collision_ids
        self.joint_motion_samples = max(1, int(joint_motion_samples))
        self.device = device
        self.object_args_override = object_args_override
        self.contact_force_eps = float(bin_contact_force_eps)
        self.bin_contact_settle_substeps = max(1, int(bin_contact_settle_substeps))
        m = str(bin_hit_method or "geometry").strip().lower()
        if m not in ("geometry", "contact"):
            raise ValueError(f"bin_hit_method must be 'geometry' or 'contact', got {bin_hit_method!r}")
        self.bin_hit_method = m
        self._geom_mesh_corners: Optional[List[Tuple[str, np.ndarray]]] = None
        self._geom_mesh_torch: Optional[List[Tuple[str, torch.Tensor]]] = None
        self._load_grasp_yaml()

    def _load_grasp_yaml(self):
        with open(self.grasp_file, "r") as f:
            self.yaml_data = yaml.unsafe_load(f)
        if not self.yaml_data or "grasps" not in self.yaml_data:
            raise ValueError("Invalid grasp file: missing top-level 'grasps'")

        grasp_items = list(self.yaml_data["grasps"].items())
        if self.max_num_grasps > 0:
            grasp_items = grasp_items[: self.max_num_grasps]

        num = len(grasp_items)
        if num == 0:
            raise ValueError("No grasps in file")

        self.grasp_keys = [k for k, _ in grasp_items]
        self.grasps_wp = wp.zeros(num, dtype=wp.transform, device=self.device)
        self.cspace_wp = None
        self.cspace_start_wp = None
        self.cspace_joint_names: List[str] = []

        first = grasp_items[0][1]
        if "cspace_position" not in first or not isinstance(first["cspace_position"], dict):
            raise ValueError("Each grasp must contain a 'cspace_position' dictionary (grasp_sim output).")

        self.cspace_joint_names = list(first["cspace_position"].keys())
        cspace_goal_np = np.zeros((num, len(self.cspace_joint_names)), dtype=np.float32)
        cspace_start_np = np.zeros((num, len(self.cspace_joint_names)), dtype=np.float32)
        g_np = np.zeros((num, 7), dtype=np.float32)

        for i, (_, g) in enumerate(grasp_items):
            use_pre_body = (
                self.start_with_pregrasp_cspace_position
                and "pregrasp_position" in g
                and "pregrasp_orientation" in g
            )
            if use_pre_body:
                pos = np.array(g["pregrasp_position"], dtype=np.float32)
                xyz = np.array(g["pregrasp_orientation"]["xyz"], dtype=np.float32)
                w = float(g["pregrasp_orientation"]["w"])
            else:
                pos = np.array(g["position"], dtype=np.float32)
                xyz = np.array(g["orientation"]["xyz"], dtype=np.float32)
                w = float(g["orientation"]["w"])
            g_np[i, :3] = pos
            g_np[i, 3:6] = xyz
            g_np[i, 6] = w
            pos_goal = g["cspace_position"]
            pre = g.get("pregrasp_cspace_position")
            for j, jn in enumerate(self.cspace_joint_names):
                key_s = str(jn)
                gv = pos_goal.get(jn, pos_goal.get(key_s))
                if gv is None:
                    raise ValueError(f"Grasp missing cspace_position entry for joint {jn!r}")
                cspace_goal_np[i, j] = float(gv)
                if isinstance(pre, dict) and pre:
                    sv = pre.get(jn, pre.get(key_s))
                    cspace_start_np[i, j] = float(sv) if sv is not None else float(gv)
                else:
                    cspace_start_np[i, j] = float(gv)

        self.grasps_wp = wp.array(g_np, dtype=wp.transform, device=self.device)
        self.cspace_wp = wp.array(cspace_goal_np, dtype=wp.float32, device=self.device)
        self.cspace_start_wp = wp.array(cspace_start_np, dtype=wp.float32, device=self.device)

        self.gripper_file = resolve_usd_path(str(self.yaml_data.get("gripper_file", "")))

        frame = str(self.yaml_data.get("gripper_frame_link", "base_link")).strip()
        fingers = [str(x).strip() for x in self.yaml_data.get("finger_colliders", []) if str(x).strip()]
        self.bin_contact_link_names: List[str] = []
        seen: set[str] = set()
        for n in [frame] + fingers:
            if n not in seen:
                seen.add(n)
                self.bin_contact_link_names.append(n)
        if not self.bin_contact_link_names:
            raise ValueError(
                "grasp_file needs gripper_frame_link and/or finger_colliders so bin contact can target Robot links."
            )
        _max_bin_contact_links = 16
        if len(self.bin_contact_link_names) > _max_bin_contact_links:
            print_yellow(
                f"collision_check: truncating bin contact links from {len(self.bin_contact_link_names)} "
                f"to {_max_bin_contact_links} (ContactSensor per link)."
            )
            self.bin_contact_link_names = self.bin_contact_link_names[:_max_bin_contact_links]
        self.bin_contact_sensor_keys = [f"contact_bin_{i}" for i in range(len(self.bin_contact_link_names))]

        if self.object_args_override is not None:
            self.object_config = ObjectConfig.from_isaac_grasp_dict(self.yaml_data, self.object_args_override)
        else:
            self.object_config = ObjectConfig.from_isaac_grasp_dict(self.yaml_data, None)

        self._geom_mesh_torch = None
        if self.bin_hit_method == "geometry":
            self._geom_mesh_corners = _load_gripper_collision_mesh_corners_body_frame(self.gripper_file)
            if not self._geom_mesh_corners:
                raise ValueError(
                    f"geometry mode: no collision-enabled UsdGeom.Mesh prims under the default prim in "
                    f"{self.gripper_file!r} (need mesh points + UsdPhysics.CollisionAPI)."
                )
        else:
            self._geom_mesh_corners = None

    def get_usd_path(self, file_path: str) -> str:
        file_path = os.path.expanduser(file_path)
        if file_path.lower().endswith((".usd", ".usda", ".usdz")):
            return resolve_usd_path(file_path)
        usd_file = os.path.splitext(file_path)[0] + ".usd"
        if self.object_config.obj2usd_use_existing_usd and os.path.exists(resolve_usd_path(usd_file)):
            return resolve_usd_path(usd_file)
        from graspgen_utils import get_simulation_app
        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
        from mesh_utils import convert_mesh_to_usd

        _ = get_simulation_app(__file__, force_headed=False, wait_for_debugger_attach=False)
        physics_material = RigidBodyMaterialCfg(
            static_friction=self.object_config.obj2usd_friction,
            dynamic_friction=self.object_config.obj2usd_friction,
        )
        out = resolve_usd_path(usd_file)
        return convert_mesh_to_usd(
            out,
            file_path,
            overwrite=True,
            mass=1.0,
            collision_approximation=self.object_config.obj2usd_collision_approximation,
            physics_material=physics_material,
        )

    def build_scene_cfg(self, num_envs: int, usd_path: str):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        scene_members = {
            "dome_light": AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            ),
            "gripper": ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                init_state=ArticulationCfg.InitialStateCfg(),
                actuators={
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=None,
                        damping=None,
                    ),
                },
            ),
            "object": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
                ),
            ),
        }
        if self.bin_hit_method == "contact":
            from isaaclab.sensors import ContactSensorCfg

            # One :class:`ContactSensorCfg` per Robot link (filtered ``force_matrix_w`` is unreliable when one
            # sensor covers multiple rigid bodies — e.g. ``Robot/.*`` on a single sensor).
            box_filters = list(_BOX_CONTACT_FILTER_PRIM_PATHS)
            for i, link in enumerate(self.bin_contact_link_names):
                scene_members[f"contact_bin_{i}"] = ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    filter_prim_paths_expr=box_filters,
                    max_contact_data_count_per_prim=16,
                )

        BinCheckSceneCfg = configclass(type("BinCheckSceneCfg", (InteractiveSceneCfg,), scene_members))
        scene_cfg = BinCheckSceneCfg(
            num_envs=num_envs,
            env_spacing=self.env_spacing,
            filter_collisions=True,
            replicate_physics=False,
        )
        # Do not set kinematic on the gripper USD: it turns every link static and PhysX cannot build articulation joints.
        scene_cfg.gripper.spawn = sim_utils.UsdFileCfg(
            usd_path=self.gripper_file,
            activate_contact_sensors=(self.bin_hit_method == "contact"),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )
        scene_cfg.object.spawn = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(
                self.object_config.object_scale,
                self.object_config.object_scale,
                self.object_config.object_scale,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            activate_contact_sensors=False,
        )
        return scene_cfg

    def _env_prim_path(self, scene, env_id: int) -> str:
        if hasattr(scene, "env_prim_paths") and len(scene.env_prim_paths) > env_id:
            return scene.env_prim_paths[env_id]
        return f"/World/envs/env_{env_id}"

    def _spawn_bin_walls_for_env(
        self,
        env_path: str,
        env_origin: torch.Tensor,
        sim_utils,
    ) -> List[str]:
        """Spawn five kinematic cuboids (bin) under ``env_path``; return root prim paths for PhysX id lookup."""
        ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
        min_x, min_y, min_z, max_x, max_y, max_z = _aligned_world_aabb6(f"{env_path}/Object")
        cx_w = (min_x + max_x) * 0.5
        cy_w = (min_y + max_y) * 0.5
        w = (max_x - min_x) + self.box_horizontal_padding
        d = (max_y - min_y) + self.box_horizontal_padding
        h = (max_z - min_z) + self.box_vertical_padding
        floor_w = float(min_z)
        thick = self.box_wall_thickness

        cx_l = cx_w - ox
        cy_l = cy_w - oy
        base_z_l = (floor_w + thick * 0.5) - oz
        wall_z_l = (floor_w + h * 0.5) - oz

        def wall_cfg(sx: float, sy: float, sz: float):
            return sim_utils.CuboidCfg(
                size=(sx, sy, sz),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )

        bases = [
            (f"{env_path}/Box/Base", (w, d, thick), (cx_l, cy_l, base_z_l)),
            (f"{env_path}/Box/W1", (thick, d, h), (cx_l + w * 0.5 - thick * 0.5, cy_l, wall_z_l)),
            (f"{env_path}/Box/W2", (thick, d, h), (cx_l - w * 0.5 + thick * 0.5, cy_l, wall_z_l)),
            (f"{env_path}/Box/W3", (w, thick, h), (cx_l, cy_l + d * 0.5 - thick * 0.5, wall_z_l)),
            (f"{env_path}/Box/W4", (w, thick, h), (cx_l, cy_l - d * 0.5 + thick * 0.5, wall_z_l)),
        ]
        prim_paths: List[str] = []
        for prim_path, size, trans in bases:
            sim_utils.spawn_cuboid(prim_path, wall_cfg(*size), translation=trans)
            prim_paths.append(prim_path)
        return prim_paths

    def run(self, force_headed: bool, wait_for_debugger_attach: bool) -> np.ndarray:
        from graspgen_utils import get_simulation_app

        simulation_app = get_simulation_app(
            __file__, force_headed=force_headed, wait_for_debugger_attach=wait_for_debugger_attach
        )
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import build_simulation_context

        num_grasps = int(self.grasps_wp.shape[0])
        results = np.ones(num_grasps, dtype=bool)
        self._geom_mesh_torch = None

        usd_path = self.get_usd_path(self.object_config.object_file)
        print_blue(f"Object USD: {usd_path}")
        print_blue(f"Gripper USD: {self.gripper_file}")
        if self.bin_hit_method == "geometry":
            print_blue(
                "Bin hit test: geometry — collision mesh local AABBs × body_link_pose_w vs /Box/Base,W1–W4 "
                "(mug collision OFF)."
            )
        else:
            print_blue(
                "Bin hit test: contact — filtered PhysX force_matrix_w on gripper_frame_link + finger_colliders "
                "vs /Box/* (mug collision OFF)."
            )

        grasp_torch = wp.to_torch(self.grasps_wp).clone()
        pos_rel = grasp_torch[:, :3]
        quat_xyzw = grasp_torch[:, 3:7]
        cspace_torch = wp.to_torch(self.cspace_wp)
        cspace_start_torch = wp.to_torch(self.cspace_start_wp)

        start = 0
        batch_idx = 0
        total_batches = int(math.ceil(num_grasps / float(self.max_num_envs)))
        do_render = not simulation_app.DEFAULT_LAUNCHER_CONFIG["headless"]

        while start < num_grasps:
            batch_idx += 1
            n_env = min(self.max_num_envs, num_grasps - start)
            print_purple(f"\r  Bin collision batch {batch_idx}/{total_batches} (size {n_env}){' ' * 20}", end="", flush=True)

            sim_cfg = sim_utils.SimulationCfg(
                device=self.device,
                gravity=(0.0, 0.0, 0.0),
                physx=sim_utils.PhysxCfg(enable_ccd=False),
            )
            with build_simulation_context(
                device=self.device, gravity_enabled=False, auto_add_lighting=True, sim_cfg=sim_cfg
            ) as sim:
                print("\033[0m", end="")
                sim._app_control_on_stop_handle = None

                scene_cfg = self.build_scene_cfg(n_env, usd_path)
                scene = InteractiveScene(scene_cfg)
                try:
                    kit_major = int(simulation_app._app.get_build_version()[:3])
                except Exception:
                    kit_major = 0
                if not (
                    (not scene.cfg.replicate_physics and scene.cfg.filter_collisions)
                    or (kit_major >= 107 and self.device == "cpu")
                ):
                    scene.filter_collisions()

                # Spawn bin *before* ``sim.reset()`` so ``/Box/*`` prims exist when contact views initialize.
                origins = scene.env_origins[:n_env].clone()
                for i in range(n_env):
                    env_path = self._env_prim_path(scene, i)
                    self._spawn_bin_walls_for_env(env_path, origins[i], sim_utils)

                bin_piece_names = ("Base", "W1", "W2", "W3", "W4")
                bin_aabbs_per_env: List[List[Tuple[float, float, float, float, float, float]]] = []
                for i in range(n_env):
                    ep = self._env_prim_path(scene, i)
                    bin_aabbs_per_env.append([_aligned_world_aabb6(f"{ep}/Box/{n}") for n in bin_piece_names])

                sim_dt = sim.get_physics_dt()
                sim.reset()
                # After physics parses the stage, try to clear CCD on kinematic bin bodies (reduces log spam).
                _try_disable_ccd_on_bin_prims(scene, n_env)
                scene.write_data_to_sim()
                # ``sim.forward()`` only updates articulation kinematics for rendering — it does **not** run PhysX,
                # so contact sensors stay at zero. Use ``sim.step`` like ``grasp_sim.py``.
                # Geometry mode always uses a single step per pose so dynamics cannot tunnel the gripper through
                # the bin floor; ``bin_contact_settle_substeps`` applies only to ``contact`` mode.
                settle_subs = self.bin_contact_settle_substeps if self.bin_hit_method == "contact" else 1
                _physics_step_scene(sim, scene, sim_dt, render=do_render, substeps=settle_subs)

                if self.bin_hit_method == "contact":
                    fm_probe = scene[self.bin_contact_sensor_keys[0]].data.force_matrix_w
                    if fm_probe is None:
                        raise RuntimeError(
                            "Contact sensor has no force_matrix_w (empty filter_prim_paths_expr?). "
                            "Bin-vs-gripper check requires filtered contact reporting."
                        )

                gripper = scene["gripper"]
                if self.bin_hit_method == "geometry" and self._geom_mesh_torch is None:
                    dev = gripper.data.default_joint_pos.device
                    dt = gripper.data.default_joint_pos.dtype
                    if self._geom_mesh_corners is None:
                        raise RuntimeError("internal: geometry mode missing precomputed mesh corners")
                    self._geom_mesh_torch = [
                        (ln, torch.tensor(c, device=dev, dtype=dt)) for ln, c in self._geom_mesh_corners
                    ]
                object_asset = scene["object"]
                joint_pos = gripper.data.default_joint_pos.clone()
                joint_vel = torch.zeros_like(gripper.data.default_joint_pos)

                grasp_pos_batch = pos_rel[start : start + n_env]
                grasp_quat_batch = quat_xyzw[start : start + n_env]
                root_pose = compose_object_relative_grasps_world(origins, grasp_pos_batch, grasp_quat_batch)

                obj_state = object_asset.data.default_root_state.clone()
                obj_state[:, :3] = origins
                obj_state[:, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=obj_state.device, dtype=obj_state.dtype
                ).expand(n_env, 4)

                c_goal = cspace_torch[start : start + n_env].to(device=joint_pos.device, dtype=joint_pos.dtype)
                c_start = cspace_start_torch[start : start + n_env].to(device=joint_pos.device, dtype=joint_pos.dtype)
                if self.joint_motion_samples <= 1:
                    # Single check at closed grasp only (not α=0 pregrasp).
                    alphas = torch.tensor([1.0], device=joint_pos.device, dtype=joint_pos.dtype)
                else:
                    alphas = torch.linspace(
                        0.0, 1.0, self.joint_motion_samples, device=joint_pos.device, dtype=joint_pos.dtype
                    )

                batch_ok = torch.ones(n_env, dtype=torch.bool, device=self.device)
                for alpha in alphas:
                    jmix = (1.0 - alpha) * c_start + alpha * c_goal
                    for env_i in range(n_env):
                        for j_idx, jname in enumerate(self.cspace_joint_names):
                            ji = gripper.data.joint_names.index(str(jname))
                            joint_pos[env_i, ji] = jmix[env_i, j_idx]

                    gripper.write_root_pose_to_sim(root_pose)
                    gripper.write_root_velocity_to_sim(torch.zeros_like(gripper.data.default_root_state[:, 7:]))
                    object_asset.write_root_pose_to_sim(obj_state[:, :7])
                    object_asset.write_root_velocity_to_sim(obj_state[:, 7:])
                    gripper.write_joint_state_to_sim(joint_pos, joint_vel)
                    scene.write_data_to_sim()
                    settle_subs = self.bin_contact_settle_substeps if self.bin_hit_method == "contact" else 1
                    _physics_step_scene(sim, scene, sim_dt, render=do_render, substeps=settle_subs)

                    if self.bin_hit_method == "geometry":
                        poses_w = gripper.data.body_link_pose_w
                        body_names = gripper.data.body_names
                        name_to_i = {n: i for i, n in enumerate(body_names)}
                        if self._geom_mesh_torch is None:
                            raise RuntimeError("internal: geometry mode missing torch mesh specs")
                        if not any(name_to_i.get(ln) is not None for ln, _ in self._geom_mesh_torch):
                            raise RuntimeError(
                                "geometry mode: collision mesh link names from USD do not match any "
                                f"articulation body_names (first bodies: {body_names[:12]!r})."
                            )

                    for env_i in range(n_env):
                        if not batch_ok[env_i]:
                            continue
                        if self.bin_hit_method == "geometry":
                            if _world_aabb_from_body_corners(
                                poses_w[env_i],
                                name_to_i,
                                self._geom_mesh_torch,
                                bin_aabbs_per_env[env_i],
                            ):
                                batch_ok[env_i] = False
                        elif _env_gripper_bin_contact_from_filtered_sensors(
                            scene, env_i, self.contact_force_eps, self.bin_contact_sensor_keys
                        ):
                            batch_ok[env_i] = False

                if do_render:
                    batch_ok_cpu = batch_ok.detach().cpu().numpy()
                    for env_i in range(n_env):
                        env_path = self._env_prim_path(scene, env_i)
                        rgb = (0.0, 1.0, 0.0) if bool(batch_ok_cpu[env_i]) else (1.0, 0.0, 0.0)
                        _apply_bin_wall_preview_colors(env_path, rgb)
                    try:
                        sim.render()
                    except Exception:
                        pass

                results[start : start + n_env] = batch_ok.detach().cpu().numpy()

                if do_render and force_headed:
                    print_purple("force_headed: stepping until window closes (batch scene)...", flush=True)
                    while simulation_app.is_running():
                        sim.step(render=True)
                        scene.update(sim_dt)

            start += n_env

        print(f"\r", end="", flush=True)
        n_clear = int(results.sum())
        if self.joint_motion_samples <= 1:
            mode = "closed grasp cspace only"
        else:
            mode = f"pregrasp→grasp joint lerp ({self.joint_motion_samples} samples)"
        settle_note = (
            f"{self.bin_contact_settle_substeps} settle substep(s)/pose."
            if self.bin_hit_method == "contact"
            else "1 PhysX step/pose + mesh AABB × link pose vs bin (geometry mode)."
        )
        print_green(
            f"Gripper–bin check done ({self.bin_hit_method}): {n_clear} / {num_grasps} clear — {mode}; {settle_note}"
        )
        return results


def main(args):
    simulation_app = start_isaac_lab_if_needed(
        file_name=__file__,
        headless=False if args.force_headed else args.headless,
        wait_for_debugger_attach=args.wait_for_debugger_attach,
    )
    apply_gripper_configuration(args)

    checker = GraspInBinCollisionChecker(
        grasp_file=args.grasp_file,
        max_num_envs=args.max_num_envs,
        max_num_grasps=args.max_num_grasps,
        env_spacing=args.env_spacing,
        box_wall_thickness=args.box_wall_thickness,
        box_horizontal_padding=args.box_horizontal_padding,
        box_vertical_padding=args.box_vertical_padding,
        start_with_pregrasp_cspace_position=args.start_with_pregrasp_cspace_position,
        skip_missing_collision_ids=args.skip_missing_collision_ids,
        joint_motion_samples=args.joint_motion_samples,
        bin_contact_force_eps=args.bin_contact_force_eps,
        bin_contact_settle_substeps=args.bin_contact_settle_substeps,
        bin_hit_method=args.bin_hit_method,
        device=args.device,
        object_args_override=args,
    )
    flags = checker.run(force_headed=args.force_headed, wait_for_debugger_attach=args.wait_for_debugger_attach)

    out_yaml = copy.deepcopy(checker.yaml_data)
    for i, key in enumerate(checker.grasp_keys):
        out_yaml["grasps"][key]["in_bin_no_wall_collision"] = bool(flags[i])
    out_yaml["created_with_bin_collision_pass"] = "collision_check"

    grasp_abs = os.path.abspath(os.path.expanduser(args.grasp_file))
    grasp_base_no_ext = os.path.splitext(grasp_abs)[0]
    clear_path = grasp_base_no_ext + "_bin_clear.yaml"
    wall_hit_path = grasp_base_no_ext + "_bin_wall_hit.yaml"

    clear_keys = [k for i, k in enumerate(checker.grasp_keys) if flags[i]]
    wall_keys = [k for i, k in enumerate(checker.grasp_keys) if not flags[i]]

    def _write_grasp_subset(path: str, keys: List[str], subset_tag: str) -> None:
        sub = copy.deepcopy(out_yaml)
        sub["grasps"] = {k: out_yaml["grasps"][k] for k in keys}
        sub["created_with_bin_collision_pass"] = "collision_check"
        sub["collision_check_grasp_subset"] = subset_tag
        sub["collision_check_subset_count"] = len(keys)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        save_yaml(sub, path)
        print_green(f"Wrote {path} ({len(keys)} grasps, {subset_tag})")

    _write_grasp_subset(clear_path, clear_keys, "in_bin_no_wall_collision_true")
    _write_grasp_subset(wall_hit_path, wall_keys, "in_bin_no_wall_collision_false")

    if not args.force_headed and simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gripper–bin check (zero-g, mug collision off): default USD AABB overlap vs bin; optional contact mode."
    )
    add_collision_check_args(parser, globals(), **collect_collision_check_args(globals()))
    main(parser.parse_args())
