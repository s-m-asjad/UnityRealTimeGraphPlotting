#!/usr/bin/env python3
"""Grasp a mug in the fixed bin, then **lift the hand** along world :math:`+Z` with **gravity** on and a
**dynamic** object.

Typical input: ``mug_bin_clear.yaml`` (or any isaac_grasp file). Use
``--only_in_bin_no_wall_collision true`` (default) to match grasps that already passed
:mod:`collision_check` in the bin. Each grasp gets its own environment; the gripper
holds the **closed cspace** while the base is stepped upward so the sim does not drop
the hand as a free-floating body.

**Success** for an env: object com world :math:`z` is above the bin **rim** (from wall
AABB max :math:`z` of W1--W4) by at least ``--liftout_rim_clearance_m`` at the end of
the lift segment.

This uses PhysX with gravity; object collisions are **on** (unlike the bin geometry
pass). Requires a conda env with Isaac Lab / ``torch`` (e.g. ``grasp_data_gen``).

Run (from the repo or ``scripts/graspgen``): the script adds its own directory to
``sys.path`` so you can use ``python3 path/to/bin_liftout.py``. The sim device and
``--headless`` come from Isaac’s ``add_isaac_lab_args`` (``--force_headed`` opens
the viewport; pass ``--max_num_envs 1`` to inspect one env). With
``--force_headed``, the **first** ``sim.step`` **waits** until the timeline is
**playing** (use **Pause** in the editor if the sim is already running, then
**Play**). We do **not** call ``timeline.stop()`` (that would invalidate the
gripper’s PhysX view and crash ``scene.update``). Between substeps, joint commands
re-assert the **closed cspace**. After each physics substep, the **mug** is written
so its root stays in the same **plan position** and **upright** as
``collision_check`` (``env_origin`` in :math:`xy`, no tilt in world—same as ``obj_state`` in
:mod:`collision_check` ``run``) while
:math:`z` still follows the dynamics—same as having the object at the bin’s AABB
center, without roll/slide. A **fixed** ``j_template`` (snapshot of the articulation
default at reset) is used so finger re-asserts are not based on the sim’s updated
``default_joint_pos`` (which would **open** the hand). In ``--force_headed`` the
**lift** renders on **every** substep so the gripper is seen rising from the bin, not
only at the final height.
"""
from __future__ import annotations

# Ensure this directory is importable (``python3 path/to/bin_liftout.py`` from any cwd).
import sys
from pathlib import Path

_d = str(Path(__file__).resolve().parent)
if _d not in sys.path:
    sys.path.insert(0, _d)

import argparse
import copy
import math
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import warp as wp
import yaml

from collision_check import (
    GraspInBinCollisionChecker,
    _aligned_world_aabb6,
    _physics_step_scene,
    _try_disable_ccd_on_bin_prims,
    compose_object_relative_grasps_world,
)
from graspgen_utils import (
    add_arg_to_group,
    add_isaac_lab_args_if_needed,
    get_simulation_app,
    print_blue,
    print_green,
    print_purple,
    print_yellow,
    register_argument_group,
    save_yaml,
    str_to_bool,
    start_isaac_lab_if_needed,
)
from gripper import add_gripper_args, apply_gripper_configuration, collect_gripper_args
from object import add_object_args, collect_object_args

default_grasp_file = os.path.join(
    os.environ.get("GRASP_DATASET_DIR", ""), "grasp_sim_data/robotiq_2f_85/mug_bin_clear.yaml"
)
default_max_num_envs = 32
default_max_num_grasps = 0
default_env_spacing = 1.0
default_settle_substeps = 80
default_lift_substeps = 200
default_lift_height_m = 0.5
default_liftout_rim_clearance_m = 0.01
default_only_in_bin_clear = True
default_start_with_pregrasp_cspace_position = False
default_box_wall_thickness = 0.01
default_box_horizontal_padding = 0.15
default_box_vertical_padding = 0.06

# PhysX / actuator limits (see ``ImplicitActuatorCfg.effort_limit_sim``): max effort per actuated DOF; units follow
# the joint (linear: N, revolute: N·m). Object mass (kg) applied via spawn ``MassPropertiesCfg`` when the root has Mass API.
BIN_LIFTOUT_GRIPPER_EFFORT_LIM = 8.0
BIN_LIFTOUT_OBJECT_MASS_KG = 0.2
# ``ImplicitActuatorCfg`` PD gains for the gripper. With ``stiffness=None`` (USD defaults) the actuator
# barely opposes the mug, so the fingers never grip and the hand "ladder-climbs" away from the mug.
BIN_LIFTOUT_ACTUATOR_STIFFNESS = 600.0
BIN_LIFTOUT_ACTUATOR_DAMPING = 40.0
# **PD overdrive past the cspace contact** (rad) — how far past the per-grasp ``cspace_position`` the
# *driver* PD aims (mimics still target the soft-closed end). At overdrive ≈ 0.4 rad with stiffness=600
# the driver torque saturates ``effort_limit_sim`` for typical cspace poses, matching the pattern used by
# ``grasp_sim.py:1128`` (which targets the deep soft-closed end). The penetration is then bounded
# *geometrically* by the gripper's contact offset (see ``BIN_LIFTOUT_GRIPPER_CONTACT_OFFSET_M``) rather
# than by giving up the squeeze.
BIN_LIFTOUT_GRIPPER_PD_OVERDRIVE_RAD = 0.40
# Per-pad **contact offset** (m): PhysX starts applying contact force when bodies are this far apart, so a
# larger value spreads the contact response over more solver steps and visibly reduces "pads drive into
# mug" artefacts at high squeeze force without giving up holding force. The effect on success rate is
# negligible but the visual penetration is clearly less. (PhysX requires
# ``contact_offset > rest_offset``; ``rest_offset`` stays at 0.)
BIN_LIFTOUT_GRIPPER_CONTACT_OFFSET_M = 0.001
BIN_LIFTOUT_GRIPPER_REST_OFFSET_M = 0.0
# Tack: static/dynamic μ on **mug↔gripper** (keep in a sane 1–4 range; extreme μ harms the solver and can look like slip).
BIN_LIFTOUT_CONTACT_FRICTION = 3.0
# Box uses a *separate*, near-frictionless material so the mug only "rests" on it (no fall-through) but does
# **not** stick: at high gripper friction, a tacky bin floor/wall would (a) glue mugs that touch a wall during
# settle so the lift can't peel them off ("did not budge") and (b) misdirect the spawn-time depenetration
# impulse so other mugs are flung along the wall normal ("flew away"). PhysX uses the *combined* μ per pair
# (we use ``min`` so the bin's near-zero μ wins for box↔mug while ``max`` between mug↔gripper still gives
# the high tack we want for grasping).
BIN_LIFTOUT_BIN_FRICTION = 0.02
# Bin **floor depth** (m): we extend the kinematic base slab *downward* so the visible floor still sits
# at ``box_wall_thickness`` (the rim/wall geometry is unchanged) but the collider is thick enough that
# (a) a stray downward depenetration impulse can't tunnel a mug through it (CCD is off on GPU in this
# Isaac Sim build), and (b) settle dynamics stabilise — empirically a thin slab produces enough contact
# noise during settle that the mug ends up a few mm off where ``cspace_position`` assumes, which makes
# most grasps slip during lift. The base *top face* z is unchanged so mug rest pose is identical.
BIN_LIFTOUT_BIN_FLOOR_DEPTH_M = 0.05
# Analytical floor pin (m) — minimum allowed mug ``pz - origin_z``. Belt-and-suspenders against
# tunneling: even with the thicker collider above, a hard pin guarantees no escape downward.
BIN_LIFTOUT_MUG_MIN_PZ_BELOW_ORIGIN_M = -0.01
# Keep ``ImplicitActuatorCfg`` stiffness/damping as ``None`` (USD/defaults): hard-coded large PD gains with
# a small ``BIN_LIFTOUT_PHYSICS_DT`` can destabilize the solve so the hand stops tracking the closed cspace
# reasserts. Contact quality comes from ``dt``, phys substeps, solver, and ``BIN_LIFTOUT_CONTACT_FRICTION``.
# Physics time step (simulation, not display): smaller ``dt`` improves contact / friction. Rendering cadence is unchanged
# (governed by which ``sim.step`` pass requests ``render=``; ``scene.update`` uses this ``dt`` each substep).
BIN_LIFTOUT_PHYSICS_DT = 1.0 / 240.0
# Chosen so ``SUBSTEPS * PHYSICS_DT == 1/30`` s (same as 4×(1/120) and 2×(1/60))—finer sub-steps, unchanged sim time/outer.
BIN_LIFTOUT_PHYS_SUBSTEPS = 8
# Diagnostic knob: ``BIN_LIFTOUT_DISABLE_MUG_PIN=1`` skips the xy/orient pin and the pz/vz clamps so you
# can *see* the raw spawn-time depenetration explosion (mug shoots km/s upward — see commit notes).
BIN_LIFTOUT_DISABLE_MUG_PIN = bool(int(os.environ.get("BIN_LIFTOUT_DISABLE_MUG_PIN", "0")))
# Settle: suppress penetration “explosions” that launch the root upward (|v_z| and p_z stay bounded).
BIN_LIFTOUT_MUG_MAX_Z_RISE_SETTLE_M = 0.35
BIN_LIFTOUT_MUG_VZ_CAP_M_S = 2.5
BIN_LIFTOUT_MUG_VZ_CAP_LIFT_M_S = 6.0
# PhysX: limit depenetration *velocity*; default can be very large and ejects the mug.
BIN_LIFTOUT_MUG_MAX_DEPEN_VEL_M_S = 0.35


def collect_bin_liftout_args(globs: dict) -> dict:
    keys = [
        "default_grasp_file",
        "default_max_num_envs",
        "default_max_num_grasps",
        "default_env_spacing",
        "default_settle_substeps",
        "default_lift_substeps",
        "default_lift_height_m",
        "default_liftout_rim_clearance_m",
        "default_only_in_bin_clear",
        "default_start_with_pregrasp_cspace_position",
        "default_box_wall_thickness",
        "default_box_horizontal_padding",
        "default_box_vertical_padding",
    ]
    return {k: globs.get(k, globals()[k]) for k in keys}


def add_bin_liftout_args(parser, param_dict, **kwargs) -> None:
    register_argument_group(
        parser, "bin_liftout", "bin_liftout", "Lift mug out of bin (gravity + dynamic object)"
    )
    add_gripper_args(parser, param_dict, **collect_gripper_args(param_dict))
    add_object_args(parser, param_dict, **collect_object_args(param_dict))
    # Must be before we add a conflicting ``--device`` so AppLauncher registers ``--headless`` etc.
    add_isaac_lab_args_if_needed(parser)

    add_arg_to_group(
        "bin_liftout",
        parser,
        "--grasp_file",
        type=str,
        default=kwargs.get("default_grasp_file", default_grasp_file),
        help="Path to an isaac_grasp YAML (e.g. *_bin_clear.yaml).",
    )
    add_arg_to_group(
        "bin_liftout",
        parser,
        "--max_num_envs",
        type=int,
        default=kwargs.get("default_max_num_envs", default_max_num_envs),
    )
    add_arg_to_group(
        "bin_liftout",
        parser,
        "--max_num_grasps",
        type=int,
        default=kwargs.get("default_max_num_grasps", default_max_num_grasps),
        help="0 = all grasps after the in-bin filter.",
    )
    add_arg_to_group("bin_liftout", parser, "--env_spacing", type=float, default=kwargs.get("default_env_spacing", default_env_spacing))
    add_arg_to_group("bin_liftout", parser, "--settle_substeps", type=int, default=kwargs.get("default_settle_substeps", default_settle_substeps))
    add_arg_to_group("bin_liftout", parser, "--lift_substeps", type=int, default=kwargs.get("default_lift_substeps", default_lift_substeps))
    add_arg_to_group("bin_liftout", parser, "--lift_height_m", type=float, default=kwargs.get("default_lift_height_m", default_lift_height_m))
    add_arg_to_group(
        "bin_liftout",
        parser,
        "--liftout_rim_clearance_m",
        type=float,
        default=kwargs.get("default_liftout_rim_clearance_m", default_liftout_rim_clearance_m),
    )
    add_arg_to_group(
        "bin_liftout",
        parser,
        "--only_in_bin_no_wall_collision",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=kwargs.get("default_only_in_bin_clear", default_only_in_bin_clear),
        help="If true, only grasps with in_bin_no_wall_collision: true are simulated.",
    )
    add_arg_to_group(
        "bin_liftout",
        parser,
        "--start_with_pregrasp_cspace_position",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=kwargs.get("default_start_with_pregrasp_cspace_position", default_start_with_pregrasp_cspace_position),
    )
    add_arg_to_group("bin_liftout", parser, "--box_wall_thickness", type=float, default=kwargs.get("default_box_wall_thickness", default_box_wall_thickness))
    add_arg_to_group("bin_liftout", parser, "--box_horizontal_padding", type=float, default=kwargs.get("default_box_horizontal_padding", default_box_horizontal_padding))
    add_arg_to_group("bin_liftout", parser, "--box_vertical_padding", type=float, default=kwargs.get("default_box_vertical_padding", default_box_vertical_padding))
    # Sim device: use AppLauncher ``--device`` (added by add_isaac_lab above).


def _apply_closed_cspace(
    joint_pos: torch.Tensor,
    c_goal: torch.Tensor,
    cspace_joint_names: List[str],
    gripper,
) -> None:
    for j_idx, jname in enumerate(cspace_joint_names):
        ji = gripper.data.joint_names.index(str(jname))
        joint_pos[:, ji] = c_goal[:, j_idx]


def _soft_quadrant_index_for_finger_close(loader: GraspInBinCollisionChecker) -> int:
    """Grasp file ``open_limit: lower|upper`` tells which *limit* in ``[soft_lower, soft_upper]`` is the open end.

    We take the *other* column as the “fully closed on the range” default for *non-cspace* actuated dofs, then
    overwrite cspace dofs with per-grasp :attr:`cspace_position` (same idea as :mod:`grasp_sim`).
    """
    s = str((loader.yaml_data or {}).get("open_limit", "lower") or "lower").strip().lower()
    return 0 if s == "upper" else 1  # open at lower end → “closed on range” = upper; ``upper`` open → 0


def _gripper_joint_template_soft_closed(
    grip,
    m: int,
    loader: GraspInBinCollisionChecker,
) -> torch.Tensor:
    """For every actuated DOF, start from the soft *closed* end of the limit segment (mimic fingers track driver).

    Using only :attr:`grip.data.default_joint_pos` for non-cspace joints leaves *mimic* and helper joints in the
    USD *open* bind pose, while ``cspace_position`` might only set ``finger_joint`` – causing wildly inconsistent
    pad contact and “some grasps slip, others stick” under the same friction settings.

    This template is also the **PD position target** during settle/lift: by aiming for the deeper "fully
    closed on range" end (rather than the per-grasp ``cspace_position`` where the fingers merely *touch* the
    mug), we generate a real PD error against the mug → real squeeze force (compare ``grasp_sim.py`` which
    sets the PD target to ``soft_joint_pos_limits[..., grasp_mode]`` for the same reason).
    """
    q = _soft_quadrant_index_for_finger_close(loader)
    if getattr(grip.data, "soft_joint_pos_limits", None) is not None and grip.data.soft_joint_pos_limits.numel() > 0:
        return grip.data.soft_joint_pos_limits[:m, :, q].clone()
    return grip.data.default_joint_pos[:m].clone()


def _gripper_joint_at_closed_cspace(
    j_template: torch.Tensor,
    c_b: torch.Tensor,
    cspace_joint_names: List[str],
    grip,
) -> torch.Tensor:
    """Build the **initial** DOF position: per-env soft-closed template, with per-grasp ``cspace_position``
    overlaid for the cspace dofs (so the fingers spawn already at the grasp pose, touching the mug).

    Used **only** for the one-shot ``write_joint_state_to_sim`` at scene init — *not* as the PD target.
    The PD target is built separately (see :func:`_gripper_pd_target_with_overdrive`) so the squeeze force
    is bounded. Do *not* read :attr:`grip.data.default_joint_pos` every frame: it is refreshed from the sim
    and drifts to an **open** hand after the first substeps, making re-asserts look wrong and fingers open.
    """
    j = j_template.clone()
    _apply_closed_cspace(j, c_b, cspace_joint_names, grip)
    return j


def _gripper_pd_target_with_overdrive(
    j_template: torch.Tensor,
    c_b: torch.Tensor,
    cspace_joint_names: List[str],
    grip,
    overdrive_rad: float,
) -> torch.Tensor:
    """PD position target: ``j_template`` (soft-closed) for mimic dofs, but ``cspace_position + overdrive``
    for the cspace driver dofs.

    Targeting the deep ``j_template`` for the *driver* drove PD error up to ~0.4 rad which, at
    ``stiffness = BIN_LIFTOUT_ACTUATOR_STIFFNESS``, saturates ``effort_limit_sim`` and crushes soft mug
    bodies (visible as gripper-into-mug penetration on certain grasps). With a small overdrive the
    steady-state error is bounded by ``overdrive_rad`` regardless of where the per-grasp ``cspace_position``
    sits in the joint range, giving consistent and gentle squeeze force across all grasps.
    """
    j = j_template.clone()
    od = float(overdrive_rad)
    for j_idx, jname in enumerate(cspace_joint_names):
        ji = grip.data.joint_names.index(str(jname))
        j[:, ji] = c_b[:, j_idx] + od
    return j


_MUG_DEBUG_STATS: dict = {
    "max_raw_pz_above_origin": -1e9,
    "max_raw_abs_vz": -1e9,
    "max_raw_horiz_dist": -1e9,
    "settle_max_pz_above_origin": -1e9,
    "settle_max_abs_vz": -1e9,
}


def _reset_mug_debug_stats() -> None:
    for k in _MUG_DEBUG_STATS:
        _MUG_DEBUG_STATS[k] = -1e9


def _print_mug_debug_stats(prefix: str) -> None:
    s = _MUG_DEBUG_STATS
    print_purple(
        f"{prefix} max raw pz over origin = {s['max_raw_pz_above_origin']:+.3f} m, "
        f"max |vz| raw = {s['max_raw_abs_vz']:.3f} m/s, "
        f"max raw horiz dist from origin = {s['max_raw_horiz_dist']:.3f} m, "
        f"settle pz above origin = {s['settle_max_pz_above_origin']:+.3f} m, "
        f"settle max |vz| = {s['settle_max_abs_vz']:.3f} m/s",
        flush=True,
    )


def _reassert_mug_plan_centered_upright(
    obj,
    origins: torch.Tensor,
    m: int,
    *,
    settle: bool = False,
) -> None:
    """Match ``collision_check`` object root: stay at ``origins`` in :math:`xy`, identity quat, no spin.

    Retains the sim :math:`z` so the body can follow gravity, **but** pass ``settle=True`` during
    the settle window so a bad first contact (interpenetration) cannot launch the body in :math:`z`—we
    clamp :math:`v_z` and cap :math:`p_z` above the env. During **lift** (``settle=False``) the cap
    is off so the com can follow the hand upward.
    """
    p = obj.data.root_pos_w[:m]
    o = origins[:m]
    raw_pz_above = (p[:, 2] - o[:, 2]).detach()
    raw_horiz = torch.linalg.norm(p[:, :2] - o[:, :2], dim=-1).detach()
    vlin = obj.data.root_lin_vel_w[:m]
    raw_vz = vlin[:, 2].detach()
    s = _MUG_DEBUG_STATS
    s["max_raw_pz_above_origin"] = max(s["max_raw_pz_above_origin"], float(raw_pz_above.max().item()))
    s["max_raw_abs_vz"] = max(s["max_raw_abs_vz"], float(raw_vz.abs().max().item()))
    s["max_raw_horiz_dist"] = max(s["max_raw_horiz_dist"], float(raw_horiz.max().item()))
    if settle:
        s["settle_max_pz_above_origin"] = max(s["settle_max_pz_above_origin"], float(raw_pz_above.max().item()))
        s["settle_max_abs_vz"] = max(s["settle_max_abs_vz"], float(raw_vz.abs().max().item()))
    if BIN_LIFTOUT_DISABLE_MUG_PIN:
        return
    # Lower floor pin: keep ``pz`` at-or-above the bin floor so the mug cannot tunnel through the (now
    # original-thickness) base if a single substep ever produces a stray downward impulse. Acts as the
    # geometric replacement for the deep floor we used briefly.
    pz = p[:, 2].clamp(min=o[:, 2] + float(BIN_LIFTOUT_MUG_MIN_PZ_BELOW_ORIGIN_M))
    if settle:
        pz = torch.minimum(pz, o[:, 2] + float(BIN_LIFTOUT_MUG_MAX_Z_RISE_SETTLE_M))
    wxyz = torch.tensor([1.0, 0.0, 0.0, 0.0], device=p.device, dtype=p.dtype).expand(m, 4)
    new_pose = torch.cat((torch.stack((o[:, 0], o[:, 1], pz), dim=-1), wxyz), dim=-1)
    vcap = float(BIN_LIFTOUT_MUG_VZ_CAP_M_S if settle else BIN_LIFTOUT_MUG_VZ_CAP_LIFT_M_S)
    vzu = vlin[:, 2:3].clamp(min=-vcap, max=vcap)
    new_vel6 = torch.cat(
        (torch.zeros(m, 2, device=p.device, dtype=p.dtype), vzu, torch.zeros(m, 3, device=p.device, dtype=p.dtype)),
        dim=-1,
    )
    obj.write_root_pose_to_sim(new_pose)
    obj.write_root_velocity_to_sim(new_vel6)


def _reassert_gripper_in_sim(
    grip,
    j_template: torch.Tensor,
    c_b: torch.Tensor,
    cspace_joint_names: List[str],
    zv: torch.Tensor,
    zr: torch.Tensor,
    root_pose: torch.Tensor,
    scene,
    object_asset: Any = None,
    object_st: Optional[torch.Tensor] = None,
    *,
    write_joint_state: bool = False,
) -> None:
    """Per-substep gripper re-assert: lock root to the controlled pose and re-apply the **PD position target**.

    The PD target for cspace driver dofs is ``cspace_position + BIN_LIFTOUT_GRIPPER_PD_OVERDRIVE_RAD`` (a
    small, bounded squeeze past contact); mimic dofs target ``j_template`` (soft-closed end). This produces
    a steady, gentle holding force on the mug (~tens of N at 0.05 rad / 600 N·m·rad⁻¹) instead of the
    saturated 235 N effort that resulted from targeting the deep ``soft-closed`` end and which crushed soft
    mug bodies on some grasps.

    The joint *state* is intentionally **not** written every substep: doing so fights the PD controller and
    lets the fingers spring open between teleports, which is exactly what was leaving 29/30 mugs in the bin.
    Pass ``write_joint_state=True`` (and the ``object_*`` args) only for the one-shot scene initialisation,
    where we want the fingers spawned at the per-grasp ``cspace_position``.
    """
    grip.write_root_pose_to_sim(root_pose)
    grip.write_root_velocity_to_sim(zr)
    if object_asset is not None and object_st is not None:
        object_asset.write_root_pose_to_sim(object_st[:, :7])
        object_asset.write_root_velocity_to_sim(object_st[:, 7:])
    grip.set_joint_position_target(j_template)
    if write_joint_state:
        j_init = _gripper_joint_at_closed_cspace(j_template, c_b, cspace_joint_names, grip)
        grip.write_joint_state_to_sim(j_init, zv)
    scene.write_data_to_sim()


def _bump_gripper_velocity_limits_for_teleport(grip) -> None:
    """Loosen per-DOF max joint velocity so :meth:`write_joint_state_to_sim` is not throttled (see ``grasp_sim``)."""
    w = torch.full_like(grip.data.joint_vel_limits, 1.0e9)
    grip.write_joint_velocity_limit_to_sim(w)


def _print_force_headed_play_hint(sim_app, do_render: bool, force_headed: bool) -> None:
    """Explain how headed runs wait for Play without calling ``timeline.stop()`` (that invalidates PhysX / articulation).

    :class:`SimulationContext.step` already blocks in ``while not is_playing(): render()``; we only print guidance.
    """
    if not (bool(force_headed) and bool(do_render) and sim_app.is_running()):
        return
    print_purple(
        "force_headed: the **first** settle substep will **block** until the timeline is **playing** "
        "(``sim.step``). If the sim is already running, use **Pause** in the editor first, then **Play (▶)** "
        "to start settle + lift.",
        flush=True,
    )


def _spawn_bin_walls(
    env_path: str,
    env_origin: torch.Tensor,
    sim_utils,
    *,
    box_wall_thickness: float,
    box_horizontal_padding: float,
    box_vertical_padding: float,
) -> None:
    ox, oy, oz = float(env_origin[0]), float(env_origin[1]), float(env_origin[2])
    min_x, min_y, min_z, max_x, max_y, max_z = _aligned_world_aabb6(f"{env_path}/Object")
    cx_w = (min_x + max_x) * 0.5
    cy_w = (min_y + max_y) * 0.5
    w = (max_x - min_x) + box_horizontal_padding
    d = (max_y - min_y) + box_horizontal_padding
    h = (max_z - min_z) + box_vertical_padding
    floor_w = float(min_z)
    thick = box_wall_thickness
    floor_thick = max(float(thick), float(BIN_LIFTOUT_BIN_FLOOR_DEPTH_M))
    cx_l = cx_w - ox
    cy_l = cy_w - oy
    base_top_w = floor_w + thick
    base_center_w = base_top_w - floor_thick * 0.5
    base_z_l = base_center_w - oz
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58)),
        )

    for path, siz, tr in (
        (f"{env_path}/Box/Base", (w, d, floor_thick), (cx_l, cy_l, base_z_l)),
        (f"{env_path}/Box/W1", (thick, d, h), (cx_l + w * 0.5 - thick * 0.5, cy_l, wall_z_l)),
        (f"{env_path}/Box/W2", (thick, d, h), (cx_l - w * 0.5 + thick * 0.5, cy_l, wall_z_l)),
        (f"{env_path}/Box/W3", (w, thick, h), (cx_l, cy_l + d * 0.5 - thick * 0.5, wall_z_l)),
        (f"{env_path}/Box/W4", (w, thick, h), (cx_l, cy_l - d * 0.5 + thick * 0.5, wall_z_l)),
    ):
        sim_utils.spawn_cuboid(path, wall_cfg(*siz), translation=tr)


def _env_path_for(scene, env_id: int) -> str:
    if hasattr(scene, "env_prim_paths") and len(scene.env_prim_paths) > env_id:
        return scene.env_prim_paths[env_id]
    return f"/World/envs/env_{env_id}"


def _apply_bin_liftout_tack_friction(scene, m: int) -> None:
    """Bind two physics materials per env: a **high-μ tack** on Object + Robot (for the grasp), and a
    **near-frictionless** material on Box (so the mug rests on the bin without sticking to it).

    PhysX combines per-pair μ; we use ``max`` for the gripper-side tack (so mug↔gripper picks up the tack
    even if the gripper's USD has lower μ) and ``min`` for the bin (so the box's near-zero μ wins for any
    box↔* contact, regardless of the other body's friction). This stops the "mug glued to the bin floor"
    cases where a settled mug refuses to lift, and also prevents large depenetration impulses misdirected
    along the bin walls from flinging the mug along the wall normal.
    """
    from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
    from isaaclab.sim.utils import bind_physics_material
    from isaaclab.sim.utils.stage import get_current_stage

    st = get_current_stage()
    tack_cfg = RigidBodyMaterialCfg(
        static_friction=BIN_LIFTOUT_CONTACT_FRICTION,
        dynamic_friction=BIN_LIFTOUT_CONTACT_FRICTION,
        restitution=0.0,
        friction_combine_mode="max",
    )
    bin_cfg = RigidBodyMaterialCfg(
        static_friction=BIN_LIFTOUT_BIN_FRICTION,
        dynamic_friction=BIN_LIFTOUT_BIN_FRICTION,
        restitution=0.0,
        friction_combine_mode="min",
    )
    for j in range(m):
        pth = _env_path_for(scene, j)
        tack_path = f"{pth}/_liftout_tack_physics"
        bin_path = f"{pth}/_liftout_bin_physics"
        if not st.GetPrimAtPath(tack_path).IsValid():
            tack_cfg.func(tack_path, tack_cfg)
        if not st.GetPrimAtPath(bin_path).IsValid():
            bin_cfg.func(bin_path, bin_cfg)
        for root in (f"{pth}/Object", f"{pth}/Robot"):
            if st.GetPrimAtPath(root).IsValid():
                bind_physics_material(root, tack_path, stage=st)
        bin_root = f"{pth}/Box"
        if st.GetPrimAtPath(bin_root).IsValid():
            bind_physics_material(bin_root, bin_path, stage=st)


def _wall_rim_max_z(wall_aabbs: List[Tuple[float, float, float, float, float, float]]) -> float:
    return max(b[5] for b in wall_aabbs)


def _grasp_index_list(loader: GraspInBinCollisionChecker, raw: dict, only_clear: bool, cap: int) -> List[int]:
    n = int(loader.grasps_wp.shape[0])
    out: List[int] = []
    for i in range(n):
        g = raw["grasps"][loader.grasp_keys[i]]
        if only_clear and g.get("in_bin_no_wall_collision", None) is not True:
            continue
        out.append(i)
    if cap > 0 and len(out) > cap:
        out = out[:cap]
    return out


def _build_scene(loader: GraspInBinCollisionChecker, m: int, usd: str, sim_utils):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass

    oc = loader.object_config
    members = {
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
                    effort_limit_sim=BIN_LIFTOUT_GRIPPER_EFFORT_LIM,
                    stiffness=BIN_LIFTOUT_ACTUATOR_STIFFNESS,
                    damping=BIN_LIFTOUT_ACTUATOR_DAMPING,
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
    C = configclass(type("BinLiftoutSc", (InteractiveSceneCfg,), members))
    sc = C(
        num_envs=m,
        env_spacing=loader.env_spacing,
        filter_collisions=True,
        replicate_physics=False,
    )
    sc.gripper.spawn = sim_utils.UsdFileCfg(
        usd_path=loader.gripper_file,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    sc.object.spawn = sim_utils.UsdFileCfg(
        usd_path=usd,
        scale=(oc.object_scale, oc.object_scale, oc.object_scale),
        mass_props=sim_utils.MassPropertiesCfg(mass=BIN_LIFTOUT_OBJECT_MASS_KG),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # rigid_body_enabled=True,
            # kinematic_enabled=False,
            # disable_gravity=False,
            # max_depenetration_velocity=5.0,
            solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=True,
    )
    return sc


def _run(loader: GraspInBinCollisionChecker, args: argparse.Namespace, inds: List[int]) -> np.ndarray:
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import build_simulation_context
    import isaaclab.sim as sim_utils

    n_total = len(inds)
    if n_total < 1:
        raise ValueError("No grasps after filter.")
    sim_device = str(getattr(args, "device", "cuda:0"))
    g_all = wp.to_torch(loader.grasps_wp).float()
    c_all = wp.to_torch(loader.cspace_wp).float()
    usd = loader.get_usd_path(loader.object_config.object_file)
    print_blue(f"Object USD: {usd}")
    print_blue(f"Gripper USD: {loader.gripper_file}")
    sim_app = get_simulation_app(
        __file__, force_headed=bool(getattr(args, "force_headed", False)), wait_for_debugger_attach=bool(getattr(args, "wait_for_debugger_attach", False))
    )
    do_render = not sim_app.DEFAULT_LAUNCHER_CONFIG["headless"]
    settle = max(1, int(getattr(args, "settle_substeps", 80)))
    lift_n = max(1, int(getattr(args, "lift_substeps", 200)))
    lift_dz = float(getattr(args, "lift_height_m", 0.15)) / float(lift_n) if lift_n else 0.0
    rim_m = max(0.0, float(getattr(args, "liftout_rim_clearance_m", 0.01)))
    print_blue(f"bin_liftout: {n_total} grasps; settle {settle} subs; {lift_n} lift subs; total dz = {float(getattr(args, 'lift_height_m', 0.15))} m")
    bsz = max(1, int(getattr(args, "max_num_envs", 32)))
    bnum = int(math.ceil(n_total / bsz))
    out = np.zeros(n_total, dtype=bool)
    s0 = 0
    bi = 0
    sim_dt = float(BIN_LIFTOUT_PHYSICS_DT)
    while s0 < n_total:
        bi += 1
        m = min(bsz, n_total - s0)
        sub = inds[s0 : s0 + m]
        print_purple(f"\r  batch {bi}/{bnum}  (m={m}){' ' * 20}", end="", flush=True)
        g_b = g_all[sub]
        c_b = c_all[sub]
        pr, qe = g_b[:, :3], g_b[:, 3:7]
        c_b = c_b.to(sim_device, dtype=g_b.dtype)
        sim_cfg = sim_utils.SimulationCfg(
            device=sim_device,
            dt=BIN_LIFTOUT_PHYSICS_DT,
            gravity=(0.0, 0.0, -9.81),
            physx=sim_utils.PhysxCfg(
                enable_ccd=True,
                # Favor contact resolution for grippers: articulation constraints can otherwise starve contact friction.
                solve_articulation_contact_last=True,
                # More iterations per contact solve (TGS will clamp to actor max but floor is higher).
                min_position_iteration_count=8,
                min_velocity_iteration_count=2,
                # TGS: recompute ext. forces each position iteration; often stabilizes contact/friction.
                enable_external_forces_every_iteration=True,
                # Slightly stricter: friction is applied when contact points are this close to the surface (m).
                friction_offset_threshold=0.01,
            ),
        )
        with build_simulation_context(
            device=sim_device,
            gravity_enabled=True,
            auto_add_lighting=bool(do_render or getattr(args, "force_headed", False)),
            sim_cfg=sim_cfg,
        ) as simx:
            simx._app_control_on_stop_handle = None
            sim_dt = simx.get_physics_dt()
            sc = InteractiveScene(_build_scene(loader, m, usd, sim_utils))
            try:
                km = int(sim_app._app.get_build_version()[:3])
            except Exception:
                km = 0
            if not ((not sc.cfg.replicate_physics and sc.cfg.filter_collisions) or (km >= 107 and sim_device == "cpu")):
                sc.filter_collisions()
            origins = sc.env_origins[:m].clone()
            for j in range(m):
                pth = _env_path_for(sc, j)
                _spawn_bin_walls(
                    pth, origins[j], sim_utils, box_wall_thickness=float(getattr(args, "box_wall_thickness", 0.01)), box_horizontal_padding=float(getattr(args, "box_horizontal_padding", 0.15)), box_vertical_padding=float(getattr(args, "box_vertical_padding", 0.06))
                )
            _apply_bin_liftout_tack_friction(sc, m)
            simx.reset()
            _try_disable_ccd_on_bin_prims(sc, m)
            root0 = compose_object_relative_grasps_world(origins, pr, qe)
            grip = sc["gripper"]
            obj = sc["object"]
            # “Closed” template for *all* DOFs from soft joint limits, then cspace (YAML) overrides. Using only
            # ``default_joint_pos`` leaves non-cspace mimics in the USD open bind — inconsistent pad contact
            # across grasps that all list the same :file:`cspace` keys.
            j_template = _gripper_joint_template_soft_closed(grip, m, loader)
            obj_st = obj.data.default_root_state.clone()
            obj_st[:, :3] = origins
            obj_st[:, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=obj_st.device, dtype=obj_st.dtype
            ).expand(m, 4)
            zv = torch.zeros_like(grip.data.default_joint_pos)
            zr = torch.zeros_like(grip.data.default_root_state[:, 7:])
            _bump_gripper_velocity_limits_for_teleport(grip)
            _reassert_gripper_in_sim(
                grip,
                j_template,
                c_b,
                loader.cspace_joint_names,
                zv,
                zr,
                root0,
                sc,
                obj,
                obj_st,
                write_joint_state=True,
            )
            _print_force_headed_play_hint(
                sim_app, do_render, bool(getattr(args, "force_headed", False))
            )
            fh = bool(getattr(args, "force_headed", False))
            _reset_mug_debug_stats()
            for si in range(settle):
                _reassert_gripper_in_sim(
                    grip, j_template, c_b, loader.cspace_joint_names, zv, zr, root0, sc, None, None
                )
                if not do_render:
                    st_render = False
                elif fh:
                    st_render = (si % 8 == 0) or (si + 1 == settle)
                else:
                    st_render = bool(si + 1 == settle)
                _physics_step_scene(simx, sc, sim_dt, render=st_render, substeps=max(1, int(BIN_LIFTOUT_PHYS_SUBSTEPS)))
                _reassert_mug_plan_centered_upright(obj, origins, m, settle=True)
                _reassert_gripper_in_sim(
                    grip, j_template, c_b, loader.cspace_joint_names, zv, zr, root0, sc, None, None
                )
            _print_mug_debug_stats(f"\n[bin_liftout debug] post-settle batch {bi}/{bnum}:")
            wall_tops: List[float] = []
            for j in range(m):
                epp = _env_path_for(sc, j)
                aab = [_aligned_world_aabb6(f"{epp}/Box/{n}") for n in ("W1", "W2", "W3", "W4")]
                wall_tops.append(_wall_rim_max_z(aab))
            # Lift: teleport the gripper root **every physics substep** by ``lift_dz / SUBSTEPS`` (≈ 0.3 mm
            # at the defaults), not in 2.5 mm chunks per outer step. With per-outer-step teleports the
            # gripper sat stationary for 8 substeps after each jump, and any grasp on a *vertically curving*
            # part of the mug (handle, taper) lost contact in the gap → mug fell out → "NO_BUDGE". Smaller
            # per-substep jumps keep the contact width nearly constant so PD friction can carry the mug up.
            n_substeps = max(1, int(BIN_LIFTOUT_PHYS_SUBSTEPS))
            lif = root0.clone()
            sub_dz = float(lift_dz) / float(n_substeps)
            total_substeps = int(lift_n) * n_substeps
            for k in range(total_substeps):
                lif = lif.clone()
                lif[:, 2] = lif[:, 2] + sub_dz
                _reassert_gripper_in_sim(
                    grip, j_template, c_b, loader.cspace_joint_names, zv, zr, lif, sc, None, None
                )
                # In ``--force_headed`` render every Nth substep so the hand visibly rises from the box;
                # headless: only the very last substep.
                render_lift = bool(do_render and ((fh and (k % n_substeps == 0)) or k + 1 == total_substeps))
                simx.step(render=render_lift)
                sc.update(sim_dt)
                _reassert_mug_plan_centered_upright(obj, origins, m, settle=False)
                _reassert_gripper_in_sim(
                    grip, j_template, c_b, loader.cspace_joint_names, zv, zr, lif, sc, None, None
                )
            sc.update(sim_dt)
            _print_mug_debug_stats(f"[bin_liftout debug] post-lift  batch {bi}/{bnum}:")
            com_pos = obj.data.root_pos_w[:m].detach().cpu().numpy()
            ori_np = origins[:m].detach().cpu().numpy()
            com_z = com_pos[:, 2]
            n_lift_ok = 0
            n_no_budge = 0
            n_flew = 0
            for j in range(m):
                gpos = s0 + j
                dz = float(com_pos[j, 2] - ori_np[j, 2])
                horiz = float(((com_pos[j, 0] - ori_np[j, 0]) ** 2 + (com_pos[j, 1] - ori_np[j, 1]) ** 2) ** 0.5)
                if float(com_z[j]) > float(wall_tops[j]) + rim_m:
                    out[gpos] = True
                    n_lift_ok += 1
                    cls = "LIFTED"
                elif dz < -0.10:
                    n_flew += 1
                    cls = "FELL_THROUGH"
                elif horiz > 0.10 or dz > float(BIN_LIFTOUT_MUG_MAX_Z_RISE_SETTLE_M) - 0.05:
                    n_flew += 1
                    cls = "FLEW"
                elif abs(dz) < 0.02 and horiz < 0.02:
                    n_no_budge += 1
                    cls = "NO_BUDGE"
                else:
                    cls = "PARTIAL"
                print_purple(
                    f"  env {gpos:3d}: dz={dz:+.3f} m  horiz={horiz:.3f} m  rim_top={float(wall_tops[j]):+.3f} m  -> {cls}",
                    flush=True,
                )
            print_purple(
                f"  batch {bi}/{bnum} summary: LIFTED={n_lift_ok}  NO_BUDGE={n_no_budge}  FLEW={n_flew}  m={m}",
                flush=True,
            )
            if bool(getattr(args, "force_headed", False)) and do_render and s0 + m >= n_total:
                print_purple("force_headed: stepping until window closes (last batch scene)...", flush=True)
                while sim_app.is_running():
                    _reassert_gripper_in_sim(
                        grip, j_template, c_b, loader.cspace_joint_names, zv, zr, lif, sc, None, None
                    )
                    simx.step(render=True)
                    sc.update(sim_dt)
                    _reassert_mug_plan_centered_upright(obj, origins, m, settle=False)
                    _reassert_gripper_in_sim(
                        grip, j_template, c_b, loader.cspace_joint_names, zv, zr, lif, sc, None, None
                    )
        s0 += m
    print()
    print_green(
        f"bin_liftout: {int(out.sum())} / {n_total} lifted (object com z > wall rim + {rim_m:g} m)"
    )
    return out


def main(args: argparse.Namespace) -> None:
    fh = bool(getattr(args, "force_headed", False))
    sim_app = start_isaac_lab_if_needed(
        file_name=__file__,
        headless=False if fh else bool(getattr(args, "headless", True)),
        wait_for_debugger_attach=bool(getattr(args, "wait_for_debugger_attach", False)),
    )
    apply_gripper_configuration(args)
    path = os.path.abspath(os.path.expanduser(args.grasp_file))
    with open(path, "r") as f:
        raw0 = yaml.unsafe_load(f)
    if not raw0 or "grasps" not in raw0:
        raise SystemExit("grasp file missing 'grasps'")

    if bool(args.only_in_bin_no_wall_collision) and not any(
        (v or {}).get("in_bin_no_wall_collision") is True for v in raw0["grasps"].values()
    ):
        print_yellow("No in_bin_no_wall_collision: true — will simulate all grasps in file.")
        object.__setattr__(args, "only_in_bin_no_wall_collision", False)

    loader = GraspInBinCollisionChecker(
        grasp_file=path,
        max_num_envs=max(1, int(args.max_num_envs)),
        max_num_grasps=0,
        env_spacing=float(args.env_spacing),
        box_wall_thickness=float(args.box_wall_thickness),
        box_horizontal_padding=float(args.box_horizontal_padding),
        box_vertical_padding=float(args.box_vertical_padding),
        start_with_pregrasp_cspace_position=bool(args.start_with_pregrasp_cspace_position),
        skip_missing_collision_ids=True,
        joint_motion_samples=1,
        bin_contact_force_eps=1.0,
        bin_contact_settle_substeps=1,
        bin_hit_method="contact",
        device=str(getattr(args, "device", "cuda:0")),
        object_args_override=args,
    )

    inds0 = _grasp_index_list(loader, raw0, bool(args.only_in_bin_no_wall_collision), int(args.max_num_grasps) or 0)
    if not inds0:
        raise SystemExit("no grasps to simulate after filter")

    flags = _run(loader, args, inds0)
    out = copy.deepcopy(raw0)
    out["created_with_bin_liftout"] = "bin_liftout"
    oks, bads = {}, {}
    for out_i, gi in enumerate(inds0):
        k = loader.grasp_keys[gi]
        ok = bool(flags[out_i])
        out["grasps"][k]["liftout_success"] = ok
        (oks if ok else bads)[k] = out["grasps"][k]
    base = f"{os.path.splitext(path)[0]}"
    p_ok, p_bad, p_all = f"{base}_liftout_ok.yaml", f"{base}_liftout_fail.yaml", f"{base}_liftout_annotated.yaml"
    save_yaml(out, p_all)
    print_green(f"Wrote {p_all} (all grasps with liftout_success on those simulated)")
    if oks:
        t = copy.deepcopy(out)
        t["grasps"] = oks
        save_yaml(t, p_ok)
        print_green(f"Wrote {p_ok} ({len(oks)} grasps)")
    if bads:
        t = copy.deepcopy(out)
        t["grasps"] = bads
        save_yaml(t, p_bad)
        print_green(f"Wrote {p_bad} ({len(bads)} grasps)")
    if not fh and sim_app is not None:
        sim_app.close()


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Lift a dynamic mug from bin-clear grasps; gravity on; one grasp per env."
    )
    add_bin_liftout_args(pa, globals(), **collect_bin_liftout_args(globals()))
    main(pa.parse_args())