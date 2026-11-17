"""BONES units, rotation continuity, loader compatibility and feasibility gates."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from mjlab.utils.forward_kinematics import ForwardKinematics

from supermimic.scripts.prepare_bones import (
  CSV_COLUMNS,
  BonesMotionLoader,
  FeasibilityFilter,
)


def write_motion(path, frames=121):
  times = np.arange(frames) / 120
  raw = np.zeros((frames, len(CSV_COLUMNS)))
  raw[:, 0] = np.arange(frames)
  raw[:, 1:4] = (np.array([0.3, -0.2, 0.8]) + times[:, None] * [2, -1, 0.1]) * 100
  raw[:, 6] = (179 + 10 * times + 180) % 360 - 180
  raw[:, 7:] = np.rad2deg(0.25 * times[:, None])
  np.savetxt(path, raw, delimiter=",", header=",".join(CSV_COLUMNS), comments="")
  return raw


@pytest.fixture(scope="module")
def checker():
  spec = Path(__file__).parents[1] / "configs/bones_seed_preprocessing.json"
  return FeasibilityFilter(json.loads(spec.read_text()))


def test_conversion_units_velocities_and_loader(tmp_path, checker):
  source = tmp_path / "linear.csv"
  write_motion(source)
  converted = BonesMotionLoader(str(source), input_fps=120, output_fps=50)
  arrays = converted.arrays()
  assert converted.output_frames == 51
  times = np.arange(51) / 50
  np.testing.assert_allclose(
    arrays["root_pos_w"],
    [0.3, -0.2, 0.8] + times[:, None] * [2, -1, 0.1],
    atol=2e-7,
    rtol=0,
  )
  expected_joint_pos = np.broadcast_to(0.25 * times[:, None], (51, 29))
  np.testing.assert_allclose(arrays["joint_pos"], expected_joint_pos, atol=2e-8, rtol=0)
  expected_vel = np.broadcast_to([2, -1, 0.1], (51, 3))
  np.testing.assert_allclose(arrays["root_lin_vel_w"], expected_vel, atol=1e-5, rtol=0)
  np.testing.assert_allclose(arrays["joint_vel"], 0.25, atol=1e-6, rtol=0)
  expected_omega = np.broadcast_to([0, 0, np.deg2rad(10)], (51, 3))
  np.testing.assert_allclose(
    arrays["root_ang_vel_w"], expected_omega, atol=2e-6, rtol=0
  )
  half_yaw = np.deg2rad(179 + 10 * times) / 2
  expected_quat = np.column_stack(
    [np.cos(half_yaw), np.zeros(51), np.zeros(51), np.sin(half_yaw)]
  )
  # Quaternion signs may differ across the Euler wrap; physical rotations must not.
  np.testing.assert_allclose(
    np.abs(np.sum(arrays["root_quat_w"] * expected_quat, axis=1)),
    1,
    atol=1e-7,
    rtol=0,
  )
  npz = tmp_path / "linear.npz"
  np.savez(npz, **arrays)
  model = checker.model
  fk = ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.from_numpy(model.jnt_qposadr[1:]),
    torch.arange(model.nbody - 1),
  )
  loaded = MotionLoader(str(npz), fk)
  assert loaded.motion_names == ["linear"]
  assert loaded.motion_ends.tolist() == [51]
  positions, quats, linear, angular = loaded.body_state(torch.tensor([0, 25, 50]))
  assert positions.shape == (3, model.nbody - 1, 3)
  for values in (positions, quats, linear, angular):
    assert torch.isfinite(values).all()
  np.testing.assert_allclose(positions[:, 0], arrays["root_pos_w"][[0, 25, 50]])


def test_non_grid_endpoint_is_not_extrapolated(tmp_path):
  source = tmp_path / "short.csv"
  write_motion(source, frames=120)
  converted = BonesMotionLoader(str(source), input_fps=120, output_fps=50)
  assert converted.output_frames == 50
  np.testing.assert_allclose(
    converted.arrays()["root_pos_w"][-1],
    [0.3 + 2 * 0.98, -0.2 - 0.98, 0.898],
    atol=2e-7,
    rtol=0,
  )


@pytest.mark.parametrize("problem", ["columns", "frame_indices", "nonfinite"])
def test_invalid_input_fails(tmp_path, problem):
  source = tmp_path / "invalid.csv"
  raw = write_motion(source)
  columns = CSV_COLUMNS.copy()
  if problem == "columns":
    columns[1] = "unknown_position_units"
  elif problem == "frame_indices":
    raw[5, 0] = 7
  else:
    raw[5, 1] = np.nan
  np.savetxt(source, raw, delimiter=",", header=",".join(columns), comments="")
  with pytest.raises(ValueError):
    BonesMotionLoader(str(source), input_fps=120, output_fps=50)


def stationary_motion(checker, clearance, frames=50):
  checker.data.qpos[:] = checker.model.qpos0
  mujoco.mj_kinematics(checker.model, checker.data)
  root = checker.data.qpos[:3].copy()
  root[2] += clearance - checker.foot_clearance()
  return {
    "root_pos_w": np.tile(root, (frames, 1)),
    "root_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
    "joint_pos": np.zeros((frames, 29)),
    "root_lin_vel_w": np.zeros((frames, 3)),
    "joint_vel": np.zeros((frames, 29)),
  }


@pytest.mark.parametrize(
  "clearance,frames,reason",
  [
    (0, 50, ""),
    (0.04, 24, ""),
    (0.04, 25, "floating_feet"),
    (0.3, 29, ""),
    (0.3, 30, "elevated_support"),
    (-0.2, 1, "ground_penetration"),
  ],
)
def test_feasibility_height_and_time_boundaries(checker, clearance, frames, reason):
  arrays = stationary_motion(checker, clearance, frames)
  assert checker(arrays) == reason


def test_floating_requires_stationary_gate(checker):
  arrays = stationary_motion(checker, 0.04)
  arrays["root_lin_vel_w"][:, 0] = 0.11
  assert checker(arrays) == ""
  arrays["root_lin_vel_w"][:] = 0
  arrays["joint_vel"][:] = 0.51
  assert checker(arrays) == ""


def test_penetration_fraction_boundary(checker):
  arrays = stationary_motion(checker, 0, frames=100)
  # With neutral joints, the lowest body link lies 0.035 m above the foot sole.
  arrays["root_pos_w"][:80, 2] -= 0.13
  assert checker(arrays) == ""
  arrays["root_pos_w"][80, 2] -= 0.13
  assert checker(arrays) == "ground_penetration"


@pytest.mark.parametrize("clearance", [-0.01, 0.0, 0.04])
def test_geom_distance_matches_capsule_bottom(checker, clearance):
  arrays = stationary_motion(checker, clearance)
  checker._set_frame(arrays, 0)
  bottoms = []
  for geom_id in checker.foot_ids:
    assert checker.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE
    radius, half_length = checker.model.geom_size[geom_id, :2]
    axis_z = checker.data.geom_xmat[geom_id].reshape(3, 3)[2, 2]
    bottoms.append(
      checker.data.geom_xpos[geom_id, 2] - radius - abs(axis_z) * half_length
    )
  assert checker.foot_clearance() == pytest.approx(min(bottoms), abs=1e-10)
  assert checker.foot_clearance() == pytest.approx(clearance, abs=1e-10)
