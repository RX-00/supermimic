"""Convert and filter BONES-SEED G1 CSVs for the research mjlab loader.

The chosen rules live in configs/bones_seed_preprocessing.json. Raw files are
never modified. Only retained clips are written to npz/; manifest.csv records
every input and exclusion. The existing mjlab pack_motions script packs npz/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import re
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.scripts.csv_to_npz import JOINT_NAMES
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from scipy.spatial.transform import Rotation, Slerp

CSV_COLUMNS = (
  ["Frame"]
  + [f"root_translate{axis}" for axis in "XYZ"]
  + [f"root_rotate{axis}" for axis in "XYZ"]
  + [f"{joint}_dof" for joint in JOINT_NAMES]
)


class BonesMotionLoader(CsvMotionLoader):
  """Reuse mjlab's velocity calculation and output schema with BONES input."""

  def _load_motion(self) -> None:
    with open(self.motion_file, newline="") as source:
      columns = next(csv.reader(source))
      if columns != CSV_COLUMNS:
        raise ValueError(f"Unexpected CSV columns: {self.motion_file}")
      raw = np.loadtxt(source, delimiter=",", ndmin=2)
    if raw.shape[1] != len(CSV_COLUMNS) or not np.isfinite(raw).all():
      raise ValueError(f"Invalid numeric motion data: {self.motion_file}")
    if not np.array_equal(raw[:, 0], np.arange(len(raw))):
      raise ValueError(f"Nonconsecutive frame indices: {self.motion_file}")
    self.input_frames = len(raw)
    self.duration = (self.input_frames - 1) * self.input_dt
    if self.duration < 2 * self.output_dt:
      raise ValueError(f"Motion is too short to derive velocities: {self.motion_file}")
    self._positions = raw[:, 1:4] / 100.0
    self._rotations = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True)
    self._joints = np.deg2rad(raw[:, 7:])

  def _interpolate_motion(self) -> None:
    # Float64 time indices avoid crossing the final input frame through rounding.
    self.output_frames = math.floor(self.duration / self.output_dt + 1e-6) + 1
    times = np.arange(self.output_frames, dtype=np.float64) * self.output_dt
    times = np.minimum(times, self.duration)
    input_times = np.arange(self.input_frames, dtype=np.float64) * self.input_dt

    def linear(values: np.ndarray) -> torch.Tensor:
      result = np.column_stack(
        [np.interp(times, input_times, column) for column in values.T]
      )
      return torch.from_numpy(result.astype(np.float32))

    self.motion_base_poss = linear(self._positions)
    self.motion_dof_poss = linear(self._joints)
    # SciPy provides batch SLERP, including nearly equal and antipodal rotations.
    quats = Slerp(input_times, self._rotations)(times).as_quat()[:, [3, 0, 1, 2]]
    self.motion_base_rots = torch.from_numpy(quats.astype(np.float32))


def longest_run(mask: np.ndarray) -> int:
  edges = np.diff(np.r_[False, mask, False].astype(np.int8))
  starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
  return int(np.max(ends - starts, initial=0))


def name_filter(name: str, spec: dict) -> str:
  matches = [word for word in spec["name_filter"]["keywords"] if word in name.lower()]
  return ";".join(matches)


class FeasibilityFilter:
  def __init__(self, spec: dict) -> None:
    self.rules = spec["paper_feasibility_rules"]
    self.gates = spec["assumed_floating_feet_gates"]
    model_spec = mujoco.MjSpec.from_file(str(G1_XML))
    model_spec.worldbody.add_geom(
      name="preprocessing_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1]
    )
    self.model = model_spec.compile()
    self.data = mujoco.MjData(self.model)
    self.floor_id = self.model.geom("preprocessing_floor").id
    self.foot_ids = [
      i
      for i in range(self.model.ngeom)
      if re.fullmatch(r"(left|right)_foot[1-7]_collision", self.model.geom(i).name)
    ]
    if len(self.foot_ids) != 14:
      raise ValueError("Expected all 14 G1 foot collision geometries")
    joints = [self.model.joint(i).name for i in range(1, self.model.njnt)]
    if joints != list(JOINT_NAMES):
      raise ValueError("G1 model and CSV joint orders differ")

  def _set_frame(self, arrays: dict, frame: int) -> None:
    self.data.qpos[:3] = arrays["root_pos_w"][frame]
    self.data.qpos[3:7] = arrays["root_quat_w"][frame]
    self.data.qpos[7:] = arrays["joint_pos"][frame]
    mujoco.mj_kinematics(self.model, self.data)

  def foot_clearance(self) -> float:
    return min(
      mujoco.mj_geomDistance(self.model, self.data, i, self.floor_id, 10.0, None)
      for i in self.foot_ids
    )

  def __call__(self, arrays: dict) -> str:
    frames = len(arrays["joint_pos"])
    heights = np.empty(frames)
    for frame in range(frames):
      self._set_frame(arrays, frame)
      heights[frame] = self.data.xpos[1:, 2].min()

    elevated = self.rules["elevated_support"]
    if (
      longest_run(heights > elevated["minimum_body_height_gt_m"])
      >= elevated["consecutive_frames_gte"]
    ):
      return "elevated_support"
    penetration = self.rules["ground_penetration"]
    if (heights < penetration["any_body_height_lt_m"]).any() or (
      heights < penetration["or_body_height_lt_m"]
    ).mean() > penetration["fraction_gt"]:
      return "ground_penetration"

    q = arrays["root_quat_w"]
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    z = arrays["root_pos_w"][:, 2]
    speed = np.linalg.norm(arrays["root_lin_vel_w"], axis=1)
    joint_speed = np.sqrt(np.mean(arrays["joint_vel"] ** 2, axis=1))
    gates = self.gates
    stationary = (
      (up > gates["pelvis_up_dot_world_up_gt"])
      & (z > gates["pelvis_height_gt_m"])
      & (z < gates["pelvis_height_lt_m"])
      & (speed < gates["pelvis_linear_speed_lt_m_per_s"])
      & (joint_speed < gates["joint_velocity_rms_lt_rad_per_s"])
    )
    floating = self.rules["floating_feet"]
    if longest_run(stationary) >= floating["consecutive_frames_gte"]:
      airborne = np.zeros(frames, dtype=bool)
      for frame in np.flatnonzero(stationary):
        self._set_frame(arrays, int(frame))
        airborne[frame] = self.foot_clearance() > floating["both_foot_clearances_gt_m"]
      if longest_run(airborne) >= floating["consecutive_frames_gte"]:
        return "floating_feet"
    return ""


_SPEC: dict = {}
_FILTER: FeasibilityFilter | None = None
_RAW = Path()
_NPZ = Path()


def _init_worker(spec: dict, raw: Path, npz: Path) -> None:
  global _SPEC, _FILTER, _RAW, _NPZ
  torch.set_num_threads(1)
  _SPEC, _RAW, _NPZ = spec, raw, npz
  _FILTER = FeasibilityFilter(spec)


def _process(row: dict) -> dict:
  name = row["filename"]
  input_frames = int(row["move_duration_frames"])
  matched = name_filter(name, _SPEC)
  result = {
    "filename": name,
    "csv_path": row["move_g1_path"],
    "input_frames": input_frames,
    "output_frames": 0,
    "keep": 0,
    "reason": "name" if matched else "",
    "matched_keywords": matched,
    "npz_path": "",
  }
  if matched:
    return result
  loader = BonesMotionLoader(
    str(_RAW / row["move_g1_path"]),
    input_fps=_SPEC["conversion"]["input_fps"],
    output_fps=_SPEC["conversion"]["output_fps"],
  )
  if loader.input_frames != input_frames:
    raise ValueError(f"CSV/metadata frame count mismatch: {name}")
  arrays = loader.arrays()
  for key, value in arrays.items():
    if value.dtype.kind in "fiu" and not np.isfinite(value).all():
      raise ValueError(f"Nonfinite {key}: {name}")
  if not np.allclose(
    np.linalg.norm(arrays["root_quat_w"], axis=1), 1, atol=2e-6, rtol=0
  ):
    raise ValueError(f"Nonunit root quaternion: {name}")
  assert _FILTER is not None
  reason = _FILTER(arrays)
  result.update(output_frames=loader.output_frames, reason=reason, keep=int(not reason))
  if not reason:
    path = _NPZ / f"{name}.npz"
    with path.open("xb") as stream:
      np.savez(stream, **arrays)
    result["npz_path"] = str(path)
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--raw-root", type=Path, required=True)
  parser.add_argument("--metadata", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--spec", type=Path, default=Path("configs/bones_seed_preprocessing.json")
  )
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument("--limit", type=int, help="Limit inputs for a conversion check")
  args = parser.parse_args()
  if args.workers < 1 or (args.limit is not None and args.limit < 1):
    parser.error("workers and limit must be positive")
  spec_bytes = args.spec.read_bytes()
  spec = json.loads(spec_bytes)
  if spec["conversion"]["output_fps"] != 50:
    raise ValueError("The selected feasibility run lengths require 50 Hz")
  with args.metadata.open(newline="") as source:
    rows = list(csv.DictReader(source))
  rows.sort(key=lambda row: row["move_g1_path"])
  if len({row["filename"] for row in rows}) != len(rows):
    raise ValueError("Duplicate motion names would overwrite NPZ files")
  if args.limit is not None:
    rows = rows[: args.limit]
  if not rows:
    raise ValueError("No input motions")
  for row in rows:
    relative = Path(row["move_g1_path"])
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError(f"Invalid metadata path: {relative}")
    if not (args.raw_root / relative).is_file():
      raise FileNotFoundError(args.raw_root / relative)
  args.output.mkdir(parents=True, exist_ok=False)
  npz = args.output / "npz"
  npz.mkdir()
  (args.output / "preprocessing_spec.json").write_bytes(spec_bytes)
  counts: Counter = Counter()
  total_frames = 0
  start = last_report = time.monotonic()
  fields = [
    "filename",
    "csv_path",
    "input_frames",
    "output_frames",
    "keep",
    "reason",
    "matched_keywords",
    "npz_path",
  ]
  with (
    (args.output / "manifest.csv").open("w", newline="") as stream,
    mp.get_context("spawn").Pool(
      args.workers, initializer=_init_worker, initargs=(spec, args.raw_root, npz)
    ) as pool,
  ):
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for result in pool.imap(_process, rows, chunksize=16):
      writer.writerow(result)
      counts[result["reason"] or "kept"] += 1
      if result["keep"]:
        total_frames += result["output_frames"]
      now = time.monotonic()
      if now - last_report >= 20:
        stream.flush()
        print(f"Processed {counts.total():,}/{len(rows):,}; {dict(counts)}", flush=True)
        last_report = now
  report = {
    "complete": True,
    "limited_check": args.limit is not None,
    "input_clips": len(rows),
    "after_name_filter": len(rows) - counts["name"],
    "counts": dict(counts),
    "retained_frames": total_frames,
    "retained_hours": total_frames / spec["conversion"]["output_fps"] / 3600,
    "specification_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "robot_xml_sha256": hashlib.sha256(G1_XML.read_bytes()).hexdigest(),
    "elapsed_seconds": time.monotonic() - start,
  }
  (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
  main()
