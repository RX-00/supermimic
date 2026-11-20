"""Verify the comparison split and pack a single partition for the native trainer."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from mjlab.scripts.pack_motions import FIELDS


def pack(manifest, canonical, converted, split, output):
  with manifest.open() as f:
    rows = list(csv.DictReader(f))
  for key in ("motion", "sha256", "actor", "content"):
    memberships = defaultdict(set)
    for r in rows:
      memberships[r[key]].add(r["split"])
    if any(len(parts) != 1 for parts in memberships.values()):
      raise ValueError(f"Cross-partition {key}")
  selected = {r["motion"]: r for r in rows if r["split"] == split}
  if not selected or set(selected) != {
    p.stem for p in (converted / split / "npz").glob("*.npz")
  }:
    raise ValueError("Mismatched comparison partition")
  with tempfile.TemporaryDirectory(dir=output.parent) as workdir:
    temp = Path(workdir)
    inputs = temp / "npz"
    inputs.mkdir()
    for name, r in selected.items():
      path = canonical / (name + ".npz")
      with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != r["sha256"]:
          raise ValueError(f"Canonical reference hash mismatch for {name}")
      with (
        np.load(path) as ref,
        np.load(converted / split / "npz" / path.name, allow_pickle=True) as wbc,
      ):
        for k in ("joint_names", "joint_pos", "joint_vel", "fps"):
          np.testing.assert_array_equal(ref[k], wbc[k])
        for k in ("pos_w", "lin_vel_w", "ang_vel_w"):
          np.testing.assert_allclose(
            ref["root_" + k], wbc["body_" + k][:, 0], atol=2e-5
          )
        qdot = (ref["root_quat_w"] * wbc["body_quat_w"][:, 0]).sum(axis=1)
        np.testing.assert_allclose(np.abs(qdot), 1, atol=2e-6)
      (inputs / path.name).symlink_to(path.resolve())
    packed = temp / "packed"
    subprocess.run(
      [
        sys.executable,
        "-m",
        "mjlab.scripts.pack_motions",
        "--input",
        str(inputs / "*.npz"),
        "--output",
        str(packed),
      ],
      check=True,
    )
    names = json.loads((packed / "motion_names.json").read_text())
    ends = np.load(packed / "motion_ends.npy")
    np.testing.assert_array_equal(names, sorted(selected))
    np.testing.assert_array_equal(
      np.diff(ends, prepend=0), [int(selected[n]["frames"]) for n in names]
    )
    arrays = {k: np.load(packed / (k + ".npy"), mmap_mode="r") for k in FIELDS}
    for name, start, end in zip(names, np.r_[0, ends[:-1]], ends, strict=True):
      with np.load(inputs / (name + ".npz")) as ref:
        for k, v in arrays.items():
          np.testing.assert_array_equal(v[start:end], ref[k])
    packed.rename(output)


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  for k in ("manifest", "canonical", "converted", "output"):
    p.add_argument(k, type=Path)
  p.add_argument("--split", required=True, choices=("train", "validation"))
  pack(**vars(p.parse_args()))
