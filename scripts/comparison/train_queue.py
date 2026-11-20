"""Run independent GenCoord seeds serially under the workstation user service."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

RECIPE_SHA = "a4d4e9506b3433f301ae5f701a761885b8e31c407b678f3a4f80628088181a87"


def write(path, data):
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(data, indent=2) + "\n")
  tmp.replace(path)


def sha(path):
  with path.open("rb") as f:
    return hashlib.file_digest(f, "sha256").hexdigest()


def train(args):
  if not os.environ.get("INVOCATION_ID"):
    raise RuntimeError("Submit through a persistent systemd service")
  repo = Path(__file__).resolve().parents[2]
  out, packed = args.output.resolve(), args.packed.resolve()
  if sha(args.recipe) != RECIPE_SHA:
    raise ValueError("Expected archived accepted GenCoord run command")
  template = shlex.split(args.recipe.read_text())
  args_start = template.index("-m")
  out.mkdir(parents=True, exist_ok=False)
  shutil.copytree(
    repo / "src", out / "source", ignore=shutil.ignore_patterns("__pycache__")
  )
  shutil.copy2(args.recipe, out / "accepted-command.sh")
  shutil.copy2(repo / "uv.lock", out / "uv.lock")
  manifest = repo / "docs/pipeline-comparison/bones-motions.csv"
  shutil.copy2(manifest, out / manifest.name)
  env = dict(
    os.environ,
    PYTHONPATH=str(out / "source"),
    WANDB_ENTITY="a2r-lab",
    WANDB_RESUME="never",
    PYTHONUNBUFFERED="1",
    CUDA_VISIBLE_DEVICES="0",
  )
  plans = []
  for seed in args.seeds:
    cmd = [sys.executable, "-u", *template[args_start:]]
    for key, value in {
      "--motion-file": str(packed),
      "--env.seed": str(seed),
      "--runner.run-name": f"{out.name}-seed{seed}",
    }.items():
      cmd[cmd.index(key) + 1] = value
    plans.append({"seed": seed, "command": cmd})
  write(
    out / "queue.json",
    dict(
      plans=plans,
      service=args.service,
      recipe_sha256=RECIPE_SHA,
      manifest_sha256=sha(manifest),
      packed_sha256={p.name: sha(p) for p in sorted(packed.iterdir())},
      versions={
        k: version(k)
        for k in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "wandb")
      },
      source_commits={
        p.name: subprocess.check_output(
          ["git", "rev-parse", "HEAD"], cwd=p, text=True
        ).strip()
        for p in (repo, repo.parent / "mjlab_research")
      },
    ),
  )
  for plan in plans:
    run = out / f"seed-{plan['seed']}"
    run.mkdir()
    started = time.time()
    with (run / "training.log").open("w") as log, (run / "gpu.csv").open("w") as gpu:
      with subprocess.Popen(
        plan["command"], cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT
      ) as proc:
        state = dict(
          seed=plan["seed"],
          command=plan["command"],
          started=started,
          pid=proc.pid,
          cgroup=Path(f"/proc/{proc.pid}/cgroup").read_text().strip(),
        )
        write(run / "status.json", state)
        while proc.poll() is None:
          apps = subprocess.check_output(
            [
              "nvidia-smi",
              "--query-compute-apps=pid,used_memory,gpu_uuid",
              "--format=csv,noheader,nounits",
            ],
            text=True,
          )
          for line in apps.splitlines():
            if line.partition(",")[0].strip() == str(proc.pid):
              gpu.write(f"{time.time()},{line}\n")
              gpu.flush()
          try:
            proc.wait(timeout=10)
          except subprocess.TimeoutExpired:
            continue
        state.update(exit_code=proc.returncode, finished=time.time())
        state["allocated_gpu_hours"] = (state["finished"] - started) / 3600
        write(run / "status.json", state)
        if proc.returncode:
          raise SystemExit(proc.returncode)
    final = list(run.glob("logs/*/*/iteration-300000/snapshot.json"))
    if (
      len(final) != 1
      or json.loads(final[0].read_text())["completed_iterations"] != 300000
    ):
      raise RuntimeError(f"Missing final snapshot in {run}")


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  for k in ("recipe", "packed", "output"):
    p.add_argument(k, type=Path)
  p.add_argument("--service", required=True)
  p.add_argument("--seeds", nargs="+", type=int, required=True)
  train(p.parse_args())
