"""Export a verified native GenCoord actor and check recorded closed-loop observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from supermimic.representations import extract_actor_goal
from supermimic.runners.exporter import _SingleEncoderOnnx
from supermimic.runners.run_config import load_runner_cfg
from supermimic.runners.supermimic_runner import build_actor, build_world_model
from supermimic.scripts.evaluate import (
  advance_history,
  configure_evaluation,
  reset_clips,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.wandb_artifacts import verified_snapshot


@torch.inference_mode()
def export(snapshot, motion, output, device):
  task = "SuperMimic-GenCoord-G1"
  meta = verified_snapshot(snapshot, task)
  cfg = load_runner_cfg(meta)
  env_cfg = load_env_cfg(task, play=True)
  configure_evaluation(env_cfg)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 0
  env_cfg.commands["motion"].motion_file = str(motion)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  output.mkdir(parents=True, exist_ok=False)
  try:
    wm = build_world_model(cfg, RslRlVecEnvWrapper(env), device)
    actor = build_actor(cfg, wm, device).eval()
    actor.load_state_dict(torch.load(snapshot / "actor.pt", map_location=device))
    actor.set_slot_mix({"motion": 1.0})
    widths = [3, 3, 29, 29, 29]
    wrapper = _SingleEncoderOnnx(actor, "motion", widths).eval()
    group = env.action_manager.get_term("joint_pos")
    ids = torch.zeros(1, dtype=torch.long, device=device)
    obs = reset_clips(env, ids, ids)
    rows = {
      k: [] for k in ("qpos", "qvel", "frame", "history", "goal", "action", "input")
    }
    history = None
    reset = ids.bool()
    for _ in range(512):
      sensor = obs["actor"]
      history = advance_history(history, sensor, actor.history_length, reset)
      goal = extract_actor_goal(obs["goal"])
      action = actor(build_actor_obs_td({"motion": goal}, history))
      x = torch.cat(
        [*(c.flatten(1) for c in history.split(widths, dim=-1)), goal], dim=-1
      )
      torch.testing.assert_close(wrapper(x), action, rtol=1e-4, atol=1e-5)
      for k, v in dict(
        qpos=env.sim.data.qpos,
        qvel=env.sim.data.qvel,
        frame=env.command_manager.get_term("motion").time_steps,
        history=history,
        goal=goal,
        action=action,
        input=x,
      ).items():
        rows[k].append(v[0].cpu().numpy().copy())
      obs, _, term, trunc, _ = env.step(action)
      if bool((term | trunc).any()):
        break
    torch.jit.trace(wrapper.cpu(), x.cpu()).save(str(output / "actor.ts"))
    np.savez_compressed(
      output / "parity.npz", **{k: np.asarray(v) for k, v in rows.items()}
    )
    pd = env.cfg.scene.entities["robot"].articulation.actuators
    contract = dict(
      task=task,
      sha256=meta["sha256"],
      history_length=actor.history_length,
      names=group.target_names,
      offset=group.offset[0].tolist(),
      scale=group.scale[0].tolist(),
      state_joint_names=env.scene["robot"].joint_names,
      widths=widths,
      native_frames=len(rows["action"]),
      gains=[
        dict(patterns=a.target_names_expr, kp=a.stiffness, kd=a.damping) for a in pd
      ],
    )
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__)
  for arg in ("snapshot", "motion", "output"):
    ap.add_argument(arg, type=Path)
  ap.add_argument("--device", default="cpu")
  opts = ap.parse_args()
  export(**vars(opts))
