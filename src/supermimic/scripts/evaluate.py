"""Evaluate each reference clip once using the paper's Appendix F protocol."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg, types
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

import supermimic.tasks  # noqa: F401
from supermimic.representations import extract_actor_goal
from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
from supermimic.runners.supermimic_runner import (
  WorldModel,
  build_actor,
  build_world_model,
)
from supermimic.tasks.task_spec import build_actor_obs_td
from supermimic.utils.wandb_artifacts import (
  checkpoint_hashes,
  download_model_dir,
  verified_snapshot,
)


@dataclass
class EvalConfig:
  motion_file: str
  output: Path
  wandb_path: str | None = None
  snapshot: Path | None = None
  num_envs: int = 256
  device: str = "cuda:0"
  seed: int = 0
  start_motion: int = 0
  limit: int | None = None
  """Evaluate this many consecutive clips; omit for the full loaded library."""
  video: bool = False
  """Record playback.mp4. Requires num_envs=1; includes resets between clips."""


def configure_evaluation(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep context fields for checkpoint compatibility, but make them constant."""
  if set(cfg.events) != {"foot_friction", "base_com", "encoder_bias"}:
    raise ValueError("Evaluation expects the three SuperMimic physics events")
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["base_com"].params["ranges"] = dict.fromkeys(range(3), (0.0, 0.0))
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  for group in cfg.observations.values():
    group.enable_corruption = False
  command = cfg.commands["motion"]
  assert isinstance(command, MotionCommandCfg)
  command.pose_range = {}
  command.velocity_range = {}
  command.joint_position_range = (0.0, 0.0)
  command.sampling_mode = "start"
  command.resampling_time_range = (1e9, 1e9)
  cfg.episode_length_s = 1e9
  cfg.auto_reset = False


def advance_history(
  history: torch.Tensor | None,
  sensor: torch.Tensor,
  length: int,
  reset: torch.Tensor,
) -> torch.Tensor:
  """Repeat the first observation on reset; never carry history across clips."""
  if history is None:
    return sensor[:, None].repeat(1, length, 1)
  history = torch.cat((history[:, 1:], sensor[:, None]), dim=1)
  history[reset] = sensor[reset, None]
  return history


def reset_clips(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, motion_ids: torch.Tensor
) -> types.VecEnvObs:
  """Reset all managers, then place each slot on its requested first frame.

  The generic env.reset advances the motion command even at dt=0. Override
  that frame *after* reset, using the same state writer as reset_to_frame.
  This preserves the release's soft joint-limit clipping at initialization.
  """
  env.reset(env_ids=env_ids)
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  motion = command.motion
  starts = torch.cat((motion.motion_ends.new_zeros(1), motion.motion_ends[:-1]))
  frames = starts[motion_ids]
  command.set_time_steps(env_ids, frames)
  command._write_reference_state_to_sim(
    env_ids,
    motion.root_pos_w[frames] + env.scene.env_origins[env_ids],
    motion.root_quat_w[frames],
    motion.root_lin_vel_w[frames],
    motion.root_ang_vel_w[frames],
    motion.joint_pos[frames],
    motion.joint_vel[frames],
  )
  env.scene.write_data_to_sim(env_ids=env_ids)
  env.sim.forward()
  command.update_relative_body_poses()
  env.sim.sense()
  # env.reset() cached observations before the requested frame was written.
  # Clear that cache and reseed only reset slots' observation buffers; live
  # slots must keep their existing history and delay timelines.
  env.observation_manager.reset(env_ids=env_ids)
  return env.observation_manager.compute(update_history=True, env_ids=env_ids)


@dataclass
class ClipResult:
  motion_index: int
  motion_name: str
  expected_steps: int
  steps: int
  failed: bool
  termination_terms: list[str]
  episode_return: float
  normalized_return: float


def clip_result(
  index: int,
  name: str,
  expected: int,
  steps: int,
  reward: float,
  peak_reward: float,
  failed: bool,
  motion_end: bool,
  terms: list[str],
) -> ClipResult:
  if not 0 < steps <= expected:
    raise RuntimeError(f"{name}: invalid rollout length {steps}/{expected}")
  if motion_end != (steps == expected):
    raise RuntimeError(f"{name}: motion boundary disagrees with rollout length")
  if not failed and not motion_end:
    raise RuntimeError(f"{name}: unexpected truncation before motion end")
  return ClipResult(
    index,
    name,
    expected,
    steps,
    failed,
    terms,
    reward,
    reward / (expected * peak_reward),
  )


def require_finite(name: str, value: torch.Tensor) -> None:
  if not torch.isfinite(value).all():
    raise FloatingPointError(f"Non-finite {name} during evaluation")


@torch.no_grad()
def evaluate_policy(
  env: ManagerBasedRlEnv,
  wm: WorldModel,
  actor,
  motion_ids: list[int],
  on_frame: Callable[[], None] | None = None,
) -> Iterator[ClipResult]:
  """Fill vacated slots with the next clip; discard idle slots at the end."""
  if env.cfg.auto_reset:
    raise ValueError("Evaluation requires auto_reset=False")
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  motion = command.motion
  if not motion_ids or len(set(motion_ids)) != len(motion_ids):
    raise ValueError("Select a nonempty list of distinct motions")
  if min(motion_ids) < 0 or max(motion_ids) >= len(motion.motion_names):
    raise ValueError("Motion index outside the loaded library")
  if env.num_envs > len(motion_ids):
    raise ValueError("num_envs must not exceed the selected motion count")
  lengths = torch.diff(motion.motion_ends, prepend=motion.motion_ends.new_zeros(1))
  if (lengths < 2).any():
    raise ValueError("Motion clips must have at least two frames")
  peak_reward = sum(max(term.weight, 0.0) for term in env.cfg.rewards.values())
  if peak_reward != 5.0:
    raise ValueError("Expected the SuperMimic reward with maximum rate 5.0")
  if env.cfg.scale_rewards_by_dt:
    peak_reward *= env.step_dt
  actor.eval()
  actor.set_slot_mix({"motion": 1.0})
  wm.dynamics.eval()
  selected = torch.tensor(motion_ids, device=env.device)
  slots = selected[: env.num_envs].clone()
  cursor = env.num_envs
  steps = torch.zeros_like(slots)
  returns = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
  reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  obs = reset_clips(env, torch.arange(env.num_envs, device=env.device), slots)
  history = None
  while (slots >= 0).any():
    full_state, actor_obs, goal = obs["full_state"], obs["actor"], obs["goal"]
    assert isinstance(full_state, torch.Tensor)
    assert isinstance(actor_obs, torch.Tensor)
    assert isinstance(goal, torch.Tensor)
    for name, value in (
      ("full_state", full_state),
      ("actor", actor_obs),
      ("goal", goal),
    ):
      require_finite(name, value)
    if on_frame is not None:
      on_frame()
    sensor = wm.representation.compute_actor_sensor(
      wm.representation.represent(full_state),
      wm.sim_context.read(env),
      sim_actor_obs=(
        actor_obs if wm.representation.actor_obs_mode == "proprio" else None
      ),
      noisy=False,
    )
    history = advance_history(history, sensor, actor.history_length, reset)
    actions = actor(build_actor_obs_td({"motion": extract_actor_goal(goal)}, history))
    require_finite("actions", actions)
    obs, rewards, terminated, truncated, _ = env.step(actions)
    # Check raw physics too: mjlab's reward manager replaces nonfinite terms
    # with zero, so finite returned rewards alone cannot validate a rollout.
    require_finite("physics qpos", env.sim.data.qpos)
    require_finite("physics qvel", env.sim.data.qvel)
    require_finite("rewards", rewards)
    active = slots >= 0
    steps[active] += 1
    returns[active] += rewards[active].double()
    if (steps[active] > lengths[slots[active]]).any():
      raise RuntimeError("Motion completed without a termination signal")
    reset = terminated | truncated
    for slot in (reset & active).nonzero().flatten().tolist():
      index = int(slots[slot])
      terms = [
        name
        for name in env.termination_manager.active_terms
        if env.termination_manager.get_term(name)[slot]
      ]
      yield clip_result(
        index,
        motion.motion_names[index],
        int(lengths[index]),
        int(steps[slot]),
        float(returns[slot]),
        peak_reward,
        bool(terminated[slot]),
        bool(env.termination_manager.get_term("motion_end")[slot]),
        terms,
      )
      slots[slot] = -1
    done_ids = reset.nonzero().flatten()
    count = min(len(done_ids), len(selected) - cursor)
    slots[done_ids[:count]] = selected[cursor : cursor + count]
    cursor += count
    steps[done_ids] = 0
    returns[done_ids] = 0
    if (slots >= 0).any() and len(done_ids):
      # Idle slots still need reset to satisfy mjlab's manual-reset contract.
      # Their rollouts never contribute to metrics or consume another clip.
      assigned = slots[done_ids]
      assigned = torch.where(assigned >= 0, assigned, selected[0])
      obs = reset_clips(env, done_ids, assigned)
  if on_frame is not None:
    on_frame()


def evaluate(task_id: str, cfg: EvalConfig) -> None:
  if (
    cfg.num_envs < 1
    or cfg.start_motion < 0
    or (cfg.limit is not None and cfg.limit < 1)
  ):
    raise ValueError("num_envs/limit must be positive and start_motion nonnegative")
  if cfg.video and cfg.num_envs != 1:
    raise ValueError("Video recording requires --num-envs 1")
  start = time.monotonic()
  configure_torch_backends(deterministic=True)
  if (cfg.snapshot is None) == (cfg.wandb_path is None):
    raise ValueError("select one checkpoint with --snapshot or --wandb-path")
  if cfg.snapshot is not None:
    model_dir = cfg.snapshot
    record = verified_snapshot(model_dir, task_id)
    hashes = record["sha256"]
    runner_cfg = load_runner_cfg(record)
    wandb_path, run_state = record["wandb_path"], "local snapshot"
  else:
    import wandb

    run = wandb.Api().run(cfg.wandb_path)
    runner_cfg = load_runner_cfg(run.config)
    model_dir = download_model_dir(run)
    hashes = checkpoint_hashes(model_dir)
    wandb_path, run_state = cfg.wandb_path, run.state
  check_representation_matches(task_id, runner_cfg.representation)
  env_cfg = load_env_cfg(task_id, play=True)
  configure_evaluation(env_cfg)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = min(cfg.num_envs, cfg.limit or cfg.num_envs)
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = str(Path(cfg.motion_file).expanduser())
  cfg.output.mkdir(parents=True, exist_ok=False)
  metadata = {
    "complete": False,
    "task_id": task_id,
    "wandb_path": wandb_path,
    "run_state": run_state,
    "runner_cfg": asdict(runner_cfg),
    "evaluation_cfg": {
      **asdict(cfg),
      "output": str(cfg.output),
      "snapshot": str(cfg.snapshot) if cfg.snapshot else None,
    },
    "checkpoint_sha256": hashes,
    "protocol": {
      "paper": "https://arxiv.org/abs/2610.28480v2",
      "sections": "4 (normalized return), A (reward/termination), F (benchmark)",
      "foot_friction": 1.0,
      "domain_randomization": False,
      "observation_noise": False,
      "initialization": "first frame, release reset soft joint-limit clipping",
      "normalization": "full clip frame count * 5 * reward time scale",
      "aggregation": "report both equal-clip and frame-weighted means",
    },
  }
  report_path = cfg.output / "report.json"
  report_path.write_text(json.dumps(metadata, indent=2) + "\n")
  env = ManagerBasedRlEnv(
    env_cfg, device=cfg.device, render_mode="rgb_array" if cfg.video else None
  )
  writer = None
  try:
    wm = build_world_model(runner_cfg, RslRlVecEnvWrapper(env), cfg.device)
    wm.load_bundle(torch.load(model_dir / "world_model.pt", map_location=cfg.device))
    actor = build_actor(runner_cfg, wm, cfg.device)
    actor.load_state_dict(torch.load(model_dir / "actor.pt", map_location=cfg.device))
    command = env.command_manager.get_term("motion")
    assert isinstance(command, MotionCommand)
    total = len(command.motion.motion_names)
    end = total if cfg.limit is None else min(total, cfg.start_motion + cfg.limit)
    motion_ids = list(range(cfg.start_motion, end))
    if cfg.video:
      import imageio.v2 as imageio

      writer = imageio.get_writer(
        cfg.output / "playback.mp4", fps=round(1 / env.step_dt)
      )

    def record_frame() -> None:
      assert writer is not None
      frame = env.render()
      if frame is None:
        raise RuntimeError("Renderer returned no frame")
      writer.append_data(frame)

    results = []
    with (cfg.output / "clips.jsonl").open("x") as out:
      for result in evaluate_policy(
        env, wm, actor, motion_ids, record_frame if cfg.video else None
      ):
        out.write(json.dumps(asdict(result), allow_nan=False) + "\n")
        out.flush()
        results.append(result)
        if len(results) % 1000 == 0:
          print(f"[INFO] Evaluated {len(results)}/{len(motion_ids)} clips", flush=True)
    if sorted(r.motion_index for r in results) != motion_ids:
      raise RuntimeError("Evaluation did not cover each selected clip exactly once")
    if writer is not None:
      writer.close()
      writer = None
    frames = sum(r.expected_steps for r in results)
    metadata.update(
      {
        "complete": True,
        "elapsed_seconds": time.monotonic() - start,
        "loaded_motions": total,
        "evaluated_motions": len(results),
        "failed_motions": sum(r.failed for r in results),
        "expected_steps": frames,
        "executed_steps": sum(r.steps for r in results),
        "mean_normalized_return": sum(r.normalized_return for r in results)
        / len(results),
        "frame_weighted_normalized_return": sum(
          r.normalized_return * r.expected_steps for r in results
        )
        / frames,
      }
    )
    report_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"[INFO] Evaluation complete: {report_path}")
  finally:
    if writer is not None:
      writer.close()
    env.close()


def main() -> None:
  task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  evaluate(task, tyro.cli(EvalConfig, args=remaining_args, config=mjlab.TYRO_FLAGS))


if __name__ == "__main__":
  main()
