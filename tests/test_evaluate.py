"""Benchmark accounting: full-clip denominators, boundaries and reset history."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from mjlab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand

from supermimic.scripts import evaluate as ev


def test_early_failure_keeps_full_clip_denominator():
  result = ev.clip_result(0, "fall", 100, 10, 1.0, 0.1, True, False, ["anchor_ori"])
  assert result.failed
  assert result.normalized_return == pytest.approx(0.1)


def test_failure_on_last_frame_is_still_failure():
  result = ev.clip_result(
    0, "last", 100, 100, 9.0, 0.1, True, True, ["motion_end", "anchor_pos"]
  )
  assert result.failed
  assert result.normalized_return == pytest.approx(0.9)


@pytest.mark.parametrize(
  "steps,failed,end",
  [
    (9, False, False),
    (9, False, True),
    (10, True, False),
    (11, True, True),
  ],
)
def test_bad_boundaries_cannot_be_reported_as_complete(steps, failed, end):
  with pytest.raises(RuntimeError):
    ev.clip_result(0, "bad", 10, steps, 0, 0.1, failed, end, [])


def test_history_reset_does_not_mix_clips_or_reset_live_slots():
  initial = torch.tensor([[1.0], [10.0]])
  history = ev.advance_history(None, initial, 3, torch.ones(2, dtype=torch.bool))
  history = ev.advance_history(
    history, initial + 1, 3, torch.zeros(2, dtype=torch.bool)
  )
  assert history[:, :, 0].tolist() == [[1, 1, 2], [10, 10, 11]]
  history = ev.advance_history(
    history, torch.tensor([[100.0], [12.0]]), 3, torch.tensor([True, False])
  )
  assert history[:, :, 0].tolist() == [[100, 100, 100], [10, 11, 12]]


@pytest.mark.parametrize("task", ["SuperMimic-GenCoord-G1", "SuperMimic-MaxCoord-G1"])
def test_eval_has_constant_physics_but_preserves_training_termination(task):
  cfg = load_env_cfg(task, play=True)
  ev.configure_evaluation(cfg)
  assert cfg.events["foot_friction"].params["ranges"] == (1.0, 1.0)
  assert set(cfg.events["base_com"].params["ranges"].values()) == {(0.0, 0.0)}
  assert cfg.events["encoder_bias"].params["bias_range"] == (0.0, 0.0)
  assert not any(group.enable_corruption for group in cfg.observations.values())
  assert cfg.commands["motion"].joint_position_range == (0.0, 0.0)
  assert not cfg.auto_reset
  assert cfg.terminations["anchor_pos"].params["threshold"] == 0.25
  assert cfg.terminations["anchor_ori"].params["threshold"] == 0.8
  assert "ee_body_pos" not in cfg.terminations


def test_nonfinite_physics_is_not_hidden_by_finite_reward():
  with pytest.raises(FloatingPointError, match="physics"):
    ev.require_finite("physics qvel", torch.tensor([float("nan")]))


def test_clip_reset_refreshes_cached_observations_without_advancing_live_history():
  command = MagicMock(spec=MotionCommand)
  command.time_steps = torch.zeros(2, dtype=torch.long)
  command.motion = SimpleNamespace(
    motion_ends=torch.tensor([3, 7]),
    root_pos_w=torch.arange(21).float().reshape(7, 3),
    root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(7, 1),
    root_lin_vel_w=torch.zeros(7, 3),
    root_ang_vel_w=torch.zeros(7, 3),
    joint_pos=torch.zeros(7, 1),
    joint_vel=torch.zeros(7, 1),
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    position=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene=SimpleNamespace(env_origins=torch.zeros(2, 3), write_data_to_sim=MagicMock()),
    sim=SimpleNamespace(forward=MagicMock(), sense=MagicMock()),
  )
  env.observation_manager = ObservationManager(
    {
      "full_state": ObservationGroupCfg(
        terms={"position": ObservationTermCfg(func=lambda e: e.position)}
      ),
      "goal": ObservationGroupCfg(
        terms={
          "frame": ObservationTermCfg(
            func=lambda e: command.time_steps.float().unsqueeze(-1)
          )
        }
      ),
      "actor": ObservationGroupCfg(
        terms={
          "position": ObservationTermCfg(
            func=lambda e: e.position, history_length=3, flatten_history_dim=False
          )
        }
      ),
    },
    env,
  )
  env.observation_manager.compute(update_history=True)
  env.position += 1
  live_history = env.observation_manager.compute(update_history=True)["actor"][
    1
  ].clone()

  def generic_reset(*, env_ids):
    env.position[env_ids] = -5
    command.time_steps[env_ids] = 1
    env.observation_manager.reset(env_ids)
    return env.observation_manager.compute(update_history=True, env_ids=env_ids), {}

  def set_frames(env_ids, frames):
    command.time_steps[env_ids] = frames

  def write_state(env_ids, root_pos, *args):
    env.position[env_ids] = root_pos

  env.reset = generic_reset
  command.set_time_steps.side_effect = set_frames
  command._write_reference_state_to_sim.side_effect = write_state
  obs = ev.reset_clips(env, torch.tensor([0]), torch.tensor([1]))
  expected = command.motion.root_pos_w[3]
  torch.testing.assert_close(obs["full_state"][0], expected)
  assert obs["goal"][0].item() == 3
  torch.testing.assert_close(obs["actor"][0], expected.repeat(3, 1))
  torch.testing.assert_close(obs["actor"][1], live_history)
  assert env.observation_manager.compute() is obs


def test_scheduling_visits_every_clip_once_and_resets_each_history(monkeypatch):
  # Slot 0 finishes a short clip while slot 1 runs a longer one. Its next clip
  # fails early; slot 0 is then idle while the last assigned clip completes.
  lengths = torch.tensor([2, 5, 4, 6])
  command = MagicMock(spec=MotionCommand)
  command.motion = SimpleNamespace(
    motion_ends=lengths.cumsum(0), motion_names=["a", "b", "c", "d"]
  )
  assigned = torch.zeros(2, dtype=torch.long)
  elapsed = torch.zeros(2, dtype=torch.long)
  terms = {name: torch.zeros(2, dtype=torch.bool) for name in ("motion_end", "fall")}
  resets = []

  def observations():
    values = (assigned * 100 + elapsed).float().unsqueeze(1)
    return {"full_state": values, "actor": values, "goal": torch.zeros(2, 670)}

  def reset_clips(env, env_ids, ids):
    assigned[env_ids] = ids
    elapsed[env_ids] = 0
    resets.append((env_ids.clone(), ids.clone()))
    return observations()

  def step(actions):
    elapsed.add_(1)
    terms["motion_end"] = elapsed == lengths[assigned]
    terms["fall"] = (assigned == 2) & (elapsed == 1)
    return observations(), torch.full((2,), 0.1), terms["fall"], terms["motion_end"], {}

  env = SimpleNamespace(
    cfg=SimpleNamespace(
      auto_reset=False,
      scale_rewards_by_dt=True,
      rewards={"tracking": SimpleNamespace(weight=5.0)},
    ),
    command_manager=SimpleNamespace(get_term=lambda name: command),
    termination_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.get),
    num_envs=2,
    device="cpu",
    step_dt=0.02,
    step=step,
    sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2), qvel=torch.zeros(2))),
  )
  wm = SimpleNamespace(
    representation=SimpleNamespace(
      actor_obs_mode="proprio",
      represent=lambda value: value,
      compute_actor_sensor=lambda *args, sim_actor_obs, noisy: sim_actor_obs,
    ),
    sim_context=SimpleNamespace(read=lambda env: torch.zeros(2, 1)),
    dynamics=MagicMock(),
  )
  histories = []
  actor = MagicMock()
  actor.history_length = 3

  def act(obs):
    histories.append(obs["history"].clone())
    assert not torch.is_grad_enabled()
    return torch.zeros(2, 1)

  actor.side_effect = act
  monkeypatch.setattr(ev, "reset_clips", reset_clips)
  results = list(ev.evaluate_policy(env, wm, actor, [0, 1, 2, 3]))
  by_id = {r.motion_index: r for r in results}
  assert len(results) == len(by_id) == 4
  assert [by_id[i].steps for i in range(4)] == [2, 5, 1, 6]
  assert [by_id[i].failed for i in range(4)] == [False, False, True, False]
  assert by_id[2].normalized_return == pytest.approx(0.25)
  assert by_id[3].normalized_return == pytest.approx(1.0)
  assert histories[2][0, :, 0].tolist() == [200, 200, 200]
  assert histories[2][1, :, 0].tolist() == [100, 101, 102]
  assert histories[3][0, :, 0].tolist() == [300, 300, 300]
  # More resets than scored episodes: idle slots are reset but never counted.
  assert sum(len(ids) for ids, _ in resets) > 4
