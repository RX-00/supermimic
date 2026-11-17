"""The world-model display horizon must not reset the real policy's memory."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from supermimic.scripts.viewers import SuperMimicViewerMixin


def _viewer(horizon=2):
  viewer = SuperMimicViewerMixin()
  command = SimpleNamespace(time_steps=torch.zeros(1, dtype=torch.long), motion=None)
  state = torch.zeros(1, 1)
  done = torch.tensor([False])

  def observations():
    return {"full_state": state.clone(), "goal": torch.zeros(1, 1)}

  def step(actions):
    state.add_(1)
    command.time_steps.add_(1)
    return observations(), torch.zeros(1), done.clone(), {}

  viewer.env = SimpleNamespace(
    get_observations=observations,
    step=step,
    unwrapped=SimpleNamespace(
      command_manager=SimpleNamespace(get_term=lambda name: command),
      sim=SimpleNamespace(mj_model=SimpleNamespace(nq=7)),
    ),
  )
  viewer.representation = SimpleNamespace(
    actor_obs_mode="rep_state",
    extract_aux_state=lambda x: x.clone(),
    represent=lambda x: x.clone(),
    compute_actor_sensor=lambda x, *args, **kwargs: x.clone(),
    compute_wm_goal=lambda *args: (torch.zeros(1, 1), None),
    joint_state=lambda x: x,
    reconstruct_full_state=lambda *args: torch.tensor([[0, 0, 1, 1, 0, 0, 0]]),
  )
  viewer.sim_context = SimpleNamespace(read=lambda env: None)
  viewer.dynamics = MagicMock(side_effect=lambda rep, act, aux, ctx: (aux, rep + 1))
  viewer.dynamics.num_actions = 1
  viewer.policy = MagicMock(side_effect=lambda obs: torch.zeros(1, 1))
  viewer.policy.history_length = 3
  viewer._build_actor_goal = lambda *args: torch.zeros(1, 1)
  viewer.current_task = "motion"
  viewer.num_steps_rollout = horizon
  viewer.free_joint_q_adr = np.arange(7)
  viewer.joint_q_adr = np.array([], dtype=int)
  viewer._is_paused = False
  viewer._step_count = viewer._stats_steps = 0
  viewer.pause = MagicMock()
  viewer.reset_env_supermimic()
  return viewer, state, done


def _step(viewer):
  assert viewer.step_and_predict(), getattr(viewer, "_last_error", None)


def _latest_sim_history(viewer):
  # Each step invokes the actor for the real sim, then for the WM ghost.
  return viewer.policy.call_args_list[-2].args[0]["history"][0, :, 0].tolist()


def test_world_model_horizon_preserves_real_policy_history():
  viewer, _, _ = _viewer(horizon=2)
  for _ in range(3):
    _step(viewer)
  assert _latest_sim_history(viewer) == [0, 1, 2]
  # The restarted prediction begins at the same state with the same memory.
  assert viewer._wm_history[0, :, 0].tolist() == [0, 1, 2]


def test_episode_and_manual_resets_reseed_real_policy_history():
  viewer, state, done = _viewer(horizon=100)
  _step(viewer)
  done[:] = True
  _step(viewer)
  done[:] = False
  state[:] = 20  # The environment has reset to a different motion.
  _step(viewer)
  assert _latest_sim_history(viewer) == [20, 20, 20]
  viewer.reset_env_supermimic()
  state[:] = 40
  _step(viewer)
  assert _latest_sim_history(viewer) == [40, 40, 40]
