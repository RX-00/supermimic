import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from supermimic.learners.policy_learner import (
  ActorCriticPolicyLearner,
  ActorCriticPolicyLearnerCfg,
  BpttPolicyLearner,
  BpttPolicyLearnerCfg,
  gae_returns,
)

NUM_ENVS = 2


class _DummyActor(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(()))

  def forward(self, obs: TensorDict) -> torch.Tensor:
    return obs["actor"] * self.weight


class _DummyRepresentation(nn.Module):
  dt = 1.0


class _TwoStepEnv:
  """Reward 1 per step; the first step terminates with probability 0.5."""

  num_envs = NUM_ENVS
  num_actions = 1
  max_step_reward = float("inf")

  def __init__(self):
    self.dynamics = nn.Linear(1, 1)
    self.representation = _DummyRepresentation()
    self.step_idx = 0

  def sample_initial_states(self) -> TensorDict:
    return TensorDict({}, batch_size=[NUM_ENVS])

  def reset(self, batch: TensorDict) -> TensorDict:
    del batch
    self.step_idx = 0
    return self._obs()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    rewards = 1.0 + actions.squeeze(-1)
    term_prob = torch.full((NUM_ENVS,), 0.5 if self.step_idx == 0 else 0.0)
    self.step_idx += 1
    return self._obs(), rewards, term_prob, {}

  def _obs(self) -> TensorDict:
    obs = {"actor": torch.zeros(NUM_ENVS, 1), "critic": torch.zeros(NUM_ENVS, 1)}
    return TensorDict(obs, batch_size=[NUM_ENVS])


class _ConstantCritic(nn.Module):
  def __init__(self, value: float):
    super().__init__()
    self.value = nn.Parameter(torch.tensor(value))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return obs * 0.0 + self.value


def test_survival_masking_counts_current_transition_reward_before_done():
  learner = BpttPolicyLearner(
    _DummyActor(),
    _TwoStepEnv(),
    BpttPolicyLearnerCfg(horizon=2, gamma=1.0, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + (1 - 0.5) * 1, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx(1.5 / 2)


def test_actor_critic_bootstraps_the_surviving_last_state():
  learner = ActorCriticPolicyLearner(
    _DummyActor(),
    _ConstantCritic(1.0),
    _TwoStepEnv(),
    ActorCriticPolicyLearnerCfg(horizon=2, gamma=0.99, survival_masking=True),
  )

  stats = learner.update()

  # Return 1 + 0.99 * 0.5 * 1, plus the value 1 discounted by 0.99^2 and by
  # survival 0.5, averaged over the horizon.
  assert stats["Policy/reward"].item() == pytest.approx((1.495 + 0.49005) / 2)
  assert torch.isfinite(stats["Policy/value_function"])


def test_gae_returns_discount_by_soft_done():
  returns = gae_returns(
    values=torch.zeros(2, 1),
    last_value=torch.zeros(1),
    rewards=torch.ones(2, 1),
    dones=torch.tensor([[0.5], [0.0]]),
    gamma=0.99,
    lam=1.0,
  )

  assert torch.allclose(returns, torch.tensor([[1.495], [1.0]]))


def test_paper_cycle_has_one_actor_step_and_twenty_critic_steps():
  cfg = ActorCriticPolicyLearnerCfg(horizon=2, all_adam=True, shuffle_each_epoch=True)
  learner = ActorCriticPolicyLearner(
    _DummyActor(), _ConstantCritic(1.0), _TwoStepEnv(), cfg
  )
  learner.update()
  assert learner.actor_optimizers[0].state[learner.actor.weight]["step"].item() == 1
  assert learner.critic_optimizers[0].state[learner.critic.value]["step"].item() == 20


class _FixedRollout:
  """Fixed per-sample sensor noise and a differentiable, action-dependent state."""

  max_step_reward = 100.0

  def __init__(self, device):
    self.dynamics = nn.Linear(1, 1).to(device)
    self.representation = _DummyRepresentation()

  def reset(self, batch):
    self.x = batch["x"]
    self.noise = batch["noise"]
    return self._obs()

  def _obs(self):
    return TensorDict({"critic": self.x, "actor": self.x + self.noise}, [len(self.x)])

  def step(self, actions):
    self.x = self.x + 0.2 * actions
    reward = 5 * self.x.sum(dim=-1)
    return self._obs(), reward, self.x[:, 0].sigmoid(), {}


@pytest.mark.parametrize("captured", [False, True])
def test_accumulation_matches_unsplit_gradients_and_adam(monkeypatch, captured):
  import copy

  from supermimic.learners import policy_learner as module

  if captured and not torch.cuda.is_available():
    pytest.skip("CUDA graph comparison requires a GPU")
  device = "cuda" if captured else "cpu"
  torch.manual_seed(41)
  actor = nn.Linear(3, 3).to(device)

  class SensorActor(nn.Sequential):
    def forward(self, obs):
      output = super().forward(obs["actor"])
      self.aux = 0.02 * output.square().mean(-1)
      return output

    def auxiliary_loss(self):
      return self.aux

  actor = SensorActor(actor)
  critic = nn.Linear(3, 1).to(device)
  starts = TensorDict(
    {
      "x": torch.randn(8, 3, device=device),
      "noise": torch.randn(8, 3, device=device) * 0.05,
    },
    [8],
  )
  # Use the same nontrivial permutation in both implementations, including replay.
  monkeypatch.setattr(
    torch, "randperm", lambda n, device: torch.arange(n - 1, -1, -1, device=device)
  )
  full, split = [
    ActorCriticPolicyLearner(
      copy.deepcopy(actor),
      copy.deepcopy(critic),
      _FixedRollout(device),
      ActorCriticPolicyLearnerCfg(
        batch_size=8, microbatch_size=micro, horizon=3, all_adam=True
      ),
      device=device,
    )
    for micro in (None, 4)
  ]
  measured = {}
  real_step = module.step_with_gradients

  def capture(optimizers, params, max_norm, clip_value):
    key = id(optimizers[0])
    before = torch.cat([p.grad.flatten() for p in params]).clone()
    norm = real_step(optimizers, params, max_norm, clip_value)
    measured[key] = (before, torch.cat([p.grad.flatten() for p in params]).clone())
    return norm

  monkeypatch.setattr(module, "step_with_gradients", capture)
  for _ in range(6 if captured else 2):
    # The reference is eager even during comparison with CUDA graph replay.
    ref = full._update_from(starts.unsqueeze(0))
    actual = split._captured_update(starts.unsqueeze(0))
    for name in ref:
      torch.testing.assert_close(ref[name], actual[name], atol=2e-5, rtol=2e-5)
    for a, b in zip(
      measured[id(full.actor_optimizers[0])],
      measured[id(split.actor_optimizers[0])],
      strict=True,
    ):
      torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for network, optim in [
      ("actor", "actor_optimizers"),
      ("critic", "critic_optimizers"),
    ]:
      for a, b in zip(
        getattr(full, network).parameters(),
        getattr(split, network).parameters(),
        strict=True,
      ):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        ref_state = getattr(full, optim)[0].state[a]
        got_state = getattr(split, optim)[0].state[b]
        for key in ["step", "exp_avg", "exp_avg_sq"]:
          torch.testing.assert_close(
            ref_state[key], got_state[key], atol=2e-5, rtol=2e-5
          )
