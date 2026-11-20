import torch
import torch.nn as nn

from supermimic.learners.learner_utils import _optimize_step, build_optimizers
from supermimic.models import build_mlp


def _ids(params) -> list[int]:
  return [id(p) for p in params]


def test_muon_steps_the_hidden_matrices_and_adam_the_rest():
  torch.manual_seed(0)
  mlp = build_mlp(5, (8, 8, 8), 3)
  linears = [m for m in mlp if isinstance(m, nn.Linear)]
  hidden = [linears[1].weight, linears[2].weight]
  adam, muon = build_optimizers([mlp], 1e-3)
  assert isinstance(muon, torch.optim.Muon)
  assert _ids(muon.param_groups[0]["params"]) == _ids(hidden)
  rest = [p for p in mlp.parameters() if id(p) not in _ids(hidden)]
  assert _ids(adam.param_groups[0]["params"]) == _ids(rest)
  before = [p.detach().clone() for p in mlp.parameters()]
  _optimize_step([adam, muon], mlp(torch.randn(4, 5)).square().mean())
  for param, old in zip(mlp.parameters(), before, strict=True):
    assert not torch.equal(param, old)


def test_cosine_collection_clock_and_floor():
  from supermimic.learners.learner_utils import cosine_lr

  for iteration, fraction in [(0, 1), (125000, 0.515), (250000, 0.03), (320000, 0.03)]:
    assert cosine_lr(2e-4, iteration, 250000) == 2e-4 * fraction


def test_captured_all_adam_reads_changed_lr():
  import pytest
  from tensordict import TensorDict

  from supermimic.learners.learner_utils import CapturedUpdate, graph_state

  if not torch.cuda.is_available():
    pytest.skip("CUDA graph validation requires a GPU")
  layer = build_mlp(2, (4, 4), 1).cuda()
  (optimizer,) = build_optimizers([layer], 0.01, device="cuda", all_adam=True)
  assert isinstance(optimizer, torch.optim.Adam)
  assert _ids(optimizer.param_groups[0]["params"]) == _ids(layer.parameters())

  def train(batch):
    loss = layer(batch["x"]).sum()
    _optimize_step([optimizer], loss)
    return {"loss": loss.detach()}

  update = CapturedUpdate(train, lambda: graph_state([layer], [optimizer]))
  batch = TensorDict({"x": torch.ones(4, 2, device="cuda")}, [4])
  for _ in range(4):
    update(batch)
  bias = layer[-1].bias
  before = bias.detach().clone()
  optimizer.param_groups[0]["lr"].fill_(0.0003)
  update(batch)
  torch.testing.assert_close(
    before - bias, torch.full_like(bias, 0.0003), atol=1e-7, rtol=1e-4
  )
