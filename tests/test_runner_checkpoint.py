"""The last completed iteration must be available for playback."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from supermimic.runners import supermimic_runner as module


@pytest.mark.parametrize("iterations,expected", [(5, [1, 4, 5]), (4, [1, 4])])
def test_final_checkpoint_contains_final_weights(
  tmp_path, monkeypatch, iterations, expected
):
  actor = torch.nn.Linear(1, 1, bias=False)
  runner = MagicMock()
  runner.cfg = module.SuperMimicRunnerCfg(
    max_iterations=iterations, save_interval=3, log_interval=1
  )
  runner.env.num_envs = 1
  runner.replay_buffer.num_transitions = 0
  runner._create_log_dir.return_value = tmp_path
  runner._representation.actor_obs_mode = "rep_state"
  runner.policy_learner = SimpleNamespace(actor=actor)
  completed = 0

  def step():
    nonlocal completed
    completed += 1
    with torch.no_grad():
      actor.weight.fill_(completed)
    return 1

  runner.step.side_effect = step
  runner._wm.bundle.side_effect = lambda: {"completed": completed}
  monkeypatch.setattr(module.wandb, "run", SimpleNamespace(name="test"))
  uploaded = []

  def upload(paths):
    world_model = torch.load(paths[0], weights_only=True)
    weights = torch.load(paths[1], weights_only=True)
    assert weights["weight"].item() == world_model["completed"]
    uploaded.append(world_model["completed"])

  monkeypatch.setattr(module, "log_model_artifact", upload)
  module.SuperMimicRunner.run(runner)
  assert uploaded == expected
  assert (
    torch.load(tmp_path / "actor.pt", weights_only=True)["weight"].item() == iterations
  )
  runner._cleanup.assert_called_once()


def test_snapshot_verification_rejects_substitution(tmp_path):
  import json

  from supermimic.utils.wandb_artifacts import checkpoint_hashes, verified_snapshot

  for name in ("actor.pt", "world_model.pt", "critic.pt"):
    (tmp_path / name).write_bytes(name.encode())
  record = {"task_id": "task", "sha256": checkpoint_hashes(tmp_path)}
  (tmp_path / "snapshot.json").write_text(json.dumps(record))
  assert verified_snapshot(tmp_path, "task") == record
  with pytest.raises(ValueError, match="mismatch"):
    verified_snapshot(tmp_path, "other")
  (tmp_path / "actor.pt").write_bytes(b"later actor")
  with pytest.raises(ValueError, match="mismatch"):
    verified_snapshot(tmp_path, "task")
