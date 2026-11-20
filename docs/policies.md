# Running and testing SuperMimic policies

Use the [README quick start](../README.md#quick-start) to install the locked environment, obtain the accepted checkpoint, and run one clip. Examples below use Bash from the repository root. `TASK` is the registered mjlab environment, `SNAPSHOT` is a checkpoint directory, `MOTION` is one converted G1 NPZ, and `PACKED` is the full packed motion directory. Set those paths to files on your machine.

## How the policies work

Imagine asking the G1 to follow a forward flip. The clip supplies the timing and target poses. Every 20 ms the motion encoder reads ten reference frames at offsets 0, 5, …, 45 from the current frame. Near the end, offsets clamp to the last reference frame. For the accepted checkpoint the encoder sees reference joint positions and velocities plus reference orientations expressed relative to the current pelvis. It compresses this goal into 32 numbers.

The actor combines that goal with ten consecutive frames of robot observations. Each 93-value frame contains projected gravity, angular velocity, relative joint positions, joint velocities and the previous action. At reset the initial observation fills the history. On each subsequent control step the newest frame replaces the oldest. The actor outputs 29 scaled joint-position commands. The simulator applies its PD controller for four 5 ms physics steps, then the actor observes the result. Here “sensor-only” describes the robot-state input. The policy also needs the reference motion and the pelvis orientation used to form its tracking goal. [Actor](../src/supermimic/models.py) · [Task and goal construction](../src/supermimic/tasks/task_spec.py) · [Environment settings](../src/supermimic/tasks/env_cfgs.py).

During **training**, three components learn together:

1. The actor collects transitions in MuJoCo. Replay stores the simulated states, actions, motion frames and physical context, including friction and encoder offsets. The reset sampler gives extra practice to poorly tracked parts of the library.
2. The world model learns to predict what the simulator will do. The accepted GenCoord recipe scores eight consecutive state transitions from nine replay records. Its predicted accelerations are integrated to obtain the next state. The representation computes sensor observations and body poses from that state.
3. Starting from replay states, the actor takes 32 imagined steps through the differentiable world model. Gradients from tracking rewards, action penalties and predicted survival improve the actor. A critic estimates the return beyond the imagination horizon. Only the simulator generates the real transitions used to fit the model.

During **evaluation**, MuJoCo advances the robot on every step. The actor uses its saved parameters and observation processing. During **interactive playback**, a second, overlaid robot can show a world-model prediction starting from the simulated robot. Drift between the two is a model diagnostic. Motion completion and return are measured on the simulated robot. [Training loop](../src/supermimic/runners/supermimic_runner.py) · [Imagined environment](../src/supermimic/envs/world_model_env.py) · [Viewer implementation](../src/supermimic/scripts/viewers.py).

## Choose a checkpoint

**GenCoord** describes the world in generalized coordinates (root and joints) and uses forward kinematics to obtain body poses. **MaxCoord** describes it through per-body kinematics and can reconstruct joint readings for the sensor actor. Both support a `proprio` actor or a `rep_state` actor that reads privileged simulation state. Choose the task that produced the weights. The loader reconstructs the actor input layout and history from the saved runner configuration. [Representations](../src/supermimic/representations.py) · [Checkpoint manifest for the accepted run](checkpoints/gencoord-300k/snapshot.json).

| Run ID | Task | Actor input | Failed clips | Equal-clip return |
| --- | --- | --- | ---: | ---: |
| **vt2ualwi** · accepted 300k | GenCoord | 10 sensor frames | **3 / 5** | **0.95981 / 0.95981** |
| 2hm1hoxb · 320k with collection noise | GenCoord | 10 sensor frames | 461 / 466 | 0.88871 / 0.88863 |
| tq5a8xrg · earlier 320k baseline | GenCoord | 10 sensor frames | 54,488 | 0.62871 |
| 6e0845qv · earlier 320k baseline | MaxCoord | 10 sensor frames | 83,747 | 0.51121 |
| xz2lrdej · 1M baseline | GenCoord | Full state, 1 frame | 101 | 0.93642 |
| ft29lwog · 1M baseline | MaxCoord | Full state, 1 frame | 8 | 0.96562 |

The November 17 baseline notes retain the earlier memory experiments, service details and viewer checks. [Archived baseline notes](https://github.com/RX-00/supermimic/blob/64d5f7ad375230adfff12c2916b50b4b04bbf7e5/README.md#single-5090-training-decision-2026-11-12).

Each figure is from a sweep of all 129,491 local clips. Paired values denote repeat sweeps. The million-iteration baselines have access to simulation state that the sensor actor does not receive. For all rows the W&B run path is `a2r-lab/supermimic/<run ID>`. [Experiment record and baseline provenance](../REPRODUCTION_5090.md).

There are two ways to select weights:

- **Exact snapshot.** Pass `--snapshot "$SNAPSHOT"`. The folder contains `snapshot.json`, `actor.pt`, `world_model.pt` and, for these actor-critic runs, `critic.pt`. The loader checks the task and SHA-256 hashes. The README pairs the committed manifest with artifact `model-vt2ualwi:v29`. Once downloaded, this path uses local files for policy loading.
- **W&B run.** Pass `--wandb-path` with the run path. The loader gets its configuration and the model artifact carrying the run’s `latest` alias. Use this convenient route for the older baselines. Supply local motions for the runs in this table. The artifact name with a version belongs in the download command, while the run path belongs in the playback or evaluation command.

For example, to watch the full-state MaxCoord baseline on your G1 clip:

```bash
uv run --locked python -m supermimic.scripts.play SuperMimic-MaxCoord-G1 \
  --wandb-path a2r-lab/supermimic/ft29lwog --motion-file "$MOTION" \
  --viewer viser --num-steps-rollout 32
```

## Playback and video

Use `play` for interactive exploration. The Viser viewer prints a browser address. A quoted NPZ glob lets you supply a small selection of named clips. With local motions the viewer offers a clip selector. Selecting a clip resets the simulator to a sampled frame in it. The accepted policy trained on motion tracking, so leave the task set to motion. The `--num-steps-rollout 32` option sets how often the world-model overlay restarts from the simulator; playback continues until you stop it. On a workstation with a display, `--viewer native` opens the native viewer.

Use `evaluate` for a **first-frame replay** under benchmark settings. The following writes a movie plus its matching clip record. `--video True` requires one environment. On the Linux 5090 host, NVIDIA EGL supports headless rendering.

```bash
OUT=$(mktemp -d)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --locked python -m supermimic.scripts.evaluate "$TASK" \
  --snapshot "$SNAPSHOT" --motion-file "$MOTION" --num-envs 1 --limit 1 \
  --seed 0 --video True --output "$OUT/video"
ls "$OUT/video"
```

To replay a failure from a full sweep, read its `motion_index` and `motion_name` from `clips.jsonl`. Use the **same packed library** with `--motion-file "$PACKED" --start-motion 78209 --limit 1 --num-envs 1 --video True` and a new output directory. Index 78209 was the mirrored kneeling-stop motion in the accepted sweeps. A single-environment replay can differ from a batched sweep. The viewer also enables observation corruption, so compare scores using the evaluator.

## Test a policy

**Code check.** Run the full suite after changing a learner, representation, environment or loader. A focused check exercises clip scheduling, reset histories, immutable saves and gradient accumulation:

```bash
uv run --locked pytest -q
uv run --locked pytest -q tests/test_evaluate.py tests/test_viewer_history.py \
  tests/test_runner_checkpoint.py tests/test_policy_learner.py
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

**One-clip check.** The README example exercises the real loader, simulator and actor. Look for the expected name, finite return and termination reasons in `clips.jsonl`. A normal completion has `failed: false` and `motion_end`. A clip that triggers a pelvis failure can have `failed: true` even on its last step. A completed report can contain failed motions.

**Full acceptance test.** Use the frozen 129,491-clip library, the exact same snapshot, seed 0 and 4,096 environments. Run each sweep in a new process, one after the other on the free GPU. Each took about six minutes on the reproduction host. The output directory for each invocation must be new. For a short subset check, add `--start-motion 0 --limit 16` and reduce `--num-envs`; the loader still reads the whole supplied library.

```bash
PACKED=/home/roy/datasets/bones-seed/processed-v1/packed  # change on another host
OUT=$(mktemp -d)
for sweep in 1 2; do
  uv run --locked python -m supermimic.scripts.evaluate "$TASK" \
    --snapshot "$SNAPSHOT" --motion-file "$PACKED" --seed 0 --num-envs 4096 \
    --output "$OUT/sweep-$sweep" || break
done
uv run --locked python - "$OUT" "$SNAPSHOT" <<'PY'
import json, sys
from pathlib import Path

out, snapshot = map(Path, sys.argv[1:])
weights = json.loads((snapshot / "snapshot.json").read_text())["sha256"]
for sweep in (1, 2):
    folder = out / f"sweep-{sweep}"
    report = json.loads((folder / "report.json").read_text())
    assert report["complete"] and report["checkpoint_sha256"] == weights
    assert (report["evaluation_cfg"]["seed"], report["evaluation_cfg"]["num_envs"]) == (0, 4096)
    assert report["evaluated_motions"] == report["loaded_motions"] == 129491
    assert report["expected_steps"] == 47345789
    with (folder / "clips.jsonl").open() as source:
        indices = sorted(json.loads(row)["motion_index"] for row in source)
    assert indices == list(range(129491))
    failed, mean = report["failed_motions"], report["mean_normalized_return"]
    print(sweep, failed, mean, report["frame_weighted_normalized_return"])
    assert failed <= 30 and mean >= 0.85
PY
```

Each motion starts at its first frame, with fixed foot friction 1.0, domain randomization disabled, and clean observations. A pelvis height or orientation termination counts as failure. The per-clip return is the sum of rewards earned divided by the maximum reference reward for the **entire** clip (0.1 times its reference frame count at 50 Hz). An early fall thus loses credit for the remaining frames. The equal-clip mean gives a short flip the same weight as a long walk. The frame-weighted mean gives more weight to longer clips. Only the equal-clip mean and failure count determine acceptance, independently in each sweep. `benchmark.py` measures throughput. [Scoring implementation](../src/supermimic/scripts/evaluate.py) · [Per-metric results and comparison to the paper](../REPRODUCTION_5090.md).

## Train on one GPU

To reproduce the accepted configuration, use GenCoord, ten proprioceptive frames, motion-only slots, Adam, cosine decay to 3% of the initial rate at 250k iterations, eight scored world-model transitions and zero collection action noise. Sensor noise remains active during training. Splitting the 4,096 imagined starts into two microbatches of 2,048 preserves one actor update and fits this 5090. The completed run peaked at about 27,268 MiB of GPU process memory and took 14 h 13 min 30 s. A bare invocation of `train` uses the release defaults.

**On the original training host**, use the tracked [training launcher](../scripts/train_5090.sh). It verifies the original packing manifest, writes the exact command and environment versions, and samples device memory. It contains the original checkout, corpus, uv path and W&B entity. Start it through systemd with a unique name:

```bash
RUN="gencoord-$(date +%Y%m%d-%H%M%S)"
UNIT="supermimic-$RUN"
systemd-run --user --unit="$UNIT" --property=Type=exec --property=Restart=no \
  --working-directory="$PWD" "$PWD/scripts/train_5090.sh" "$RUN" 300000 10000 \
  --runner.policy.microbatch-size 2048 --runner.policy.noise 0
```

**On another Linux training host**, point `PACKED` at your prepared corpus and set `WANDB_ENTITY` to a team you can write to. Log in with `uv run --locked wandb login`. The equivalent training command below uses the local uv executable and working directory. Choose one launch method per run.

```bash
RUN="gencoord-$(date +%Y%m%d-%H%M%S)"
UNIT="supermimic-$RUN"
systemd-run --user --unit="$UNIT" --property=Type=exec --property=Restart=no \
  --working-directory="$PWD" --setenv="WANDB_ENTITY=${WANDB_ENTITY:?set your W&B entity}" \
  --setenv=PYTHONUNBUFFERED=1 "$(command -v uv)" run --locked python -u \
  -m supermimic.scripts.train SuperMimic-GenCoord-G1 \
  --device cuda:0 --env.seed 0 --num-envs 4096 --motion-file "$PACKED" \
  --runner.run-name "$RUN" --runner.max-iterations 300000 --runner.save-interval 10000 \
  --runner.log-interval 50 --runner.actor-obs proprio --runner.history-length 10 \
  --runner.slot-mix.motion 1 --runner.slot-mix.velocity 0 --runner.policy.noise 0 \
  --runner.wm.all-adam True --runner.policy.all-adam True \
  --runner.policy.batch-size 4096 --runner.policy.num-batches 1 \
  --runner.policy.microbatch-size 2048 --runner.wm.horizon 8 \
  --runner.wm.horizon-counts-transitions True --runner.policy.shuffle-each-epoch True \
  --runner.cosine-decay-iterations 250000 --runner.gencoord-encoder-bias True \
  --runner.episode-sim-context True --runner.share-reset-sensor True \
  --runner.return-sampler.bin-seconds 1 --runner.return-sampler.square-historical-weights False \
  --runner.immutable-snapshots True
```

After `systemd-run` returns, verify that the actual GPU training PID appears inside the user service. Record the unit name, log location, checkpoint directory and final exit status. For unattended use across logout, check that the user service manager stays available (`loginctl show-user "$USER" -p Linger` on the original host). Run GPU evaluations after training releases the device.

```bash
systemctl --user status "$UNIT"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# Substitute the Python trainer PID from nvidia-smi. Its cgroup should end in $UNIT.service.
# cat /proc/<trainer-pid>/cgroup
systemctl --user show "$UNIT" -p ControlGroup -p ActiveState -p Result -p ExecMainStatus
journalctl --user -u "$UNIT" -n 40
```

The portable command sends training output to the user journal. The host launcher sends it to `logs/reproduction-5090-20261118/$RUN/training.log` and records `exit_code` when training ends. Look for `[INFO] Logging to` in the output to find `logs/<task>/<timestamp>/`. With immutable saves, each completed `iteration-NNNNNN/` below it is directly usable as `--snapshot`, even after later saves. These checkpoints contain model weights, normalization state and configuration. Exact training continuation would also require optimizer, replay, sampler, random-generator and simulator state. `--runner.wandb-run-path` warm-starts the world model only.

## Source map

| Question | Read |
| --- | --- |
| Which commands start training, run the viewer or score a policy? | [train](../src/supermimic/scripts/train.py), [play](../src/supermimic/scripts/play.py), [evaluate](../src/supermimic/scripts/evaluate.py) |
| What does each task measure, reward and terminate on? | [Task registration](../src/supermimic/tasks/__init__.py), [environment configuration](../src/supermimic/tasks/env_cfgs.py), [tracking rewards](../src/supermimic/utils/tracking_reward.py) |
| How do observations and physical parameters reach the actor and model? | [Representations](../src/supermimic/representations.py), [simulator context](../src/supermimic/sim_context.py), [motion goals](../src/supermimic/tasks/task_spec.py) |
| Where do sampling, history and training updates happen? | [Runner](../src/supermimic/runners/supermimic_runner.py), [replay buffer](../src/supermimic/replay_buffer.py), [return-based resets](../src/supermimic/runners/reset_sampling.py) |
| Where are the networks and losses? | [Network definitions](../src/supermimic/models.py), [world-model learner](../src/supermimic/learners/wm_learner.py), [actor and critic learner](../src/supermimic/learners/policy_learner.py) |
| Which tools serve other experiments? | [PPO training entry point](../src/supermimic/scripts/train_ppo.py), [throughput benchmark](../src/supermimic/scripts/benchmark.py), [export code](../src/supermimic/runners/exporter.py) |
| Where are the independent checks? | [Tests](../tests/), [locked dependencies](../uv.lock), [experiment record](../REPRODUCTION_5090.md) |

## Common problems

| Symptom | What to check |
| --- | --- |
| uv cannot resolve mjlab, or importing MuJoCo fails | The locked editable dependency lives at `../mjlab_research`. Use the pinned revision and `uv sync --locked`. This branch pins mjviser 0.0.16 for its MuJoCo 3.x environment. |
| W&B cannot find the model | Check the signed-in account’s read access. Use the full artifact name with `:v29` when downloading or the full run path with `--wandb-path`. |
| Snapshot task or digest mismatch | Use GenCoord for the accepted manifest and fetch all three files from the pinned version. Mixing weights from different saves invalidates the hash check. |
| Motion file missing or joint layout rejected | Use a converted 50 Hz G1 NPZ or the packed directory. See the [data guide](motions.md#get-the-inputs) for raw CSV conversion. |
| Output directory already exists | Choose a fresh child of `mktemp -d`. A partial run may have left a report with `complete: false`. |
| GPU runs out of memory | Check for another GPU job. For a smoke test use one NPZ and one environment. For training the accepted settings require policy microbatch 2,048 and effective batch 4,096. Reducing the latter changes the experiment. |
| Headless video fails to initialize EGL | Verify the NVIDIA EGL driver is installed. On the reproduction host, setting `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json` selects that driver. Scoring without `--video` does not need an offscreen renderer. |
