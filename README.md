# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

The `dev_5090` sensor-only GenCoord policy passed the local benchmark on one RTX 5090.
After 300k training iterations with collection action noise disabled, the same checkpoint
failed 3 and 5 of 129,491 clips in two full sweeps. Both averaged 0.95981 equal-clip
normalized return. Training took 14 h 13 min. The original 320k run with collection
action noise failed 461 and 466 clips. [Checkpoint, commands, results and playback](REPRODUCTION_5090.md).

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
uv sync
```

The uv configuration pins `mjviser==0.0.16` for compatibility with the research
repo's MuJoCo 3.x dependency. `mjviser` 0.0.17 pulls in `mujoco-core` 4.x, which
overwrites files in the same Python package and breaks MuJoCo imports.

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run-name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

## BONES-SEED preprocessing

The main paper benchmark uses the G1-retargeted [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed).
Appendix E reports 142,220 raw clips, 129,667 after name filtering, and 129,514
after feasibility filtering (264.56 hours). The
[paper and its source](https://arxiv.org/abs/2610.28480v2), project website, and
official repositories do not publish the complete keyword list, retained-clip
manifest, or numeric definitions of upright, standing height, and nearly stationary.

Our selected rules and pinned sources are recorded in
[configs/bones_seed_preprocessing.json](configs/bones_seed_preprocessing.json).
This file documents the preprocessing specification; the training CLI does not
read it. Use the prepared motions through the existing `--motion-file` argument.

For the unspecified choices, use the full SONIC keyword list plus the additions
in [SuperMimic-Reproduced's filter](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/scripts/data/filter_bones_seed.py).
Match case-insensitive substrings against filename stems. The same implementation
supplies these assumptions for the floating-feet test:

| Gate | Assumed condition |
| --- | --- |
| Upright | Pelvis up-axis dot world up > 0.9 |
| Standing height | 0.60 m < pelvis height < 0.85 m |
| Nearly stationary | Pelvis linear speed < 0.1 m/s and joint-velocity RMS < 0.5 rad/s |

Keep the paper's published feasibility rules: lowest body-link height above
0.20 m for at least 0.6 s; penetration below -0.12 m at any frame or below -0.08 m
for more than 80% of a clip; and both feet more than 0.02 m above the floor for at
least 0.5 s while all three gates hold. Use `mj_geomDistance` for foot collision
geometry against the floor, as specified by the paper, rather than ankle origins.
At 50 Hz, consecutive-run thresholds are 30 and 25 frames respectively.

The raw CSVs are 120 Hz, with root translations in centimeters and root Euler
angles and joint angles in degrees. Convert to meters, radians and wxyz root
quaternions, then resample to 50 Hz and derive velocities. Retain every raw clip
and record the reason for each exclusion in the processed-data manifest.

On dataset revision `2f59b2077b9da34dd4e43618e705c7cb962c9a66`, a local check of
all 142,220 metadata rows retains **129,647** clips after the selected name filter,
20 fewer than the paper. Full local conversion and feasibility filtering retains
**129,491 clips**, totaling **47,345,789 frames at 50 Hz (263.03 hours)**. It excludes
100 elevated-support clips, 44 with ground penetration, and 12 with floating feet.
The clip count matches the third-party implementation's reported count, but is 23
below the paper's. Equal counts do not prove identical clip membership; this is a
documented approximation of the paper's benchmark.

After extracting `g1.tar.gz` into a raw-data directory, convert and filter with:

```bash
uv run python -m supermimic.scripts.prepare_bones \
  --raw-root /path/to/bones-seed/raw \
  --metadata /path/to/bones-seed/metadata/seed_metadata_v004.csv \
  --output /path/to/bones-seed/processed --workers 12
uv run python -m mjlab.scripts.pack_motions \
  --input '/path/to/bones-seed/processed/npz/*.npz' \
  --output /path/to/bones-seed/processed/packed --workers 12
```

The converter requires a new output directory and leaves raw files intact. Its
`manifest.csv` records every clip and exclusion, `preprocessing_spec.json` records
the chosen rules, and `report.json` records counts and source checksums. `--limit`
is available for a small conversion check; omit it for the full dataset. Pass the
packed directory to both training and playback using `--motion-file`.

Set `WANDB_ENTITY` to your W&B team before training. Checkpoints are saved at the
configured interval and after the final iteration. These contain model weights
for playback; `--runner.wandb-run-path` only warm-starts the world model and does
not resume policy training, optimizer state, or the replay buffer.

Two training presets are being compared for each task ID:

| Preset | Iterations | Actor observations | History length |
| --- | ---: | --- | ---: |
| Paper duration and history | 320,000 | `proprio` | 10 |
| Released defaults | 1,000,000 | `rep_state` | 1 |

Select these with `--runner.max-iterations`, `--runner.actor-obs`, and
`--runner.history-length`. Both retain the release's Muon/Adam optimizers and
constant learning rates. The paper specifies Adam and cosine decay, so the first
preset reproduces its duration and history settings only. Run the four jobs
sequentially on a single GPU; validate full-dataset memory use before starting.

### Single-5090 training decision (2026-11-12)

First train and evaluate all four task/preset combinations with smaller policy
batches, initially **2,048** instead of 4,096 (`--runner.policy.batch-size 2048`).
Retain the full filtered dataset and the existing FP32 motion storage. Evaluate
the resulting policies before deciding whether to implement memory-saving code.
Both presets receive the same policy-batch override; "released defaults" above
therefore describes the duration and observation settings, with this explicit
hardware adjustment. A full-dataset MaxCoord check with a 2,048 policy batch
completed 64 iterations, including repeated learning updates, with finite logged
metrics and finite saved actor, critic, and world-model tensors. This validates
the short run's memory fit, not the quality of a trained policy.

The relevant single-GPU precedent is **SuperMimic-Reproduced's e5 experiment**,
which reports one RTX 5090, 4,096 environments, a 32-step imagination horizon,
and a policy batch reduced to 2,048 because of memory limits. It kept the full
then-filtered BONES-SEED library (129,943 clips) in FP16, about 6.74 GB, and
reported **22 GB total GPU memory usage**. This run used extra anchor-relative
observations. Its later headline result used **four** 5090s with about 20 GB per
card; that is a separate experiment, not evidence for single-card training time
or final policy quality. See the pinned
[experiment log](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/docs/EXPERIMENTS.md).

That reproduction stores a 71-component root/joint state and derives body states
with forward kinematics. Its
[motion loader](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/src/supermimic/envs/mdp/motion_library.py)
casts selected FP16 reference frames back to FP32 for computation. Our failed
MaxCoord run used a 182-component body state and an additional 93 sensor values
in replay, so its memory result does not establish that our MaxCoord configuration
fits. Our 4,096-policy-batch check exhausted memory at 30.18 GiB process usage;
the motion tensors and replay account for approximately 12.52 and 5.62 GiB.

Deferred options are FP16 reference storage (validate quantization error) and
policy microbatching with gradient accumulation (validate gradients, clipping,
and update frequency). HoloMotion has an existing
[microbatch implementation](https://github.com/HorizonRobotics/HoloMotion/blob/f52dd81ae935a828b0296f5ee0f833beb7378bf0/holomotion/src/algo/supermimic/imagination.py)
and recommends it for actor-memory limits. Its reported 5090 experiment used
40 LAFAN1 clips and a 24-step horizon, so it is an implementation reference, not
a validation of full BONES-SEED or our MaxCoord task. The four baseline runs used unsplit batches. The 2026-11-18 reproduction adds
validated actor gradient accumulation for the effective batch of 4,096.

### One-GPU sensor-only reproduction

The paper-recipe launcher selects GenCoord, ten proprioceptive frames, motion-only
slots, eight scored WM transitions, Adam and cosine decay. The legacy CLI defaults
continue to load the four baseline configurations. On this 5090 the unsplit 4,096
profile exhausted memory. The accumulation mode splits imagination across
microbatches, clips the full gradient once, takes one actor step, and fits the
critic to all imagined samples. The [run plan](REPRODUCTION_5090.md) records
measured feasibility, progress and the current acceptance status.

Use `--runner.immutable-snapshots True` to save a separate weights directory
at each completed save interval. To evaluate that exact checkpoint, pass
`--snapshot /path/to/iteration-004000` in place of `--wandb-path` in the
evaluation command below. The evaluator verifies the saved weight hashes and
rebuilds the recorded runner configuration before starting the sweep.

### Unattended training

Run long-term training under a persistent service, never in a Codex-managed
terminal. On 2026-11-13, a Codex daemon auto-update interrupted the original
terminal-owned queue at about iteration 174,410 of its first run. Its checkpoints
were discarded at the user's request; the replacement queue starts from scratch.

The local replacement uses the systemd user service
`supermimic-training.service`, defined in
`/home/roy/.config/systemd/user/supermimic-training.service`. Its queue, per-run
logs, exit codes, and start/finish markers live in
`logs/bones-seed-four-runs-20261113/`. Inspect the actual service as well as the
files; a stale `status` file alone does not prove that training is running:

```bash
systemctl --user status supermimic-training.service
tail -n 20 logs/bones-seed-four-runs-20261113/service.log
cat logs/bones-seed-four-runs-20261113/status
```

The user service manager has lingering enabled, and the service runs independently
of the launching terminal and Codex daemon. The queue stops on failure and does
not automatically restart an unfinished run: these checkpoints lack full training
state. This arrangement survives Codex exits and updates, but does not resume
training after a machine reboot. Pausing the Codex goal must leave the training
service running unless the user also asks to stop training.

## Policy evaluation and playback checks

Evaluate the finished checkpoints after the training queue releases the GPU.
`benchmark.py` measures throughput; use `evaluate.py` to measure policy quality:

```bash
uv run --locked python -m supermimic.scripts.evaluate \
  SuperMimic-GenCoord-G1 --wandb-path a2r-lab/supermimic/<RUN_ID> \
  --motion-file /home/roy/datasets/bones-seed/processed-v1/packed \
  --output logs/evaluation/<RUN_ID> --num-envs 256
```

Run this for both task IDs and both presets. The evaluator loads the run's latest
model artifact and its saved architecture, observations and history length. The
output directory must be new. `report.json` records checkpoint hashes, settings,
failure count, and normalized returns; `clips.jsonl` records each clip and its
termination reasons. A report becomes complete only after every selected clip
finishes; partial results retain `complete: false`.
Use `--limit` and `--start-motion` for explicitly labeled subset checks; omit them
for the full benchmark. A limit changes evaluation coverage, not the library's
memory footprint. Small NPZ inputs can be used for a lightweight loading check.

The protocol follows paper §4 and Appendices A/F: each motion starts at its first
frame, foot friction is 1.0, COM perturbations and encoder bias are zero, and
observation/reset noise is disabled. The released gains, action scales, reward,
and pelvis failure conditions remain in use (height error >0.25 m or projected
gravity z-error >0.8). Reaching the motion boundary is successful only if no
failure condition fires on that step. Policy history is padded with the first
observation at each reset and advances on every step. No frames from a previous
clip enter the next clip's history.
After placing a clip at its first frame, the evaluator clears the observations
cached by the generic reset and rebuilds the reset slots' observations. This
keeps the actor's initial history aligned with the requested motion while
preserving observation histories in slots that are still running.

Explicit evaluation conventions: initialization retains the release's soft
joint-limit clipping; a clip with N frames produces N reward steps under the
release's motion-end condition. Normalize the undiscounted return by N × 5 ×
step duration (the environment scales rewards by time), including unplayed
frames after a failure. Scores are not clipped. Since the paper does not specify
aggregation across different clip lengths, report both the mean of per-clip
normalized returns and the frame-weighted mean. These are training-set coverage
metrics on our documented filtered dataset, not held-out generalization results.
Fixed settings and a fixed seed do not establish bitwise reproducibility across
reruns, environment counts, GPUs, or simulator versions. Reported failure counts
are observations of one full sweep, not a statistical robustness estimate.

For a recorded playback check, select a few clips and one environment:

```bash
uv run --locked python -m supermimic.scripts.evaluate \
  SuperMimic-MaxCoord-G1 --wandb-path a2r-lab/supermimic/<RUN_ID> \
  --motion-file /home/roy/datasets/bones-seed/processed-v1/packed \
  --output logs/evaluation/<RUN_ID>-video --num-envs 1 --limit 3 --video True
```

This writes `playback.mp4` alongside the metrics, using the same evaluation
settings. Also inspect the trained actor and world-model overlay interactively
with `play.py --viewer viser` and the matching task, W&B path and motion file.
The interactive viewer retains its own noise/randomization settings; its visual
playback is a separate check from the fixed-settings evaluation metrics. Inspect
representative walking, running, turning and difficult motions, plus any reported
failures, before deciding whether smaller policy batches need further changes.
`play.py --num-steps-rollout 32` restarts the world-model prediction from the
simulator every 32 steps; it does not limit the playback duration. Restarting the
prediction preserves the real policy's observation history. That history resets
only when the simulated episode or viewer environment resets. The restarted
world-model rollout copies the real policy's current history before predicting
its next sensor frames.

### Local training and evaluation results (2026-11-17)

All four training jobs finished with exit code 0. The final actor, world-model,
and critic artifacts match the local final checkpoints. All runs used the
single-5090 policy batch of 2,048 documented above.

| Task | Preset | W&B run | Failed / 129,491 motions | Mean normalized return |
| --- | --- | --- | --- | --- |
| GenCoord | 320k, proprio history 10 | [tq5a8xrg](https://wandb.ai/a2r-lab/supermimic/runs/tq5a8xrg) | 54,488 (42.08%) | 0.628710 |
| MaxCoord | 320k, proprio history 10 | [6e0845qv](https://wandb.ai/a2r-lab/supermimic/runs/6e0845qv) | 83,747 (64.67%) | 0.511212 |
| GenCoord | 1M, full-state observation | [xz2lrdej](https://wandb.ai/a2r-lab/supermimic/runs/xz2lrdej) | 101 (0.078%) | 0.936421 |
| MaxCoord | 1M, full-state observation | [ft29lwog](https://wandb.ai/a2r-lab/supermimic/runs/ft29lwog) | 8 (0.0062%) | 0.965617 |

These are full-library, equal-clip means under the evaluation conventions above.
The frame-weighted means and every clip's outcome are retained in
`logs/evaluation-20261117/<RUN_ID>-full-v2/`. `benchmark-audit.json` independently
checks clip names, lengths, coverage, termination accounting, and aggregates
against the packed dataset. Earlier subset reports without `v2` are invalidated
because their reset path returned stale cached observations; do not use their
scores. The corrected evaluator passed training-input parity checks, and a
separate check found synthesized proprioception agrees with real-simulator
observations to numerical precision on 2,400 states per representation.

Both million-iteration policies completed full walking, turning/jogging, running,
flip, side-roll, and standing-up playback clips through `play.py`, with the
world-model overlay restarting every 32 steps. The same viewer works with the
320k checkpoints, but those policies fail many motions. Browser screenshots,
videos, and step-by-step root prediction errors are in
`logs/evaluation-20261117/<RUN_ID>-playback-full/`. Playback uses the viewer's own
noise/reset settings and is separate from benchmark scoring. Two GenCoord
benchmark failures (crouching and kneeling) also failed in individual evaluation
replays; see `xz2lrdej-failure-video/` in the same evaluation directory.
Of two sampled MaxCoord benchmark failures, the mirrored crouching motion failed
again, while its unmirrored counterpart completed in a one-environment replay.
`failure-review.json` preserves both outcomes; individual replay videos are in
`ft29lwog-failure-video/`. Do not assume per-clip outcomes are invariant to batching
or rerunning the simulator.

The paper-duration/history policies did not reach the paper's reported motion
coverage. Stable training rewards were insufficient evidence of convergence.
The million-iteration policies receive privileged simulator state, so their
results do not establish deployable proprioceptive-policy quality. Duration,
observations, optimizer/schedule, and dataset differences prevent attributing the
gap to policy batch size. Keep the working 2,048 batch baseline for comparison.
The next experiment tests the documented one-GPU recipe as a whole. Its first
full-library memory pilot determines whether the actor needs gradient accumulation.

Play the million-iteration checkpoints locally:

```bash
uv run --locked python -m supermimic.scripts.play SuperMimic-GenCoord-G1 \
  --wandb-path a2r-lab/supermimic/xz2lrdej \
  --motion-file /home/roy/datasets/bones-seed/processed-v1/packed \
  --viewer viser --num-steps-rollout 32

uv run --locked python -m supermimic.scripts.play SuperMimic-MaxCoord-G1 \
  --wandb-path a2r-lab/supermimic/ft29lwog \
  --motion-file /home/roy/datasets/bones-seed/processed-v1/packed \
  --viewer viser --num-steps-rollout 32
```

All four full evaluations finished with exit code 0, running sequentially under
`supermimic-full-evaluation-20261117.service`; full playback checks ran under
`supermimic-full-playback-checks-20261117.service` and also finished with exit code
0. No training or evaluation jobs remain queued. Their queue status, service
ownership, process exit codes, and logs are recorded in
`logs/evaluation-20261117/`. Simulator line-search-limit warnings remain in the
logs; solver settings were not changed or warnings suppressed for evaluation.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
