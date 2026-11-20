# Reproducing SuperMimic on one RTX 5090

Prepared 2026-11-17. Working directory: /home/roy/supermimic. Branch: dev_5090.

## Confirmed sensor-only result on one RTX 5090 · November 20, 2026

**The 300k GenCoord checkpoint passed the local acceptance test twice.** In separate evaluator processes, the same weights failed 3 and 5 of 129,491 clips and averaged 0.959808 in both sweeps. Both clear the limits of 30 failures and 0.85 equal-clip return. The successful run used ten proprioceptive frames, motion-only slots and zero action noise during real-simulator collection. The verified 320k run with the paper’s collection noise SD 0.1 missed the failure target. [Accepted snapshot](logs/SuperMimic-GenCoord-G1/2026-11-19_14-53-44/iteration-300000/snapshot.json) · [Verified W&B artifact `model-vt2ualwi:v29`](logs/reproduction-5090-20261118/zero-action-noise-300k/wandb-final.json) · [Run](https://wandb.ai/a2r-lab/supermimic/runs/vt2ualwi).

| Iterations | Training hours at save | Failed clips | Equal-clip return | Frame-weighted return | Sweep seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 (sweep 1) | 2.377 | 131 | 0.928415 | 0.930377 | 374.03 |
| 100,000 (sweep 1) | 4.747 | 82 | 0.939963 | 0.941465 | 374.07 |
| 110,000 (sweep 1) | 5.221 | 50 | 0.945184 | 0.947502 | 373.70 |
| 150,000 (sweep 1) | 7.116 | 24 | 0.950852 | 0.953551 | 368.14 |
| 200,000 (sweep 1) | 9.483 | 12 | 0.956523 | 0.959344 | 374.69 |
| 250,000 (sweep 1) | 11.851 | 3 | 0.959388 | 0.963147 | 372.81 |
| 300,000 (sweep 1) | 14.218 | 3 | 0.959808 | 0.963613 | 374.03 |
| 300,000 (sweep 2) | 14.218 | 5 | 0.959808 | 0.963610 | 371.18 |

The first tested follow-up checkpoint under 25 failures was 150k (24 in one sweep at 7.12 hours). At the predeclared 250k comparison, removing collection noise reduced failures from 504 to 3. The authors report 25 after 3 hours on one H100 and a final chart return of 0.9600 at 9.368 hours. Our confirmed checkpoint had 22 and 20 fewer failures than their 3-hour point and was about 0.00019 below their chart endpoint after 14.22 hours at save. These comparisons use different training times and corpora (129,491 local clips versus 129,514 published), and the authors’ one-GPU representation was unverified. [Published benchmark](https://supermimic.github.io/) · [Chart values](logs/training-curve-review-20261114/project-site.js) · [Local return curve](logs/reproduction-5090-20261118/zero-action-noise-300k/evaluation/return-curve.pdf).

**Compute.** Follow-up training took 14 h 13 min 30 s of the allowed 15 h 17 min 21 s, leaving 1 h 03 min 51 s unused. The two main training runs totaled 29 h 30 min 51 s. The two fit profiles and learning pilot added 20 min 43 s. The eight follow-up sweeps took 49 min 43 s, including 12 min 25 s for the two confirmations. All 15 full sweeps across both runs took 1 h 33 min 06 s, excluding playback, subset diagnostics and profiling. All seven follow-up service stages exited 0. The trainer completed 119,997 actor updates, 1,199,970 world-model updates and 2,399,940 critic updates. [Training audit](logs/reproduction-5090-20261118/zero-action-noise-300k/training-audit.json) · [Stage times and commands](logs/reproduction-5090-20261118/diagnostic/followup-stages.json) · [Training command](logs/reproduction-5090-20261118/zero-action-noise-300k/command.sh).

**Verification.** Both processes ran seed 0 and 4,096 environments with fixed friction and noise-free observations. We checked every motion index, name, length, termination, normalized return and file hash against the corpus and saved manifests, and recomputed both means. Each sweep covered all 47,345,789 reference frames. The three weight files matched the immutable snapshot and the pinned W&B artifact. Kneeling-stop clip 78209 and resting clip 101577 failed in both sweeps. The other four failures across the pair appeared in just one sweep each. [Sweep 1 per clip](logs/reproduction-5090-20261118/zero-action-noise-300k/evaluation/iteration-300000-sweep-1/clips.jsonl) · [Sweep 2 per clip](logs/reproduction-5090-20261118/zero-action-noise-300k/evaluation/iteration-300000-sweep-2/clips.jsonl) · [Failure groups, hashes and scores](logs/reproduction-5090-20261118/zero-action-noise-300k/evaluation/sweeps.json) · [Software versions](logs/reproduction-5090-20261118/zero-action-noise-300k/versions.txt).

All six representative motions reached their ends in the simulation viewer. Separate one-environment replays reproduced the kneeling-stop position failure and spider-crawl orientation failure. The final suite passed 261 tests with two upstream warnings. [Walk](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/walk_forward_amateur_001__A001.mp4) · [Turn](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/Turn_Start_Jog_0000_001__A017.mp4) · [Run](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/run_like_crazy_R_001__A533.mp4) · [Flip](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/flip_360_001__A304.mp4) · [Roll](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/side_roll_R_001__A304.mp4) · [Stand-up](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/stand_up_lying_R_002__A472.mp4) · [Kneeling-stop failure](logs/reproduction-5090-20261118/zero-action-noise-300k/failure-video-78209/playback.mp4) · [Spider-crawl failure](logs/reproduction-5090-20261118/zero-action-noise-300k/failure-video-107275/playback.mp4) · [Playback settings and diagnostics](logs/reproduction-5090-20261118/zero-action-noise-300k/playback/report.json).

## Original 320k run with collection action noise

Training took 55,041 seconds (15.29 RTX 5090 hours) under `supermimic-5090-320k.service` and exited 0. Both fresh-process final sweeps covered all 129,491 clips. They measured 461 and 466 failures, equal-clip returns of 0.88871 and 0.88863, and frame-weighted returns of 0.87815 and 0.87801. The failure limit is 30. The 100k checkpoint had 64 failures in one full sweep. [Stage record](logs/reproduction-5090-20261118/pipeline.json) · [Full sweeps and commands](logs/reproduction-5090-20261118/verified-320k/evaluation/sweeps.json) · [Playback](logs/reproduction-5090-20261118/verified-320k/playback/report.json).

### Original run milestones

| Collection iterations | Training hours at save | Failed clips | Equal-clip return | Frame-weighted return | Sweep minutes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 (sweep 1) | 0.481 | 786 | 0.879323 | 0.882859 | 6.40 |
| 50,000 (sweep 1) | 2.386 | 179 | 0.928619 | 0.930430 | 6.17 |
| 100,000 (sweep 1) | 4.771 | 64 | 0.927014 | 0.923557 | 6.19 |
| 250,000 (sweep 1) | 11.938 | 504 | 0.891068 | 0.880806 | 6.15 |
| 320,000 (sweep 1) | 15.283 | 461 | 0.888709 | 0.878149 | 6.15 |
| 320,000 (sweep 2) | 15.283 | 466 | 0.888627 | 0.878010 | 6.19 |

On this sparse schedule, 50k was the first scored checkpoint above the published 0.90 SONIC return level. It averaged 0.92862 after 2.39 hours and still failed 179 clips. The exact first crossing between evaluations is unknown.

The six sweeps took 37 min 14 s in total, including 12 min 20 s for the final pair. They used fresh processes, seed 0, 4,096 environments and the same frozen corpus. The 320k run performed 127,997 actor updates, 1,279,970 world-model updates and 2,559,940 critic updates. The final local snapshot matches W&B artifact `a2r-lab/supermimic/model-2hm1hoxb:v31` for all three weight files. [Audited scores and per-clip hashes](logs/reproduction-5090-20261118/verified-320k/evaluation/sweeps.json) · [Return curve](logs/reproduction-5090-20261118/verified-320k/evaluation/return-curve.pdf) · [Artifact verification](logs/reproduction-5090-20261118/verified-320k/wandb-final.json).

The authors report 25 failures after 3 hours on one H100. Our final sweeps had 436 and 441 more failures. The earlier local 100k checkpoint had 39 more after 4.77 hours. The published one-H100 return curve ends near 0.9600 after 9.37 hours, about 0.0713 above our final equal-clip returns. The paper reports 129,514 clips, 23 more than the local corpus. The published failure count and final return are from different points on the training curve. [Published results](https://supermimic.github.io/) · [Saved chart data](logs/training-curve-review-20261114/project-site.js).

Of the final failures, 414 clip IDs failed in both sweeps. The `high_jump_R` and `reach_jump_R` groups accounted for 387 and 393 failures. The six representative clips (walk, turn, run, flip, roll and stand-up) reached their ends in simulation with playback noise enabled. Two separate benchmark replays reproduced failures. [Per-clip sweep 1](logs/reproduction-5090-20261118/verified-320k/evaluation/iteration-320000-sweep-1/clips.jsonl) · [Per-clip sweep 2](logs/reproduction-5090-20261118/verified-320k/evaluation/iteration-320000-sweep-2/clips.jsonl) · [Playback](logs/reproduction-5090-20261118/verified-320k/playback/report.json) · [Failure 35121](logs/reproduction-5090-20261118/verified-320k/failure-video-35121/playback.mp4) · [Failure 86026](logs/reproduction-5090-20261118/verified-320k/failure-video-86026/playback.mp4).

## Checkpoint selection and conditional follow-up, November 19

We screened all 32 immutable checkpoints on 1,450 motions of at most 700 frames, selected from previously failed clips and 256 randomly selected controls. The 110k checkpoint had the fewest failures (52), followed by 100k (54). We selected 110k for a full noise-free sweep, with a second fresh sweep required if it passes. The selected subset cannot estimate the corpus-wide failure rate. [Selection and exact indices](logs/reproduction-5090-20261118/diagnostic/selection.json) · [Diagnostic implementation](logs/reproduction-5090-20261118/diagnostic/compare.py) · [All 34 comparisons](logs/reproduction-5090-20261118/diagnostic/scores.json) · [110k full evaluation](logs/reproduction-5090-20261118/diagnostic/full-110k/sweeps.json).

The full 110k sweep finished in 369.59 seconds at 70 failures, 0.925270 equal-clip return and 0.922029 frame-weighted return after 5.25 training hours. This missed the failure limit by 40. The 100k checkpoint had the fewest observed full-sweep failures in the original run (64). The seven clean full sweeps of this training run took 43 min 23 s in total, separate from the selected-clip diagnostic. [110k evidence](logs/reproduction-5090-20261118/diagnostic/full-110k/sweeps.json).

Applying the original collection action noise (Gaussian SD 0.1) to the frozen 320k actor rescued 425 previously failing clips and caused zero new failures in this subset. Failures fell from 430 to 5 and mean control return rose from 0.89194 to 0.94552. At 100k, the same intervention yielded 54 versus 49 failures (15 rescued, 10 new). This points to a late dependence on action perturbations. It does not establish that removing noise in training will cure it.

**Predeclared follow-up if 110k misses.** Retrain from scratch with collection action SD 0 and the same seed, architecture, proprioceptive sensor noise, optimizer and cosine clock. This intentionally changes the paper's collection setting. Compare full, clean scores at 50k, 100k, 110k and 250k with the original run. If the 250k checkpoint has at least the original 504 failures, the predicted late reliability improvement is falsified at that milestone. The final decision still requires both original acceptance gates in two fresh, clean sweeps. We also score 150k, 200k and 300k, repeat 300k, then confirm an earlier qualifying candidate if needed.

The follow-up is limited to 300k iterations to leave time margin. A separate wall-clock cutoff of 54,900 seconds plus 10 seconds for termination caps training below the first run's measured 55,041 seconds. A timeout or other training error stops the queue for review and retains completed snapshots. Evaluation time is recorded separately. [Follow-up driver](logs/reproduction-5090-20261118/diagnostic/run_followup.py) · [Audit](logs/reproduction-5090-20261118/diagnostic/audit_no_action_noise.py) · [Full-sweep verifier](logs/reproduction-5090-20261118/diagnostic/score_candidates.py).


**Service record.** GPU trainer PID 1407444 ran under `supermimic-5090-zero-action-noise.service` after the launcher returned. W&B recorded seed 0, 4,096 environments and the expected runner configuration. The service finished at 5:59 a.m. EST on November 20. Its final state was inactive with main PID 0, result success and exit code 0. All completed checkpoints are retained for evaluation and playback. [Ownership](logs/reproduction-5090-20261118/zero-action-noise-300k/ownership.json) · [Startup verification](logs/reproduction-5090-20261118/zero-action-noise-300k/startup-verification.json) · [Training log](logs/reproduction-5090-20261118/zero-action-noise-300k/training.log) · [Final stages](logs/reproduction-5090-20261118/diagnostic/followup-stages.json).

Profile A terminated with exit 1 during the first 4,096-rollout imagination.
PyTorch reported 28.71 GiB allocated, process usage 30.12 GiB and 42.44 MiB free.
The replacement accumulates gradients over two sequential blocks of 2,048.
The controlled test compares raw and clipped gradients, the actor and critic
weights, step counters and both Adam moments with an unsplit reference. The
same test executes eager and captured updates with identical starts and sensor
noise. [257-test run](logs/reproduction-5090-20261118/pytest-2.log) ·
[Randomized sensor audit](logs/reproduction-5090-20261118/context-audit.json) ·
[Profile B service](logs/reproduction-5090-20261118/profile-b/ownership.log).

The two-block pilot completed 1,200 iterations, 477 actor steps, 4,770 WM
steps and 9,540 critic steps. At full replay the four logging windows measured
23,886–23,956 simulator transitions/s. Peak allocated/reserved memory was
23.57/25.06 GiB, maximum sampled process memory was 27,166 MiB, and minimum
sampled free device memory was 3,736 MiB. The resulting 320k estimate is
15.2 training hours plus startup and saves. See the [measurements](logs/reproduction-5090-20261118/profile-b/profile.json).
The Python run finished and synced W&B versions v0 and v1. Its shell wrapper
then exited 127 because its source was edited in place while Bash was running.
The failed unit and its original logs are preserved. Subsequent units execute
their own read-only launcher copies.

After the 1,200 save, fresh evaluator processes loaded both the 600 and 1,200
snapshots. Each verified all three SHA-256 hashes against its save record.
On the 512-clip diagnostic, the 600 actor had 480 failures and equal-clip /
frame-weighted returns 0.160448 / 0.148936. The 1,200 actor had zero failures
and 0.442181 / 0.442078. The latest local actor matched 1,200 and differed
from 600. Neither subset qualifies for acceptance. [Reload evidence](logs/reproduction-5090-20261118/snapshot-reload.json).

The separate 4,000-iteration learning pilot started from scratch with seed 0 and
the same recipe, under `supermimic-5090-learning-pilot.service`. Its W&B ID is
`qm9ra2gp`. The full test suite passed 261 tests with two upstream warnings
before launch. [Training command](logs/reproduction-5090-20261118/learning-pilot/command.sh) ·
[Service ownership](logs/reproduction-5090-20261118/learning-pilot/ownership.log) ·
[Test output](logs/reproduction-5090-20261118/pytest-3.log).

The original unit `supermimic-5090-320k.service` waited for the learning pilot’s
process to exit. Its [driver](logs/reproduction-5090-20261118/run_verified_320k.py)
checks weights, counters, hashes and timing, evaluates iterations 2,000 and
4,000 on the fixed subset, then launches exactly one fresh 320k experiment.
It records the live trainer PID and cgroup independently of the original
systemd-run client. After training it scores the sparse milestones, performs
confirmation, saves per-clip files, plots the curve, and records playback.
A stage failure stops the queue. Additional training requires a separate
diagnostic decision. [Stage and exit records](logs/reproduction-5090-20261118/pipeline.json).

To repeat the original 320k training on this checkout, use a fresh run and unit name. These
commands select the frozen launcher and the measured microbatch configuration.

```bash
cd /home/roy/supermimic
systemd-run --user --unit=supermimic-5090-repeat --property=Type=exec \
  --property=Restart=no --working-directory=/home/roy/supermimic \
  /home/roy/supermimic/logs/reproduction-5090-20261118/full-run-launcher.sh \
  repeat-320k 320000 10000 --runner.policy.microbatch-size 2048
systemctl --user status supermimic-5090-repeat
```

### Reproduce the accepted checkpoint’s training, evaluation and playback

To train the successful configuration again on this checkout, start the frozen launcher under a new user service on the free GPU. This command includes the follow-up training time limit.

```bash
cd /home/roy/supermimic
root="$PWD/logs/reproduction-5090-20261118"
run_name="repeat-zero-noise-$(date +%Y%m%d-%H%M%S)"
systemd-run --user --unit="supermimic-$run_name" --property=Type=exec \
  --property=Restart=no --working-directory="$PWD" \
  /usr/bin/timeout --signal=TERM --kill-after=10s 54900s \
  "$root/full-run-launcher.sh" "$run_name" 300000 10000 \
  --runner.policy.microbatch-size 2048 --runner.policy.noise 0
systemctl --user status "supermimic-$run_name"
```

To score the saved, accepted weights directly, run these commands on the free GPU. They use a fresh output directory and independent evaluator processes, then play the six named motions from the first frame. The recorded stage commands also reproduce the individual failure videos.

```bash
cd /home/roy/supermimic
set -e
root="$PWD/logs/reproduction-5090-20261118"
job="$root/zero-action-noise-300k"
snapshot=$(jq -er .selected_snapshot "$job/evaluation/decision.json")
out=$(mktemp -d "$job/recheck-XXXXXXXX")
for repeat in 1 2; do
  .venv/bin/python -m supermimic.scripts.evaluate SuperMimic-GenCoord-G1 \
    --snapshot "$snapshot" --motion-file /home/roy/datasets/bones-seed/processed-v1/packed \
    --seed 0 --num-envs 4096 --output "$out/sweep-$repeat"
done
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  .venv/bin/python "$root/check_playback.py" "$snapshot" "$out/playback"
```

### Decisions and evidence from this run

| Paper v2 requirement | Local implementation and evidence |
| --- | --- |
| One-H100 checkpoint identity | Neither the upstream README, the release list (empty), the published demo constants nor the WASM worker identifies the training run of the browser weights. The worker copies 71 simulator generalized state values, emits 29 controls and loads an engine-specific binary. That ABI does not establish actor inputs, architecture or the training representation. No compatible one-H100 checkpoint or retained-clip manifest was identified. Selected GenCoord under the fallback rule. |
| Eq. 1 and Table A.5 | The GenCoord dynamics MLP predicts accelerations, integrates the state and uses analytic observation, reward and termination functions. The new WM setting takes nine consecutive state records and scores successors 1 through 8. Replay excludes a terminal or truncated action in the first eight records. The action on the ninth record has no successor in the window. Group weights and FK integration follow the release. |
| Eqs. 2–3 and Tables A.4–5 | One actor step over 4,096 starts, 32 steps with differentiable soft survival and a discounted terminal critic. Five freshly shuffled critic epochs with four minibatches each. Actor, critic and WM use fused Adam with tensor learning rates. The cosine clock uses the zero-based collection iteration and stops at 0.03× at 250k. Actor clips norm first, then values. Ten WM steps per learner cycle. Reference schedule gives 128,000 scheduled cycles, three skipped before nine records exist, and 127,997 completed at 320k. |
| Appendix A, observation and motion goal | Oldest to newest sensor frames, 93 values per frame in gravity, gyro, relative joint position, joint velocity and applied-action order. The configured decoder receives 930 sensor values plus a 32-dimensional tanh latent. The motion encoder takes 580 reference joint values plus 60 root-relative orientation values using the current simulated pelvis quaternion. Offsets 0,5,…,45 clamp to the last frame of each clip. The auxiliary root-translation component goes to the critic. The actor and WM normalize their inputs online. Sensor noise is added once per current frame. At replay starts with no in-episode predecessors, the paper run repeats that same noisy realization in the padded history. |
| Appendix A, dynamics and control | Dynamics widths 2048/2048/2048, motion encoder and decoder 2048/1024/512/256, critic 256/128. The simulator action is a scaled position target with the released G1 PD gains. Four 5 ms substeps per action. The four body and two root tracking terms, their widths and weights, the −0.1 smoothness and −2 joint-limit penalties, and the 0.25 m / 0.8 pelvis termination thresholds agree with the published tables. Numerical PD gains, feature ordering, exact normalization, loss scaling, and the soft termination width are release choices where the paper gives insufficient detail. |
| Appendix A.3, context | The recipe uses mjlab reset events for shared foot friction, torso COM and 29 joint encoder offsets. Replay records 33 pre-step values. COM input is normalized absolute body-frame COM position and includes the fixed nominal COM. The random offset is drawn about the original model field. An additional GenCoord discrepancy was found during profiling: the released imagined joint position omits the sampled encoder bias. The earlier sensor-parity audit disabled randomization. Profile B enables the calibration correction. Across 64 partial resets, four live slots retained their context, every reset slot redrew each field and all observed COM offsets stayed within bounds. A collected record exactly matched the pre-step context. With randomized encoder bias the largest error across sensor channels was 4.77e−7. |
| Appendix D.1, reset distribution | The fork used roughly 0.2 s bins and squared historical softmax weights. The new configuration selects roughly one-second equal-width timeline bins, preserving mjlab’s integer frame map, with 50% uniform, 40% current focus and 10% cumulative linear softmax weight. A regression reverses the ordering of difficult bins. Running-return memory, bootstrap choices, incomplete final bin and tie handling are implementation conventions. |
| Appendices E–F | Recomputed SHA-256 for the manifest and nine packed files against packing verification. All ten matched. The exact authors’ retained membership remains unknown. Evaluation keeps fixed physics and the existing reward denominator. Use the same snapshot, 4,096 environments, seed 0 and fresh processes for both confirmation sweeps. |
| Local provenance | Training checkout fc1aa9ed6b408f30713b0a7e9c4c2d85524293f2 plus each job’s saved patch. Unchanged mjlab dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd. The upstream change dropping collection noise reports different results for Adam and Muon. The verified 320k reproduction retained the paper’s action noise SD 0.1. |

The [verified dataset hashes](logs/reproduction-5090-20261118/corpus-sha256.json),
[initial regression results](logs/reproduction-5090-20261118/pytest-1.log),
[profile command](logs/reproduction-5090-20261118/profile-a/command.sh) and
[service ownership](logs/reproduction-5090-20261118/profile-a/ownership.log)
are local evidence. The [training launcher](scripts/train_5090.sh) records
the command, software revision, source diff, GPU samples, exit code and timestamps
for each new run. Snapshot directories contain separately saved model weights and
a completion record with SHA-256 and collection counters. These support scored
playback only. Follow-up training budget begins after the verified 320k run.

## Start a fresh chat

Recommended: start a new chat in this same local checkout and use the /goal
prompt below. This separates the reproduction effort from the long setup,
dependency, training, and evaluation history. This document and the saved
evidence provide the handoff. A fresh chat is an organizational choice, not a
technical requirement or a way to replenish account credits.

Use **GPT-6.2 Astra, High reasoning** (model gpt-6.2-astra, effort high). Both are
listed in the local model catalog checked on 2026-11-17. High is my recommended
starting point for this research/code task; use Extra High (xhigh) for difficult
gradient-equivalence or CUDA-graph failures. Routine status checks need less
reasoning. The current configuration was GPT-6 Astra / Extra High; this plan
does not change it. See the official [model guidance](https://learn.chatgpt.com/docs/models).

In the CLI, /new starts a new chat in the same session; select the model and
effort with /model. See [Codex commands](https://learn.chatgpt.com/docs/developer-commands).
Keep the old chat available for reference. Use this checkout: a fresh clone or
worktree would omit the ignored experiment files and locally prepared dataset.
Do not have both chats editing the checkout concurrently.

## Objective and evidence required

Train one motion-tracking policy from scratch on the single RTX 5090 using
proprioception, ten-frame history, reference goals, and the paper's one-GPU
recipe. Pursuing zero failures and training the other representation are
follow-up experiments. The initial goal ends with the local tracking result.

Use the authors' representation if provenance ties it to the published single-GPU
tracking checkpoint before training. Otherwise select `SuperMimic-GenCoord-G1`.
Its earlier sensor-only run scored 0.628710 versus 0.511212 for MaxCoord under
the same duration/history preset. This supports choosing a baseline, without
establishing how either will perform under the revised recipe.

The task ID selects the world model's state representation. GenCoord uses joint
coordinates and forward kinematics. MaxCoord uses per-body kinematics and
analytic inverse kinematics to synthesize joint readings during imagination.
Both can train a sensor-only actor or a privileged actor. Pin `actor_obs=proprio`,
`history_length=10`, and motion-only slots. The critic and world model use
simulator state during training. For simulation playback verify the actor's
actual history and reference goals, including the heading input described below.

The [official project results](https://supermimic.github.io/) report 25 failed
motions after 3 hours on one H100 and zero after 9.8 hours on two H100s, over
129,514 clips. Its published chart data end near 0.9600 and 0.9641 normalized
return respectively. These are different checkpoints and hardware settings;
the 25-failure count is not the final one-H100 checkpoint's reported count.
The local chart-data copy is
[project-site.js](logs/training-curve-review-20261114/project-site.js).

**Local acceptance requires at most 30 failed motions and an equal-clip mean
normalized return of at least 0.85 on the same sensor-only checkpoint.** Each of
two full sweeps must meet both thresholds on the frozen local corpus of 129,491 clips. Report the frame-weighted
mean alongside the acceptance statistics. Record the margins to the authors'
one-H100 results separately. A passing local score below the reported SONIC
baseline of 0.90 would still leave that paper comparison unmet.

The authors evaluated 129,514 clips. Establishing benchmark equivalence would
require their retained-clip list, preprocessing, and scoring conventions. Report
measured 5090 time to each scored checkpoint and the uncertainty between
checkpoints. The H100 times provide comparison points.

Physical transfer is a separate claim. Export and simulation checks cannot prove
performance on a real Unitree G1. Hardware access and supervised robot tests are
required to reproduce that part; their availability is not established here.

## Preserve the existing baseline

dev_5090 was created from main at
abc124c1bd5976341034668af3d12c087f2f1f55. The previously uncommitted baseline code,
tests, configuration, and documentation are included in this branch along with
this plan. Preserve them and record any future working-tree patch alongside
experiment metadata; do not reset, clean, or replace this checkout.

The editable dependency is /home/roy/mjlab_research at
dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd. Keep the existing lockfile and
mjviser 0.0.16 compatibility constraint. Replacing the research fork with regular
mjlab or upgrading dependencies is not part of the initial reproduction attempt.

Existing W&B project: [a2r-lab/supermimic](https://wandb.ai/a2r-lab/supermimic).
W&B and gated BONES-SEED access already work; reuse credentials without printing
or copying them into documentation.

All four baselines used policy batch 2048, released Muon/Adam optimization, and
constant learning rates. The 320k preset matched duration/history only.

| Task | Actor / duration | W&B run | Failures / 129,491 | Mean normalized return |
| --- | --- | --- | ---: | ---: |
| GenCoord | proprio, history 10 / 320k | [tq5a8xrg](https://wandb.ai/a2r-lab/supermimic/runs/tq5a8xrg) | 54,488 | 0.628710 |
| MaxCoord | proprio, history 10 / 320k | [6e0845qv](https://wandb.ai/a2r-lab/supermimic/runs/6e0845qv) | 83,747 | 0.511212 |
| GenCoord | privileged state / 1M | [xz2lrdej](https://wandb.ai/a2r-lab/supermimic/runs/xz2lrdej) | 101 | 0.936421 |
| MaxCoord | privileged state / 1M | [ft29lwog](https://wandb.ai/a2r-lab/supermimic/runs/ft29lwog) | 8 | 0.965617 |

The strong privileged-state scores do not validate the sensor-only actor. The
baseline experiment does not isolate batch size as the cause of poor performance.
Keep every completed checkpoint, including the weak sensor-only policies.

Evidence directory: [logs/evaluation-20261117](logs/evaluation-20261117/).
Links into logs/ resolve in the local workspace only. Experiment logs, videos,
checkpoints, and the prepared dataset are not included in the GitHub branch.

- Each RUN_ID-full-v2/report.json and clips.jsonl contains the valid full sweep.
  [benchmark-audit.json](logs/evaluation-20261117/benchmark-audit.json) independently
  checks coverage, clip identity, lengths, normalization, and failure accounting.
- [checkpoint-verification.json](logs/evaluation-20261117/checkpoint-verification.json)
  records artifact verification. RUN_ID-run.json files retain saved run settings.
- [input-parity.json](logs/evaluation-20261117/input-parity.json) and
  [sensor-parity.json](logs/evaluation-20261117/sensor-parity.json) check training
  versus evaluation inputs and synthesized versus simulator proprioception.
  Those sampled checks passed; they do not prove the entire learning algorithm.
- [playback-audit.json](logs/evaluation-20261117/playback-audit.json) and the
  RUN_ID-playback-full directories contain viewer checks and videos.
- Earlier subset scores without v2 were invalidated after a stale reset-observation
  bug. Do not reuse them. Preserve the evaluator and viewer history fixes.
- Last recorded full test result: 249 passed, 2 warnings in
  [pytest-viewer-fix.log](logs/evaluation-20261117/pytest-viewer-fix.log).
  This is prior validation, not validation of future training changes.

## 1. Resolve the recipe and dataset uncertainties

Read [paper v2](https://arxiv.org/pdf/2610.28480v2), especially equations 1–2 and
Appendices A, D, E, and F, against the current implementation. Record each
requirement as verified, different, or unspecified, with its source and code path.

The initial recipe to implement is the paper's one-GPU setting:

| Component | Target |
| --- | --- |
| Actor | Proprioception, 10-frame history; reference-motion goals |
| World model | 8 predicted transitions; 10 batches of 2048 windows |
| Policy | 4096 imagined rollouts; horizon 32; discount 0.99 |
| Optimizers | Adam throughout; actor betas (0.9, 0.95) |
| Learning rates | Start 2e-4; cosine to 3% over 250k iterations, then constant |
| Clipping | Global norm 1; actor additionally element-wise 0.1 |
| Critic | GAE lambda 0.95; 5 epochs, 4 shuffled minibatches per epoch |
| Collection | 4096 environments; 1000 replay steps/environment; action-noise SD 0.1 |
| Update cadence | 0.4 learner cycles/collection iteration; one actor update/cycle |
| Duration | 320k collection iterations |

Source: paper Table A.5 and observation specification. Verify iteration/cadence
semantics explicitly. Check network architecture, history order, normalization,
reward, termination, control timing, gains, noise, randomization, and reset
sampling against their corresponding paper tables before claiming recipe parity.

The [official browser worker](https://supermimic.github.io/demo/worker.mjs) loads
a weights.bin asset. Its [constants file](https://supermimic.github.io/demo/constants.json)
lists 36 motions and a 0.02 s control interval. Establish the input contract,
architecture, and checkpoint provenance before using it as a reference. Check
upstream assets for a training configuration and retained-clip manifest. The public
GitHub Releases listing for HybridRobotics/supermimic was empty on 2026-11-17.
If a compatible reference policy
exists, run it through our evaluator to distinguish pipeline differences from
training failures. Do not treat a demo policy as the paper checkpoint without
provenance. If details remain unavailable, document assumptions and continue
with a clearly qualified reproduction; do not invent settings or silently tune
the benchmark to match a count.

Dataset location: /home/roy/datasets/bones-seed/processed-v1/packed.
It contains 129,491 clips and 47,345,789 frames at 50 Hz (263.03 hours).
Its pinned source revision is 2f59b2077b9da34dd4e43618e705c7cb962c9a66.
Read [preprocessing assumptions](configs/bones_seed_preprocessing.json) and the
existing [README](README.md). A difference of 23 clips does not establish that
the two retained sets differ by only 23 members. Use
`processed-v1/packing_verification.json` as the hash reference. On 2026-11-17 the manifest and all nine packed files matched its SHA-256 digests.
Record the verified digests in run metadata. Preserve this corpus for acceptance
if the authors later release a different manifest.

Deliverable: a concise paper/code comparison with explicit unresolved assumptions
and a frozen run specification. Update this document as evidence changes.

Freeze the actor's complete input contract. Both tasks use ten frames of
93-dimensional proprioception for this recipe. The motion encoder also receives
joint positions, joint velocities and root-relative reference orientations at
offsets 0, 5, …, 45 frames. The goal builder uses the current pelvis orientation
for that transform. Record its simulator source and distinguish it from the five
proprioceptive terms.
Check goal construction, ordering, and clip-end clamping during turns and
immediately after resets in collection and evaluation. See
[representations.py](src/supermimic/representations.py).

## 2. Implement and validate the required training changes

| Area | Verified current behavior | Work needed |
| --- | --- | --- |
| Optimizers | [learner_utils.py](src/supermimic/learners/learner_utils.py) uses Muon for interior linear weights and Adam elsewhere | Add an explicit all-Adam recipe and serialize the choice |
| LR schedule | [supermimic_runner.py](src/supermimic/runners/supermimic_runner.py) has no training LR decay | Apply the schedule consistently to actor, critic, and world model |
| WM horizon | [wm_learner.py](src/supermimic/learners/wm_learner.py) scores L-1 predictions from L replay records | Define horizon as transitions; sample enough states/actions for 8 predictions |
| Policy batching | [policy_learner.py](src/supermimic/learners/policy_learner.py) takes an optimizer step for each configured batch | Add accumulation only if needed to fit effective batch 4096 |
| Reset sampling | [reset_sampling.py](src/supermimic/runners/reset_sampling.py) mixes 50% uniform, 40% current softmax and 10% normalized historical **squared** softmax weights | Appendix D.1 describes historical softmax weight. Document the interpretation and test the paper mode on bins whose relative difficulty reverses. Preserve the baseline mode. |

Upstream commit `8edc582` contains the pre-Muon fused, capturable Adam setup. Reuse
that parameter and tensor-LR convention inside the current optimizer factory.

The pinned mjlab fork configures foot friction, torso COM offset, and encoder
bias as startup events. Their bounds agree with Table A.3, but the paper
describes per-episode context. Verify draws across resets and record the
context stored with replay windows. If implementing reset-time sampling,
test repeated resets for cumulative COM drift and confirm live environments
retain their parameters.

The horizon discrepancy is confirmed in code; it is not an established cause of
the poor scores. Fix sequence boundaries, terminal handling, warmup, and cadence
together. The present cadence already equals 0.4:
4096 / (10 × 2048 × 8) × replay_ratio 16. Changing the storage window to nine
records must not inadvertently change update frequency. At 320k collection iterations and 4,096 environments expect
1,310,720,000 simulator transitions, with approximately 128k actor steps. Log
exact warmup skips and completed cycles, plus ten WM and twenty critic steps
per completed cycle.

The CUDA-graph helper captures optimizer state and learning-rate tensor addresses.
Use existing capture-safe conventions: mutate device LR tensors in place rather
than replacing them or changing a Python float after capture. Test the schedule
at initialization, midpoint, 250k, and 320k, and demonstrate that graph replay
actually applies changed rates. Confirm the schedule's collection-iteration
counter is separate from optimizer steps and W&B logging steps.

Write meaningful regressions for eight scored transitions, reset boundaries,
optimizer selection, update counts, and old checkpoint/config compatibility.
Reuse the existing learner, replay, runner-history, and evaluation tests. Check
whether critic shuffling should change each epoch; the current permutation is
reused across all five epochs. Resolve this against the recipe before running.

Deliverable: tested implementation plus fully serialized paper configuration.
Do not change legacy run loading so that old artifacts silently acquire new
training or inference semantics.

## 3. Fit the effective batch on the 5090

The earlier full-dataset MaxCoord policy batch4096 check ran out of memory near
30.18 GiB process usage. Reference motion tensors account for approximately
12.52 GiB; replay for approximately 5.62 GiB. Batch2048 ran successfully. Switching
Muon parameters to Adam changes optimizer-state memory, so measure the new recipe.

First profile the actual full library and filled replay buffer through graph
capture, repeated updates, and checkpoint saving for the selected representation. Record
peak allocated/reserved memory, process memory, and remaining device headroom.
A tiny-motion or early-warmup test cannot establish full training memory fit.

If the full batch does not fit, use two 2048 microbatches per effective4096 actor
update, reducing the microbatch further only if measurements require it:

- Keep actor/critic parameters fixed while generating the effective batch.
  Accumulate sample-weighted mean-loss gradients, then clip once and step once.
- Aggregate detached critic data across all microbatches. Preserve the intended
  critic epochs, minibatches, targets, and optimizer-step count.
- Preserve history, normalization, stochastic inputs, survival masks, and auxiliary
  loss scaling. Ensure consumed activation graphs are freed between microbatches.
- Compare gradients, clipped gradients, optimizer states, and parameter updates
  against an unsplit small reference batch with identical samples and controlled
  noise. Then check eager versus captured CUDA behavior and full-dataset memory.

Setting policy.num_batches to 2 currently makes two optimizer steps; it does
not implement accumulation. Likewise, keeping both microbatch activation graphs
alive would defeat the memory saving.

Keep reference storage FP32 initially. If accumulation is insufficient, assess
lower-precision storage or CPU staging separately; measure induced state, reward,
termination, action, and throughput differences before using it in the recipe.
The earlier user preference was to evaluate smaller batches first; that baseline
evaluation is now complete. These changes are a proposed next step, not a
conclusion that batch size caused the performance gap.

Single-5090 third-party results are implementation references only: the pinned
[SuperMimic-Reproduced experiment](https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/3602aed6a397b3202ac052d688d47300f443a3ea/docs/EXPERIMENTS.md)
used different storage and actor inputs, while
[HoloMotion's accumulation code](https://github.com/HorizonRobotics/HoloMotion/blob/f52dd81ae935a828b0296f5ee0f833beb7378bf0/holomotion/src/algo/supermimic/imagination.py)
comes from a different training setup. Neither validates our full recipe's fit.

## 4. Run controlled experiments under persistent services

1. Run relevant regressions, then the full test suite and formatting/lint checks.
   Follow with a bounded full-dataset CUDA pilot covering filled replay, graph
   capture, changing LR, checkpoint saving/loading, and finite updates.
2. Use the pilot to measure throughput and estimate the new full-run duration.
   Prior 320k runs took about 11–11.4 hours each; all-Adam and effective4096
   throughput are unknown. Do not reuse an H100 or old-run ETA as a promise.
3. Run a bounded learning pilot from scratch and evaluate it after it releases the
   GPU. Confirm the recipe is active and compare diagnostic curves. Early poor
   performance alone does not prove the full schedule will fail, but numerical
   or semantic failures must be corrected before a long run.
4. Before a full run, implement immutable milestone snapshots and an evaluator
   selector for a local snapshot or pinned W&B artifact version. The runner
   overwrites local filenames, W&B prunes older versions, and the evaluator
   resolves only the latest alias. Record completed collection iterations,
   training wall time, and file hashes at save time. In the pilot, save twice,
   then evaluate the first snapshot and verify its hashes against the save record.
   Figure A.1 scores checkpoints every 4,000 iterations. Budget evaluation time
   separately and label any sparser curve. Candidate sparse milestones are 10k,
   50k, 100k, 250k, and 320k. Label subset checks and selection criteria. Evaluate
   these sequentially after training releases the GPU unless a validated in-process
   evaluation path preserves all training state and fits memory.
5. Start one clean 320k run of the selected representation. Evaluate it on the
   frozen corpus. Record any correctness failure or intentional recipe revision.

If the first verified 320k run misses acceptance, inspect the full-sweep failures,
learning curve and model diagnostics. Record one hypothesis, the baseline
comparison, and the observation that would falsify it. Only a diagnostic that
supports a specific change authorizes further training. Budget **at most one
additional full-run equivalent** across follow-up pilots and training, using
the measured training GPU-hours of the verified 320k run as the reference.
Report cumulative training and evaluation hours separately. Stop for review
when that budget is spent, the evidence gives no justified next experiment,
or a checkpoint passes twice. Report an unmet target explicitly.

Long-term training must be owned by a systemd user service, never a Codex terminal
or exec session. Reuse the established queue conventions, with new service names
and output directories. The old service is supermimic-training.service; its queue
is [logs/bones-seed-four-runs-20261113](logs/bones-seed-four-runs-20261113/).
That service and the full-evaluation/full-playback services were inactive with
Result=success, ExecMainStatus=0, and MainPID=0 when this plan was written.

Before leaving a new run unattended, verify that the service owns the training
PID/cgroup after the launcher exits. Record the unit name, command, logs, run ID,
start time, exit/failure state, checkpoint locations, configuration, source diff,
dependency commits, dataset hashes, and measured ETA. Verify W&B logging to
a2r-lab/supermimic. Queue GPU work sequentially; do not evaluate alongside training
that occupies the card. A goal pause leaves the service running unless explicitly
asked to stop training.

Current checkpoints are weights only, not exact training-resume state. Do not
warm-start one and call it a resumed reproduction. The queue must stop on failure
and surface it; no silent restart from scratch. Exact resume is a separate feature
requiring all relevant optimizer, replay, simulator, RNG, normalization, sampler,
and counter state, with interruption-equivalence validation. Do not claim it
exists. Persistent service ownership prevents the earlier Codex-session failure,
but does not make training survive a machine reboot.

## 5. Evaluate the checkpoint and verify playback

Reuse [evaluate.py](src/supermimic/scripts/evaluate.py) and its existing audit
conventions. Evaluate every clip from its first frame with fixed friction and
noise/randomization disabled. Preserve the documented initialization, motion-end,
reward-denominator, and failure rules. Record any deviation from the paper.
Report complete per-clip results, failure groups, both mean-return aggregations,
checkpoint hashes, source/configuration, simulator versions, and environment count.

Compare curves using the same metric: full evaluation scores versus the paper's
evaluation scores, not training reward versus paper evaluation return. Plot
collection iterations and measured GPU-hours, distinguish training time from
evaluation overhead, and state any unmatched sample/update counts. Do not convert
H100 GPU-hours into 5090 hours using an assumed multiplier.

Run each confirmation in a fresh evaluator process with the same checkpoint
hashes, seed, clip order, and environment count. The full-v2 baselines used
4,096 environments and seed 0. Pin these settings for comparable results.
Report both per-clip files and every sweep attempted. If either sweep misses
either threshold, record the discrepancy and investigate. A one-environment
replay can help diagnose a failure.

Inspect full walking, running, turning, flip, side-roll, and stand-up clips, plus
failures. Reuse [play.py](src/supermimic/scripts/play.py), its corrected history
handling, and the saved playback harness where applicable. Keep short world-model
overlay diagnostics separate from policy completion and paper WM-error metrics.

Record exact copyable commands for training under the service, selecting the
checkpoint, both evaluation sweeps, and replaying the named clips. Include the
working directory, resolved motion path, run ID, artifact version, hashes,
configuration and software revisions. ONNX export, 50 Hz inference timing,
and physical deployment are deferred follow-ups.

## Completion checklist

- [x] One selected task trained from scratch with the verified sensor-only recipe.
- [x] Hardware adaptations preserve the intended update, or differences are measured and disclosed.
- [x] Complete benchmark evidence meets the declared local target for the selected policy, in each of two complete sweeps on the same checkpoint.
- [x] Representative playback is recorded and exact training, evaluation, and playback commands are provided.
- [x] Dataset/evaluation differences and the boundary of the physical-deployment claim remain explicit.
- [x] Configurations, code provenance, checkpoints, curves, and a runnable workflow are recorded.

If targets remain unmet, report the shortfall and the next evidence-driven
experiment. Do not mark the performance objective complete merely because all
jobs exited successfully. An exact-paper claim also requires resolution of the
benchmark-equivalence gaps. Changing an acceptance target requires an explicit
user decision, not an undocumented adjustment to make a run pass.

## Copy into the new chat

Select GPT-6.2 Astra with High reasoning, open /home/roy/supermimic, and paste:

```text
/goal Reproduce SuperMimic's sensor-only policy performance on this computer's single RTX 5090, working in /home/roy/supermimic on dev_5090. Read REPRODUCTION_5090.md, README.md, the applicable AGENTS instructions, and the saved evaluation evidence first. Preserve all existing local changes and checkpoints.

Use the representation tied by provenance to the authors' one-GPU tracking checkpoint if known before training, otherwise GenCoord. Explicitly select proprio observations, ten history frames and motion-only slots. Train one motion-tracking policy from scratch using the one-GPU sensor-only recipe. Verify paper v2 against the implementation and audit the reset sampler, sim context, observation history and motion goals. Implement and test all-Adam, cosine decay and eight scored world-model transitions. Profile the full corpus and filled replay on this card. Preserve effective batch 4096 and one actor update per cycle, validating gradient accumulation if needed. Prove you can reload and evaluate an immutable intermediate checkpoint after a later save.

Acceptance requires at most 30 failed clips AND an equal-clip mean normalized return of at least 0.85 in EACH of two full sweeps of the same checkpoint on all 129,491 frozen local clips. Report frame-weighted scores, actual training and evaluation time, and gaps to the published one-H100 benchmark. Deliver representative playback, per-clip evidence and exact commands to reproduce the result. Defer zero failures, a second representation, export, and hardware trials.

Run one verified 320k experiment. If it misses, at most one additional full-run equivalent of training is authorized, conditional on a diagnostic supporting a specific change. Count follow-up training against the first verified run's measured GPU-hours. Stop for review if there is no supported next experiment or the budget is exhausted. Keep the performance objective explicitly unmet if either acceptance threshold is missed.

All long training must run under a persistent systemd user service. Verify service ownership after the launcher exits, record logs and failure status, serialize work on the one GPU, and never claim weights-only checkpoints support exact resume. Use measured memory and throughput before full runs. Keep this plan updated with decisions and evidence, and preserve a concise handoff when pausing.
```

### Scoring choices fixed before the full run

The bounded pilot diagnostic uses frozen library indices 64,000 through 64,511.
Both pilot snapshots use that same contiguous subset, 512 environments and seed
0. The subset samples one region of the sorted library and serves only as a
loading and learning diagnostic. The final 320k checkpoint is the primary
acceptance candidate. Score the 10k, 50k, 100k, 250k and 320k milestones on the
full corpus for a sparse training curve. The final confirmation repeats 320k in
a fresh process. If an earlier milestone passes both thresholds on its first
sweep while the final checkpoint misses, confirm that earlier checkpoint
separately and publish all attempted sweeps.
