# Reproducing SuperMimic on one RTX 5090

Prepared 2026-11-17. Working directory: /home/roy/supermimic. Branch: dev_5090.

Status: planning only. The four earlier runs are finished baselines. Their
training and playback succeeded, but paper-level sensor-only policy performance
has not been reproduced. No new training was started when this plan was written.
The previous goal's completion flag and completion-audit.json cover the earlier
train/play work; they do not establish reproduction of the paper's performance.

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

Train both SuperMimic-GenCoord-G1 and SuperMimic-MaxCoord-G1 from scratch on the
single 32 GB RTX 5090, using sensor-only actors and a documented implementation
of the paper recipe. Establish policy quality with complete motion-library
evaluation, representative playback, and export/runtime parity.

The [official project results](https://supermimic.github.io/) report 25 failed
motions after 3 hours on one H100 and zero after 9.8 hours on two H100s, over
129,514 clips. Its published chart data end near 0.9600 and 0.9641 normalized
return respectively. These are different checkpoints and hardware settings;
the 25-failure count is not the final one-H100 checkpoint's reported count.
The local chart-data copy is
[project-site.js](logs/training-curve-review-20261114/project-site.js).

Proposed project acceptance target: each sensor-only task reaches at least 0.96
equal-clip mean normalized return and zero failures on the frozen local corpus,
confirmed in two complete sweeps. Report both sweeps and frame-weighted scores.
This is an explicit local target, not a claim that the paper used our aggregation
or reported identical performance for both representations. Identify which
representation/configuration produced the published headline results first.

Our corpus differs from the paper's. Even meeting that target would establish a
documented reproduction on our approximate corpus. Exact benchmark equivalence
requires the authors' retained-clip list, preprocessing, and scoring conventions.
A useful intermediate milestone is at most 25 local failures; do not confuse
that with completion of the zero-failure target. H100 wall-clock claims are
reference points, not deadlines for the 5090.

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

Inspect the [official browser demo](https://supermimic.github.io/demo/) and upstream
release assets for an author policy, full configuration, and filter manifest.
Their availability has not been established. If a compatible reference policy
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
the two retained sets differ by only 23 members. Freeze our manifest and hashes
before comparing runs. Preserve the old dataset if a better manifest is found.

Deliverable: a concise paper/code comparison with explicit unresolved assumptions
and a frozen run specification. Update this document as evidence changes.

## 2. Implement and validate the required training changes

| Area | Verified current behavior | Work needed |
| --- | --- | --- |
| Optimizers | [learner_utils.py](src/supermimic/learners/learner_utils.py) uses Muon for interior linear weights and Adam elsewhere | Add an explicit all-Adam recipe and serialize the choice |
| LR schedule | [supermimic_runner.py](src/supermimic/runners/supermimic_runner.py) has no training LR decay | Apply the schedule consistently to actor, critic, and world model |
| WM horizon | [wm_learner.py](src/supermimic/learners/wm_learner.py) scores L-1 predictions from L replay records | Define horizon as transitions; sample enough states/actions for 8 predictions |
| Policy batching | [policy_learner.py](src/supermimic/learners/policy_learner.py) takes an optimizer step for each configured batch | Add accumulation only if needed to fit effective batch 4096 |

The horizon discrepancy is confirmed in code; it is not an established cause of
the poor scores. Fix sequence boundaries, terminal handling, warmup, and cadence
together. The present cadence already equals 0.4:
4096 / (10 × 2048 × 8) × replay_ratio 16. Changing the storage window to nine
records must not inadvertently change update frequency.

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
capture, repeated updates, and checkpoint saving for both representations. Record
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
4. Start a clean full GenCoord run, evaluate it, then a clean full MaxCoord run
   and evaluate it. Keep the run specification consistent. Investigate a failed
   correctness check before proceeding; record any intentional recipe revision.
5. Save intermediate checkpoints for later offline curves, for example at 10k,
   50k, 100k, 250k, and 320k. Label subset checks and selection criteria. Evaluate
   these sequentially after training releases the GPU unless a validated in-process
   evaluation path preserves all training state and fits memory.

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

## 5. Evaluate quality, then export and check playback

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

Near the target, repeat the complete evaluation with unchanged settings. Some
previous individual replay outcomes differed from batched results; a fixed seed
does not guarantee bitwise reproducibility. Report variability rather than choosing
the better sweep. If training reproducibility is still uncertain after success,
use another training seed as a separately reported robustness experiment.

Inspect full walking, running, turning, flip, side-roll, and stand-up clips, plus
failures. Reuse [play.py](src/supermimic/scripts/play.py), its corrected history
handling, and the saved playback harness where applicable. Keep short world-model
overlay diagnostics separate from policy completion and paper WM-error metrics.

Use [exporter.py](src/supermimic/runners/exporter.py) for sensor-only ONNX export.
Full-library runs skip automatic export because the current motion-embedding
export path has a 15,000-frame guard. Export with an appropriately small reference
set or a validated separate-reference interface. Compare PyTorch and ONNX actions
on identical histories/goals, verify term ordering and joint/action metadata, and
measure sustained runtime against the 50 Hz control deadline on the declared
deployment computer. Export/runtime success establishes compatibility only;
physical G1 performance remains unverified until actual robot trials.

## Completion checklist

- [ ] Both task IDs trained from scratch with the verified sensor-only recipe.
- [ ] Hardware adaptations preserve the intended update, or differences are measured and disclosed.
- [ ] Complete benchmark evidence meets the declared local target for both policies in repeated sweeps.
- [ ] Full playback, export action parity, and real-time runtime checks pass.
- [ ] Dataset/evaluation differences and the boundary of the physical-deployment claim remain explicit.
- [ ] Configurations, code provenance, checkpoints, curves, and a runnable workflow are recorded.

If targets remain unmet, report the shortfall and the next evidence-driven
experiment. Do not mark the performance objective complete merely because all
jobs exited successfully. An exact-paper claim also requires resolution of the
benchmark-equivalence gaps. Changing an acceptance target requires an explicit
user decision, not an undocumented adjustment to make a run pass.

## Copy into the new chat

Select GPT-6.2 Astra with High reasoning, open /home/roy/supermimic, and paste:

```text
/goal Reproduce SuperMimic's sensor-only policy performance on this computer's single RTX 5090, working in /home/roy/supermimic on dev_5090. Read REPRODUCTION_5090.md, README.md, the applicable AGENTS instructions, and the saved evaluation evidence first. Preserve all existing local changes and checkpoints.

Carry out the plan: verify paper v2 against code, investigate author reference artifacts and dataset assumptions, implement and test the paper optimizer/schedule and world-model transition horizon, and make effective policy batch4096 fit with validated gradient accumulation if necessary. Then train both GenCoord and MaxCoord from scratch, evaluate the complete frozen corpus, compare evaluation curves fairly, and validate playback plus sensor-only export/runtime parity. Proceed through the documented checks without asking me to reapprove routine implementation steps or the planned runs.

Use the plan's explicit local performance target: mean normalized return at least 0.96 and zero failures for each task in two complete sweeps. This is an approximate-corpus reproduction until exact dataset/evaluation equivalence is established. Do not substitute privileged-state results, lower the target silently, or mark the goal complete just because training and playback run. Report remaining shortfalls honestly; real G1 transfer is a separate hardware-dependent validation.

All long training must run under a persistent systemd user service. Verify service ownership after the launcher exits, record logs and failure status, serialize work on the one GPU, and never claim weights-only checkpoints support exact resume. Use measured memory and throughput before full runs. Keep this plan updated with decisions and evidence, and preserve a concise handoff when pausing.
```
