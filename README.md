# SuperMimic

SuperMimic teaches a simulated Unitree G1 to follow a reference motion. At each control step, the policy reads recent robot observations and a short window of the target motion, then commands 29 joints. During training, a learned world model lets the actor practice many possible steps between simulator updates. [Project](https://supermimic.github.io/) · [Paper](https://arxiv.org/abs/2610.28480v2).

**Start with the 300k sensor-only GenCoord policy on `dev_5090`.** It trained in 14 h 13 min on one RTX 5090. In two evaluations of the same weights it failed **3 and 5 of 129,491 motions** and averaged **0.95981 normalized return** in both. These numbers measure coverage of the local training-motion library in simulation. [Reproduction report](REPRODUCTION_5090.md).

## Quick start

Use Bash on Linux. The tested environment uses Python 3.13, an NVIDIA GPU with a working CUDA driver, `uv`, and the research fork of mjlab checked out **beside** this repository. You need read access to both source repositories and, for the download below, the W&B model artifact. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
mkdir -p ~/supermimic-ws && cd ~/supermimic-ws
git clone --branch dev_5090 https://github.com/RX-00/supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git
git -C mjlab_research checkout dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
cd supermimic
uv sync --locked --python 3.13
uv run --locked pytest -q
```

**Get the policy.** From the repo root, download the pinned weights (about 79 MB) and copy their matching manifest. The loader uses it to rebuild the network and verify the weight hashes. If you are already on the reproduction host, you can skip the download and set `SNAPSHOT="$PWD/logs/SuperMimic-GenCoord-G1/2026-11-19_14-53-44/iteration-300000"`.

```bash
uv run --locked wandb login
SNAPSHOT="$PWD/artifacts/gencoord-300k-v29"
uv run --locked wandb artifact get --root "$SNAPSHOT" a2r-lab/supermimic/model-vt2ualwi:v29
cp docs/checkpoints/gencoord-300k/snapshot.json "$SNAPSHOT/snapshot.json"
```

**Run one motion.** Set `MOTION` to one converted G1 motion file. The example path is present on the reproduction host. On a new machine, use an existing compatible NPZ or follow the [motion preparation guide](docs/motions.md#get-the-inputs). Run these commands in the same shell:

```bash
TASK=SuperMimic-GenCoord-G1
MOTION=/home/roy/datasets/bones-seed/processed-v1/npz/flip_360_001__A304.npz
OUT=$(mktemp -d)
uv run --locked python -m supermimic.scripts.evaluate "$TASK" \
  --snapshot "$SNAPSHOT" --motion-file "$MOTION" --num-envs 1 --seed 0 --output "$OUT/clip"
cat "$OUT/clip/clips.jsonl"
```

A successful smoke test writes one clip record and a `report.json` with `complete: true` and `evaluated_motions: 1`. The clip record includes its return, how many steps it ran, and whether it failed. To **watch** the same policy, start the browser viewer, open the address printed in the terminal, and press Ctrl+C when finished:

```bash
uv run --locked python -m supermimic.scripts.play "$TASK" \
  --snapshot "$SNAPSHOT" --motion-file "$MOTION" --viewer viser --num-steps-rollout 32
```

## Where to go next

| I want to… | Start here |
| --- | --- |
| Understand the actor, motion goals, world model and training loop | [How the policies work](docs/policies.md#how-the-policies-work) |
| Choose GenCoord, MaxCoord, a sensor-only policy or a full-state baseline | [Checkpoint catalog](docs/policies.md#choose-a-checkpoint) |
| Inspect a motion, record a video or replay a failure | [Playback and video](docs/policies.md#playback-and-video) |
| Run two full sweeps, interpret a report or test code changes | [Testing a policy](docs/policies.md#test-a-policy) |
| Train under systemd and find the next snapshot | [Training and checkpoint lifecycle](docs/policies.md#train-on-one-gpu) |
| Rebuild the 50 Hz reference library | [Data, conversion and filtering](docs/motions.md) |
| See the measured memory, run times, ablations and paper comparison | [RTX 5090 experiment record](REPRODUCTION_5090.md) |
| Find the implementation behind a result | [Source map](docs/policies.md#source-map) |

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
