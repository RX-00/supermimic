#!/usr/bin/env bash
# Run via systemd-run --user as documented in REPRODUCTION_5090.md.
set -euo pipefail
cd /home/roy/supermimic
: "${INVOCATION_ID:?launch this script under a systemd user service}"
run_name=${1:?run name}
iterations=${2:?collection iterations}
save_every=${3:?snapshot interval}
shift 3
run_dir="logs/reproduction-5090-20261118/$run_name"
mkdir -p "$(dirname "$run_dir")"
mkdir "$run_dir"
exec >>"$run_dir/training.log" 2>&1
export WANDB_ENTITY=a2r-lab WANDB_RESUME=never PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
date -u +%FT%TZ > "$run_dir/started"
cp "$0" "$run_dir/launcher.sh"
cp uv.lock "$run_dir/"
git diff HEAD > "$run_dir/source.diff"
git rev-parse HEAD > "$run_dir/git-head"
git -C ../mjlab_research rev-parse HEAD > "$run_dir/mjlab-head"
dataset=/home/roy/datasets/bones-seed/processed-v1
cp "$dataset/packing_verification.json" "$run_dir/"
printf '%s  %s\n' ab10a4df46fa8cee85544df66f12334c945e88edb4fcc379db192fe6578badad "$run_dir/packing_verification.json" | sha256sum --check
jq -r '(.packed_sha256 | to_entries[] | [.value, ("packed/"+.key)] | join("  ")), (.manifest_sha256+"  manifest.csv")' "$run_dir/packing_verification.json" > "$run_dir/corpus.sha256"
hash_list=$(realpath "$run_dir/corpus.sha256")
(cd "$dataset" && sha256sum --check "$hash_list")
.venv/bin/python -c 'from importlib.metadata import version; print({p:version(p) for p in ("torch","mujoco","mujoco-warp","warp-lang","mjlab","tensordict","wandb")})' > "$run_dir/versions.txt"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 2 > "$run_dir/device.csv" &
monitor=$!
nvidia-smi --query-compute-apps=timestamp,pid,used_memory --format=csv -l 2 > "$run_dir/process-memory.csv" &
process_monitor=$!
trap 'code=$?; kill "$monitor" "$process_monitor" 2>/dev/null || true; echo "$code" > "$run_dir/exit_code"; date -u +%FT%TZ > "$run_dir/finished"' EXIT
cmd=(/home/roy/.local/bin/uv run --locked python -u -m supermimic.scripts.train
  SuperMimic-GenCoord-G1 --device cuda:0 --num-envs 4096 --env.seed 0
  --motion-file /home/roy/datasets/bones-seed/processed-v1/packed
  --runner.run-name "$run_name" --runner.max-iterations "$iterations"
  --runner.save-interval "$save_every" --runner.log-interval 50
  --runner.actor-obs proprio --runner.history-length 10
  --runner.slot-mix.motion 1 --runner.slot-mix.velocity 0
  --runner.wm.all-adam True --runner.policy.all-adam True
  --runner.policy.batch-size 4096 --runner.policy.num-batches 1
  --runner.wm.horizon 8 --runner.wm.horizon-counts-transitions True
  --runner.policy.shuffle-each-epoch True --runner.cosine-decay-iterations 250000
  --runner.gencoord-encoder-bias True --runner.episode-sim-context True --runner.share-reset-sensor True
  --runner.return-sampler.bin-seconds 1 --runner.return-sampler.square-historical-weights False
  --runner.immutable-snapshots True "$@")
printf '%q ' "${cmd[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
"${cmd[@]}"
