#!/usr/bin/env bash
# Single workstation companion for the GenCoord seed queue. Runs without a GPU.
set -euo pipefail
root=$(pwd)
campaign=$(realpath "${1:?training queue directory}")
service=$(jq -er .service "$campaign/queue.json")
wbc=$(realpath ../wbc-mjlab)
art="$root/artifacts/pipeline-comparison"
eval_py="$HOME/miniconda3/envs/wbc-mjlab/bin/python"
base="$wbc/logs/stability-final-v12/completed/Wbc-G1-20236"
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
validation="$campaign/validation"
mkdir "$validation"
cp "$wbc"/scripts/comparison/{sweep,common,planner,convert_reference}.py "$validation/"
cp scripts/comparison/export_actor.py "$validation/"
cp "$art/lafan/manifest.csv" "$validation/lafan.csv"
sha256sum "$validation/"*.py "$validation/lafan.csv" > "$validation/source.sha256"
exec >>"$validation/validation.log" 2>&1
trap 'echo "$?" > "$validation/exit_code"' EXIT

sweep() {
  local actor=$1 out=$2
  PYTHONPATH="$art/py-extra" "$eval_py" "$validation/sweep.py" "$base" \
    "$campaign/bones-motions.csv" "$art/wbc-bones" "$out/bones" --actor "$actor" \
    --split validation --rates 5 --horizons 15 45 --revisions none --workers 2
  PYTHONPATH="$art/py-extra" "$eval_py" "$validation/sweep.py" "$base" \
    "$validation/lafan.csv" "$art/wbc-lafan" "$out/lafan" --actor "$actor" \
    --split validation --rates 2 5 10 --horizons 5 15 45 --revisions none slow --workers 3
}

# Exercise the BONES sweep with the existing policy while the first checkpoint trains.
PYTHONPATH="$art/py-extra" "$eval_py" "$validation/sweep.py" "$base" \
  "$campaign/bones-motions.csv" "$art/wbc-bones" "$validation/accepted/bones" \
  --actor "$art/sm-300k" --split validation --rates 5 --horizons 15 45 --revisions none --workers 2

mapfile -t seeds < <(jq -r '.plans[].seed' "$campaign/queue.json")
shopt -s nullglob
for seed in "${seeds[@]}"; do
  for iteration in 010000 050000 100000 150000 200000 250000 300000; do
    while true; do
      meta=("$campaign/seed-$seed/logs/SuperMimic-GenCoord-G1"/*/"iteration-$iteration/snapshot.json")
      if (( ${#meta[@]} == 1 )) && jq -e '.sha256["actor.pt"]' "${meta[0]}" >/dev/null; then break; fi
      if [[ $(systemctl --user show "$service" -p SubState --value) != running ]]; then
        echo "Missing snapshot for seed $seed iteration $iteration after training stopped"
        exit 1
      fi
      sleep 30
    done
    dest="$validation/seed-$seed/iteration-$iteration"
    mkdir -p "$dest"
    echo "$(date -Iseconds) validating seed=$seed iteration=$iteration"
    PYTHONPATH="$campaign/source" "$root/.venv/bin/python" "$validation/export_actor.py" \
      "$(dirname "${meta[0]}")" /home/roy/datasets/bones-seed/processed-v1/npz/jump_and_land_heavy_001__A001.npz \
      "$dest/actor" --device cpu >"$dest/export.log" 2>&1
    sweep "$dest/actor" "$dest"
    date -Iseconds > "$dest/finished"
  done
done
