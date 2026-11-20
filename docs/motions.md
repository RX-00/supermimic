# Prepare G1 reference motions

The main paper benchmark uses the G1-retargeted [BONES-SEED dataset](https://huggingface.co/datasets/bones-studio/seed).
Appendix E reports 142,220 raw clips, 129,667 after name filtering, and 129,514
after feasibility filtering (264.56 hours). The
[paper and its source](https://arxiv.org/abs/2610.28480v2), project website, and
official repositories do not publish the complete keyword list, retained-clip
manifest, or numeric definitions of upright, standing height, and nearly stationary.

Our selected rules and pinned sources are recorded in
[configs/bones_seed_preprocessing.json](../configs/bones_seed_preprocessing.json).
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

## Get the inputs

Use an existing converted G1 NPZ for a one-clip run, or the packed directory for a full sweep. On the reproduction host the data root is `/home/roy/datasets/bones-seed/processed-v1`. It contains `npz/flip_360_001__A304.npz` and `packed/`. A fresh Git checkout contains the converter and filtering rules. Motion data are distributed separately by BONES Studio.

For a new conversion, request dataset access on Hugging Face, sign in to the Hugging Face CLI (`uvx --from huggingface-hub hf auth login`), and download the two files from the pinned revision. The compressed G1 archive alone is 23.5 GB. Allow space for extraction, the converted NPZs and the packed copy. Run the following in Bash from the SuperMimic repository root. [BONES-SEED data and access terms](https://huggingface.co/datasets/bones-studio/seed).

```bash
DATA="$HOME/datasets/bones-seed"
uvx --from huggingface-hub hf download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset \
  --revision 2f59b2077b9da34dd4e43618e705c7cb962c9a66 \
  --local-dir "$DATA"
mkdir -p "$DATA/raw"
tar -xzf "$DATA/g1.tar.gz" -C "$DATA/raw"
```

The metadata field `move_g1_path` is relative to `raw` (for example, beneath `raw/g1/`). The converter needs both the metadata and those CSVs. For a small first conversion, add `--limit 10` to the converter and use a separate output directory. Choose a retained NPZ from that directory for the README quick start. For the benchmark, convert and pack the full library:

```bash
uv run --locked python -m supermimic.scripts.prepare_bones \
  --raw-root "$DATA/raw" --metadata "$DATA/metadata/seed_metadata_v004.csv" \
  --output "$DATA/processed-v1" --workers 12
uv run --locked python -m mjlab.scripts.pack_motions \
  --input "$DATA/processed-v1/npz/*.npz" \
  --output "$DATA/processed-v1/packed" --workers 12
```

The converter requires a new output directory and leaves raw files intact. Its
`manifest.csv` records every clip and exclusion, `preprocessing_spec.json` records
the chosen rules, and `report.json` records counts and source checksums. `--limit`
is available for a small conversion check; omit it for the full dataset. Pass the
packed directory to both training and playback using `--motion-file`.

The source CSV and the packed library use different representations. Give `--motion-file` a converted G1 NPZ or the **packed directory** containing the arrays. A clip index is relative to the particular library loaded. For portable replays record its name, the library revision, and the index. The original run’s per-file corpus digests are in the training host’s `logs/reproduction-5090-20261118/zero-action-noise-300k/corpus.sha256`.
