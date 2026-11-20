"""Select unaugmented 50 Hz BONES clips for the two from-scratch trainers."""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

# Disjoint command vocabularies provide a second guard across performers.
# Each full expression denotes a motion, never a substring of a compound take.
VOCABULARY = {
  "locomotion": {
    "train": r"(walk_ff|jog_ff)_(start_|stop_|loop_)?\d{3}(_[RL])?",
    "validation": r"walk_arc_cw_(start|stop|loop)_R(_normal_pace)?",
    "test": r"(run_(start|stop|loop)_180_R|walk_hands_on_back_180_(start|stop|loop)_R)",
  },
  "turns": {
    "train": r"(neutral_)?idle_turn_\d{3}_[RL]",
    "validation": r"(crouch_turn|turn_walk)_\d{3}_[RL]",
    "test": r"(turn_jog|turn_run)_\d{3}_[RL]",
  },
  "floor": {
    "train": r"(kneeling_(start_R|loop|stop_R)|on_all_fours_(180|360)_(start|stop|loop)_R)",
    "validation": r"(idle_to_on_all_fours_idle|idle_on_all_fours_to_idle)",
    "test": r"(idle_to_crawl_idle|crawl_idle_to_idle|crawl_ff_(start|loop|stop)_\d{3}_R)",
  },
  "upper_body": {
    "train": r"(dancing_routine_opt_[12]|clap_(forced|enthusiastic)|brush_off_dust|looking_around_(up|horizontally|on_ground))",
    "validation": r"(wave_R|wave_two_hands_R_opt_[12]|dance_latino_[a-zA-Z0-9_]+)",
    "test": r"(dance_basic_[a-zA-Z0-9_]+|show_bicep_R|don_t_know_(no|with)_hands)",
  },
  "jumping": {
    "train": r"(jump_ff_\d{3}(_R)?|jump_up_loop_R|reach_jump_R)",
    "validation": r"(jumping_jack|jump_twice_R|jump_for_joy_R)",
    "test": r"(turn_jump_\d{3}(_R)?|high_jump_R_opt_[12])",
  },
}
QUOTA = {"train": 128, "validation": 20, "test": 20}


def digest(value):
  return hashlib.sha256(value.encode()).hexdigest()


def actor_partition(actor):
  bucket = int(digest("actor/" + actor), 16) % 10
  return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def select(root):
  processed = root / "processed-v1"
  with (processed / "manifest.csv").open() as stream:
    retained = {r["filename"]: r for r in csv.DictReader(stream) if r["keep"] == "1"}
  packed = json.loads((processed / "packed/motion_names.json").read_text())
  if len(packed) != len(retained) or set(packed) != retained.keys():
    raise ValueError("Native training membership differs from conversion manifest")
  rules = {s: [(f, re.compile(v[s])) for f, v in VOCABULARY.items()] for s in QUOTA}
  pools = defaultdict(list)
  with (root / "metadata/seed_metadata_v004.csv").open() as stream:
    for r in csv.DictReader(stream):
      name, actor = r["filename"], r["take_actor"]
      if (
        not actor
        or name not in retained
        or r["is_mirror"] != "False"
        or r["content_props"] != "0"
      ):
        continue
      split = actor_partition(actor)
      for family, pattern in rules[split]:
        if pattern.fullmatch(r["content_name"]):
          pools[family, split].append(r)
  rows = []
  for family in VOCABULARY:
    for split in QUOTA:
      # Prefer one take per actor and content before reusing a combination.
      counts = defaultdict(int)
      ranked = []
      for r in sorted(pools[family, split], key=lambda r: digest(r["filename"])):
        key = r["take_actor"], r["content_name"]
        ranked.append((counts[key], digest(r["filename"]), r))
        counts[key] += 1
      group = sorted(ranked)[: QUOTA[split]]
      if not group:
        raise ValueError(f"Empty {family} {split}")
      for _, _, r in group:
        name = r["filename"]
        with (processed / "npz" / (name + ".npz")).open("rb") as stream:
          checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        rows.append(
          dict(
            split=split,
            family=family,
            motion=name,
            actor=r["take_actor"],
            take=r["take_org_name"],
            date=r["take_date"],
            content=r["content_name"],
            frames=int(retained[name]["output_frames"]),
            fps=50,
            sha256=checksum,
            supermimic_300k_member=True,
          )
        )
  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("bones", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  rows = select(args.bones)
  with args.output.open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
