"""Derive the norm stats required by the StreamingVLA (SVLA) checkpoint.

The released HF checkpoint (`Yiran-Shi/StreamingVLA_LIBERO`) does **not** ship the
`assets/<asset_id>/norm_stats.json` that `policy_config.create_trained_policy` loads
unconditionally. Without it the server either crashes (`normalize.load` raises
FileNotFoundError) or, if the file is present but incomplete, silently feeds
unnormalized tensors to the model.

SVLA additionally normalizes `action_states`, and `scripts/compute_norm_stats_new.py`
fills that entry from the *actions* array:

    stats["state"].update(np.asarray(batch["state"]))
    stats["actions"].update(np.asarray(batch["actions"]))
    stats["action_states"].update(np.asarray(batch["actions"]))

i.e. `action_states` shares the statistics of `actions` (paper Eq. 9). This script
reproduces that: it takes the official pi05_libero stats and derives `action_states`
as a copy of `actions`.

Usage:
    python scripts/make_norm_stats.py                 # uses the default paths
    python scripts/make_norm_stats.py --src ... --dst ...
"""

import argparse
import copy
import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_SRC = REPO_ROOT / "ckpt/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
DEFAULT_DST = REPO_ROOT / "ckpt/StreamingVLA_LIBERO/58300/assets/libero/norm_stats.json"

# Must match the key set that Policy.streaming_infer / Policy.infer emit, because
# Unnormalize is applied with strict=True.
REQUIRED_KEYS = ("state", "actions", "action_states")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=pathlib.Path, default=DEFAULT_SRC, help="Source norm_stats.json")
    parser.add_argument("--dst", type=pathlib.Path, default=DEFAULT_DST, help="Destination norm_stats.json")
    args = parser.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()
    if not src.exists():
        raise SystemExit(f"Source norm stats not found: {src}")

    data = json.loads(src.read_text())
    norm_stats = data.get("norm_stats")
    if norm_stats is None:
        raise SystemExit(f"{src} does not contain a top-level 'norm_stats' object")

    for key in ("state", "actions"):
        if key not in norm_stats:
            raise SystemExit(f"{src} is missing the '{key}' entry")

    # action_states shares the actions statistics (see module docstring).
    norm_stats["action_states"] = copy.deepcopy(norm_stats["actions"])

    # Drop anything that is not part of the model's input/output key set.
    norm_stats = {k: v for k, v in norm_stats.items() if k in REQUIRED_KEYS}

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"norm_stats": norm_stats}, indent=2))

    print(f"Wrote {dst}")
    for key in REQUIRED_KEYS:
        entry = norm_stats[key]
        for stat in ("mean", "std", "q01", "q99"):
            if stat in entry:
                print(f"  {key}.{stat}: dim={len(entry[stat])}")


if __name__ == "__main__":
    main()
