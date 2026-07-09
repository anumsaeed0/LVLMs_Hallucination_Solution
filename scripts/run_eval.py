#!/usr/bin/env python
"""Run a benchmark with GLIMPSE.

Example:
  python scripts/run_eval.py --model llava15 --bench pope \
      --pope-json data/pope/coco_pope_adversarial.json \
      --image-dir data/coco/val2014 --out results/pope_adv.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.profiler import Profiler  # noqa: E402
from glimpse import GlimpseConfig, GlimpsePipeline  # noqa: E402


def build_adapter(name: str):
    if name == "llava15":
        from glimpse.models import Llava15Adapter
        return Llava15Adapter()
    raise NotImplementedError(
        f"adapter '{name}' not implemented yet (see glimpse/models/__init__.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava15",
                    choices=["llava15", "llava_next", "instructblip", "qwenvl"])
    ap.add_argument("--bench", default="pope", choices=["pope"])
    ap.add_argument("--pope-json")
    ap.add_argument("--image-dir")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha-vhr", type=float, default=2.0)
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    pipeline = GlimpsePipeline(build_adapter(args.model),
                               GlimpseConfig(alpha_vhr=args.alpha_vhr))
    prof = Profiler()

    if args.bench == "pope":
        from eval import pope
        m = pope.run(pipeline, args.pope_json, args.image_dir,
                     profiler=prof, limit=args.limit)
        result = m.__dict__

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"metrics": result, "efficiency": prof.summary()}, f, indent=2)
    print(json.dumps({"metrics": result, "efficiency": prof.summary()}, indent=2))


if __name__ == "__main__":
    main()
