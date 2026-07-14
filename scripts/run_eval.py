#!/usr/bin/env python
"""Run a benchmark with GLIMPSE.

Examples:
  # POPE
  python scripts/run_eval.py --model llava15 --bench pope \
      --pope-json data/pope/coco_pope_adversarial.json \
      --image-dir data/coco/val2014

  # CHAIR (500 imgs, MSCOCO val2014)
  python scripts/run_eval.py --model llava_next --bench chair \
      --image-dir data/coco/val2014 \
      --instances-json data/coco/annotations/instances_val2014.json

  # A-OKVQA (auto-downloads from HF hub)
  python scripts/run_eval.py --model qwenvl --bench aokvqa --limit 500

  # TextVQA
  python scripts/run_eval.py --model instructblip --bench textvqa \
      --questions-json data/textvqa/TextVQA_0.5.1_val.json \
      --image-dir data/textvqa/train_images
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.profiler import Profiler  # noqa: E402
from glimpse import GlimpseConfig, GlimpsePipeline  # noqa: E402

MODELS = {
    "llava15": ("glimpse.models", "Llava15Adapter", {}),
    "llava_next": ("glimpse.models", "LlavaNextAdapter", {}),
    "instructblip": ("glimpse.models", "InstructBlipAdapter",
                     {"reinforced_last_n": 18}),  # VHR paper setting
    "qwenvl": ("glimpse.models", "QwenVLAdapter", {}),
}


def build(name: str, alpha_vhr: float, model_path: str | None = None,
          force_route: str | None = None):
    import importlib
    module, cls, cfg_over = MODELS[name]
    adapter_cls = getattr(importlib.import_module(module), cls)
    adapter = adapter_cls(model_id=model_path) if model_path else adapter_cls()
    print(adapter.device)
    print(next(adapter.model.parameters()).device)
    cfg = GlimpseConfig(alpha_vhr=alpha_vhr, force_route=force_route,
                        **cfg_over)
    return GlimpsePipeline(adapter, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava15", choices=sorted(MODELS))
    ap.add_argument("--bench", default="pope",
                    choices=["pope", "chair", "aokvqa", "textvqa"])
    ap.add_argument("--image-dir", help="pope/chair: COCO val2014; "
                                        "textvqa: train_images")
    ap.add_argument("--pope-json", help="pope: POPE annotation jsonl")
    ap.add_argument("--instances-json", help="chair: instances_val2014.json")
    ap.add_argument("--questions-json", help="textvqa: TextVQA_0.5.1_val.json")
    ap.add_argument("--n-images", type=int, default=500, help="chair sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--alpha-vhr", type=float, default=2.0)
    ap.add_argument("--model-path", default=None,
                    help="local model directory (falls back to HF hub "
                         "download if omitted/not found); also checks "
                         "GLIMPSE_MODELS_DIR and ./models/ automatically")
    ap.add_argument("--force-route", default=None,
                    choices=["easy", "medium", "hard"],
                    help="bypass the difficulty router (ablations: easy=VHR "
                         "only, hard=full LASER-style path)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pipeline = build(args.model, args.alpha_vhr, model_path=args.model_path,
                     force_route=args.force_route)
    prof = Profiler()
    records: list = []

    if args.bench == "pope":
        from eval import pope
        m = pope.run(pipeline, args.pope_json, args.image_dir,
                     profiler=prof, limit=args.limit, records=records)
    elif args.bench == "chair":
        from eval import chair
        m = chair.run(pipeline, args.image_dir, args.instances_json,
                      n_images=args.n_images, seed=args.seed, profiler=prof,
                      records=records)
    elif args.bench == "aokvqa":
        from eval import aokvqa
        m = aokvqa.run(pipeline, limit=args.limit, profiler=prof,
                       records=records)
    elif args.bench == "textvqa":
        from eval import textvqa
        m = textvqa.run(pipeline, args.questions_json, args.image_dir,
                        limit=args.limit, profiler=prof, records=records)

    out = args.out or f"results/{args.bench}_{args.model}.json"
    preds_out = str(Path(out).with_name(Path(out).stem + "_predictions.json"))
    payload = {"model": args.model, "bench": args.bench,
               "metrics": m.__dict__, "efficiency": prof.summary(),
               "predictions_file": preds_out}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    with open(preds_out, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2))
    print(f"[predictions] {len(records)} records -> {preds_out}")


if __name__ == "__main__":
    main()
