# GLIMPSE — LVLM Hallucination Mitigation Pipeline

Training-free pipeline unifying **VHR** (vision-aware head reinforcement, arXiv:2412.13949) and **LASER** (layer-adaptive localization + contrastive decoding, arXiv:2602.04304), with three novel components: Head-Gated Contrastive Attention, a difficulty router, and event-triggered token verification. See `PROPOSAL.md` for the full methodology, novelty claims, cost model, and experimental plan.

## Layout

```
PROPOSAL.md          research proposal (read this first)
glimpse/             pipeline package
  hooks.py           head capture / VHR scaling forward hooks
  metrics.py         VHD, T-VHD, VAQ, HGCA
  router.py          difficulty routing (EASY/MEDIUM/HARD)
  localize.py        crop box + counterfactual masking
  decoding.py        ETV selective contrastive decoding
  pipeline.py        Stage 0-4 orchestration
  models/            adapters (llava.py implemented; 3 stubs documented)
eval/                pope.py + profiler.py implemented; chair/mme/aokvqa/textvqa/refcoco stubs
scripts/             run_eval.py, calibrate_router.py
```

## Setup

```bash
pip install -r requirements.txt
python scripts/run_eval.py --model llava15 --bench pope --pope-json data/pope/coco_pope_adversarial.json --image-dir data/coco/val2014 --limit 100 --out results/pope_adv.json
```

## Implementation status / next steps

1. Validate `Llava15Adapter.build_ucp_batch` visual-position alignment (unit test: attention slice shapes).
2. Reproduce VHR numbers with router forced to EASY; reproduce LASER with router forced to HARD.
3. Implement the per-step T-VHD shadow stream for ETV (currently triggers every step = LASER-equivalent).
4. Add remaining adapters and eval harnesses (stubs document the work items).
5. `scripts/calibrate_router.py` after a Stage-0-only feature dump.
