"""POPE evaluation: Accuracy / Precision / Recall / F1 over yes-no answers.

Expects the standard POPE json files (random/popular/adversarial) and a
COCO val2014 image directory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PIL import Image


@dataclass
class PopeMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_ratio: float


def parse_answer(text: str) -> str:
    t = text.strip().lower()
    return "yes" if t.startswith("yes") or " yes" in t[:16] else "no"


def score(preds: list[str], golds: list[str]) -> PopeMetrics:
    tp = sum(p == "yes" and g == "yes" for p, g in zip(preds, golds))
    fp = sum(p == "yes" and g == "no" for p, g in zip(preds, golds))
    fn = sum(p == "no" and g == "yes" for p, g in zip(preds, golds))
    tn = sum(p == "no" and g == "no" for p, g in zip(preds, golds))
    n = max(len(preds), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return PopeMetrics(
        accuracy=(tp + tn) / n,
        precision=prec,
        recall=rec,
        f1=2 * prec * rec / max(prec + rec, 1e-9),
        yes_ratio=(tp + fp) / n,
    )


def run(pipeline, pope_json: str, image_dir: str, profiler=None,
        limit: int | None = None) -> PopeMetrics:
    preds, golds = [], []
    with open(pope_json) as f:
        items = [json.loads(l) for l in f]
    for item in items[:limit]:
        img = Image.open(os.path.join(image_dir, item["image"])).convert("RGB")
        if profiler is not None:
            with profiler.track() as t:
                out = pipeline.run(img, item["text"])
                rho = out.etv_stats.utilization if out.etv_stats else None
                t.done(n_tokens=len(out.text.split()), route=out.route.value,
                       etv_utilization=rho)
        else:
            out = pipeline.run(img, item["text"])
        preds.append(parse_answer(out.text))
        golds.append(item["label"].strip().lower())
    return score(preds, golds)
