"""TextVQA evaluation (validation split, VQA soft accuracy).

Data (https://textvqa.org/dataset/):
  questions: https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
  images:    https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
             (unzips to train_images/, shared by train and val)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm

from .vqa_utils import soft_score

PROMPT = "{q}\nAnswer the question using a single word or phrase."


@dataclass
class VqaMetrics:
    accuracy: float
    n_samples: int


def run(pipeline, questions_json: str, image_dir: str,
        limit: int | None = None, profiler=None,
        records: list | None = None) -> VqaMetrics:
    with open(questions_json) as f:
        items = json.load(f)["data"]
    if limit:
        items = items[:limit]

    total = n = 0
    for item in tqdm(items, desc="TextVQA", unit="q", dynamic_ncols=True):
        path = os.path.join(image_dir, f"{item['image_id']}.jpg")
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        q = PROMPT.format(q=item["question"])
        if profiler is not None:
            with profiler.track() as t:
                out = pipeline.run(img, q)
                rho = out.etv_stats.utilization if out.etv_stats else None
                t.done(n_tokens=len(out.text.split()), route=out.route.value,
                       etv_utilization=rho)
        else:
            out = pipeline.run(img, q)
        s = soft_score(out.text, item["answers"])
        total += s
        n += 1
        if records is not None:
            records.append({
                "image": f"{item['image_id']}.jpg",
                "question": item["question"],
                "gold_answers": item["answers"],
                "model_answer": out.text,
                "score": s,
                "route": out.route.value,
            })
    return VqaMetrics(accuracy=total / max(n, 1), n_samples=n)
