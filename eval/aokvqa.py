"""A-OKVQA evaluation, Direct Answer setting (as in LASER Table 1).

Loads HuggingFaceM4/A-OKVQA from the HF hub (images embedded in the
dataset, no separate download). Score = VQA soft accuracy against the
10 direct_answers.
"""
from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from .vqa_utils import soft_score

PROMPT = "{q}\nAnswer the question using a single word or phrase."


@dataclass
class VqaMetrics:
    accuracy: float
    n_samples: int


def run(pipeline, split: str = "validation", limit: int | None = None,
        profiler=None, records: list | None = None) -> VqaMetrics:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    total = 0.0
    for item in tqdm(ds, desc="A-OKVQA", unit="q", dynamic_ncols=True):
        img = item["image"].convert("RGB")
        answers = item["direct_answers"]
        if isinstance(answers, str):        # some mirrors store a repr string
            import ast
            answers = ast.literal_eval(answers)
        q = PROMPT.format(q=item["question"])
        if profiler is not None:
            with profiler.track() as t:
                out = pipeline.run(img, q)
                rho = out.etv_stats.utilization if out.etv_stats else None
                t.done(n_tokens=len(out.text.split()), route=out.route.value,
                       etv_utilization=rho)
        else:
            out = pipeline.run(img, q)
        s = soft_score(out.text, answers)
        total += s
        if records is not None:
            records.append({
                "question_id": item.get("question_id"),
                "question": item["question"],
                "gold_answers": answers,
                "model_answer": out.text,
                "score": s,
                "route": out.route.value,
            })
    n = len(ds)
    return VqaMetrics(accuracy=total / max(n, 1), n_samples=n)
