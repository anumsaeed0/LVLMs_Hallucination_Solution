"""CHAIR evaluation (Rohrbach et al., 2018) on MSCOCO val2014.

  CHAIR_I = hallucinated object mentions / all object mentions
  CHAIR_S = captions with >=1 hallucination / all captions
  Recall  = ground-truth objects mentioned / all ground-truth objects

Data needed:
  images:      http://images.cocodataset.org/zips/val2014.zip
  annotations: http://images.cocodataset.org/annotations/annotations_trainval2014.zip
               (instances_val2014.json for ground-truth object sets)

Protocol (matches VHR paper): 500 random images, prompt
"Please describe this image in detail.", max_new_tokens 512, 5 random splits.

NOTE: the synonym table below covers the standard 80 COCO categories with
the common synonyms from the official chair.py. For camera-ready numbers,
diff against the official implementation
(https://github.com/LisaAnne/Hallucination) on a small sample.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass

from PIL import Image
from tqdm import tqdm

PROMPT = "Please describe this image in detail."

# synonym -> canonical COCO category (compact port of official chair.py)
SYNONYMS: dict[str, str] = {}


def _add(cat: str, *syns: str) -> None:
    SYNONYMS[cat] = cat
    for s in syns:
        SYNONYMS[s] = cat


_add("person", "man", "woman", "men", "women", "people", "child", "children",
     "boy", "girl", "kid", "kids", "guy", "lady", "player", "rider", "human")
_add("bicycle", "bike", "bicycles", "bikes", "cyclist")
_add("car", "cars", "taxi", "sedan", "suv", "jeep", "van")
_add("motorcycle", "motorbike", "motorcycles", "moped", "scooter")
_add("airplane", "plane", "airplanes", "aircraft", "jet")
_add("bus", "buses")
_add("train", "trains", "locomotive", "railway")
_add("truck", "trucks", "pickup", "lorry")
_add("boat", "boats", "ship", "ships", "sailboat", "canoe", "kayak", "ferry")
_add("traffic light", "stoplight", "traffic signal")
_add("fire hydrant", "hydrant")
_add("stop sign")
_add("parking meter")
_add("bench", "benches")
_add("bird", "birds", "seagull", "pigeon", "duck", "ducks", "goose", "geese")
_add("cat", "cats", "kitten", "kitty")
_add("dog", "dogs", "puppy")
_add("horse", "horses", "pony", "foal")
_add("sheep", "lamb", "lambs", "ram")
_add("cow", "cows", "cattle", "bull", "calf", "ox")
_add("elephant", "elephants")
_add("bear", "bears", "panda")
_add("zebra", "zebras")
_add("giraffe", "giraffes")
_add("backpack", "backpacks", "knapsack", "rucksack")
_add("umbrella", "umbrellas", "parasol")
_add("handbag", "handbags", "purse", "purses")
_add("tie", "necktie", "bowtie")
_add("suitcase", "suitcases", "luggage", "briefcase")
_add("frisbee")
_add("skis", "ski")
_add("snowboard", "snowboards")
_add("sports ball", "ball", "balls", "baseball", "basketball", "soccer ball",
     "football", "volleyball", "tennis ball")
_add("kite", "kites")
_add("baseball bat", "bat")
_add("baseball glove", "glove", "mitt")
_add("skateboard", "skateboards")
_add("surfboard", "surfboards")
_add("tennis racket", "racket", "racquet")
_add("bottle", "bottles")
_add("wine glass", "wine glasses", "wineglass")
_add("cup", "cups", "mug", "mugs")
_add("fork", "forks")
_add("knife", "knives")
_add("spoon", "spoons")
_add("bowl", "bowls")
_add("banana", "bananas")
_add("apple", "apples")
_add("sandwich", "sandwiches", "burger", "hamburger", "cheeseburger")
_add("orange", "oranges")
_add("broccoli")
_add("carrot", "carrots")
_add("hot dog", "hotdog", "hot dogs")
_add("pizza", "pizzas")
_add("donut", "donuts", "doughnut", "doughnuts")
_add("cake", "cakes", "cupcake")
_add("chair", "chairs", "stool", "armchair", "seat", "seats")
_add("couch", "sofa", "couches", "loveseat")
_add("potted plant", "plant", "plants", "houseplant", "flower pot")
_add("bed", "beds", "mattress")
_add("dining table", "table", "tables", "desk", "desks")
_add("toilet")
_add("tv", "television", "monitor", "monitors", "screen", "tv screen")
_add("laptop", "laptops", "computer", "notebook computer")
_add("mouse", "computer mouse")
_add("remote", "remotes", "remote control")
_add("keyboard", "keyboards")
_add("cell phone", "phone", "phones", "cellphone", "mobile phone",
     "smartphone", "iphone")
_add("microwave")
_add("oven", "ovens", "stove")
_add("toaster")
_add("sink", "sinks")
_add("refrigerator", "fridge", "freezer")
_add("book", "books")
_add("clock", "clocks", "watch")
_add("vase", "vases")
_add("scissors")
_add("teddy bear", "teddy", "stuffed animal", "stuffed bear")
_add("hair drier", "hair dryer", "hairdryer", "blow dryer")
_add("toothbrush", "toothbrushes")

_MAX_NGRAM = 3


def extract_objects(caption: str) -> list[str]:
    """Return canonical COCO categories mentioned in the caption
    (greedy longest-match n-grams, each token consumed once)."""
    words = [w.strip(".,!?;:'\"()") for w in caption.lower().split()]
    found, i = [], 0
    while i < len(words):
        for n in range(_MAX_NGRAM, 0, -1):
            gram = " ".join(words[i:i + n])
            if gram in SYNONYMS:
                found.append(SYNONYMS[gram])
                i += n
                break
        else:
            i += 1
    return found


@dataclass
class ChairMetrics:
    chair_s: float
    chair_i: float
    recall: float
    avg_len: float
    n_images: int


def load_gt_objects(instances_json: str) -> dict[int, set[str]]:
    """image_id -> set of canonical category names present."""
    with open(instances_json) as f:
        data = json.load(f)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    gt: dict[int, set[str]] = {}
    for ann in data["annotations"]:
        gt.setdefault(ann["image_id"], set()).add(cats[ann["category_id"]])
    return gt


def score(captions: dict[int, str], gt: dict[int, set[str]]) -> ChairMetrics:
    n_hal_caps = n_mentions = n_hal_mentions = 0
    n_gt_total = n_gt_hit = length_sum = 0
    for img_id, cap in captions.items():
        objs = extract_objects(cap)
        gt_objs = gt.get(img_id, set())
        hal = [o for o in objs if o not in gt_objs]
        n_mentions += len(objs)
        n_hal_mentions += len(hal)
        n_hal_caps += bool(hal)
        n_gt_total += len(gt_objs)
        n_gt_hit += len(gt_objs & set(objs))
        length_sum += len(cap.split())
    n = max(len(captions), 1)
    return ChairMetrics(
        chair_s=n_hal_caps / n,
        chair_i=n_hal_mentions / max(n_mentions, 1),
        recall=n_gt_hit / max(n_gt_total, 1),
        avg_len=length_sum / n,
        n_images=len(captions),
    )


def run(pipeline, image_dir: str, instances_json: str, n_images: int = 500,
        seed: int = 42, profiler=None,
        records: list | None = None) -> ChairMetrics:
    gt = load_gt_objects(instances_json)
    rng = random.Random(seed)
    img_ids = rng.sample(sorted(gt), n_images)
    captions: dict[int, str] = {}
    for img_id in tqdm(img_ids, desc="CHAIR", unit="img", dynamic_ncols=True):
        fname = f"COCO_val2014_{img_id:012d}.jpg"
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        if profiler is not None:
            with profiler.track() as t:
                out = pipeline.run(img, PROMPT)
                rho = out.etv_stats.utilization if out.etv_stats else None
                t.done(n_tokens=len(out.text.split()), route=out.route.value,
                       etv_utilization=rho)
        else:
            out = pipeline.run(img, PROMPT)
        captions[img_id] = out.text
        if records is not None:
            mentioned = extract_objects(out.text)
            gt_objs = gt.get(img_id, set())
            records.append({
                "image": fname,
                "question": PROMPT,
                "model_answer": out.text,
                "gt_objects": sorted(gt_objs),
                "mentioned_objects": mentioned,
                "hallucinated_objects": [o for o in mentioned
                                         if o not in gt_objs],
                "route": out.route.value,
            })
    return score(captions, gt)
