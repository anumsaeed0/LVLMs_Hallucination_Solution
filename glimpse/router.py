"""Stage 2: training-free difficulty router.

Features (all free byproducts of the Unified Counterfactual Prefill):
  d1: normalized depth of argmax-VAQ layer      (late peak  -> harder)
  d2: entropy of the layer-wise VAQ profile     (flat       -> harder)
  d3: T-VHD of the first answer token           (low        -> language-prior risk)
  d4: HGCA mass inside the best crop box        (low        -> cropping valuable)

Thresholds are per-model PERCENTILES calibrated once on ~200 held-out
samples (scripts/calibrate_router.py); no training, no labels needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class Route(str, Enum):
    EASY = "easy"       # VHR only
    MEDIUM = "medium"   # + HGCA crop re-prefill
    HARD = "hard"       # + counterfactual masking + ETV decoding


@dataclass
class RouterThresholds:
    """Percentile-based thresholds; values below are LLaVA-1.5 placeholders
    to be replaced by calibrate_router.py output."""
    d1_late: float = 0.55       # l*/L above this counts as "late peak"
    d2_flat: float = 0.80       # normalized entropy above this = diffuse
    d3_low: float = 0.0         # 20th-percentile T-VHD from calibration
    d4_low: float = 0.35        # crop-mass below this = dispersed evidence


@dataclass
class RouterFeatures:
    d1_depth: float
    d2_entropy: float
    d3_tvhd: float
    d4_crop_mass: float


def features(vaq_per_layer: torch.Tensor, best_layer: int, tvhd: float,
             crop_mass: float) -> RouterFeatures:
    L = vaq_per_layer.shape[0]
    p = vaq_per_layer.clamp_min(1e-9)
    p = p / p.sum()
    ent = float(-(p * p.log()).sum() / torch.log(torch.tensor(float(L))))
    return RouterFeatures(
        d1_depth=best_layer / (L - 1),
        d2_entropy=ent,
        d3_tvhd=tvhd,
        d4_crop_mass=crop_mass,
    )


def decide(f: RouterFeatures, th: RouterThresholds) -> Route:
    late_or_flat = (f.d1_depth > th.d1_late) or (f.d2_entropy > th.d2_flat)
    prior_risk = f.d3_tvhd < th.d3_low
    crop_useful = f.d4_crop_mass < th.d4_low

    if prior_risk and late_or_flat:
        return Route.HARD
    if late_or_flat or crop_useful:
        return Route.MEDIUM
    return Route.EASY
