"""HGCA map -> constrained crop box (I+) and counterfactual masked image (I-).

Follows LASER's Con-ViCrop constraints: crop = half the original size,
lower-bounded at 224x224 (CLIP receptive-field floor), centered on the
attention mass of the HGCA map.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class CropResult:
    image_pos: Image.Image     # I+  zoomed crop
    image_neg: Image.Image | None  # I- evidence-masked (HARD route only)
    box: tuple[int, int, int, int]
    crop_mass: float           # HGCA mass inside box (router feature d4)


def _to_grid(hgca: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    h, w = grid_hw
    assert hgca.numel() == h * w, f"map size {hgca.numel()} != grid {h}x{w}"
    return hgca.view(h, w)


def crop_box_from_map(hgca: torch.Tensor, grid_hw: tuple[int, int],
                      image_size: tuple[int, int],
                      min_side: int = 224) -> tuple[tuple[int, int, int, int], float]:
    """Center a half-size (>=224px) box on the attention centroid."""
    W, H = image_size
    g = _to_grid(hgca, grid_hw)
    gh, gw = g.shape
    ys = torch.arange(gh, dtype=torch.float32)
    xs = torch.arange(gw, dtype=torch.float32)
    total = g.sum().clamp_min(1e-9)
    cy = float((g.sum(dim=1) * ys).sum() / total) / gh * H
    cx = float((g.sum(dim=0) * xs).sum() / total) / gw * W

    bw = max(W // 2, min_side)
    bh = max(H // 2, min_side)
    x0 = int(min(max(cx - bw / 2, 0), max(W - bw, 0)))
    y0 = int(min(max(cy - bh / 2, 0), max(H - bh, 0)))
    box = (x0, y0, min(x0 + bw, W), min(y0 + bh, H))

    # fraction of attention mass inside the box (router feature d4)
    px0 = int(box[0] / W * gw); px1 = max(int(box[2] / W * gw), px0 + 1)
    py0 = int(box[1] / H * gh); py1 = max(int(box[3] / H * gh), py0 + 1)
    mass = float(g[py0:py1, px0:px1].sum() / total)
    return box, mass


def mask_top_patches(image: Image.Image, hgca: torch.Tensor,
                     grid_hw: tuple[int, int], k_patch: int = 32) -> Image.Image:
    """Counterfactual I-: gray out the top-K most query-relevant patches
    (LASER Eq. 7) on the ORIGINAL image, then the caller applies the same
    crop transform."""
    W, H = image.size
    gh, gw = grid_hw
    pw, ph = W / gw, H / gh
    out = image.copy()
    top = hgca.topk(min(k_patch, hgca.numel())).indices
    from PIL import ImageDraw
    draw = ImageDraw.Draw(out)
    for idx in top.tolist():
        r, c = divmod(idx, gw)
        draw.rectangle([c * pw, r * ph, (c + 1) * pw, (r + 1) * ph],
                       fill=(127, 127, 127))
    return out


def localize(image: Image.Image, hgca: torch.Tensor, grid_hw: tuple[int, int],
             make_counterfactual: bool = False, k_patch: int = 32) -> CropResult:
    # tiny map; do all geometry on CPU to avoid device mismatches
    hgca = hgca.detach().float().cpu()
    box, mass = crop_box_from_map(hgca, grid_hw, image.size)
    image_pos = image.crop(box)
    image_neg = None
    if make_counterfactual:
        image_neg = mask_top_patches(image, hgca, grid_hw, k_patch).crop(box)
    return CropResult(image_pos, image_neg, box, mass)
