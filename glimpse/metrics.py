"""Counterfactual divergence metrics: VHD, T-VHD (VHR paper), VAQ (LASER paper),
and the novel Head-Gated Contrastive Attention (HGCA).

All metrics are computed from a single Unified Counterfactual Prefill (UCP):
a batch of three variants  [full (x_V,x_T), image-ablated (x_T), query-ablated (x_V)]
captured via glimpse.hooks.HeadCapture.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FULL, NO_IMAGE, NO_QUERY = 0, 1, 2  # batch indices in the UCP batch


# ---------------------------------------------------------------- VHD (heads)

def vhd_scores(head_outputs: dict[int, torch.Tensor]) -> torch.Tensor:
    """VHD_{l,i} = || A_{l,i}(full) - A_{l,i}(no_image) ||_2  at the last
    prefill position (first generation step).  VHR Eq. 4.

    head_outputs[l]: (3, 1, n_heads, head_dim) from UCP (last_token_only).
    Returns (L, n_heads).
    """
    layers = sorted(head_outputs)
    out = []
    for l in layers:
        h = head_outputs[l].float().nan_to_num(0.0)  # (3, 1, H, D)
        d = torch.linalg.vector_norm(h[FULL, 0] - h[NO_IMAGE, 0], dim=-1)  # (H,)
        out.append(d)
    return torch.stack(out)  # (L, H)


def ta_scores(head_outputs: dict[int, torch.Tensor]) -> torch.Tensor:
    """TA_{l,i} = || A_{l,i}(no_image) ||_2 -- used for outlier pruning (VHR Eq. 6)."""
    layers = sorted(head_outputs)
    return torch.stack([
        torch.linalg.vector_norm(
            head_outputs[l][NO_IMAGE, 0].float().nan_to_num(0.0), dim=-1)
        for l in layers
    ])


def prune_outliers(vhd: torch.Tensor, ta: torch.Tensor) -> torch.Tensor:
    """Zero out heads whose divergence stems from an activation surge upon
    image removal (negative vision sensitivity).  VHR Eq. 6."""
    vhd = vhd.clone()
    mu_v, sd_v = vhd.mean(dim=1, keepdim=True), vhd.std(dim=1, keepdim=True)
    mu_t, sd_t = ta.mean(dim=1, keepdim=True), ta.std(dim=1, keepdim=True)
    outlier = (vhd > mu_v + sd_v) & (ta > mu_t + sd_t)
    vhd[outlier] = 0.0
    return vhd


def select_vhr_heads(vhd: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    """H_l = heads above per-layer median VHD (VHR Eq. 7), restricted to
    the reinforced layers (typically the last 14)."""
    med = vhd.median(dim=1, keepdim=True).values
    mask = vhd > med
    return {l: mask[l] for l in layers}


def t_vhd(vhd: torch.Tensor, k: int = 8) -> float:
    """T-VHD = sum over layers of top-k VHD scores (VHR Eq. 5).
    Indicator of visual reliance vs language prior for the current token."""
    topk = vhd.topk(min(k, vhd.shape[1]), dim=1).values
    return topk.sum().item()


# ---------------------------------------------------------------- VAQ (layers)

@dataclass
class VaqResult:
    vaq_per_layer: torch.Tensor          # (L,)
    best_layer: int                      # argmax
    contrastive_maps: torch.Tensor       # (L, n_heads, P) ReLU(a^q - a^no_q)
    top_heads: torch.Tensor              # (L, K) indices of top-K heads per layer


def vaq_scores(attn_maps: dict[int, torch.Tensor],
               visual_slice: slice | tuple[slice, slice],
               k_head: int = 8) -> VaqResult:
    """Contrastive attention and VAQ (LASER Eqs. 3-5), computed at t=1.

    attn_maps[l]: (3, n_heads, T_q, T_kv) UCP-batch attention of last query
    position over all keys.  visual_slice selects visual-token key positions;
    pass a (slice_full, slice_no_query) tuple when the two variants' visual
    spans sit at different absolute positions (left-padding shifts them).
    """
    if not attn_maps:
        raise RuntimeError(
            "no attention maps captured during the UCP forward; ensure the "
            "model runs with output_attentions=True and "
            "attn_implementation='eager'")
    if isinstance(visual_slice, tuple):
        sl_full, sl_noq = visual_slice
    else:
        sl_full = sl_noq = visual_slice
    layers = sorted(attn_maps)
    maps, scores, tops = [], [], []
    for l in layers:
        # float32 + nan_to_num: left-padded UCP variants yield NaN rows in
        # fp16 eager attention; guard before any reduction
        row = attn_maps[l][:, :, -1, :].float().nan_to_num(0.0)  # (3, H, T_kv)
        a_full = row[FULL, :, sl_full]                    # (H, P)
        a_noq = row[NO_QUERY, :, sl_noq]                  # (H, P)
        con = torch.relu(a_full - a_noq)                  # (H, P)  Eq. 3
        s = torch.linalg.vector_norm(con, dim=-1)         # (H,)    Eq. 4
        k = min(k_head, s.shape[0])
        top = s.topk(k).indices
        maps.append(con)
        tops.append(top)
        scores.append(s[top].mean())                      # Eq. 5 (t=1 only)
    vaq = torch.stack(scores)
    return VaqResult(vaq, int(vaq.argmax()), torch.stack(maps), torch.stack(tops))


# ------------------------------------------------------- HGCA (novel, Stage 1)

def hgca_map(vaq: VaqResult, vhd_pruned: torch.Tensor, layer: int) -> torch.Tensor:
    """Head-Gated Contrastive Attention: average the contrastive map at
    `layer` over heads that are BOTH in LASER's top-K by VAQ AND pass the
    VHR gate (above-median VHD after outlier pruning).

    Returns a patch-level map (P,) normalized to a distribution.
    Falls back to the plain VAQ top-K average if the intersection is empty.
    """
    vhr_gate = vhd_pruned[layer] > vhd_pruned[layer].median()
    top = vaq.top_heads[layer]
    keep = top[vhr_gate[top]]
    if keep.numel() == 0:
        keep = top
    m = vaq.contrastive_maps[layer][keep].mean(dim=0).nan_to_num(0.0)  # (P,)
    total = m.sum()
    if total > 0:
        return m / total
    # degenerate map (no query-driven attention): uniform fallback
    return torch.full_like(m, 1.0 / max(m.numel(), 1))
