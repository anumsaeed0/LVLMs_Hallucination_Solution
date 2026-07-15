"""GLIMPSE orchestration: Stage 0 (UCP) -> 1 (HGCA) -> 2 (Router) -> 3/4 (decode).

The ModelAdapter protocol isolates architecture differences (LLaVA grid,
Qwen-VL any-res, InstructBLIP Q-Former back-projection). See models/.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from PIL import Image

from . import metrics
from .decoding import EtvConfig, EtvStats, etv_generate, hf_generate
from .hooks import HeadCapture, HeadScaler
from .localize import localize
from .router import Route, RouterThresholds, decide, features


class ModelAdapter(Protocol):
    model: object
    tokenizer: object
    n_layers: int
    n_heads: int
    head_dim: int

    def attn_modules(self) -> list:  ...
    def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
        """Batch of [full, no-image, no-query] with aligned visual positions."""
    def visual_token_slice(self, batch: dict) -> slice:  ...
    def grid_hw(self, batch: dict) -> tuple[int, int]:  ...
    def build_inputs(self, image: Image.Image | None, query: str) -> dict:  ...
    def tvhd_proxy_fn(self, capture: HeadCapture):  ...
    def project_map(self, hgca: "torch.Tensor", batch: dict) -> "torch.Tensor":
        """Map the HGCA map from LLM visual-token space to the image patch
        grid. Identity for grid-isomorphic models (LLaVA, Qwen); Q-Former
        back-projection for InstructBLIP (PROPOSAL.md novelty #5)."""


@dataclass
class GlimpseConfig:
    alpha_vhr: float = 2.0
    reinforced_last_n: int = 14          # 18 for InstructBLIP
    k_head: int = 8
    k_patch: int = 32
    force_route: str | None = None       # "easy"/"medium"/"hard" ablations
    etv: EtvConfig = None
    router: RouterThresholds = None

    def __post_init__(self):
        self.etv = self.etv or EtvConfig()
        self.router = self.router or RouterThresholds()


@dataclass
class GlimpseOutput:
    text: str
    route: Route
    best_layer: int
    tvhd_first: float
    etv_stats: EtvStats | None
    crop_box: tuple | None


class GlimpsePipeline:
    def __init__(self, adapter: ModelAdapter, cfg: GlimpseConfig):
        self.a = adapter
        self.cfg = cfg
        self.scaler = HeadScaler(adapter.attn_modules(), adapter.n_heads,
                                 adapter.head_dim, alpha=cfg.alpha_vhr)
        self.scaler.attach()
        self.scaler.enabled = False

    @torch.no_grad()
    def run(self, image: Image.Image, query: str) -> GlimpseOutput:
        cfg, a = self.cfg, self.a
        import os
        import time
        timing = os.environ.get("GLIMPSE_TIMING") == "1"

        def _tick():
            if timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        t0 = _tick()

        # ---- Stage 0: Unified Counterfactual Prefill (batched, hooks on) ----
        ucp = a.build_ucp_batch(image, query)
        with HeadCapture(a.attn_modules(), a.n_heads, a.head_dim) as cap:
            with self.scaler.disabled():
                # output_attentions only for this probe pass (decode stays
                # fast); required so attention hooks see real weights
                a.model(**ucp, output_attentions=True)
            vhd_raw = metrics.vhd_scores(cap.state.head_outputs)
            ta = metrics.ta_scores(cap.state.head_outputs)
            vhd = metrics.prune_outliers(vhd_raw, ta)
            # prefer per-variant slices (left-padding shifts the NO_QUERY span)
            if hasattr(a, "visual_token_slices"):
                vslice = a.visual_token_slices(ucp)
            else:
                vslice = a.visual_token_slice(ucp)
            vaq = metrics.vaq_scores(cap.state.attn_maps, vslice, cfg.k_head)

        t_ucp = _tick()
        reinforced = list(range(a.n_layers - cfg.reinforced_last_n, a.n_layers))
        self.scaler.set_heads(metrics.select_vhr_heads(vhd, reinforced))
        tvhd_first = metrics.t_vhd(vhd, cfg.k_head)

        # ---- Stage 1: HGCA localization map (projected to patch grid) ----
        hgca = a.project_map(metrics.hgca_map(vaq, vhd, vaq.best_layer), ucp)
        crop = localize(image, hgca, a.grid_hw(ucp), make_counterfactual=False,
                        k_patch=cfg.k_patch)

        # ---- Stage 2: route by difficulty ----
        f = features(vaq.vaq_per_layer, vaq.best_layer, tvhd_first, crop.crop_mass)
        route = Route(cfg.force_route) if cfg.force_route else decide(f, cfg.router)

        # ---- Stage 3/4: route-dependent decoding (VHR active throughout) ----
        self.scaler.enabled = True
        try:
            if route is Route.EASY:
                pos = a.build_inputs(image, query)
                neg = None
                box = None
            else:
                make_cf = route is Route.HARD
                crop = localize(image, hgca, a.grid_hw(ucp),
                                make_counterfactual=make_cf, k_patch=cfg.k_patch)
                pos = a.build_inputs(crop.image_pos, query)
                neg = a.build_inputs(crop.image_neg, query) if make_cf else None
                box = crop.box

            t_pre = _tick()

            # text, stats = etv_generate(
            #     getattr(a, "decode_model", a.model), pos, neg,
            #     tvhd_fn=a.tvhd_proxy_fn(None),  # adapter-provided per-step proxy
            #     cfg=cfg.etv, tokenizer=a.tokenizer)

            if neg is None:
                # EASY/MEDIUM: no VAT contrast -> use the model's own
                # .generate() (correct M-RoPE/cache handling guaranteed;
                # VHR hooks still fire inside it)
                text, stats = hf_generate(
                    a.model, pos, a.tokenizer,
                    max_new_tokens=cfg.etv.max_new_tokens,
                    eos_token_ids=getattr(a, "eos_token_ids", None))
            else:
                # HARD: ETV needs per-token logits from two streams
                text, stats = etv_generate(
                    getattr(a, "decode_model", a.model), pos, neg,
                    tvhd_fn=a.tvhd_proxy_fn(None),
                    cfg=cfg.etv, tokenizer=a.tokenizer,
                    create_decode_session=getattr(a, "create_decode_session",
                                                  None),
                    eos_token_ids=getattr(a, "eos_token_ids", None))
        finally:
            self.scaler.enabled = False

        if timing:
            t_end = _tick()
            n_tok = stats.n_tokens if stats else 0
            print(f"[timing] ucp={t_ucp - t0:.2f}s "
                  f"loc/crop={t_pre - t_ucp:.2f}s "
                  f"decode={t_end - t_pre:.2f}s ({n_tok} tok) "
                  f"route={route.value} "
                  f"vram={torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
                  if torch.cuda.is_available() else "")

        return GlimpseOutput(text=text, route=route, best_layer=vaq.best_layer,
                             tvhd_first=tvhd_first,
                             etv_stats=stats if neg is not None else None,
                             crop_box=box)
