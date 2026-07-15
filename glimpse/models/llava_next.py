"""LLaVA-NeXT (llava-hf/llava-v1.6-vicuna-7b-hf) adapter.

Any-res: the image is encoded as a 336x336 GLOBAL view (576 tokens, 24x24)
followed by high-res tiles with newline separators. In both the original
LLaVA-NeXT code and the HF port, the global/base features come FIRST in the
packed visual sequence, so we compute HGCA localization on the first 576
visual tokens (24x24 grid over the full image). Cropping then re-feeds the
zoomed region, which any-res re-tiles at higher effective resolution.

Requires attn_implementation="eager" for attention maps.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

PROMPT = "USER: <image>\n{q} ASSISTANT:"
PROMPT_NO_IMAGE = "USER: {q} ASSISTANT:"
PROMPT_NO_QUERY = "USER: <image>\n ASSISTANT:"

BASE_GRID = (24, 24)  # global-view grid used for localization


class LlavaNextAdapter:
    grid = BASE_GRID

    def __init__(self, model_id: str = "llava-hf/llava-v1.6-vicuna-7b-hf",
                 device: str | None = None, dtype: torch.dtype | None = None):
        from .resolve import resolve
        model_id = resolve(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda"
                               else torch.float32)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, attn_implementation="eager",
            output_attentions=True).to(self.device)
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        cfg = self.model.language_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

    def attn_modules(self):
        return [l.self_attn for l in self.model.language_model.model.layers]

    def build_inputs(self, image: Image.Image | None, query: str) -> dict:
        if image is None:
            enc = self.processor.tokenizer(PROMPT_NO_IMAGE.format(q=query),
                                           return_tensors="pt")
        else:
            enc = self.processor(text=PROMPT.format(q=query), images=image,
                                 return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
        """Left-pad three variants to equal length. Variants 0 and 2 share the
        same image so their (variable-length) visual spans are identical and
        stay position-aligned; variant 1 (no image) is text-only."""
        enc_full = self.processor(text=PROMPT.format(q=query), images=image,
                                  return_tensors="pt")
        enc_no_image = self.processor.tokenizer(PROMPT_NO_IMAGE.format(q=query),
                                                return_tensors="pt")
        enc_no_query = self.processor(text=PROMPT_NO_QUERY, images=image,
                                      return_tensors="pt")
        encodings = [enc_full, enc_no_image, enc_no_query]
        max_len = max(e["input_ids"].shape[1] for e in encodings)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        ids_list, mask_list = [], []
        for e in encodings:
            ids, mask = e["input_ids"], e["attention_mask"]
            pad = max_len - ids.shape[1]
            if pad > 0:
                ids = F.pad(ids, (pad, 0), value=pad_id)
                mask = F.pad(mask, (pad, 0), value=0)
            ids_list.append(ids)
            mask_list.append(mask)

        # pixel_values: (n_tiles, 3, 336, 336); duplicate for variants 0 and 2,
        # zeros placeholder for the text-only variant (masked out by input_ids
        # containing no image token -- the model ignores unused pixels).
        pv = enc_full["pixel_values"]
        batch = {
            "input_ids": torch.cat(ids_list),
            "attention_mask": torch.cat(mask_list),
            "pixel_values": torch.cat([pv, torch.zeros_like(pv), pv]),
            "image_sizes": torch.cat([enc_full["image_sizes"],
                                      enc_full["image_sizes"],
                                      enc_no_query["image_sizes"]]),
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    def visual_token_slice(self, batch: dict) -> slice:
        """First 576 expanded image-token positions = 24x24 global view."""
        ids = batch["input_ids"][0]
        image_token = self.model.config.image_token_index
        pos = (ids == image_token).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            raise ValueError("no image tokens found; check processor version "
                             "(needs patch_size-aware expansion)")
        start = int(pos[0])
        n = BASE_GRID[0] * BASE_GRID[1]
        if pos.numel() < n:  # unexpanded single placeholder token
            raise ValueError("input_ids not expanded; upgrade transformers or "
                             "set processor.patch_size / vision_feature_select")
        return slice(start, start + n)

    def grid_hw(self, batch: dict) -> tuple[int, int]:
        return BASE_GRID

    def project_map(self, hgca: torch.Tensor, batch: dict) -> torch.Tensor:
        return hgca  # global view is grid-isomorphic

    def tvhd_proxy_fn(self, capture):
        return lambda: float("-inf") if capture is None else capture()
