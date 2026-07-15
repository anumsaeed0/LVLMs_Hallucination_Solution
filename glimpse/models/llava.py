"""LLaVA-1.5 adapter: fixed 336x336 -> 24x24 grid -> 576 visual tokens.

Requires attn_implementation="eager" so attention maps are available.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

PROMPT = "USER: <image>\n{q} ASSISTANT:"
PROMPT_NO_IMAGE = "USER: {q} ASSISTANT:"
PROMPT_NO_QUERY = "USER: <image>\n ASSISTANT:"


class Llava15Adapter:
    grid = (24, 24)

    def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf",
                 device: str | None = None, dtype: torch.dtype | None = None):
        from .resolve import resolve
        model_id = resolve(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda"
                               else torch.float32)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype, attn_implementation="eager",
            output_attentions=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        lm = self.model.language_model
        cfg = lm.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

    def attn_modules(self):
        return [l.self_attn for l in self.model.language_model.model.layers]

    def build_inputs(self, image: Image.Image | None, query: str) -> dict:
        if image is None:
            text = PROMPT_NO_IMAGE.format(q=query)
            enc = self.processor(text=text, return_tensors="pt")
        else:
            enc = self.processor(text=PROMPT.format(q=query), images=image,
                                 return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
        """Three variants left-padded to equal length so the LAST position is
        the prediction point for all. Visual tokens of variants 0 and 2 are
        kept at identical absolute positions (image prefix first)."""
        self.processor.tokenizer.padding_side = "left"
        enc_full = self.processor(text=PROMPT.format(q=query), images=image,
                                  return_tensors="pt")
        enc_no_image = self.processor(text=PROMPT_NO_IMAGE.format(q=query),
                                      return_tensors="pt")
        enc_no_query = self.processor(text=PROMPT_NO_QUERY, images=image,
                                      return_tensors="pt")

        encodings = [enc_full, enc_no_image, enc_no_query]
        max_len = max(enc["input_ids"].shape[1] for enc in encodings)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        for enc in encodings:
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            pad = max_len - ids.shape[1]
            if pad > 0:
                ids = F.pad(ids, (pad, 0), value=pad_id)
                mask = F.pad(mask, (pad, 0), value=0)
            input_ids.append(ids)
            attention_mask.append(mask)

        ref_pixels = enc_full.get("pixel_values")
        if ref_pixels is None:
            raise ValueError("LLaVA processor did not return pixel_values")

        pixel_values = []
        for enc in encodings:
            pixels = enc.get("pixel_values")
            if pixels is None:
                pixels = torch.zeros_like(ref_pixels)
            pixel_values.append(pixels)

        batch = {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    def _img_slice(self, ids: torch.Tensor) -> slice:
        image_token = self.model.config.image_token_index
        pos = (ids == image_token).nonzero(as_tuple=True)[0]
        start = int(pos[0])
        return slice(start, start + self.grid[0] * self.grid[1])

    def visual_token_slices(self, batch: dict) -> tuple[slice, slice]:
        """Per-variant visual spans: with left-padding, the NO_QUERY variant's
        image span is shifted right relative to FULL. Using one slice for both
        misaligns the contrastive subtraction (degrades HGCA to raw attention)."""
        ids = batch["input_ids"]
        return self._img_slice(ids[0]), self._img_slice(ids[2])

    # backward-compat: single slice of the FULL variant
    def visual_token_slice(self, batch: dict) -> slice:
        return self._img_slice(batch["input_ids"][0])

    def grid_hw(self, batch: dict) -> tuple[int, int]:
        return self.grid

    def project_map(self, hgca: torch.Tensor, batch: dict) -> torch.Tensor:
        return hgca  # grid-isomorphic: visual tokens ARE the patch grid

    def tvhd_proxy_fn(self, capture):
        """Per-step T-VHD proxy for ETV. Full implementation maintains a
        cheap image-ablated shadow stream (see PROPOSAL.md Stage 3);
        this stub triggers verification on every step (LASER-equivalent,
        exact but not yet cost-reduced)."""
        return lambda: float("-inf") if capture is None else capture()


# """LLaVA-1.5 adapter: fixed 336x336 -> 24x24 grid -> 576 visual tokens.

# Requires attn_implementation="eager" so attention maps are available.
# """
# from __future__ import annotations

# import torch
# from PIL import Image
# from transformers import AutoProcessor, LlavaForConditionalGeneration

# PROMPT = "USER: <image>\n{q} ASSISTANT:"
# PROMPT_NO_IMAGE = "USER: {q} ASSISTANT:"
# PROMPT_NO_QUERY = "USER: <image>\n ASSISTANT:"


# class Llava15Adapter:
#     grid = (24, 24)

#     # def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf",
#     #              device: str = "cuda", dtype=torch.float16):
#     #     self.model = LlavaForConditionalGeneration.from_pretrained(
#     #         model_id, torch_dtype=dtype, attn_implementation="eager",
#     #         output_attentions=True).to(device)
#     #     self.processor = AutoProcessor.from_pretrained(model_id)
#     #     self.tokenizer = self.processor.tokenizer
#     #     lm = self.model.language_model
#     #     cfg = lm.config
#     #     self.n_layers = cfg.num_hidden_layers
#     #     self.n_heads = cfg.num_attention_heads
#     #     self.head_dim = cfg.hidden_size // cfg.num_attention_heads
#     #     self.device = device
#     def __init__(self, model_id: str = "llava-hf/llava-1.5-7b-hf",
#                  device: str | None = None, dtype: torch.dtype | None = None):
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = dtype or (torch.float16 if self.device == "cuda"
#                                else torch.float32)
#         self.model = LlavaForConditionalGeneration.from_pretrained(
#             model_id, torch_dtype=self.dtype, attn_implementation="eager",
#             output_attentions=True).to(self.device)
#         self.processor = AutoProcessor.from_pretrained(model_id)
#         self.tokenizer = self.processor.tokenizer
#         lm = self.model.language_model
#         cfg = lm.config
#         self.n_layers = cfg.num_hidden_layers
#         self.n_heads = cfg.num_attention_heads
#         self.head_dim = cfg.hidden_size // cfg.num_attention_heads

#     def attn_modules(self):
#         return [l.self_attn for l in self.model.language_model.model.layers]

#     def build_inputs(self, image: Image.Image | None, query: str) -> dict:
#         if image is None:
#             text = PROMPT_NO_IMAGE.format(q=query)
#             enc = self.processor(text=text, return_tensors="pt")
#         else:
#             enc = self.processor(text=PROMPT.format(q=query), images=image,
#                                  return_tensors="pt")
#         return {k: v.to(self.device) for k, v in enc.items()}

#     def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
#         """Three variants left-padded to equal length so the LAST position is
#         the prediction point for all. Visual tokens of variants 0 and 2 are
#         kept at identical absolute positions (image prefix first)."""
#         texts = [PROMPT.format(q=query), PROMPT_NO_IMAGE.format(q=query),
#                  PROMPT_NO_QUERY]
#         self.processor.tokenizer.padding_side = "left"
#         enc = self.processor(text=texts, images=[image, None, image],
#                              padding=True, return_tensors="pt")
#         return {k: v.to(self.device) for k, v in enc.items()}

#     def visual_token_slice(self, batch: dict) -> slice:
#         ids = batch["input_ids"][0]
#         image_token = self.model.config.image_token_index
#         pos = (ids == image_token).nonzero(as_tuple=True)[0]
#         # after embedding expansion, 576 visual tokens replace the placeholder
#         start = int(pos[0])
#         return slice(start, start + self.grid[0] * self.grid[1])

#     def grid_hw(self, batch: dict) -> tuple[int, int]:
#         return self.grid

#     def tvhd_proxy_fn(self, capture):
#         """Per-step T-VHD proxy for ETV. Full implementation maintains a
#         cheap image-ablated shadow stream (see PROPOSAL.md Stage 3);
#         this stub triggers verification on every step (LASER-equivalent,
#         exact but not yet cost-reduced)."""
#         return lambda: float("-inf") if capture is None else capture()
