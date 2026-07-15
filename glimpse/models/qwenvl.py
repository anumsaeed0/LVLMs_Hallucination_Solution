"""Qwen-VL adapter (Qwen/Qwen2-VL-7B-Instruct).

The LASER paper's "Qwen-VL" is the any-resolution generation (dynamic
resolution, variable-length visual tokens; Wang et al., 2024) -- i.e.
Qwen2-VL in HF. Visual tokens keep a deterministic spatial correspondence:
image_grid_thw gives (t, h, w) in 14px vision patches; after the 2x2 patch
merge the LLM sees (h//2) x (w//2) tokens between <|vision_start|> and
<|vision_end|>.

Notes:
  - Qwen2 LLM uses GQA; per-head capture still works on o_proj input
    (query heads: n_heads * head_dim).
  - Cropping is doubly beneficial here: it reduces the visual-token count,
    cutting prefill cost (LASER Sec. 5.2 observation).
  - For the classic Qwen/Qwen-VL-Chat (fixed 256 tokens, 16x16 grid,
    trust_remote_code) write a sibling adapter; only the prompt template,
    visual_token_slice, and grid change.
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .qwen_runtime import (
    Qwen2VLDecodeSession,
    build_qwen_mrope_positions,
    get_qwen_eos_token_ids,
    merged_grid_hw,
    visual_token_slice_for_row,
)

MERGE = 2  # spatial merge size


def _chat(processor, query: str, with_image: bool) -> str:
    content = ([{"type": "image"}] if with_image else []) + \
              [{"type": "text", "text": query}]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True)


class QwenVLAdapter:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
                 device: str | None = None, dtype: torch.dtype | None = None):
        from .resolve import resolve
        model_id = resolve(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda"
                               else torch.float32)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype,
            attn_implementation="eager").to(self.device)
        
        # in glimpse/models/qwenvl.py __init__, right after the model load:
        import os
        max_pixels = int(os.environ.get("GLIMPSE_QWEN_MAX_PIXELS", 512 * 28 * 28))
        min_pixels = int(os.environ.get("GLIMPSE_QWEN_MIN_PIXELS", 64 * 28 * 28))
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        
        # self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.image_token_id = cfg.image_token_id

    @property
    def eos_token_ids(self) -> set[int]:
        return get_qwen_eos_token_ids(self.model, self.tokenizer)

    def create_decode_session(self) -> Qwen2VLDecodeSession:
        """Fresh per-stream decoder; never share across positive/negative paths."""
        return Qwen2VLDecodeSession(self.model, self.tokenizer)

    def attn_modules(self):
        return [l.self_attn for l in self.model.model.layers]

    def build_inputs(self, image: Image.Image | None, query: str) -> dict:
        text = _chat(self.processor, query, with_image=image is not None)
        enc = self.processor(text=[text],
                             images=[image] if image is not None else None,
                             return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
        """Variable-length visual spans: variants 0 and 2 share the image, so
        their expanded <image_pad> runs are identical; LEFT-pad shorter
        variants so the last position is the prediction point everywhere.
        Position alignment of the visual span between variants 0 and 2 is
        enforced by also left-padding variant 2 to variant 0's visual start
        (the query only appears AFTER the vision span in the chat template,
        so the prefix [system + vision] is naturally identical)."""
        enc_full = self.build_inputs(image, query)
        enc_no_image = self.build_inputs(None, query)
        enc_no_query = self.build_inputs(image, "")
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
            ids_list.append(ids.cpu())
            mask_list.append(mask.cpu())

        batch = {
            "input_ids": torch.cat(ids_list),
            "attention_mask": torch.cat(mask_list),
            # same image twice -> identical grids; model matches pixels to
            # image_pad tokens per sample in batch order (samples 0 and 2)
            "pixel_values": torch.cat([enc_full["pixel_values"].cpu(),
                                       enc_no_query["pixel_values"].cpu()]),
            "image_grid_thw": torch.cat([enc_full["image_grid_thw"].cpu(),
                                         enc_no_query["image_grid_thw"].cpu()]),
        }
        position_ids, _ = build_qwen_mrope_positions(
            self.model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["image_grid_thw"],
        )
        batch["position_ids"] = position_ids
        if os.environ.get("GLIMPSE_QWEN_DEBUG", "").strip() in {"1", "true", "yes"}:
            import logging
            sl_full, sl_noq = self.visual_token_slices(batch)
            gh, gw = self.grid_hw(batch)
            logging.getLogger(__name__).info(
                "[qwen] ucp grid=%dx%d full_slice=%s no_query_slice=%s",
                gh, gw, sl_full, sl_noq,
            )
        return {k: v.to(self.device) for k, v in batch.items()}

    def visual_token_slice(self, batch: dict) -> slice:
        return visual_token_slice_for_row(
            batch["input_ids"][0], self.image_token_id)

    def visual_token_slices(self, batch: dict) -> tuple[slice, slice]:
        """FULL (0) and NO_QUERY (2) spans under independent left-padding."""
        ids = batch["input_ids"]
        return (
            visual_token_slice_for_row(ids[0], self.image_token_id),
            visual_token_slice_for_row(ids[2], self.image_token_id),
        )

    def grid_hw(self, batch: dict) -> tuple[int, int]:
        # Original-image UCP grid (first image row in the 2-image UCP tensor).
        return merged_grid_hw(batch["image_grid_thw"][0], merge_size=MERGE)

    def project_map(self, hgca: torch.Tensor, batch: dict) -> torch.Tensor:
        return hgca  # grid-isomorphic

    def tvhd_proxy_fn(self, capture):
        return lambda: float("-inf") if capture is None else capture()
