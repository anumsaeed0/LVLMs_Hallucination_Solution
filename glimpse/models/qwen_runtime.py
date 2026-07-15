"""Qwen2-VL runtime helpers: M-RoPE positions, decode sessions, visual slices.

Targets the Transformers 4.45 API (``get_rope_index`` positional args,
``prepare_inputs_for_generation`` incremental positions).  Each
:class:`Qwen2VLDecodeSession` owns its own KV-cache and generation kwargs so
positive and counterfactual streams never share state.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("GLIMPSE_QWEN_DEBUG", "").strip() in {"1", "true", "yes"}

if _DEBUG:
    logging.basicConfig(level=logging.INFO)


def _dbg(msg: str, *args) -> None:
    if _DEBUG:
        logger.info("[qwen] " + msg, *args)


def _normalize_eos_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return {int(v) for v in value}
    return set()


def get_qwen_eos_token_ids(model, tokenizer) -> set[int]:
    """Collect all configured EOS ids (int or list) without hard-coding."""
    ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        ids |= _normalize_eos_ids(getattr(gen_cfg, "eos_token_id", None))
    cfg = getattr(model, "config", None)
    if cfg is not None:
        ids |= _normalize_eos_ids(getattr(cfg, "eos_token_id", None))
    if tokenizer is not None:
        ids |= _normalize_eos_ids(getattr(tokenizer, "eos_token_id", None))
    ids.discard(None)  # type: ignore[arg-type]
    return ids


def merged_grid_hw(image_grid_thw, merge_size: int = 2) -> tuple[int, int]:
    """LLM patch grid (h, w) after spatial merge from one ``image_grid_thw`` row."""
    if torch.is_tensor(image_grid_thw):
        row = image_grid_thw[0] if image_grid_thw.dim() > 1 else image_grid_thw
        _, h, w = row.tolist()
    else:
        _, h, w = image_grid_thw
    return (h // merge_size, w // merge_size)


def build_qwen_mrope_positions(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute M-RoPE ``position_ids`` and ``rope_deltas`` for a batch.

    Matches Transformers 4.45 ``Qwen2VLForConditionalGeneration.get_rope_index``
    positional signature.
    """
    position_ids, rope_deltas = model.get_rope_index(
        input_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
    )
    _dbg(
        "mrope batch input_ids=%s position_ids=%s rope_deltas=%s image_grid_thw=%s",
        tuple(input_ids.shape),
        tuple(position_ids.shape),
        rope_deltas.tolist() if rope_deltas is not None else None,
        image_grid_thw.tolist() if image_grid_thw is not None else None,
    )
    return position_ids, rope_deltas


def visual_token_slice_for_row(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> slice:
    """Image-token span for one UCP row (after left padding)."""
    pos = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return slice(0, 0)
    return slice(int(pos[0]), int(pos[-1]) + 1)


class Qwen2VLDecodeSession:
    """Per-stream Qwen2-VL decoder compatible with :func:`etv_generate`.

    Callable like a Hugging Face model:

    * ``session(**processor_outputs, use_cache=True)`` — multimodal prefill
    * ``session(input_ids=ids, past_key_values=kv, use_cache=True)`` — step(s)

    State (``rope_deltas``, attention mask, cache length) lives on the
    session, not on the shared ``model``.
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_ids = get_qwen_eos_token_ids(model, tokenizer)
        self.attention_mask: torch.Tensor | None = None
        self.rope_deltas: torch.Tensor | None = None
        self.cache_position: torch.Tensor | None = None
        self.prompt_length: int = 0
        self._seq_len: int = 0
        self._prefilled: bool = False
        self._past_kv = None
        self._image_grid_thw: torch.Tensor | None = None
        self._video_grid_thw: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ):
        if not self._prefilled and past_key_values is None:
            if input_ids is not None:
                kwargs.setdefault("input_ids", input_ids)
            return self.prefill(use_cache=use_cache, **kwargs)
        if input_ids is None:
            raise ValueError("incremental decode requires input_ids")
        return self.step(
            input_ids=input_ids,
            past_key_values=past_key_values if past_key_values is not None else self._past_kv,
            use_cache=use_cache,
        )

    @torch.no_grad()
    def prefill(self, use_cache: bool = True, **inputs) -> Any:
        """Multimodal prefill with vision-aware M-RoPE positions."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        video_grid_thw = inputs.get("video_grid_thw")

        position_ids, rope_deltas = build_qwen_mrope_positions(
            self.model, input_ids, attention_mask, image_grid_thw, video_grid_thw,
        )

        self.prompt_length = int(input_ids.shape[1])
        self._seq_len = self.prompt_length
        self.attention_mask = attention_mask
        self.rope_deltas = rope_deltas
        self.cache_position = torch.arange(
            self.prompt_length, device=input_ids.device, dtype=torch.long,
        )
        self._image_grid_thw = image_grid_thw
        self._video_grid_thw = video_grid_thw

        _dbg(
            "prefill shapes input_ids=%s position_ids=%s rope_deltas=%s cache_len=%d",
            tuple(input_ids.shape),
            tuple(position_ids.shape),
            rope_deltas.tolist() if rope_deltas is not None else None,
            self.prompt_length,
        )

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            rope_deltas=rope_deltas,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
        )
        self._past_kv = out.past_key_values
        self._prefilled = True
        return out

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ) -> Any:
        """Advance one or more tokens using ``prepare_inputs_for_generation``."""
        if not self._prefilled:
            raise RuntimeError("call prefill before step")
        past = past_key_values if past_key_values is not None else self._past_kv
        n_new = input_ids.shape[1]
        device = input_ids.device

        ones = torch.ones(
            (input_ids.shape[0], n_new), device=device, dtype=self.attention_mask.dtype,
        )
        self.attention_mask = torch.cat([self.attention_mask, ones], dim=-1)

        cache_position = torch.arange(
            self._seq_len, self._seq_len + n_new, device=device, dtype=torch.long,
        )

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past,
            attention_mask=self.attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
            pixel_values=None,
            image_grid_thw=self._image_grid_thw,
            video_grid_thw=self._video_grid_thw,
            rope_deltas=self.rope_deltas,
        )

        _dbg(
            "step n_new=%d cache_position=%s attn_len=%d position_ids=%s",
            n_new,
            cache_position.tolist(),
            self.attention_mask.shape[1],
            tuple(model_inputs["position_ids"].shape)
            if model_inputs.get("position_ids") is not None
            else None,
        )

        out = self.model(**model_inputs)
        self._past_kv = out.past_key_values
        self._seq_len += n_new
        return out
