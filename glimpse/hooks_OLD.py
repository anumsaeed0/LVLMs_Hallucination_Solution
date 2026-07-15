"""Forward hooks for capturing and manipulating per-head attention outputs.

Two hook types:
  1. HeadCapture  - records per-head outputs A_{l,i} and attention maps a_{l,h}
                    during a (batched) prefill, for VHD / VAQ / HGCA computation.
  2. HeadScaler   - scales selected head outputs by alpha during decoding (VHR).

Works on HF transformers attention modules by hooking the o_proj input,
which is the concatenation of per-head outputs before the output projection.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn


@dataclass
class CapturedState:
    """Per-layer captures from one forward pass."""
    # head_outputs[l]: (B, T, n_heads, head_dim) -- pre-o_proj, split by head
    head_outputs: dict[int, torch.Tensor] = field(default_factory=dict)
    # attn_maps[l]: (B, n_heads, T_q, T_kv) -- softmax attention weights
    attn_maps: dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.head_outputs.clear()
        self.attn_maps.clear()


class HeadCapture:
    """Registers hooks on each decoder layer's attention module.

    Captures per-head outputs (input to o_proj, reshaped to heads) and,
    when `capture_attn=True`, attention probability maps. Attention maps
    require eager attention implementation (attn_implementation="eager")
    or output_attentions=True support in the model adapter.
    """

    def __init__(self, attn_modules: list[nn.Module], n_heads: int,
                 head_dim: int, capture_attn: bool = True,
                 last_token_only: bool = True):
        self.attn_modules = attn_modules
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.capture_attn = capture_attn
        self.last_token_only = last_token_only
        self.state = CapturedState()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_oproj_pre_hook(self, layer_idx: int):
        def hook(module: nn.Module, args: tuple):
            (x,) = args  # (B, T, n_heads * head_dim)
            b, t, _ = x.shape
            heads = x.view(b, t, self.n_heads, self.head_dim)
            if self.last_token_only:
                heads = heads[:, -1:, :, :]
            self.state.head_outputs[layer_idx] = heads.detach()
        return hook

    def _make_attn_fwd_hook(self, layer_idx: int):
        def hook(module: nn.Module, args, kwargs, output):
            # HF eager attention returns (attn_output, attn_weights, past_kv)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]
                if self.last_token_only:
                    # keep only the last query row -- all that VAQ reads.
                    # Storing full TxT maps for every layer costs O(T^2)
                    # memory (gigabytes at any-res T>1000, causing VRAM
                    # spill and massive slowdown); this is O(T).
                    attn = attn[:, :, -1:, :]
                self.state.attn_maps[layer_idx] = attn.detach()
        return hook

    def __enter__(self) -> "HeadCapture":
        for l, mod in enumerate(self.attn_modules):
            o_proj = getattr(mod, "o_proj", None) or getattr(mod, "out_proj", None)
            if o_proj is None:
                raise AttributeError(f"layer {l}: no o_proj/out_proj found")
            self._handles.append(o_proj.register_forward_pre_hook(
                self._make_oproj_pre_hook(l)))
            if self.capture_attn:
                self._handles.append(mod.register_forward_hook(
                    self._make_attn_fwd_hook(l), with_kwargs=True))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class HeadScaler:
    """VHR reinforcement: scale outputs of selected heads by alpha.

    selected[l] is a bool mask (n_heads,) of heads to reinforce in layer l.
    Applied as a pre-hook on o_proj so it composes with HeadCapture.
    Heads are fixed at the first generation step (KV-cache consistency,
    per VHR Sec. 3.2).
    """

    def __init__(self, attn_modules: list[nn.Module], n_heads: int,
                 head_dim: int, alpha: float = 2.0):
        self.attn_modules = attn_modules
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.alpha = alpha
        self.selected: dict[int, torch.Tensor] = {}  # layer -> bool mask
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.enabled = True

    def set_heads(self, selected: dict[int, torch.Tensor]) -> None:
        self.selected = selected

    def _make_hook(self, layer_idx: int):
        def hook(module: nn.Module, args: tuple):
            if not self.enabled or layer_idx not in self.selected:
                return None
            (x,) = args
            b, t, _ = x.shape
            heads = x.view(b, t, self.n_heads, self.head_dim).clone()
            mask = self.selected[layer_idx].to(x.device)  # (n_heads,)
            heads[:, :, mask, :] *= self.alpha
            return (heads.view(b, t, -1),)
        return hook

    def attach(self) -> None:
        for l, mod in enumerate(self.attn_modules):
            o_proj = getattr(mod, "o_proj", None) or getattr(mod, "out_proj", None)
            self._handles.append(o_proj.register_forward_pre_hook(self._make_hook(l)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @contextlib.contextmanager
    def disabled(self):
        prev, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = prev
