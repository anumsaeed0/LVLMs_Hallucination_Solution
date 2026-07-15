"""Stage 3/4 decoding: VHR reinforcement (always on) and Event-Triggered
token Verification (ETV) -- selective VAT contrastive decoding.

ETV replaces LASER/VCD's always-on second stream: the counterfactual (I-)
stream is only advanced when the per-token T-VHD proxy signals a
language-prior-driven token. Skipped steps are caught up lazily in one
batched forward, so triggered-step logits are exact.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

import os

_QWEN_DBG = os.environ.get("GLIMPSE_QWEN_DEBUG", "").strip() in {"1", "true", "yes"}

@dataclass
class EtvConfig:
    alpha_vat: float = 1.0     # LASER Eq. 11 contrast strength
    tau_tok: float = 0.0       # T-VHD trigger threshold (calibrated percentile)
    max_new_tokens: int = 512


@dataclass
class EtvStats:
    n_tokens: int = 0
    n_triggered: int = 0
    triggered_positions: list[int] = field(default_factory=list)

    @property
    def utilization(self) -> float:  # rho in the cost model
        return self.n_triggered / max(self.n_tokens, 1)


@torch.no_grad()
def hf_generate(model, inputs: dict, tokenizer, max_new_tokens: int = 512,
                eos_token_ids: set[int] | None = None) -> tuple[str, EtvStats]:
    """Single-stream greedy decoding via the model's own .generate().

    Used for EASY/MEDIUM routes where no VAT contrast is needed: there is
    no benefit to a manual KV loop, and .generate() guarantees correct
    position handling for architectures with nontrivial rotary schemes
    (Qwen2-VL M-RoPE breaks under naive past_key_values stepping --
    symptom: first token correct, then repeated garbage). VHR head-scaling
    hooks fire inside .generate() exactly as in the manual loop.
    """
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False,
                      num_beams=1, use_cache=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kwargs["pad_token_id"] = pad_id
    if eos_token_ids:
        gen_kwargs["eos_token_id"] = sorted(eos_token_ids)
    out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0][prompt_len:]
    stats = EtvStats(n_tokens=int(new_tokens.shape[0]))
    return tokenizer.decode(new_tokens, skip_special_tokens=True), stats


class CounterfactualStream:
    """Lazy negative stream over (I-, q) with its own KV-cache.

    advance(tokens) catches up on all pending tokens in ONE batched forward
    and returns logits for the latest position. Skipped steps only pay when
    a trigger occurs.
    """

    def __init__(self, model, neg_inputs: dict, 
                 create_decode_session=None):
        # self.model = model
        self.decoder = (create_decode_session()
                        if create_decode_session is not None else model)
        self.neg_inputs = neg_inputs
        self.past_kv = None
        self.pending: list[int] = []
        self._prefilled = False


    @property
    def device(self):
        dec = self.decoder
        if hasattr(dec, "device"):
            return dec.device
        return next(dec.parameters()).device


    @torch.no_grad()
    def _prefill(self) -> torch.Tensor:
        # out = self.model(**self.neg_inputs, use_cache=True)
        out = self.decoder(**self.neg_inputs, use_cache=True)
        self.past_kv = out.past_key_values
        self._prefilled = True
        return out.logits[:, -1]

    def defer(self, token_id: int) -> None:
        self.pending.append(token_id)

    @torch.no_grad()
    def advance(self) -> torch.Tensor:
        """Catch up on pending tokens; return logits z- at current position."""
        if not self._prefilled:
            logits = self._prefill()
            if not self.pending:
                return logits
        # ids = torch.tensor([self.pending], device=self.model.device)
        # out = self.model(input_ids=ids, past_key_values=self.past_kv,
        #                  use_cache=True)
        ids = torch.tensor([self.pending], device=self.device)
        out = self.decoder(input_ids=ids, past_key_values=self.past_kv,
                           use_cache=True)

        self.past_kv = out.past_key_values
        self.pending.clear()
        return out.logits[:, -1]


def _resolve_eos_ids(tokenizer, eos_token_ids: set[int] | None) -> set[int]:
    if eos_token_ids:
        return eos_token_ids
    eid = getattr(tokenizer, "eos_token_id", None)
    return {eid} if eid is not None else set()


@torch.no_grad()
def etv_generate(model, pos_inputs: dict, neg_inputs: dict | None,
                 tvhd_fn, cfg: EtvConfig, tokenizer,
                 create_decode_session=None,
                 eos_token_ids: set[int] | None = None) -> tuple[str, EtvStats]:
    """Greedy decoding with VHR active (via attached HeadScaler hooks) and
    event-triggered VAT correction.

    tvhd_fn() -> float: per-step T-VHD proxy computed from the HeadCapture
    state populated during the positive forward (the adapter re-runs the
    image-ablated last-token forward cheaply, or reuses the cached
    no-image stream -- see PROPOSAL.md Stage 3).
    """
    stats = EtvStats()
    # neg = CounterfactualStream(model, neg_inputs) if neg_inputs else None
    
    stop_ids = _resolve_eos_ids(tokenizer, eos_token_ids)
    pos_decoder = (create_decode_session()
                   if create_decode_session is not None else model)
    device = getattr(pos_decoder, "device", model.device)
    neg = (CounterfactualStream(model, neg_inputs, create_decode_session)
           if neg_inputs else None)

    # out = model(**pos_inputs, use_cache=True)
    out = pos_decoder(**pos_inputs, use_cache=True)
    past_kv, logits = out.past_key_values, out.logits[:, -1]
    generated: list[int] = []

    for step in range(cfg.max_new_tokens):
        z_pos = logits

        if neg is not None and tvhd_fn() < cfg.tau_tok:
            z_neg = neg.advance()                     # exact catch-up
            vat = z_pos - z_neg                       # LASER Eq. 10
            scores = z_pos + cfg.alpha_vat * vat      # LASER Eq. 11
            stats.n_triggered += 1
            stats.triggered_positions.append(step)
        else:
            scores = z_pos

        tok = int(scores.argmax(dim=-1))
        stats.n_tokens += 1

        if _QWEN_DBG and create_decode_session is not None and step == 0:
            import logging
            logging.getLogger(__name__).info(
                "[qwen] first_token id=%d text=%r is_eos=%s",
                tok, tokenizer.decode([tok]), tok in stop_ids,
            )
        if tok in stop_ids:
        # if tok == tokenizer.eos_token_id:
            break
        generated.append(tok)
        if neg is not None:
            neg.defer(tok)

        # step_out = model(input_ids=torch.tensor([[tok]], device=model.device),
        #                  past_key_values=past_kv, use_cache=True)
        step_out = pos_decoder(
            input_ids=torch.tensor([[tok]], device=device),
            past_key_values=past_kv, use_cache=True)

        past_kv, logits = step_out.past_key_values, step_out.logits[:, -1]

    return tokenizer.decode(generated, skip_special_tokens=True), stats
