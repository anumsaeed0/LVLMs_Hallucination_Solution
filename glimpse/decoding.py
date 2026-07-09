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


class CounterfactualStream:
    """Lazy negative stream over (I-, q) with its own KV-cache.

    advance(tokens) catches up on all pending tokens in ONE batched forward
    and returns logits for the latest position. Skipped steps only pay when
    a trigger occurs.
    """

    def __init__(self, model, neg_inputs: dict):
        self.model = model
        self.neg_inputs = neg_inputs
        self.past_kv = None
        self.pending: list[int] = []
        self._prefilled = False

    @torch.no_grad()
    def _prefill(self) -> torch.Tensor:
        out = self.model(**self.neg_inputs, use_cache=True)
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
        ids = torch.tensor([self.pending], device=self.model.device)
        out = self.model(input_ids=ids, past_key_values=self.past_kv,
                         use_cache=True)
        self.past_kv = out.past_key_values
        self.pending.clear()
        return out.logits[:, -1]


@torch.no_grad()
def etv_generate(model, pos_inputs: dict, neg_inputs: dict | None,
                 tvhd_fn, cfg: EtvConfig, tokenizer) -> tuple[str, EtvStats]:
    """Greedy decoding with VHR active (via attached HeadScaler hooks) and
    event-triggered VAT correction.

    tvhd_fn() -> float: per-step T-VHD proxy computed from the HeadCapture
    state populated during the positive forward (the adapter re-runs the
    image-ablated last-token forward cheaply, or reuses the cached
    no-image stream -- see PROPOSAL.md Stage 3).
    """
    stats = EtvStats()
    neg = CounterfactualStream(model, neg_inputs) if neg_inputs else None

    out = model(**pos_inputs, use_cache=True)
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
        if tok == tokenizer.eos_token_id:
            break
        generated.append(tok)
        if neg is not None:
            neg.defer(tok)

        step_out = model(input_ids=torch.tensor([[tok]], device=model.device),
                         past_key_values=past_kv, use_cache=True)
        past_kv, logits = step_out.past_key_values, step_out.logits[:, -1]

    return tokenizer.decode(generated, skip_special_tokens=True), stats
