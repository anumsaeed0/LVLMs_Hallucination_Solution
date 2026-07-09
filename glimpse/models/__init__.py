"""Model adapters. llava.py is the reference implementation; the others are
stubs documenting the architecture-specific work items.

- llava_next.py:   any-res tiling -> variable grid; visual_token_slice must
                   account for multi-tile layouts; reinforce last 14 layers.
- qwenvl.py:       dynamic-resolution ViT, variable-length visual tokens;
                   grid_hw derived per sample from image aspect ratio.
- instructblip.py: Q-Former (32 query tokens, no spatial isomorphism).
                   HGCA back-projection: compose LLM->query attention with
                   Q-Former query->patch cross-attention to recover a
                   patch-level map (PROPOSAL.md novelty #5). Reinforce last
                   18 layers per VHR.
"""
from .llava import Llava15Adapter

__all__ = ["Llava15Adapter"]
