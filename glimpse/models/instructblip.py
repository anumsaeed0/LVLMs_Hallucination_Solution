"""InstructBLIP (Salesforce/instructblip-vicuna-7b) adapter.

Architecture: ViT-g (224x224 -> 16x16=256 patches + CLS) -> Q-Former
(32 learned queries, instruction-conditioned) -> Vicuna-7B. The LLM sees
32 query tokens PREPENDED to the text, with NO spatial isomorphism.

GLIMPSE adaptations (PROPOSAL.md novelty #5):
  - visual_token_slice = the 32 query-token positions (LLM level);
    VHD/VAQ/HGCA are computed over these 32 tokens as usual.
  - project_map back-projects the 32-dim HGCA map to the 16x16 patch grid
    by composing it with the Q-Former's query->patch cross-attention
    (averaged over cross-attention layers and heads, CLS dropped):
        map_patch[p] = sum_q hgca[q] * A_qformer[q, p]
  - VHR reinforces the last 18 layers (per the VHR paper's InstructBLIP
    setting) -- pass reinforced_last_n=18 in GlimpseConfig.
  - "no-query" UCP variant uses an empty instruction for BOTH the Q-Former
    and the LLM, keeping the 32 query positions aligned with the full
    variant.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (InstructBlipForConditionalGeneration,
                          InstructBlipProcessor)

N_QUERY = 32
PATCH_GRID = (16, 16)
NO_QUERY_TEXT = " "


class _IBlipDecoder:
    """Adapter between GLIMPSE's decode loop and InstructBLIP.

    The composite InstructBlipForConditionalGeneration.forward() does not
    accept use_cache/past_key_values (generation is meant to go through
    .generate()). This shim computes the vision -> Q-Former -> projection
    prefix once at prefill, hands inputs_embeds to the underlying LLaMA
    (which fully supports KV caching), and forwards incremental steps to
    the LLaMA directly.
    """

    def __init__(self, blip, legacy: bool):
        self.blip = blip
        self.legacy = legacy
        self.lm = blip.language_model

    @property
    def device(self):
        return self.blip.device

    @torch.no_grad()
    def _visual_prefix(self, pixel_values, qformer_input_ids,
                       qformer_attention_mask) -> torch.Tensor:
        blip = self.blip
        dtype = next(blip.vision_model.parameters()).dtype
        vis = blip.vision_model(pixel_values.to(dtype)).last_hidden_state
        img_atts = torch.ones(vis.shape[:-1], dtype=torch.long,
                              device=vis.device)
        q_tokens = blip.query_tokens.expand(vis.shape[0], -1, -1)
        q_mask = torch.ones(q_tokens.shape[:-1], dtype=torch.long,
                            device=vis.device)
        qf_mask = torch.cat([q_mask, qformer_attention_mask], dim=1)
        qf = blip.qformer(input_ids=qformer_input_ids,
                          attention_mask=qf_mask,
                          query_embeds=q_tokens,
                          encoder_hidden_states=vis,
                          encoder_attention_mask=img_atts)
        query_output = qf[0][:, :q_tokens.size(1), :]
        return blip.language_projection(query_output)  # (B, 32, hidden)

    def __call__(self, input_ids=None, attention_mask=None, pixel_values=None,
                 qformer_input_ids=None, qformer_attention_mask=None,
                 past_key_values=None, use_cache=True, **kw):
        if past_key_values is not None or pixel_values is None:
            # incremental step or text-only prefill: straight to the LM
            return self.lm(input_ids=input_ids,
                           attention_mask=attention_mask,
                           past_key_values=past_key_values,
                           use_cache=use_cache)
        # multimodal prefill
        prefix = self._visual_prefix(pixel_values, qformer_input_ids,
                                     qformer_attention_mask)
        embed = self.lm.get_input_embeddings()
        text_embeds = embed(input_ids)
        if self.legacy:
            inputs_embeds = torch.cat(
                [prefix.to(text_embeds.dtype), text_embeds], dim=1)
            prefix_mask = torch.ones(prefix.shape[:-1], dtype=torch.long,
                                     device=prefix.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            img_idx = self.blip.config.image_token_index
            mask = (input_ids == img_idx).unsqueeze(-1).expand_as(text_embeds)
            inputs_embeds = text_embeds.masked_scatter(
                mask, prefix.to(text_embeds.dtype).flatten(0, 1))
        return self.lm(inputs_embeds=inputs_embeds,
                       attention_mask=attention_mask,
                       use_cache=use_cache)


class InstructBlipAdapter:
    grid = PATCH_GRID

    def __init__(self, model_id: str = "Salesforce/instructblip-vicuna-7b",
                 device: str | None = None, dtype: torch.dtype | None = None):
        from .resolve import resolve
        model_id = resolve(model_id)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda"
                               else torch.float32)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self.dtype,
            attn_implementation="eager").to(self.device)
        self.processor = InstructBlipProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        # Legacy checkpoints (pre-4.45 snapshots) have no image_token_index:
        # the MODEL prepends the 32 query embeddings itself, and the 4.45
        # processor's <image>-token expansion both mismatches that and hits
        # a list/Tensor concat TypeError. Disable expansion for legacy.
        self.legacy = getattr(self.model.config, "image_token_index",
                              None) is None
        if not self.legacy:
            # Mixed-vintage snapshots: new config declares image_token_index
            # but old weights lack the resized embedding row / the tokenizer
            # only gained <image> at processor init. An out-of-range
            # embedding lookup then corrupts the CUDA stream (surfaces as
            # CUBLAS_STATUS_NOT_SUPPORTED). Detect and force legacy mode.
            n_emb = self.model.get_input_embeddings().num_embeddings
            img_idx = self.model.config.image_token_index
            if img_idx >= n_emb:
                print(f"[instructblip] image_token_index {img_idx} outside "
                      f"embedding table ({n_emb} rows): old weights + new "
                      f"config detected -> forcing legacy mode")
                self.model.config.image_token_index = None
                self.legacy = True
        if self.legacy and getattr(self.processor, "num_query_tokens",
                                   None) is not None:
            self.processor.num_query_tokens = None
        cfg = self.model.language_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self._qformer_q2p: torch.Tensor | None = None  # (32, 256) cache
        # KV-cache-capable decode entry point (composite forward lacks it)
        self.decode_model = _IBlipDecoder(self.model, self.legacy)

    def attn_modules(self):
        return [l.self_attn for l in self.model.language_model.model.layers]

    def _proc(self, image: Image.Image, text: str) -> dict:
        """Processor call robust to the transformers 4.45.0 bug where
        return_tensors='pt' leaks into the <image>-token expansion and
        raises `can only concatenate list (not "Tensor") to list`.
        Falls back to list output + manual tensorization."""
        try:
            return self.processor(images=image, text=text,
                                  return_tensors="pt")
        except TypeError:
            import numpy as np
            enc = self.processor(images=image, text=text)  # lists / np
            out = {}
            for k, v in enc.items():
                if torch.is_tensor(v):
                    out[k] = v
                elif k == "pixel_values":
                    out[k] = torch.as_tensor(np.array(v))
                else:
                    t = torch.as_tensor(v)
                    out[k] = t if t.dim() > 1 else t.unsqueeze(0)
            return out

    def _encode(self, image: Image.Image | None, text: str) -> dict:
        if image is None:
            enc = self.tokenizer(text, return_tensors="pt")
        else:
            enc = self._proc(image, text)
        return {k: v.to(self.device) for k, v in enc.items()}

    def build_inputs(self, image: Image.Image | None, query: str) -> dict:
        return self._encode(image, query)

    @torch.no_grad()
    def _capture_q2p(self, image: Image.Image, query: str) -> None:
        """Cache the Q-Former query->patch map for the current sample."""
        enc = self._proc(image, query)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        vis = self.model.vision_model(
            enc["pixel_values"].to(self.dtype)).last_hidden_state
        q_tokens = self.model.query_tokens.expand(vis.shape[0], -1, -1)
        # Q-Former self-attention runs over [32 query embeds; text tokens],
        # so the mask must be extended with ones for the query positions
        # (mirrors InstructBlipForConditionalGeneration.forward)
        query_mask = torch.ones(q_tokens.shape[:-1], device=self.device,
                                dtype=torch.long)
        qf_mask = torch.cat([query_mask, enc["qformer_attention_mask"]], dim=1)
        qf = self.model.qformer(
            input_ids=enc["qformer_input_ids"],
            attention_mask=qf_mask,
            query_embeds=q_tokens,
            encoder_hidden_states=vis,
            encoder_attention_mask=torch.ones(vis.shape[:2], device=self.device,
                                              dtype=torch.long),
            output_attentions=True)
        # cross_attentions: tuple over layers (None where absent);
        # each (B, heads, 32(+text), 257). Average layers+heads, keep the 32
        # query rows, drop the CLS patch column.
        maps = [a for a in qf.cross_attentions if a is not None]
        m = torch.stack(maps).mean(dim=(0, 2))[0]        # (32+text, 257)
        m = m[:N_QUERY, 1:]                              # (32, 256)
        self._qformer_q2p = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def build_ucp_batch(self, image: Image.Image, query: str) -> dict:
        self._capture_q2p(image, query)  # side product for project_map
        enc_full = self._proc(image, query)
        enc_no_image = self.tokenizer(query, return_tensors="pt")
        enc_no_query = self._proc(image, NO_QUERY_TEXT)
        encodings = [enc_full, enc_no_image, enc_no_query]
        max_len = max(e["input_ids"].shape[1] for e in encodings)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        ids_list, mask_list = [], []
        for e in encodings:
            ids, mask = e["input_ids"], e["attention_mask"]
            pad = max_len - ids.shape[1]
            if pad > 0:  # RIGHT-pad the text; query tokens stay prepended at
                ids = F.pad(ids, (0, pad), value=pad_id)   # positions 0..31,
                mask = F.pad(mask, (0, pad), value=0)      # aligned across
            ids_list.append(ids)                           # variants 0 and 2
            mask_list.append(mask)

        # Qformer text: pad full vs no-query to same length.
        # IMPORTANT: the Q-Former is BERT-based (vocab ~30k) -- pad with ITS
        # tokenizer's pad id, NOT the LLaMA pad id (32000), which is out of
        # range for the Q-Former embedding table (device-side assert that
        # surfaces later as a cryptic CUBLAS error).
        qpad = self.processor.qformer_tokenizer.pad_token_id
        if qpad is None:
            qpad = 0
        qids = [enc_full["qformer_input_ids"], enc_no_query["qformer_input_ids"]]
        qmask = [enc_full["qformer_attention_mask"],
                 enc_no_query["qformer_attention_mask"]]
        qlen = max(q.shape[1] for q in qids)
        qids = [F.pad(q, (0, qlen - q.shape[1]), value=qpad) for q in qids]
        qmask = [F.pad(q, (0, qlen - q.shape[1]), value=0) for q in qmask]

        pv = enc_full["pixel_values"]
        if self.legacy:
            # model prepends query embeds for EVERY sample given pixels ->
            # placeholder zeros for the no-image variant (black-image
            # ablation); qformer batch must match (3)
            pixel_values = torch.cat([pv, torch.zeros_like(pv), pv])
            qformer_input_ids = torch.cat([qids[0], qids[0], qids[1]])
            qformer_attention_mask = torch.cat([qmask[0],
                                                torch.zeros_like(qmask[0]),
                                                qmask[1]])
        else:
            # features are scattered into <image> tokens; variant 1 has none,
            # so pass images -- and matching qformer text -- only for
            # variants 0 and 2 (batch order along pixel_values)
            pixel_values = torch.cat([pv, enc_no_query["pixel_values"]])
            qformer_input_ids = torch.cat([qids[0], qids[1]])
            qformer_attention_mask = torch.cat([qmask[0], qmask[1]])
        batch = {
            "input_ids": torch.cat(ids_list),
            "attention_mask": torch.cat(mask_list),
            "pixel_values": pixel_values,
            "qformer_input_ids": qformer_input_ids,
            "qformer_attention_mask": qformer_attention_mask,
        }
        return {k: v.to(self.device) for k, v in batch.items()}

    def _img_slice(self, ids: torch.Tensor) -> slice:
        image_token = self.model.config.image_token_index
        pos = (ids == image_token).nonzero(as_tuple=True)[0]
        return slice(int(pos[0]), int(pos[0]) + N_QUERY)

    def visual_token_slices(self, batch: dict) -> tuple[slice, slice]:
        if self.legacy:
            # query embeds are prepended by the model at positions 0..31 for
            # ALL variants (text is right-padded), so both slices coincide
            s = slice(0, N_QUERY)
            return s, s
        # new-style checkpoint: <image> tokens live in input_ids per variant
        ids = batch["input_ids"]
        return self._img_slice(ids[0]), self._img_slice(ids[2])

    def visual_token_slice(self, batch: dict) -> slice:
        return self.visual_token_slices(batch)[0]

    def grid_hw(self, batch: dict) -> tuple[int, int]:
        return PATCH_GRID

    def project_map(self, hgca: torch.Tensor, batch: dict) -> torch.Tensor:
        """(32,) query-space map -> (256,) patch-space map via Q-Former
        cross-attention composition."""
        assert self._qformer_q2p is not None, "call build_ucp_batch first"
        q2p = self._qformer_q2p.to(hgca.device, hgca.dtype)  # (32, 256)
        m = hgca @ q2p
        return m / m.sum().clamp_min(1e-9)

    def tvhd_proxy_fn(self, capture):
        return lambda: float("-inf") if capture is None else capture()
