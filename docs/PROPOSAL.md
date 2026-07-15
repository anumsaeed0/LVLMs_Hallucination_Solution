# GLIMPSE: Gated Layer-adaptive Integrated Multi-granular counterfactual Pipeline for Selective Enhancement

**A training-free, compute-routed pipeline for hallucination mitigation in LVLMs**

Baselines: VHR (He et al., ACL 2025, arXiv:2412.13949) and LASER (Zhu et al., 2026, arXiv:2602.04304)

---

## 1. Motivation and Gap

Both baselines are training-free and attack hallucination from complementary directions, yet neither is complete:

| | VHR | LASER |
|---|---|---|
| Granularity | Attention **heads** (internal) | Transformer **layers** + image **input** + **logits** |
| Counterfactual | Remove **image**, measure head-output divergence (VHD) | Remove **query**, measure attention-map change (VAQ) |
| Fixes language bias | Yes (reinforces vision-aware heads) | Partially (VAT contrastive decoding) |
| Fixes visual-token bottleneck | No | Yes (VAQ-guided cropping) |
| Inference cost | ~1 extra prefill; negligible | Always pays: 2 prefills + crop pass + counterfactual stream (~33% latency overhead on TextVQA: 874ms vs 656ms) |
| Adaptivity | Per-sample heads, but same treatment for every query | Per-query layer, but full machinery runs even for easy queries |

**Gap 1 (accuracy):** Neither combines head-level and layer-level grounding signals. LASER's contrastive attention still averages over top-K heads chosen by attention magnitude, which can re-admit sink-like heads; VHR's head selection ignores where (spatially) the model should look.

**Gap 2 (efficiency):** LASER's cost is query-invariant — POPE-style easy queries (where layer 14 is selected ~overwhelmingly and cropping barely helps) pay the same 3–4 forward passes as hard A-OKVQA queries. VCD-style two-stream decoding doubles decode FLOPs for *every* token, though only a minority of tokens are language-prior-driven (VHR's own T-VHD histograms show hallucinated words are a small, low-scoring subset).

**Central idea:** Both papers' probes are instances of one primitive — *modality-ablated counterfactual divergence*. Compute both signals from **one shared batched prefill**, then **route compute by measured query difficulty** so the expensive machinery (cropping, counterfactual decoding) only runs when the signals say it is needed.

---

## 2. The GLIMPSE Pipeline

### Stage 0 — Unified Counterfactual Prefill (UCP)

One batched forward pass over three input variants (padded to same length, single GPU batch):

- `(x_V, x_T)` — full input (this is the normal prefill; its KV-cache is kept for decoding)
- `(x_T)` — image-ablated → per-head divergence **VHD_{l,i}** (Eq. 4 of VHR)
- `(x_V)` — query-ablated → per-layer contrastive attention **A^con_{l,h}** and **VAQ_l** (Eqs. 3–5 of LASER)

Computed only at t=1 (end of prefill), as both papers justify. Because the two ablated variants are shorter than the full input and run in the same batch, wall-clock overhead ≈ one prefill, amortized to <5% of total generation time for captioning and <15% for short-answer VQA.

**Efficiency detail:** share the system-prompt prefix KV across the three variants (prefix caching); the ablated variants only recompute from the first differing token.

### Stage 1 — Head-Gated Contrastive Attention (HGCA) *(novel)*

LASER selects top-K heads per layer by VAQ magnitude; VHR shows some high-divergence heads have *negative* vision sensitivity and must be pruned. We intersect the two signals:

```
H_l = { i : VHD_{l,i} > median(VHD_{l,*})  ∧  not outlier per VHR Eq. 6 }
A^HGCA_l = (1/|H_l ∩ H^top_l|) Σ_{h ∈ H_l ∩ H^top_l} A^con_{l,h}
```

The localization map is built only from heads that are simultaneously (a) query-modulated (VAQ) and (b) genuinely vision-sensitive (VHD, outliers removed). Hypothesis: cleaner maps than LASER's Contra-Att (its RefCOCO Attention Aggregation of 41.77% on LLaVA leaves headroom), especially on cluttered scenes where sink heads survive the query contrast.

### Stage 2 — Difficulty Router *(novel)*

A zero-cost classifier over signals already computed in Stage 0:

- **d₁ = depth of argmax VAQ layer** (normalized ℓ*/L) — LASER shows late peaks ⇔ multi-step reasoning
- **d₂ = entropy of the layer-wise VAQ distribution** — flat profile ⇔ diffuse/uncertain grounding
- **d₃ = T-VHD of the first answer token** (VHR Eq. 5) — low ⇔ language-prior-driven start
- **d₄ = spatial concentration of A^HGCA at ℓ*** (mass inside best crop box) — low ⇔ evidence small/dispersed ⇔ cropping valuable

Route (thresholds τ calibrated once per model on ~200 held-out samples; no training):

| Route | Condition (illustrative) | Actions | Fwd passes |
|---|---|---|---|
| **EASY** | early peak, low entropy, high T-VHD | VHR head reinforcement only | 1 prefill batch + 1 decode |
| **MEDIUM** | late peak or low crop-mass | + HGCA-guided constrained crop, re-prefill on `I⁺`, decode with VHR | +1 prefill |
| **HARD** | low T-VHD and late/flat VAQ | + counterfactual masked image `I⁻`, selective VAT decoding | +1 prefill, partial 2nd stream |

This makes cost *monotone in difficulty* — the property both baselines lack. On POPE-like distributions most samples take EASY/MEDIUM; the HARD path with its extra stream is reserved for the queries where LASER's ablation shows VAT actually matters (A-OKVQA, TextVQA hard split).

### Stage 3 — Event-Triggered Token Verification (ETV) *(novel)*

On the HARD route, LASER contrasts `z⁺` and `z⁻` at **every** decode step (two full streams). We instead monitor **T-VHD_t** during decoding — it is nearly free since VHR already computes head outputs, and the image-ablated head outputs needed at step t can be approximated from the cached query-ablated stream. When

```
T-VHD_t < τ_tok   (token about to be generated from language priors)
```

we *then* run the counterfactual step for that token only and apply `s_t = z_t⁺ + α·VAT_t`. Otherwise decode single-stream with VHR reinforcement.

Since hallucinated tokens are the low-T-VHD minority (VHR Fig. 3), expected counterfactual-stream utilization is 10–30% of tokens rather than 100% ⇒ HARD-route decode FLOPs ≈ 1.1–1.3× of a single stream vs LASER/VCD's 2×.

Correctness note: the `I⁻` stream keeps its own KV-cache; on steps where it is skipped, its cache is lazily advanced in a single batched catch-up forward the next time it is needed (standard speculative-style batching), so logits remain exact.

### Stage 4 — VHR decoding everywhere

Head reinforcement (scale-up of H_l by α_VHR, layer-by-layer, last-N layers, heads fixed at t=1 for KV consistency) is active on all routes — it is essentially free and addresses language bias that survives cropping.

---

## 3. Novelty Summary (vs. both baselines)

1. **Unified Counterfactual Prefill** — VHD + VAQ from one batched pass; first work to fuse head-level and layer-level modality-ablation probes, at the cost of one.
2. **HGCA** — head-gated contrastive attention: VHD-validated heads filter LASER's attention map; expected gains on localization (RefCOCO AA) and downstream VQA.
3. **Difficulty Router** — training-free, per-instance compute allocation using internal grounding signals (not an external classifier). Turns hallucination mitigation into *anytime/elastic inference*.
4. **ETV** — token-level, event-triggered contrastive verification driven by T-VHD; breaks the 2× decode-cost floor of contrastive decoding.
5. **Q-Former generalization** — InstructBLIP has no patch-token spatial isomorphism (32 learned queries). We back-project through Q-Former cross-attention (`query→patch` maps composed with `LLM→query` attention) to recover a patch-level HGCA map. Neither baseline handles this; LASER is only demonstrated on grid-isomorphic models.

---

## 4. Cost Model (per sample, L = decode length, P = prefill cost, D = per-token decode cost)

| Method | Prefill | Decode | Notes |
|---|---|---|---|
| Greedy/Sample | 1P | L·D | reference |
| VCD | 2P | 2L·D | two streams always |
| OPERA | 1P | ≫L·D | beam + rollback, high memory |
| VHR | ~1.3P | L·D | ablated prefill batched |
| ViCrop | 2–3P | L·D | fixed layer |
| LASER | 3–4P | up to 2L·D | crop + counterfactual |
| **GLIMPSE-EASY** | ~1.5P | L·D | majority of POPE/CHAIR |
| **GLIMPSE-MEDIUM** | ~2.5P | L·D | crop re-prefill |
| **GLIMPSE-HARD** | ~3.5P | (1+ρ)L·D, ρ≈0.1–0.3 | ETV utilization ρ |

With a realistic route mix (e.g., 60/25/15 on a mixed benchmark), expected cost ≈ **1.9P + 1.05·L·D** vs LASER's ≈ 3.5P + 1.6·L·D — the headline efficiency claim. For long captioning (CHAIR, L≈100+), decode dominates and GLIMPSE ≈ VHR's cost while adding input enhancement where needed.

Memory: no beams, one extra KV-cache only on HARD route (freed after answer). Peak ≈ 1.1–1.4× base vs OPERA's >2×.

---

## 5. Experimental Plan

**Models:** LLaVA-1.5-7B (Vicuna), LLaVA-NeXT-7B, InstructBLIP (Vicuna-7B), Qwen-VL(-Chat). Covers fixed-res grid, any-res grid, and Q-Former architectures.

**Benchmarks / metrics:**

- CHAIR (MSCOCO, 500 imgs × 5 splits): CHAIR_S↓, CHAIR_I↓, Len, Recall
- POPE (random/popular/adversarial): Accuracy, Precision, Recall, F1
- MME (perception + hallucination subsets): score
- A-OKVQA (direct answer), TextVQA: Accuracy
- RefCOCO/+/g: Attention Aggregation % (localization ablation for HGCA)
- LLaVA-Bench (In-the-Wild): GPT-4o pairwise (accuracy/detail)
- **Efficiency:** ms/sample and ms/token (A100 + consumer 24GB), prefill count, ETV utilization ρ, peak VRAM, total FLOPs (fvcore/torch profiler)

**Baselines:** greedy, sampling, VCD, OPERA, ViCrop, VHR (their code), LASER (reimpl. if unreleased), SID/ICD if time permits.

**Ablations (each maps to a novelty claim):**

1. HGCA vs Contra-Att vs Rel-Att vs Raw-Att (RefCOCO AA + POPE F1)
2. Router: GLIMPSE vs always-EASY (≈VHR), always-HARD (≈LASER+VHR), random routing, oracle routing (upper bound)
3. ETV threshold sweep τ_tok ∈ {μ−σ, μ−0.5σ, μ} → accuracy-vs-ρ Pareto curve
4. UCP batching on/off (wall-clock)
5. Component removal: −VHR, −crop, −VAT, −outlier-pruning
6. Route-mix distribution per benchmark (interpretability figure mirroring LASER Fig. 5)

**Hyperparameters (from baselines, then swept):** α_VHR ∈ [1.5, 3] (VHR used ~2), last-14 layers reinforced (LLaVA family) / last-18 (InstructBLIP), K_head = 8, crop ≥ 224², α_VAT = 1.

**Success criteria (targets, to be validated empirically):**

- POPE F1: ≥ +0.5 over max(VHR, LASER) per setting; adversarial split is the differentiator
- CHAIR_S: match or beat VHR (LASER doesn't report CHAIR — free win for the combined method)
- TextVQA/A-OKVQA: ≥ LASER accuracy at ≤ 60% of its added latency
- Latency: ≤ 1.15× greedy on POPE, ≤ 1.35× on TextVQA (LASER: 1.33× and worse on hard sets)

---

## 6. Risks and Mitigations

- **Signal interference:** VHR head scaling changes attention maps used by HGCA. Mitigation: compute all Stage-0 signals *before* applying reinforcement (VHR itself is layer-by-layer for the same reason); ablate ordering.
- **Router miscalibration across models:** thresholds are model-specific. Mitigation: express thresholds as per-model percentiles (e.g., d₃ below the 20th percentile of a 200-sample calibration run), not absolute values.
- **ETV lazy KV catch-up complexity:** if batched catch-up proves fragile, fallback = recompute `I⁻` stream prefix at trigger time (still cheaper than 2× when ρ < 0.4); report both.
- **InstructBLIP back-projection quality:** Q-Former queries may mix spatial info. Mitigation: evaluate AA on RefCOCO first; if weak, fall back to VHR-only route for InstructBLIP (still a contribution: routing degrades gracefully per architecture).
- **LASER code unavailable (Feb 2026 preprint):** reimplement from paper; validate by reproducing their Table 1 within ±0.5.

---

## 7. Roadmap

1. **Wk 1–2:** VHR reproduction (code released) + LASER reimplementation; validate against published tables.
2. **Wk 3:** UCP batched prefill + hooks infrastructure (see `glimpse/` skeleton).
3. **Wk 4:** HGCA + RefCOCO localization study (fast signal on novelty #2).
4. **Wk 5:** Router calibration + EASY/MEDIUM/HARD paths end-to-end on LLaVA-1.5.
5. **Wk 6:** ETV + exactness tests (single-stream vs full two-stream logit equivalence at triggered steps).
6. **Wk 7–8:** Full benchmark sweep across 4 models; efficiency profiling.
7. **Wk 9:** Ablations, qualitative figures, paper writing.

---

## 8. Repository Layout

```
glimpse/
  hooks.py          # forward hooks: capture/scale per-head outputs, attention maps
  metrics.py        # VHD, T-VHD, VAQ, HGCA computation
  router.py         # difficulty features d1-d4, percentile thresholds, route decision
  localize.py       # HGCA map → constrained crop box (≥224²), counterfactual masking
  decoding.py       # VHR reinforcement, ETV selective contrastive decoding
  pipeline.py       # Stage 0-4 orchestration
  models/           # adapters: llava.py, llava_next.py, instructblip.py, qwenvl.py
eval/
  chair.py pope.py mme.py aokvqa.py textvqa.py refcoco.py
  profiler.py       # latency, FLOPs, VRAM, route-mix logging
scripts/
  run_eval.py calibrate_router.py
configs/
  llava15_7b.yaml llava_next_7b.yaml instructblip_7b.yaml qwenvl.yaml
```
