# GLIMPSE — Full Project Notes (Self-Explanatory)

*Gated Layer-adaptive Integrated Multi-granular counterfactual Pipeline for Selective Enhancement*

A training-free inference pipeline that reduces hallucination in vision-language models (LVLMs) by detecting, per-sample, how likely a model is to hallucinate, and adaptively spending more compute only where it's actually needed.

---

## 1. The problem it's solving

LVLMs (LLaVA, InstructBLIP, Qwen-VL, etc.) sometimes generate answers that aren't grounded in the actual image — describing objects that aren't there, or answering from language priors ("a plate usually has a fork on it") rather than from what's visible. Two existing training-free methods attack this differently:

- **VHR** (Vision-aware Head Reinforcement) — works at the **attention-head** level. Finds which heads inside the model actually respond to the image (vs. heads that fire the same regardless of the image) and amplifies the vision-sensitive ones during decoding. Cheap (~1 extra forward pass), but treats every query the same way — no adaptivity.
- **LASER** — works at the **layer + input** level. Finds which transformer layer is most "query-driven," uses that to crop the image toward the relevant region, and runs a full parallel counterfactual (masked-image) decoding stream to contrastively correct each generated token. More thorough, but expensive — always pays for 2 full prefills + a crop pass + a second decode stream, even on easy questions where none of that is needed.

**GLIMPSE's core idea:** these two methods are really computing the same *kind* of signal — "what changes in the model's internals when you remove the image vs. remove the query" — just at different granularities (head vs. layer). Compute both from **one shared batched forward pass**, then use a **cheap difficulty classifier** built from that same pass to decide, per sample, how much of the expensive machinery (cropping, counterfactual decoding) is actually worth running. Easy samples get the cheap VHR-only treatment; only genuinely hard samples pay the full LASER-style cost.

---

## 2. Core components (what exists in the codebase)

```
glimpse/
  hooks.py       — forward hooks: capture internal signals, and reinforce heads
  metrics.py     — turns captured activations into VHD / T-VHD / VAQ / HGCA scores
  router.py      — turns those scores into an EASY / MEDIUM / HARD decision
  localize.py    — turns the HGCA map into an image crop (+ optional masked negative image)
  decoding.py    — generates the answer text, with adaptive contrastive correction (ETV)
  pipeline.py    — orchestrates all of the above into one `run(image, query)` call
  models/        — per-architecture adapters (LLaVA-1.5, LLaVA-NeXT, InstructBLIP, Qwen-VL)
eval/            — benchmark drivers: POPE, CHAIR, A-OKVQA, TextVQA, + a Profiler
scripts/         — run_eval.py (CLI entry point), calibrate_router.py (threshold tuning)
```

Each of these maps to one "stage" of the pipeline (Stage 0 through Stage 4, described below).

---

## 3. How the components interact — full data flow

### Entry point
`scripts/run_eval.py` is the CLI you actually run. It picks a model adapter (`--model llava15`, etc.), builds a `GlimpseConfig` (holds hyperparameters + router thresholds) and a `GlimpsePipeline`, then hands the sample-by-sample loop to a benchmark driver (`eval/pope.py`, `eval/chair.py`, etc.) which reads the dataset, loads each image, and calls `pipeline.run(image, query)` once per example. It also owns a `Profiler` that records latency and which route each sample took.

The benchmark driver's only job is: load data from disk, call the pipeline, score the returned text against the gold label. It does **no** modeling logic itself — all of that lives in `GlimpsePipeline.run()`.

### Stage 0 — Unified Counterfactual Prefill (UCP)
**File:** `pipeline.py` (calls `adapter.build_ucp_batch`), implemented per-architecture in `models/llava.py` etc.

For a given (image, query) pair, the adapter builds **three text/image variants**, batches them together, and runs the model **once**:

| Variant | Composition | Purpose |
|---|---|---|
| **Full** | image + question | The real input — its KV-cache is kept and reused for actual decoding later |
| **No-image** | question only (image dropped) | Reveals which attention heads/values depend on the image at all |
| **No-query** | image + empty prompt (question dropped) | Reveals which attention patterns are driven by the specific question |

All three are left-padded to the same length and stacked into a single batch dimension, so the model only runs forward **once** for all three views — this is the key efficiency trick: instead of VHR's 1 extra prefill *plus* LASER's separate 2-3 prefills, GLIMPSE gets both papers' signals from one shared batch.

Inside `pipeline.py`, this happens under a `HeadCapture` context manager (see below) and with the VHR head-reinforcement hooks (`HeadScaler`) explicitly **disabled** — reinforcement must not distort the raw signal used to decide what to reinforce in the first place. That ordering is deliberate (also called out as a listed risk in the proposal: "Signal interference").

### Capturing internals — `hooks.py`
Two hook classes attach to every attention layer's `o_proj` (the linear layer each attention block uses right before merging head outputs back together):

- **`HeadCapture`** — a `forward_pre_hook` on `o_proj` that intercepts the concatenated per-head outputs *before* they get merged, reshapes them back into individual heads `(batch, tokens, n_heads, head_dim)`, and stores them (only the last token position, since prefill signals are computed at t=1). A separate `forward_hook` on the attention module also grabs the raw softmax attention weight matrices (requires `attn_implementation="eager"` in HuggingFace so attention probabilities are actually returned). This is a **read-only** hook — it doesn't change model behavior, it just exposes internals that are normally invisible.
- **`HeadScaler`** — the *write* counterpart. Also a `forward_pre_hook` on `o_proj`, but instead of just recording, it multiplies the output of selected heads by `alpha` (default 2.0) before they continue through the network — this is the actual VHR "reinforcement" mechanism. It has an `enabled` flag and a `disabled()` context manager so the pipeline can turn it off during Stage 0 (signal capture) and turn it on during Stage 3/4 (decoding).

### Stage 0 metrics — `metrics.py`
Takes the raw captured tensors from the UCP batch and turns them into four interpretable scores:

- **VHD** (Vision-Head Divergence) — for each attention head, the L2 distance between its output on the *full* input vs. the *no-image* input. Large distance = that head's behavior really depends on the image.
- **TA** (activation magnitude on no-image input alone) — used only to detect a specific failure mode: heads whose output *surges* in magnitude specifically when the image is removed. That's a red flag, not a good sign — it means the divergence is really "this head goes haywire without an image," not "this head is doing useful vision grounding."
- **Pruned VHD** — VHD with those surge-outlier heads zeroed out (`prune_outliers`), so only genuine vision-sensitive divergence remains.
- **T-VHD** — sum of the top-K per-layer VHD scores; a single scalar summarizing how vision-grounded the *current* answer token's prediction is. Low T-VHD = the token is coming mostly from language priors, not the image.
- **VAQ** (layer-level) — for each transformer layer, take the attention-weight difference between the *full* input and the *no-query* input, restricted to just the visual-token positions, ReLU'd (only keep increases, per LASER's definition), then L2-normed and averaged over the top-K heads in that layer. This produces one VAQ score per layer; the layer with the highest VAQ is `best_layer` — the layer where the model's attention is most clearly being *driven by the question* onto specific image regions.
- **HGCA** (Head-Gated Contrastive Attention, the project's own contribution) — instead of just using LASER's top-K-by-VAQ heads directly, HGCA takes the **intersection** of "top-K heads by VAQ" and "heads that pass the VHR gate" (above-median pruned VHD) at the best layer, and averages their contrastive attention maps. The idea: a head might respond strongly to the *question* (high VAQ) without actually being grounded in the *image* (could still be a language-prior head coincidentally correlated with the question). Gating by VHD as well filters those out. If the intersection happens to be empty, it falls back to the plain VAQ top-K set so the pipeline never breaks.

All four of VHD/T-VHD/VAQ/HGCA are derived from the **same single UCP forward pass** — nothing here requires a second model call.

### Stage 1 — Localization — `localize.py`
Takes the HGCA map (a probability distribution over image patches) and:
1. Computes the **attention centroid** — a weighted average (x, y) position over the patch grid, weighted by HGCA mass.
2. Centers a crop box on that centroid, sized to half the original image dimensions but never smaller than 224×224 (the CLIP vision-encoder's minimum useful receptive field).
3. Computes **crop mass** — what fraction of total HGCA attention mass actually falls inside that crop box. This becomes a router feature: if most of the "evidence" the model is attending to is already inside a natural crop, further cropping won't help much; if it's spread outside, cropping is likely to concentrate the model's attention usefully.
4. Optionally (only on the HARD route) builds a **counterfactual masked image** — the top-K most query-relevant patches (by HGCA) are grayed out on a *separate copy* of the image, which is then cropped the same way. This becomes the "negative" evidence-removed image used for contrastive decoding later.

If the HGCA map is degenerate (all zero, e.g. no clear query-driven attention anywhere), the code falls back gracefully to a center crop rather than crashing on a NaN centroid.

### Stage 2 — Routing — `router.py`
Four raw signals from the stages above are turned into normalized router features:

| Feature | Computed from | Meaning |
|---|---|---|
| `d1_depth` | `best_layer / (num_layers - 1)` | How late in the network the VAQ peak occurs (0 = first layer, 1 = last layer) |
| `d2_entropy` | Shannon entropy of the per-layer VAQ distribution, normalized to [0,1] | How spread out (vs. concentrated) the layer-wise attention signal is |
| `d3_tvhd` | T-VHD of the first answer token | How vision-grounded the very first generated token looks |
| `d4_crop_mass` | fraction of HGCA mass inside the localized crop box | Whether the crop actually captures where the model is looking |

Three boolean flags are derived, then combined into a route:

```python
late_or_flat = d1_depth > d1_late   OR   d2_entropy > d2_flat
prior_risk   = d3_tvhd  < d3_low
crop_useful  = d4_crop_mass < d4_low

if prior_risk AND late_or_flat:      → HARD
elif late_or_flat OR crop_useful:    → MEDIUM
else:                                 → EASY
```

This is a **cascading gate**: HARD is checked first and requires two conditions to co-occur (both a language-prior-driven first token *and* unstable/late attention), which is the strongest combined hallucination signature. MEDIUM is a broader catch-all (either condition alone). EASY is the default when nothing is flagged.

**Thresholds are not hardcoded absolutes.** `d1_late`, `d2_flat`, `d3_low`, `d4_low` live in a `RouterThresholds` dataclass and are meant to be calibrated per model architecture — see Section 5 (Dynamism) below.

### Stage 3/4 — Decoding — `decoding.py` + `pipeline.py`
Once a route is chosen, `pipeline.py` decides what inputs to decode on:

| Route | What gets decoded | Extra machinery |
|---|---|---|
| **EASY** | Original full image | None — just VHR head reinforcement, single decode stream |
| **MEDIUM** | The **cropped** image (`localize()` called again, `make_counterfactual=False`) | Crop re-prefill only |
| **HARD** | The cropped image, **plus** a masked/cropped counterfactual negative image | Crop re-prefill + `ETV` (Event-Triggered token Verification) contrastive decoding |

VHR head reinforcement (`HeadScaler.enabled = True`) is turned on for **all three routes** during decoding — it's treated as "essentially free" and always applied, regardless of route.

**ETV** is the mechanism that makes HARD-route decoding cheaper than LASER's always-on two-stream contrastive decoding:
- A `CounterfactualStream` object wraps the negative-image forward passes and maintains its **own KV-cache**, but doesn't advance it every step.
- At each decode step, a `tvhd_fn()` proxy checks whether the *current* token looks language-prior-driven (T-VHD below a threshold `tau_tok`).
- **Only when triggered**, the negative stream is caught up (`advance()` — replays all tokens generated since it was last advanced, in **one batched forward call**, so its logits stay exact even though it "skipped" earlier steps) and its logits are contrasted against the positive stream: `scores = z_pos + alpha_vat * (z_pos - z_neg)`.
- On non-triggered steps, decoding is just single-stream greedy generation with VHR reinforcement — no second forward pass at all.

This means the negative stream's total compute scales with how many tokens actually get flagged as risky (typically expected 10–30%), not with total generated length — the core efficiency claim over LASER/VCD-style contrastive decoding, which always pays for two full streams on every token.

### Output
`GlimpsePipeline.run()` returns a `GlimpseOutput` object: the decoded answer text, which route was taken, the best VAQ layer, the first-token T-VHD, ETV trigger statistics (how many tokens triggered the negative stream), and the crop box coordinates (if any). This is both the user-facing answer and a full diagnostic trace of *why* the pipeline decided to spend the compute it did.

### Benchmark scoring (POPE example)
`eval/pope.py` reads a POPE JSONL file (image filename, yes/no question, gold label), loads the image from disk, calls the pipeline, then parses the returned text into a binary yes/no prediction (checks if the text starts with or early-contains "yes"). Accuracy/Precision/Recall/F1/yes-ratio are computed by comparing predictions to gold labels. `eval/profiler.py` separately tracks wall-clock time per sample, route distribution (what % went EASY/MEDIUM/HARD), ETV utilization, and peak VRAM.

---

## 4. Working methodology, end-to-end (one example, traced through)

1. A benchmark driver reads one (image, question, gold-label) row from disk.
2. `GlimpsePipeline.run(image, query)` is called.
3. **Stage 0:** the adapter builds a 3-way batch (full / no-image / no-query), runs the model once with capture hooks active and reinforcement hooks disabled.
4. VHD, pruned VHD, T-VHD, and VAQ are computed from that single captured pass.
5. **Stage 1:** HGCA map is built (VAQ ∩ VHD-gated heads at the best layer), then localized into a crop box + crop-mass score.
6. **Stage 2:** the four router features are computed and fed through the EASY/MEDIUM/HARD decision logic.
7. **Stage 3/4:** depending on route, the pipeline builds the actual decode-time inputs (original image, or crop, or crop + masked negative crop), turns VHR reinforcement on, and greedily decodes the answer — using ETV's event-triggered contrastive correction only if HARD.
8. The decoded text + full diagnostic trace is returned.
9. The benchmark driver parses the text into a label, scores it against gold, and the profiler logs timing/route stats.

---

## 5. How dynamic is it?

**Per-sample adaptive, not static.** This is the central design point of the whole project — unlike VHR (same treatment every time) or LASER (same expensive machinery every time), GLIMPSE's cost and processing path change *per input*, based on signals measured from that specific input:

- **Compute is monotone in measured difficulty.** An "easy" image/question pays roughly 1 prefill + 1 decode stream. A "hard" one pays a crop re-prefill *and* a partially-active second decode stream. Nothing is fixed at configuration time — it's decided at inference time from the model's own internal signals.
- **Routing thresholds are recalibrated per model, not hand-set once.** `scripts/calibrate_router.py` runs Stage 0 only over ~200 held-out samples for a given model architecture, collects the distribution of d1–d4, and emits *percentile*-based thresholds (e.g., "d3_low = the 20th percentile of T-VHD seen in this sample of this specific model"). This matters because raw attention/entropy scales differ meaningfully between architectures (e.g., InstructBLIP's Q-Former-based attention vs. LLaVA's grid-based attention) — a fixed absolute cutoff would misclassify difficulty differently per model. No labels or training are involved anywhere in this calibration — it's purely a distributional statistic.
- **ETV's negative stream is dynamically, not uniformly, active.** Within a single HARD-route generation, some tokens trigger the counterfactual check and some don't, decided step-by-step by the live T-VHD proxy — this is dynamism *within* a single sample's decode loop, not just across samples.
- **Architecture-adaptive via the adapter Protocol.** `pipeline.py` is written against an abstract `ModelAdapter` interface (`build_ucp_batch`, `visual_token_slice`, `grid_hw`, `project_map`, etc.) — the orchestration logic itself doesn't know or care whether it's talking to LLaVA's fixed grid, Qwen-VL's any-resolution grid, or InstructBLIP's Q-Former (which has no direct patch-token spatial correspondence and needs `project_map` to back-project through Q-Former cross-attention to recover a patch-level map). Swapping models is a matter of swapping the adapter, not rewriting pipeline logic.

**What is currently static / not yet fully dynamic (per the code and README's own "Implementation status" notes):**
- `tvhd_proxy_fn` in `models/llava.py` is explicitly marked as a stub: it currently returns `-inf` (or an externally supplied capture value), meaning ETV triggers on **every** decode step right now — i.e., HARD-route decoding is currently LASER-equivalent (always-on second stream) rather than the intended sparsely-triggered version. The "cheap image-ablated shadow stream" that would make this genuinely sparse is flagged in the README as not yet implemented.
- The `RouterThresholds` defaults hardcoded in `router.py` are explicitly labeled as "LLaVA-1.5 placeholders to be replaced by calibrate_router.py output" — real calibration hasn't been run and baked in for all four supported architectures yet.
- Two eval benchmarks (MME, RefCOCO) are named in the design docs but not yet implemented in `eval/` (only POPE, CHAIR, A-OKVQA, TextVQA exist as working files).

---

## 6. Novel contributions (vs. the two papers it builds on)

1. **Unified Counterfactual Prefill (UCP)** — fuses VHR's head-level and LASER's layer-level ablation probes into one shared batched forward pass, instead of running them as two separate procedures.
2. **HGCA (Head-Gated Contrastive Attention)** — intersects LASER's query-driven head selection with VHR's vision-sensitivity gate, aiming for cleaner localization maps than either signal alone (LASER's raw top-K-by-magnitude can still admit "sink" heads that fire regardless of real visual grounding).
3. **Difficulty Router** — a zero-cost, training-free classifier that turns hallucination mitigation into elastic/anytime inference: compute scales with measured difficulty rather than being fixed per method.
4. **ETV (Event-Triggered token Verification)** — token-level, event-triggered contrastive decoding that (in its intended, non-stub form) breaks the 2× decode-cost floor that standard contrastive-decoding methods (VCD, LASER) always pay.
5. **Q-Former generalization** — a back-projection scheme (`project_map` in the InstructBLIP adapter) to recover a patch-level HGCA map for models like InstructBLIP that don't have a direct 1:1 mapping between visual tokens and image patches (because they use a fixed set of learned Q-Former queries instead).

---

## 7. Practical entry points if you want to run or extend this

- **Run an eval:** `python scripts/run_eval.py --model llava15 --bench pope --pope-json data/pope/coco_pope_adversarial.json --image-dir data/coco/val2014 --limit 100 --out results/pope_adv.json`
- **Calibrate thresholds for a model:** dump Stage-0-only features to JSON, then `python scripts/calibrate_router.py --features-json <path> --out <path>`
- **Add a new model architecture:** implement the `ModelAdapter` protocol (see `pipeline.py`'s `ModelAdapter` class) — the four architecture-specific methods to get right are `build_ucp_batch` (must keep visual-token positions aligned across the 3 variants), `visual_token_slice`, `grid_hw`, and `project_map`.
- **Tune the difficulty/compute tradeoff:** adjust `RouterThresholds` (looser thresholds → more samples routed EASY → faster but less careful) or `EtvConfig.tau_tok` (higher → ETV triggers more often on HARD route).
