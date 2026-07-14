"""Resolve a model source: local directory first, HuggingFace hub fallback.

Search order for `resolve("llava-hf/llava-1.5-7b-hf")`:
  1. the string itself as a directory (absolute or relative path)
  2. %GLIMPSE_MODELS_DIR%/llava-1.5-7b-hf   (env var, if set)
  3. ./models/llava-1.5-7b-hf               (repo-local convention)
A directory only counts if it contains config.json (i.e. a complete
snapshot, e.g. downloaded via:
  huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
).
If nothing matches, the hub ID is returned unchanged and from_pretrained
downloads it (into the normal HF cache, see HF_HOME).
"""
from __future__ import annotations

import os


def _is_snapshot(d: str) -> bool:
    return os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))


def resolve(model_id: str, local_root: str | None = None) -> str:
    if _is_snapshot(model_id):
        return model_id
    name = model_id.rstrip("/\\").split("/")[-1]
    roots = [r for r in (local_root,
                         os.environ.get("GLIMPSE_MODELS_DIR"),
                         "models") if r]
    for root in roots:
        cand = os.path.join(root, name)
        if _is_snapshot(cand):
            print(f"[resolve] using local model: {cand}")
            return cand
    print(f"[resolve] no local copy of '{model_id}' found; "
          f"will download from HuggingFace hub")
    return model_id
