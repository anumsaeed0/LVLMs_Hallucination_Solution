"""Evaluation harnesses.

Implemented:
  pope.py     - Acc / Precision / Recall / F1 (yes-no)
  chair.py    - CHAIR_S / CHAIR_I / Recall / Len on MSCOCO val2014
  aokvqa.py   - direct-answer soft accuracy (HF hub: HuggingFaceM4/A-OKVQA)
  textvqa.py  - VQA soft accuracy (TextVQA 0.5.1 val)
  vqa_utils.py- shared answer normalization + soft scoring
  profiler.py - latency / route-mix / ETV utilization / VRAM

Stubs to add:
  mme.py      - MME perception + hallucination subsets
  refcoco.py  - Attention Aggregation % for the HGCA localization ablation
"""
