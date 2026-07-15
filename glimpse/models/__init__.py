"""Model adapters.

All adapters implement the ModelAdapter protocol (glimpse/pipeline.py):
UCP batching, visual token slice, patch grid, project_map, VHR hooks target.

Model-specific notes:
  llava.py        LLaVA-1.5-7B, fixed 24x24 grid. Reference implementation.
  llava_next.py   LLaVA-NeXT (v1.6) any-res; localization on the 24x24
                  global view (first 576 visual tokens).
  instructblip.py Q-Former back-projection to a 16x16 patch grid;
                  use GlimpseConfig(reinforced_last_n=18).
  qwenvl.py       Qwen2-VL any-res; grid from image_grid_thw (2x2 merge).
"""
from .instructblip import InstructBlipAdapter
from .llava import Llava15Adapter
from .llava_next import LlavaNextAdapter
from .qwenvl import QwenVLAdapter

__all__ = ["Llava15Adapter", "LlavaNextAdapter", "InstructBlipAdapter",
           "QwenVLAdapter"]
