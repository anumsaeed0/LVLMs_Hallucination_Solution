"""Efficiency profiler: latency, route mix, ETV utilization, peak VRAM.

Every eval script wraps sample processing in `Profiler.track()` so the
paper's efficiency table (ms/sample, ms/token, rho, VRAM, route mix) comes
from one place.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

import torch


@dataclass
class SampleRecord:
    latency_ms: float
    n_tokens: int
    route: str
    etv_utilization: float | None


@dataclass
class Profiler:
    records: list[SampleRecord] = field(default_factory=list)

    def track(self):
        return _Tracker(self)

    def summary(self) -> dict:
        n = len(self.records)
        if n == 0:
            return {}
        lat = [r.latency_ms for r in self.records]
        toks = sum(r.n_tokens for r in self.records)
        rhos = [r.etv_utilization for r in self.records
                if r.etv_utilization is not None]
        return {
            "n_samples": n,
            "ms_per_sample_mean": sum(lat) / n,
            "ms_per_sample_p50": sorted(lat)[n // 2],
            "ms_per_token": sum(lat) / max(toks, 1),
            "route_mix": dict(Counter(r.route for r in self.records)),
            "etv_utilization_mean": sum(rhos) / len(rhos) if rhos else None,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9
            if torch.cuda.is_available() else None,
        }

    def dump(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary(),
                       "records": [r.__dict__ for r in self.records]}, f, indent=2)


class _Tracker:
    def __init__(self, prof: Profiler):
        self.prof = prof

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def done(self, n_tokens: int, route: str, etv_utilization: float | None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.prof.records.append(SampleRecord(
            latency_ms=(time.perf_counter() - self.t0) * 1e3,
            n_tokens=n_tokens, route=route, etv_utilization=etv_utilization))

    def __exit__(self, *exc):
        return False
