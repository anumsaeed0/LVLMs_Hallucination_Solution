#!/usr/bin/env python
"""Calibrate per-model router thresholds from ~200 held-out samples.

Runs Stage 0 only (UCP + features), collects d1-d4 distributions, and emits
percentile thresholds:
  d3_low  = 20th percentile of first-token T-VHD
  d1_late = 60th percentile of normalized peak depth
  d2_flat = 80th percentile of VAQ entropy
  d4_low  = 30th percentile of crop mass
No labels, no training -- consistent with the training-free claim.
"""
import argparse
import json

import numpy as np


def thresholds_from_features(d1, d2, d3, d4) -> dict:
    return {
        "d1_late": float(np.percentile(d1, 60)),
        "d2_flat": float(np.percentile(d2, 80)),
        "d3_low": float(np.percentile(d3, 20)),
        "d4_low": float(np.percentile(d4, 30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-json", required=True,
                    help="json list of {d1,d2,d3,d4} from a Stage-0-only run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.features_json) as f:
        rows = json.load(f)
    th = thresholds_from_features(
        [r["d1"] for r in rows], [r["d2"] for r in rows],
        [r["d3"] for r in rows], [r["d4"] for r in rows])
    with open(args.out, "w") as f:
        json.dump(th, f, indent=2)
    print(json.dumps(th, indent=2))


if __name__ == "__main__":
    main()
