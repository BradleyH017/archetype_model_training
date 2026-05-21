#!/usr/bin/env python3
"""
003_train_mil.py
Attention-based Multiple Instance Learning (MIL) with nested cross-validation.

Architecture (Ilse et al., 2018):
  - Per-cell encoder (2-layer MLP)
  - Gated attention pooling → single bag embedding
  - Bag-level predictor with softmax (simplex output)

Outputs (in --outdir):
  mil_fold_metrics.csv       — per-fold R² / RMSE + best hyperparams
  mil_oof_predictions.csv    — out-of-fold predictions (samples × archetypes)
  summary_metrics.csv        — per-archetype OOF metrics
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import (
    HAS_TORCH,
    load_mil_bags,
    nested_cv_mil,
    per_archetype_metrics,
)

if not HAS_TORCH:
    raise RuntimeError("PyTorch is required for MIL training but could not be imported.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",           required=True)
    p.add_argument("--mil_bags",          required=True)
    p.add_argument("--targets",           required=True)
    p.add_argument("--mil_params",        required=True)
    p.add_argument("--n_outer_folds",     type=int, default=5)
    p.add_argument("--n_inner_folds",     type=int, default=5)
    p.add_argument("--n_iter",            type=int, default=20)
    p.add_argument("--max_cells_per_bag", type=int, default=2000)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--outdir",            required=True)
    return p.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading bags from {args.mil_bags}")
    bags, samples = load_mil_bags(args.mil_bags)
    print(f"  {len(bags)} bags  |  cells/bag: "
          f"min={min(b.shape[0] for b in bags)} "
          f"max={max(b.shape[0] for b in bags)}")

    print(f"Loading targets from {args.targets}")
    df_tgt  = pd.read_csv(args.targets, index_col=0)
    keep    = [i for i, s in enumerate(samples) if s in df_tgt.index]
    bags    = [bags[i] for i in keep]
    samples = [samples[i] for i in keep]
    y       = df_tgt.loc[samples].values.astype(float)
    arch_cols = df_tgt.columns.tolist()
    print(f"  {len(bags)} bags × {y.shape[1]} targets")

    with open(args.mil_params) as f:
        hp_grid = json.load(f)

    torch.manual_seed(args.seed)
    fold_metrics, oof_preds = nested_cv_mil(
        bags, y, hp_grid,
        args.n_outer_folds, args.n_inner_folds, args.n_iter,
        args.max_cells_per_bag, args.seed, device,
    )

    pd.DataFrame(fold_metrics).to_csv(
        f"{args.outdir}/mil_fold_metrics.csv", index=False)

    pd.DataFrame(oof_preds, columns=arch_cols, index=samples).to_csv(
        f"{args.outdir}/mil_oof_predictions.csv")

    pd.DataFrame(
        per_archetype_metrics(y, oof_preds, arch_cols, "mil", args.cluster)
    ).to_csv(f"{args.outdir}/summary_metrics.csv", index=False)

    print(f"\nDone. Results in {args.outdir}")


if __name__ == "__main__":
    main()
