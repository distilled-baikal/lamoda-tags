#!/usr/bin/env python3
"""
Build and save per-good_type prior logits for each label into artifacts.

This is a lightweight way to "guarantee" good_type influence at inference time
without retraining:

  logit(p_final) = logit(p_model) + w * prior_logit(good_type, label)

Outputs:
  - model/artifacts/category_prior_logits.json
"""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_pylist_or_json(x) -> List[str]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        return [s]


def parse_tags_semicolon(x) -> List[str]:
    if pd.isna(x) or not str(x).strip():
        return []
    return [t.strip().lower() for t in str(x).split(";") if t.strip()]


def aggregate_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, g in df.groupby("product_sku", sort=False):
        gt = g["good_type"].iloc[0] if "good_type" in g.columns else ""

        tags = []
        for x in g["tags"]:
            tags.extend(parse_tags_semicolon(x))

        seen = set()
        tags_u = []
        for t in tags:
            if t and t not in seen:
                seen.add(t)
                tags_u.append(t)

        rows.append({"product_sku": sku, "good_type": str(gt).strip(), "tags_list": tags_u})

    out = pd.DataFrame(rows)
    out["n_tags"] = out["tags_list"].map(len)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build per-good_type prior logits for labels")
    parser.add_argument("--data", default="lamoda_reviews_sampled_with_pull_tags.csv", help="CSV with tags")
    parser.add_argument("--artifacts", default="model/artifacts", help="Artifacts directory")
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing alpha")
    args = parser.parse_args()

    data_path = Path(args.data)
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    labels_path = artifacts_dir / "labels.txt"
    if not labels_path.exists():
        raise SystemExit(f"Missing labels file: {labels_path} (train first)")

    labels = [line.strip().lower() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_to_i = {lab: i for i, lab in enumerate(labels)}
    L = len(labels)

    df = pd.read_csv(data_path, encoding="utf-8")
    agg = aggregate_by_sku(df)
    agg = agg[agg["n_tags"] > 0].reset_index(drop=True)

    gt_counts = defaultdict(int)
    gt_pos = defaultdict(lambda: np.zeros(L, dtype=np.float32))

    for gt, tags in zip(agg["good_type"].astype(str), agg["tags_list"]):
        gt = gt.strip()
        gt_counts[gt] += 1
        for t in tags:
            j = label_to_i.get(t)
            if j is not None:
                gt_pos[gt][j] += 1.0

    alpha = float(args.alpha)
    prior_logits_by_type: Dict[str, List[float]] = {}
    for gt, n in gt_counts.items():
        pos = gt_pos[gt]
        p = (pos + alpha) / (n + 2.0 * alpha)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logits = np.log(p / (1 - p))
        prior_logits_by_type[gt] = logits.astype(float).tolist()

    out_path = artifacts_dir / "category_prior_logits.json"
    out_path.write_text(
        json.dumps(
            {
                "source": str(data_path),
                "smoothing_alpha": alpha,
                "num_labels": L,
                "prior_logits_by_type": prior_logits_by_type,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✓ Saved category priors for {len(prior_logits_by_type)} good_types to {out_path}")


if __name__ == "__main__":
    main()


