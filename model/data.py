"""
Data loading, preprocessing, and dataset creation for tag prediction.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import PreTrainedTokenizer

from model.config import Config, DataConfig


def parse_pylist_or_json(x) -> List[str]:
    """Parse list from various formats (real list, JSON string, or NaN)."""
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
    """Parse tags from semicolon-separated string."""
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]


def build_text(name: str, good_type: str, good_subtype: str, comments: List[str]) -> str:
    """Build input text from product metadata and reviews."""
    parts = []
    if pd.notna(name) and str(name).strip():
        parts.append(str(name).strip())
    if pd.notna(good_type) and str(good_type).strip():
        parts.append(str(good_type).strip())
    if pd.notna(good_subtype) and str(good_subtype).strip():
        parts.append(str(good_subtype).strip())
    if comments:
        parts.append(" ".join([str(c) for c in comments if str(c).strip()]))
    return " | ".join(parts)


def aggregate_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data to 1 row per product_sku.
    - comment_text: concatenate all reviews
    - tags: union of all tags
    """
    df = df.copy()
    df["comment_text_list"] = df["comment_text"].apply(parse_pylist_or_json)
    df["tags_list"] = df["tags"].apply(parse_tags_semicolon)

    agg_rows = []
    for sku, g in df.groupby("product_sku", sort=False):
        name = g["name"].iloc[0] if "name" in g.columns else ""
        good_type = g["good_type"].iloc[0] if "good_type" in g.columns else ""
        good_subtype = g["good_subtype"].iloc[0] if "good_subtype" in g.columns else ""

        comments = []
        for lst in g["comment_text_list"].tolist():
            comments.extend(lst)

        tags = []
        for lst in g["tags_list"].tolist():
            tags.extend(lst)

        seen = set()
        tags_unique = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                tags_unique.append(t)

        text = build_text(name, good_type, good_subtype, comments)

        agg_rows.append({
            "product_sku": sku,
            "name": name,
            "good_type": good_type,
            "good_subtype": good_subtype,
            "text": text,
            "tags_list": tags_unique
        })

    return pd.DataFrame(agg_rows)


def stratified_split_three_way(
    df: pd.DataFrame,
    strat_col: str = "good_type",
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification by good_type.
    Handles rare categories gracefully.
    """
    df = df.copy()
    counts = df[strat_col].value_counts()
    rare = set(counts[counts < 3].index)
    df_rare = df[df[strat_col].isin(rare)]
    df_ok = df[~df[strat_col].isin(rare)]

    if len(df_ok) < 10 or df_ok[strat_col].nunique() < 2:
        train, tmp = train_test_split(df, test_size=(test_size + val_size), random_state=seed)
        val, test = train_test_split(tmp, test_size=test_size / (test_size + val_size), random_state=seed)
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    ok_trainval, ok_test = train_test_split(
        df_ok,
        test_size=test_size,
        random_state=seed,
        stratify=df_ok[strat_col]
    )

    val_frac_of_trainval = val_size / (1.0 - test_size)
    ok_train, ok_val = train_test_split(
        ok_trainval,
        test_size=val_frac_of_trainval,
        random_state=seed,
        stratify=ok_trainval[strat_col]
    )

    train = pd.concat([ok_train, df_rare], ignore_index=True)
    return train.reset_index(drop=True), ok_val.reset_index(drop=True), ok_test.reset_index(drop=True)


def apply_label_vocab(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_freq: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MultiLabelBinarizer]:
    """
    Build label vocabulary from train only and apply to all splits.
    Filters out labels with frequency < min_freq.
    """
    freq = pd.Series([t for tags in train_df["tags_list"] for t in tags]).value_counts()
    keep = set(freq[freq >= min_freq].index)
    
    print(f"Labels before filtering: {len(freq)}, after (freq >= {min_freq}): {len(keep)}")

    def filter_tags(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tags_list"] = df["tags_list"].apply(lambda ts: [t for t in ts if t in keep])
        df = df[df["tags_list"].map(len) > 0].reset_index(drop=True)
        return df

    train_df = filter_tags(train_df)
    val_df = filter_tags(val_df)
    test_df = filter_tags(test_df)

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train_df["tags_list"])
    y_val = mlb.transform(val_df["tags_list"])
    y_test = mlb.transform(test_df["tags_list"])

    train_df = train_df[["product_sku", "good_type", "text"]].copy()
    val_df = val_df[["product_sku", "good_type", "text"]].copy()
    test_df = test_df[["product_sku", "good_type", "text"]].copy()

    train_df["labels"] = list(y_train)
    val_df["labels"] = list(y_val)
    test_df["labels"] = list(y_test)

    return train_df, val_df, test_df, mlb


def load_allowed_tags_by_type(path_csv: str) -> Dict[str, Set[str]]:
    """
    Load allowed tags per good_type from CSV.
    CSV format: good_type, tags (semicolon-separated)
    """
    adf = pd.read_csv(path_csv)
    m = {}
    for _, r in adf.iterrows():
        gt = str(r["good_type"]).strip()
        tags = [t.strip() for t in str(r["tags"]).split(";") if t.strip()]
        m[gt] = set(tags)
    return m


def allowed_mask_for_good_type(
    good_type: str,
    mlb_classes: np.ndarray,
    allowed_tags_by_type: Dict[str, Set[str]]
) -> np.ndarray:
    """Get binary mask for allowed tags given good_type."""
    gt = str(good_type).strip()
    allowed = allowed_tags_by_type.get(gt)
    if not allowed:
        return np.ones(len(mlb_classes), dtype=np.float32)
    return np.array([1.0 if tag in allowed else 0.0 for tag in mlb_classes], dtype=np.float32)


def filter_probs_by_good_type(
    probs: np.ndarray,
    good_types: List[str],
    mlb: MultiLabelBinarizer,
    allowed_tags_by_type: Dict[str, Set[str]]
) -> np.ndarray:
    """Filter prediction probabilities by allowed tags for each good_type."""
    out = probs.copy()
    for i, gt in enumerate(good_types):
        mask = allowed_mask_for_good_type(gt, mlb.classes_, allowed_tags_by_type)
        out[i] *= mask
    return out


class ChunkedDataset:
    """Creates chunked dataset for long texts."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        stride: int = 128
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
    
    def create(self) -> Dataset:
        """Create HuggingFace Dataset with chunked examples."""
        rows = []
        for _, r in self.df.iterrows():
            text = r["text"]
            labels = np.asarray(r["labels"], dtype=np.float32)
            sku = r["product_sku"]
            good_type = r["good_type"]

            enc = self.tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_overflowing_tokens=True,
                stride=self.stride
            )

            for i in range(len(enc["input_ids"])):
                rows.append({
                    "product_sku": sku,
                    "good_type": good_type,
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": labels
                })

        ds = Dataset.from_list(rows)
        ds.set_format("torch")
        return ds


def load_and_prepare_data(
    config: Config,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset, Dataset, MultiLabelBinarizer, Dict[str, Set[str]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full data loading pipeline.
    Returns train/val/test datasets, MLB, allowed tags, and dataframes.
    """
    data_cfg = config.data
    
    print(f"Loading data from {data_cfg.data_path}...")
    df = pd.read_csv(data_cfg.data_path)
    
    print("Aggregating by product SKU...")
    dfp = aggregate_by_sku(df)
    print(f"Total products: {len(dfp)}")
    
    print("Splitting train/val/test...")
    train_raw, val_raw, test_raw = stratified_split_three_way(
        dfp,
        strat_col="good_type",
        test_size=data_cfg.test_size,
        val_size=data_cfg.val_size,
        seed=config.training.seed
    )
    
    print("Building label vocabulary...")
    train_df, val_df, test_df, mlb = apply_label_vocab(
        train_raw, val_raw, test_raw,
        min_freq=data_cfg.min_label_freq
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"Number of labels: {len(mlb.classes_)}")
    
    print(f"Loading allowed tags from {data_cfg.allowed_tags_path}...")
    allowed_tags = load_allowed_tags_by_type(data_cfg.allowed_tags_path)
    print(f"Loaded allowed tags for {len(allowed_tags)} good types")
    
    print("Creating chunked datasets...")
    train_ds = ChunkedDataset(train_df, tokenizer, data_cfg.max_length, data_cfg.stride).create()
    val_ds = ChunkedDataset(val_df, tokenizer, data_cfg.max_length, data_cfg.stride).create()
    test_ds = ChunkedDataset(test_df, tokenizer, data_cfg.max_length, data_cfg.stride).create()
    
    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}, Test chunks: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds, mlb, allowed_tags, train_df, val_df, test_df

