#!/usr/bin/env python3
"""
Main training script for multi-label tag prediction model.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import Config, get_config
from model.data import (
    filter_probs_by_good_type,
    load_and_prepare_data,
)
from model.losses import get_loss_function


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


class CustomTrainer(Trainer):
    """Trainer with custom loss function support."""
    
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.custom_loss is not None:
            loss = self.custom_loss(logits, labels)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics_factory(threshold: float = 0.35):
    """Factory for compute_metrics function with configurable threshold."""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid(logits)
        preds = (probs >= threshold).astype(int)
        
        return {
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
            "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        }
    return compute_metrics


def tune_global_threshold(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """Find optimal global threshold on validation set."""
    best_thr, best_f1 = 0.35, -1
    for thr in np.linspace(0.05, 0.95, 37):  # More granular search
        y_hat = (probs >= thr).astype(int)
        f1 = f1_score(y_true, y_hat, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def tune_per_label_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Find optimal threshold per label."""
    n_labels = y_true.shape[1]
    thrs = np.full(n_labels, 0.35, dtype=float)
    
    for j in range(n_labels):
        best_thr, best_f1 = 0.35, -1
        yj = y_true[:, j]
        pj = probs[:, j]
        
        if yj.sum() == 0:
            continue
            
        for thr in np.linspace(0.05, 0.95, 19):
            pred = (pj >= thr).astype(int)
            f1 = f1_score(yj, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        thrs[j] = best_thr
    
    return thrs


@torch.no_grad()
def predict_probs_by_sku(trainer: Trainer, ds, num_labels: int):
    """
    Predict probabilities aggregated by SKU.
    Uses max aggregation across chunks.
    """
    pred = trainer.predict(ds)
    logits = pred.predictions
    probs = sigmoid(logits)
    
    skus = ds["product_sku"]
    sku_to_max = {}
    
    for sku, p in zip(skus, probs):
        if sku not in sku_to_max:
            sku_to_max[sku] = p
        else:
            sku_to_max[sku] = np.maximum(sku_to_max[sku], p)
    
    sku_list = list(sku_to_max.keys())
    probs_by_sku = np.vstack([sku_to_max[sku] for sku in sku_list])
    
    return sku_list, probs_by_sku


def apply_thresholds(
    probs: np.ndarray,
    global_thr: Optional[float] = None,
    per_label_thr: Optional[np.ndarray] = None,
    top_k: int = 8
) -> np.ndarray:
    """Apply thresholds and ensure at least 1, at most top_k predictions."""
    n, L = probs.shape
    
    if per_label_thr is not None:
        thr = np.asarray(per_label_thr).reshape(1, L)
        preds = (probs >= thr).astype(int)
    else:
        thr = 0.35 if global_thr is None else float(global_thr)
        preds = (probs >= thr).astype(int)
    
    for i in range(n):
        idx = np.where(preds[i] == 1)[0]
        if len(idx) > top_k:
            # Keep only top_k by probability
            best = probs[i].argsort()[-top_k:]
            preds[i] = 0
            preds[i, best] = 1
        elif len(idx) == 0:
            # Ensure at least one prediction
            best = probs[i].argsort()[-1:]
            preds[i, best] = 1
    
    return preds


def save_artifacts(
    config: Config,
    mlb,
    global_thr: float,
    per_label_thr: np.ndarray,
    allowed_tags_by_type: Dict
):
    """Save model artifacts for inference."""
    import json
    
    artifacts_dir = Path(config.training.artifacts_dir)
    
    # Save thresholds
    np.save(artifacts_dir / "per_label_thresholds.npy", per_label_thr)
    
    with open(artifacts_dir / "global_threshold.txt", "w") as f:
        f.write(str(global_thr))
    
    # Save label names
    with open(artifacts_dir / "labels.txt", "w", encoding="utf-8") as f:
        for lab in mlb.classes_:
            f.write(lab + "\n")
    
    # Save allowed tags
    allowed_serializable = {k: list(v) for k, v in allowed_tags_by_type.items()}
    with open(artifacts_dir / "allowed_tags_by_type.json", "w", encoding="utf-8") as f:
        json.dump(allowed_serializable, f, ensure_ascii=False, indent=2)
    
    # Save config
    with open(artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": config.model.model_name,
            "max_length": config.data.max_length,
            "stride": config.data.stride,
            "top_k": config.training.top_k,
        }, f, indent=2)
    
    print(f"✓ Artifacts saved to {artifacts_dir}")


def main():
    """Main training pipeline."""
    config = get_config()
    
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("=" * 60)
    print("TAG PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # Load and prepare data
    train_ds, val_ds, test_ds, mlb, allowed_tags, train_df, val_df, test_df = \
        load_and_prepare_data(config, tokenizer)
    
    num_labels = len(mlb.classes_)
    config.model.num_labels = num_labels
    
    print(f"\nLoading model: {config.model.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        hidden_dropout_prob=config.model.dropout,
        attention_probs_dropout_prob=config.model.dropout,
    )
    
    base_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def multilabel_collator(features):
        batch = base_collator(features)
        batch["labels"] = batch["labels"].float()
        return batch
    
    print(f"\nUsing loss: {config.training.loss_type}")
    custom_loss = get_loss_function(config.training, num_labels)
    
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=True,
        fp16=config.training.fp16 and torch.cuda.is_available(),
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        seed=config.training.seed,
        report_to="none",
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=multilabel_collator,
        compute_metrics=compute_metrics_factory(config.training.default_threshold),
        custom_loss=custom_loss,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        ],
    )
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    trainer.train()
    
    best_model_path = Path(config.training.output_dir) / "best_model"
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"✓ Best model saved to {best_model_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    val_skus = val_df["product_sku"].tolist()
    test_skus = test_df["product_sku"].tolist()
    y_val_true = np.vstack(val_df["labels"].values)
    y_test_true = np.vstack(test_df["labels"].values)
    
    val_sku_list, val_probs = predict_probs_by_sku(trainer, val_ds, num_labels)
    test_sku_list, test_probs = predict_probs_by_sku(trainer, test_ds, num_labels)
    
    sku_to_i_val = {s: i for i, s in enumerate(val_sku_list)}
    val_probs_aligned = np.vstack([val_probs[sku_to_i_val[s]] for s in val_skus])
    
    sku_to_i_test = {s: i for i, s in enumerate(test_sku_list)}
    test_probs_aligned = np.vstack([test_probs[sku_to_i_test[s]] for s in test_skus])
    
    print("\nApplying category filter...")
    val_probs_filtered = filter_probs_by_good_type(
        val_probs_aligned, val_df["good_type"].tolist(), mlb, allowed_tags
    )
    test_probs_filtered = filter_probs_by_good_type(
        test_probs_aligned, test_df["good_type"].tolist(), mlb, allowed_tags
    )
    
    print("\nTuning thresholds on validation set...")
    best_thr, best_val_f1 = tune_global_threshold(y_val_true, val_probs_filtered)
    print(f"  Global threshold: {best_thr:.3f} (val micro_f1: {best_val_f1:.4f})")
    
    per_label_thr = tune_per_label_thresholds(y_val_true, val_probs_filtered)
    
    print("\n" + "-" * 40)
    print("TEST RESULTS")
    print("-" * 40)
    
    test_pred_global = apply_thresholds(
        test_probs_filtered, global_thr=best_thr, top_k=config.training.top_k
    )
    test_pred_pl = apply_thresholds(
        test_probs_filtered, per_label_thr=per_label_thr, top_k=config.training.top_k
    )
    
    print(f"\nGlobal threshold ({best_thr:.3f}):")
    print(f"  micro_f1:  {f1_score(y_test_true, test_pred_global, average='micro', zero_division=0):.4f}")
    print(f"  macro_f1:  {f1_score(y_test_true, test_pred_global, average='macro', zero_division=0):.4f}")
    print(f"  precision: {precision_score(y_test_true, test_pred_global, average='micro', zero_division=0):.4f}")
    print(f"  recall:    {recall_score(y_test_true, test_pred_global, average='micro', zero_division=0):.4f}")
    
    print(f"\nPer-label thresholds:")
    print(f"  micro_f1:  {f1_score(y_test_true, test_pred_pl, average='micro', zero_division=0):.4f}")
    print(f"  macro_f1:  {f1_score(y_test_true, test_pred_pl, average='macro', zero_division=0):.4f}")
    print(f"  precision: {precision_score(y_test_true, test_pred_pl, average='micro', zero_division=0):.4f}")
    print(f"  recall:    {recall_score(y_test_true, test_pred_pl, average='micro', zero_division=0):.4f}")
    
    save_artifacts(config, mlb, best_thr, per_label_thr, allowed_tags)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModel: {best_model_path}")
    print(f"Artifacts: {config.training.artifacts_dir}")


if __name__ == "__main__":
    main()

