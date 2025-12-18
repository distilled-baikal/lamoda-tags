#!/usr/bin/env python3
"""
Inference utilities for tag prediction model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def logit(p: np.ndarray) -> np.ndarray:
    """Inverse sigmoid with clipping for numerical stability."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class TagPredictor:
    """
    Tag prediction model for inference.
    
    Usage:
        predictor = TagPredictor.load("model/output/best_model", "model/artifacts")
        tags = predictor.predict(
            text="Отличные полотенца, мягкие и хорошо впитывают.",
            good_type="Home_Accs"
        )
    """
    
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        labels: List[str],
        global_threshold: float,
        per_label_thresholds: np.ndarray,
        allowed_tags_by_type: Dict[str, Set[str]],
        prior_logits_by_type: Optional[Dict[str, np.ndarray]] = None,
        category_prior_weight: float = 0.0,
        max_length: int = 256,
        stride: int = 128,
        top_k: int = 8,
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.global_threshold = global_threshold
        self.per_label_thresholds = per_label_thresholds
        self.allowed_tags_by_type = allowed_tags_by_type
        self.prior_logits_by_type = prior_logits_by_type or {}
        self.category_prior_weight = float(category_prior_weight or 0.0)
        self.max_length = max_length
        self.stride = stride
        self.top_k = top_k
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def load(
        cls,
        model_path: str,
        artifacts_path: str,
        device: str = None
    ) -> "TagPredictor":
        """
        Load predictor from saved model and artifacts.
        
        Args:
            model_path: Path to saved model directory
            artifacts_path: Path to artifacts directory
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        model_path = Path(model_path)
        artifacts_path = Path(artifacts_path)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        with open(artifacts_path / "labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        
        global_threshold = float(
            (artifacts_path / "global_threshold.txt").read_text().strip()
        )
        per_label_thresholds = np.load(artifacts_path / "per_label_thresholds.npy")
        
        with open(artifacts_path / "allowed_tags_by_type.json", "r", encoding="utf-8") as f:
            allowed_tags_raw = json.load(f)
        allowed_tags_by_type = {k: set(v) for k, v in allowed_tags_raw.items()}

        prior_logits_by_type: Dict[str, np.ndarray] = {}
        prior_path = artifacts_path / "category_prior_logits.json"
        if prior_path.exists():
            prior_data = json.loads(prior_path.read_text(encoding="utf-8"))
            raw = prior_data.get("prior_logits_by_type", {}) or {}
            for gt, logits_list in raw.items():
                try:
                    prior_logits_by_type[str(gt)] = np.asarray(logits_list, dtype=np.float32)
                except Exception:
                    continue
        
        config_path = artifacts_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            max_length = config.get("max_length", 256)
            stride = config.get("stride", 128)
            top_k = config.get("top_k", 8)
            category_prior_weight = config.get("category_prior_weight", 0.0)
        else:
            max_length, stride, top_k = 256, 128, 8
            category_prior_weight = 0.0
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            global_threshold=global_threshold,
            per_label_thresholds=per_label_thresholds,
            allowed_tags_by_type=allowed_tags_by_type,
            prior_logits_by_type=prior_logits_by_type,
            category_prior_weight=category_prior_weight,
            max_length=max_length,
            stride=stride,
            top_k=top_k,
            device=device
        )
    
    def _get_allowed_mask(self, good_type: str) -> np.ndarray:
        """Get binary mask for allowed tags given good_type."""
        gt = str(good_type).strip()
        allowed = self.allowed_tags_by_type.get(gt)
        if not allowed:
            return np.ones(len(self.labels), dtype=np.float32)
        return np.array([1.0 if tag in allowed else 0.0 for tag in self.labels], dtype=np.float32)
    
    @torch.no_grad()
    def predict_probs(self, text: str) -> np.ndarray:
        """
        Get raw prediction probabilities for text.
        Handles chunking for long texts.
        
        Returns:
            1D array of probabilities [num_labels]
        """
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_overflowing_tokens=True,
            stride=self.stride,
            return_tensors=None
        )
        
        all_probs = []
        for i in range(len(enc["input_ids"])):
            batch = {
                "input_ids": torch.tensor([enc["input_ids"][i]], device=self.device),
                "attention_mask": torch.tensor([enc["attention_mask"][i]], device=self.device),
            }
            logits = self.model(**batch).logits[0].detach().cpu().numpy()
            all_probs.append(sigmoid(logits))
        
        return np.max(np.vstack(all_probs), axis=0)
    
    def predict(
        self,
        text: str,
        good_type: Optional[str] = None,
        use_per_label_threshold: bool = True,
        return_probs: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Predict tags for text.
        
        Args:
            text: Input text (product name + reviews)
            good_type: Product category for filtering allowed tags
            use_per_label_threshold: Use per-label thresholds (True) or global (False)
            return_probs: Include probabilities in output
            
        Returns:
            List of (tag, probability) tuples, sorted by probability descending
        """
        probs = self.predict_probs(text)

        # Apply per-good_type prior in logit space (guarantees good_type influence)
        if good_type and self.category_prior_weight > 0:
            prior_logits = self.prior_logits_by_type.get(str(good_type).strip())
            if prior_logits is not None and len(prior_logits) == len(self.labels):
                probs = sigmoid(logit(probs) + self.category_prior_weight * prior_logits)

        # Apply allow-list mask (hard constraint)
        if good_type:
            mask = self._get_allowed_mask(good_type)
            probs = probs * mask
        
        if use_per_label_threshold:
            preds = (probs >= self.per_label_thresholds).astype(int)
        else:
            preds = (probs >= self.global_threshold).astype(int)
        
        idx = np.where(preds == 1)[0]
        
        if len(idx) == 0:
            idx = probs.argsort()[-1:]
        
        idx = idx[np.argsort(-probs[idx])][:self.top_k]
        
        return [(self.labels[j], float(probs[j])) for j in idx]
    
    def predict_batch(
        self,
        texts: List[str],
        good_types: Optional[List[str]] = None,
        use_per_label_threshold: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict tags for multiple texts.
        
        Args:
            texts: List of input texts
            good_types: List of product categories (same length as texts)
            use_per_label_threshold: Use per-label thresholds
            
        Returns:
            List of predictions, one per input text
        """
        if good_types is None:
            good_types = [None] * len(texts)
        
        return [
            self.predict(text, good_type, use_per_label_threshold)
            for text, good_type in zip(texts, good_types)
        ]


def main():
    """Example usage of TagPredictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict tags for product")
    parser.add_argument("--model", default="model/output/best_model", help="Path to model")
    parser.add_argument("--artifacts", default="model/artifacts", help="Path to artifacts")
    parser.add_argument("--text", required=True, help="Product text (name + reviews)")
    parser.add_argument("--good-type", default=None, help="Product category")
    args = parser.parse_args()
    
    predictor = TagPredictor.load(args.model, args.artifacts)
    
    tags = predictor.predict(
        text=args.text,
        good_type=args.good_type
    )
    
    print("Predicted tags:")
    for tag, prob in tags:
        print(f"  {tag}: {prob:.3f}")


if __name__ == "__main__":
    main()

