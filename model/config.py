"""
Configuration and hyperparameters for tag prediction model.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_path: str = "lamoda_reviews_sampled_with_pull_tags.csv"
    allowed_tags_path: str = "tag_pulls_best.csv"
    
    test_size: float = 0.1
    val_size: float = 0.1
    
    min_label_freq: int = 3
    
    max_length: int = 256
    stride: int = 128


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "ai-forever/ruBert-large"
    dropout: float = 0.1
    num_labels: int = 0


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "model/output"
    artifacts_dir: str = "model/artifacts"
    
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 6
    
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    early_stopping_patience: int = 4
    metric_for_best_model: str = "micro_f1"
    
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = 0.25
    
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    asl_clip: float = 0.05
    
    logging_steps: int = 50
    save_total_limit: int = 2
    fp16: bool = True
    seed: int = 42
    
    default_threshold: float = 0.25
    top_k: int = 8

    # Category prior (ensures good_type is used beyond allow-list masking)
    # Applied in inference: logit(p_model) + w * logit(p_prior(good_type, tag))
    category_prior_weight: float = 1.0
    category_prior_smoothing_alpha: float = 1.0


@dataclass
class Config:
    """Main configuration combining all configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.training.artifacts_dir, exist_ok=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()

