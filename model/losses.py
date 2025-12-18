"""
Custom loss functions for multi-label classification with class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    
    Reduces the contribution of easy examples and focuses on hard ones.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples. Default: 2.0
        alpha: Weighting factor. Can be scalar or per-class tensor. Default: 0.25
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] raw model outputs
            targets: [batch_size, num_labels] binary labels (0 or 1)
        """
        probs = torch.sigmoid(logits)
        
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Treats positive and negative samples differently, which helps with
    severe class imbalance in multi-label settings.
    
    Paper: "Asymmetric Loss For Multi-Label Classification" (2021)
    https://arxiv.org/abs/2009.14119
    
    Args:
        gamma_neg: Focusing parameter for negative samples. Default: 4
        gamma_pos: Focusing parameter for positive samples. Default: 1
        clip: Probability margin for hard thresholding negatives. Default: 0.05
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] raw model outputs
            targets: [batch_size, num_labels] binary labels (0 or 1)
        """
        probs = torch.sigmoid(logits)
        
        xs_pos = probs
        xs_neg = 1 - probs
        
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg
        
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w
        
        loss = -loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLabelSoftMarginLossWithWeights(nn.Module):
    """
    Weighted multi-label soft margin loss.
    Applies per-class weights to handle imbalance.
    """
    
    def __init__(self, pos_weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


def get_loss_function(config, num_labels: int, label_frequencies: torch.Tensor = None):
    """
    Factory function to get loss based on config.
    
    Args:
        config: TrainingConfig
        num_labels: number of labels
        label_frequencies: tensor of label counts for computing class weights
    """
    loss_type = config.loss_type.lower()
    
    if loss_type == "focal":
        return FocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            reduction="mean"
        )
    
    elif loss_type == "asl":
        return AsymmetricLoss(
            gamma_neg=config.asl_gamma_neg,
            gamma_pos=config.asl_gamma_pos,
            clip=config.asl_clip,
            reduction="mean"
        )
    
    elif loss_type == "weighted_bce":
        if label_frequencies is not None:
            total = label_frequencies.sum()
            pos_weight = (total - label_frequencies) / (label_frequencies + 1e-8)
            pos_weight = pos_weight.clamp(max=10.0)
        else:
            pos_weight = None
        return MultiLabelSoftMarginLossWithWeights(pos_weight=pos_weight)
    
    else:
        return nn.BCEWithLogitsLoss(reduction="mean")

