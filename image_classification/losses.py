# image_classification/losses.py
"""
커스텀 손실 함수들

- FocalLoss: 클래스 불균형 문제를 해결하는 손실 함수
- WeightedFocalLoss: 클래스 가중치가 적용된 Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _to_tensor_if_sequence(value):
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.float32)
    return value

class FocalLoss(nn.Module):
    """
    Focal Loss 구현
    
    논문: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    클래스 불균형 문제를 해결하기 위해 어려운 예제에 더 많은 가중치를 부여
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 클래스 가중치 파라미터 (0.25가 일반적)
            gamma: focusing 파라미터 (2.0이 일반적)
            reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = _to_tensor_if_sequence(alpha)
        self.gamma = _to_tensor_if_sequence(gamma)
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 예측값 [N, C] (로짓)
            targets: 정답 레이블 [N] (클래스 인덱스)
        
        Returns:
            Focal loss 값
        """
        # Cross entropy loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # p_t 계산 (정답 클래스에 대한 예측 확률)
        p_t = torch.exp(-ce_loss)
        
        # Alpha 가중치 적용
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_tensor = self.alpha.to(targets.device) if isinstance(self.alpha, torch.Tensor) else self.alpha
                alpha_t = alpha_tensor[targets]
        else:
            alpha_t = None

        if isinstance(self.gamma, (float, int)):
            gamma_t = self.gamma
        else:
            gamma_tensor = self.gamma.to(targets.device) if isinstance(self.gamma, torch.Tensor) else self.gamma
            gamma_t = gamma_tensor[targets]

        if alpha_t is not None:
            focal_loss = alpha_t * (1 - p_t) ** gamma_t * ce_loss
        else:
            focal_loss = (1 - p_t) ** gamma_t * ce_loss
        
        # Reduction 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    클래스 가중치가 적용된 Focal Loss
    
    클래스 불균형이 심한 경우 사용
    """
    
    def __init__(self, 
                 class_weights: Optional[torch.Tensor] = None,
                 alpha: float = 0.25, 
                 gamma: float = 2.0, 
                 reduction: str = 'mean'):
        """
        Args:
            class_weights: 각 클래스에 대한 가중치 [C]
            alpha: Focal loss alpha 파라미터
            gamma: Focal loss gamma 파라미터
            reduction: 'mean', 'sum', 'none'
        """
        super(WeightedFocalLoss, self).__init__()
        self.class_weights = class_weights
        self.alpha = _to_tensor_if_sequence(alpha)
        self.gamma = _to_tensor_if_sequence(gamma)
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 예측값 [N, C] (로짓)
            targets: 정답 레이블 [N] (클래스 인덱스)
        
        Returns:
            Weighted Focal loss 값
        """
        # Cross entropy loss with class weights
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # p_t 계산
        p_t = torch.exp(-ce_loss)
        
        # Focal loss 계산
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_tensor = self.alpha.to(targets.device) if isinstance(self.alpha, torch.Tensor) else self.alpha
                alpha_t = alpha_tensor[targets]
        else:
            alpha_t = None

        if isinstance(self.gamma, (float, int)):
            gamma_t = self.gamma
        else:
            gamma_tensor = self.gamma.to(targets.device) if isinstance(self.gamma, torch.Tensor) else self.gamma
            gamma_t = gamma_tensor[targets]

        if alpha_t is not None:
            focal_loss = alpha_t * (1 - p_t) ** gamma_t * ce_loss
        else:
            focal_loss = (1 - p_t) ** gamma_t * ce_loss
        
        # Reduction 적용
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Label Smoothing이 적용된 Cross Entropy Loss
    
    과적합을 방지하고 일반화 성능을 향상시킴
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            smoothing: label smoothing 정도 (0.0 ~ 1.0)
            reduction: 'mean', 'sum', 'none'
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 예측값 [N, C] (로짓)
            targets: 정답 레이블 [N] (클래스 인덱스)
        
        Returns:
            Label smoothing loss 값
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # One-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Label smoothing 적용
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / num_classes
        
        # Loss 계산
        loss = -torch.sum(targets_smooth * log_probs, dim=1)
        
        # Reduction 적용
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
