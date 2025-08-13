# image_classification/cnn_model.py
"""
CNN 기반 이미지 분류 모델

EfficientNet 백본을 사용한 이미지 분류:
- ImageClassifier: EfficientNet 기반 분류기
- 구조: EfficientNet backbone → Global Average Pooling → FC layers → output
- 사전 훈련된 모델을 활용한 전이 학습

입력: RGB 이미지 (224x224)
출력: 이진 분류 확률 (text_tag_image vs others)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# 경량 모델별 피처 차원 (efficientnet_b0 수준 이하만 유지)
EFFICIENTNET_FEATURES = {
    'efficientnet_b0': 1280,  # ~5.3M params, 기준 모델
}

# 기타 경량 모델별 피처 차원
OTHER_MODEL_FEATURES = {
    'mobilenet_v3_small': 576,    # ~2.5M params, 초경량
    'mobilenet_v3_large': 960,    # ~5.4M params, B0와 유사
    'mobilenet_v2': 1280,         # ~3.5M params, 안정성 높음
}

class ImageClassifier(nn.Module):
    """
    EfficientNet 기반 이미지 분류 모델
    
    사전 훈련된 EfficientNet을 백본으로 사용하여
    이미지가 text_tag_image인지 분류하는 이진 분류 모델
    """
    
    def __init__(self, 
                 num_classes: int,
                 backbone: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 hidden_dim: Optional[int] = None):
        super(ImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate
        
        # EfficientNet 백본 로드
        self.backbone = self._load_backbone(backbone, pretrained)
        
        # 백본의 출력 차원 계산
        self.feature_dim = self._get_feature_dim()
        
        # 분류 헤드 구성
        if hidden_dim is None:
            hidden_dim = max(512, self.feature_dim // 2)
        
        # BatchNorm 대신 LayerNorm 사용 (배치 크기 1일 때 안정성)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # BatchNorm1d 대신 LayerNorm 사용
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        logger.info(f"모델 초기화 완료: {backbone}, {num_classes}개 클래스")
        logger.info(f"피처 차원: {self.feature_dim}, 은닉 차원: {hidden_dim}")
    
    def _load_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """경량 백본 모델 로드"""
        if backbone == 'efficientnet_b0':
            if pretrained:
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                model = models.efficientnet_b0(weights=weights)
            else:
                model = models.efficientnet_b0(weights=None)
            model.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v3_small':
            if pretrained:
                weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                model = models.mobilenet_v3_small(weights=weights)
            else:
                model = models.mobilenet_v3_small(weights=None)
            model.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v3_large':
            if pretrained:
                weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
                model = models.mobilenet_v3_large(weights=weights)
            else:
                model = models.mobilenet_v3_large(weights=None)
            model.classifier = nn.Identity()
            
        elif backbone == 'mobilenet_v2':
            if pretrained:
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
                model = models.mobilenet_v2(weights=weights)
            else:
                model = models.mobilenet_v2(weights=None)
            model.classifier = nn.Identity()
            
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}. 지원 모델: {list(EFFICIENTNET_FEATURES.keys()) + list(OTHER_MODEL_FEATURES.keys())}")
        
        return model
    
    def _get_feature_dim(self) -> int:
        """백본의 출력 피처 차원 계산 (하드코딩으로 성능 개선)"""
        # EfficientNet 모델들
        if self.backbone_name in EFFICIENTNET_FEATURES:
            return EFFICIENTNET_FEATURES[self.backbone_name]
        
        # 기타 모델들
        if self.backbone_name in OTHER_MODEL_FEATURES:
            return OTHER_MODEL_FEATURES[self.backbone_name]
        
        # 알려지지 않은 모델의 경우 더미 입력으로 계산 (fallback)
        logger.warning(f"알려지지 않은 모델 {self.backbone_name}, 더미 입력으로 피처 차원 계산")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            return features.shape[1]
    
    def _initialize_weights(self):
        """분류 헤드 가중치 초기화"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 [batch_size, 3, 224, 224]
            
        Returns:
            클래스별 로짓 [batch_size, num_classes]
        """
        # 백본을 통한 피처 추출
        features = self.backbone(x)
        
        # 피처가 4차원이면 Global Average Pooling 적용
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # 분류
        logits = self.classifier(features)
        
        return logits
    
    def freeze_backbone(self):
        """백본 가중치 고정"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("백본 가중치 고정")
    
    def unfreeze_backbone(self):
        """백본 가중치 해제"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("백본 가중치 해제")
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """피처만 추출 (분류 없이)"""
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            return features


def create_image_classifier(config: Dict[str, Any], num_classes: int) -> ImageClassifier:
    """
    설정에 따라 이미지 분류기 생성
    
    Args:
        config: 모델 설정
        num_classes: 클래스 수
        
    Returns:
        ImageClassifier 인스턴스
    """
    model_config = config.get('model', {})
    
    backbone = model_config.get('backbone', 'efficientnet_b0')
    pretrained = model_config.get('pretrained', True)
    dropout_rate = model_config.get('dropout_rate', 0.3)
    hidden_dim = model_config.get('hidden_dim', None)
    
    model = ImageClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim
    )
    
    return model


def get_model_summary(model: ImageClassifier, input_size: tuple = (3, 224, 224)) -> Dict[str, Any]:
    """
    모델 요약 정보 생성
    
    Args:
        model: 모델 인스턴스
        input_size: 입력 크기
        
    Returns:
        모델 요약 정보
    """
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 계산 (MB)
    param_size = total_params * 4  # float32 = 4 bytes
    model_size_mb = param_size / (1024 * 1024)
    
    # 출력 크기 계산
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        output = model(dummy_input)
        output_size = output.shape
    
    summary = {
        'model_name': model.backbone_name,
        'num_classes': model.num_classes,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'input_size': input_size,
        'output_size': tuple(output_size),
        'feature_dim': model.feature_dim,
        'dropout_rate': model.dropout_rate
    }
    
    return summary


def print_model_summary(model: ImageClassifier, input_size: tuple = (3, 224, 224)):
    """모델 요약 정보 출력"""
    summary = get_model_summary(model, input_size)
    
    print("=" * 60)
    print("📊 모델 요약")
    print("=" * 60)
    print(f"모델명: {summary['model_name']}")
    print(f"클래스 수: {summary['num_classes']:,}")
    print(f"전체 파라미터: {summary['total_parameters']:,}")
    print(f"학습 가능 파라미터: {summary['trainable_parameters']:,}")
    print(f"모델 크기: {summary['model_size_mb']:.2f} MB")
    print(f"입력 크기: {summary['input_size']}")
    print(f"출력 크기: {summary['output_size']}")
    print(f"피처 차원: {summary['feature_dim']:,}")
    print(f"드롭아웃 비율: {summary['dropout_rate']}")
    print("=" * 60)


# 사용 가능한 경량 백본 모델들 
AVAILABLE_BACKBONES = {
    'efficientnet_b0': 'EfficientNet-B0 (~5.3M params, 기준 모델)',
    'mobilenet_v3_small': 'MobileNet-V3 Small (~2.5M params, 초경량)',
    'mobilenet_v3_large': 'MobileNet-V3 Large (~5.4M params, B0 수준)',
    'mobilenet_v2': 'MobileNet-V2 (~3.5M params, 안정적)'
}

def list_available_backbones():
    """사용 가능한 백본 모델 목록 출력"""
    print("🔧 사용 가능한 백본 모델:")
    for backbone, description in AVAILABLE_BACKBONES.items():
        print(f"  - {backbone}: {description}")