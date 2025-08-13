# utils/config_manager.py
"""
설정 관리자 클래스

ConfigValidator:
- 설정 값 검증
- 기본값 설정
- 타입 체크

ConfigManager:
- 설정 파일 로드 및 관리
- 검증된 설정 제공
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigValidator:
    """설정 값 검증 클래스"""
    
    # 기본 설정 값
    DEFAULT_CONFIG = {
        "data": {
            "csv_path": "image_data.csv",
            "base_image_path": "",
            "base_image_url": "",
            "target_column": "is_text_tag",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "random_state": 42,
            "filter_categories": None,
            "min_samples_per_class": 5,
            "max_samples_per_class": None,
            "is_remote": True
        },
        "model": {
            "backbone": "efficientnet_b0",
            "pretrained": True,
            "dropout_rate": 0.3,
            "hidden_dim": None,
            "freeze_backbone_epochs": 5,
            "unfreeze_backbone_epoch": 10
        },
        "training": {
            "batch_size": 32,
            "epochs": 20,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "scheduler_patience": 7,
            "warmup_epochs": 3,
            "patience": 10,
            "min_delta": 0.001,
            "early_stopping": True,
            "use_class_weights": True,
            "gradient_clip_norm": 1.0,
            "focal_loss": False,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0
        },
        "augmentation": {
            "image_size": 224,
            "strength": "medium",
            "horizontal_flip": True,
            "rotation": 10,
            "color_jitter": True,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1
        },
        "validation": {
            "val_frequency": 1,
            "save_best_model": True,
            "save_last_model": True,
            "metric": "f1_weighted"
        },
        "logging": {
            "log_frequency": 10,
            "save_frequency": 5,
            "print_confusion_matrix": True,
            "save_predictions": True,
            "use_wandb": True,
            "wandb_project": "image-type-classification",
            "wandb_entity": "vingle"
        },
        "inference": {
            "confidence_threshold": 0.5,
            "batch_size": 64,
            "save_features": False,
            "return_top_k": 3
        },
        "paths": {
            "data_dir": "data",
            "model_dir": "models",
            "result_dir": "results",
            "log_dir": "logs",
            "checkpoint_dir": "checkpoints",
            "auto_create_dirs": True
        },
        "system": {
            "device": "auto",
            "num_workers": 4,
            "pin_memory": True,
            "mixed_precision": False,
            "compile_model": False,
            "benchmark": True
        }
    }
    
    # 허용되는 값들
    VALID_VALUES = {
        "model.backbone": [
            "efficientnet_b0", "mobilenet_v3_small", 
            "mobilenet_v3_large", "mobilenet_v2"
        ],
        "training.optimizer": ["adam", "adamw", "sgd", "rmsprop"],
        "training.scheduler": ["cosine", "step", "reduce_on_plateau", "none"],
        "augmentation.strength": ["light", "medium", "strong", "none"],
        "validation.metric": ["accuracy", "f1_weighted", "f1_macro", "precision", "recall"],
        "system.device": ["auto", "cpu", "cuda", "mps"]
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        설정 검증 및 기본값 적용
        
        Args:
            config: 입력 설정
            
        Returns:
            검증된 설정
        """
        validated_config = cls._deep_merge(cls.DEFAULT_CONFIG, config)
        
        # 타입 및 값 검증
        cls._validate_types(validated_config)
        cls._validate_values(validated_config)
        cls._validate_ratios(validated_config)
        cls._validate_paths(validated_config)
        
        logger.info("설정 검증 완료")
        return validated_config
    
    @classmethod
    def _deep_merge(cls, default: Dict, custom: Dict) -> Dict:
        """딥 머지"""
        result = default.copy()
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def _validate_types(cls, config: Dict[str, Any]):
        """타입 검증"""
        # train_ratio, val_ratio 합이 1.0 이하인지 확인
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.1)
        
        if not isinstance(train_ratio, (int, float)) or not isinstance(val_ratio, (int, float)):
            raise ValueError("train_ratio와 val_ratio는 숫자여야 합니다")
        
        if train_ratio + val_ratio > 1.0:
            raise ValueError("train_ratio + val_ratio는 1.0 이하여야 합니다")
    
    @classmethod
    def _validate_values(cls, config: Dict[str, Any]):
        """허용값 검증"""
        for path, valid_values in cls.VALID_VALUES.items():
            keys = path.split('.')
            value = config
            
            try:
                for key in keys:
                    value = value[key]
                
                if value not in valid_values:
                    raise ValueError(f"{path}의 값 '{value}'는 허용되지 않습니다. 허용값: {valid_values}")
            except KeyError:
                # 키가 없으면 기본값이 적용됨
                pass
    
    @classmethod
    def _validate_ratios(cls, config: Dict[str, Any]):
        """비율 검증"""
        data_config = config.get('data', {})
        train_ratio = data_config.get('train_ratio', 0.8)
        val_ratio = data_config.get('val_ratio', 0.1)
        test_ratio = data_config.get('test_ratio', 0.1)
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:  # 부동소수점 오차 고려
            logger.warning(f"train:val:test 비율의 합이 1.0이 아닙니다: {total_ratio}")
    
    @classmethod
    def _validate_paths(cls, config: Dict[str, Any]):
        """경로 검증"""
        data_config = config.get('data', {})
        csv_path = data_config.get('csv_path', '')
        
        if csv_path and not os.path.exists(csv_path):
            logger.warning(f"CSV 파일을 찾을 수 없습니다: {csv_path}")


class ConfigManager:
    """설정 관리자 클래스"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """설정 파일 로드 및 검증"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                logger.info(f"설정 파일 로드: {self.config_path}")
            else:
                logger.warning(f"설정 파일을 찾을 수 없어 기본값을 사용합니다: {self.config_path}")
                raw_config = {}
            
            # 설정 검증 및 기본값 적용
            self._config = ConfigValidator.validate_config(raw_config)
            
        except json.JSONDecodeError as e:
            logger.error(f"설정 파일 파싱 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """검증된 설정 반환"""
        return self._config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """특정 섹션 설정 반환"""
        return self._config.get(section, {}).copy()
    
    def get_value(self, path: str, default: Any = None) -> Any:
        """점 표기법으로 값 가져오기 (예: 'data.train_ratio')"""
        keys = path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def save_config(self, output_path: Optional[str] = None):
        """설정을 파일로 저장"""
        save_path = output_path or self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"설정 저장: {save_path}")
