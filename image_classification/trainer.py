# image_classification/trainer.py
"""
이미지 분류 모델 학습 클래스

ImageClassifierTrainer:
- 모델 학습 및 검증
- 손실 함수 및 옵티마이저 관리
- 학습률 스케줄링
- 조기 종료 (Early Stopping)
- 체크포인트 저장
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple, List
import logging
import time
from datetime import datetime
import json
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from .losses import FocalLoss, WeightedFocalLoss

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ImageClassifierTrainer:
    """이미지 분류 모델 학습 클래스"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: torch.device = None):
        """
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            config: 학습 설정
            device: 디바이스
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        # 디바이스 관리자 사용 (import는 필요시 추가)
        if device is not None:
            self.device = device
        else:
            try:
                from ..utils.device_manager import DeviceManager
                self.device = DeviceManager.get_device()
            except ImportError:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # 학습 설정 추출
        training_config = config.get('training', {})
        self.epochs = training_config.get('epochs', 50)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.patience = training_config.get('patience', 10)
        self.min_delta = training_config.get('min_delta', 0.001)
        self.use_class_weights = training_config.get('use_class_weights', True)
        self.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
        
        # 모델 설정
        model_config = config.get('model', {})
        self.freeze_backbone_epochs = model_config.get('freeze_backbone_epochs', 5)
        self.backbone_lr = training_config.get('backbone_lr', self.learning_rate * 0.1)
        
        # 디렉토리 설정
        paths_config = config.get('paths', {})
        self.model_dir = paths_config.get('model_dir', 'models')
        self.log_dir = paths_config.get('log_dir', 'logs')
        self.checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints')
        
        # 디렉토리 생성 (안전성 체크)
        for dir_name, dir_path in [
            ('model_dir', self.model_dir),
            ('log_dir', self.log_dir), 
            ('checkpoint_dir', self.checkpoint_dir)
        ]:
            if dir_path and isinstance(dir_path, str):
                os.makedirs(dir_path, exist_ok=True)
            else:
                logger.warning(f"유효하지 않은 디렉토리 경로 {dir_name}: {dir_path}")
        
        # 손실 함수 설정
        self.criterion = self._setup_criterion()
        
        # 옵티마이저 설정
        self.optimizer = self._setup_optimizer()
        
        # 스케줄러 설정
        self.scheduler = self._setup_scheduler()
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # 조기 종료 변수
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        # wandb 설정
        self.wandb_enabled = config.get('logging', {}).get('use_wandb', False)
        self.wandb_project = config.get('logging', {}).get('wandb_project', 'image-classification')
        
        logger.info(f"트레이너 초기화 완료 - 디바이스: {self.device}")
        if self.wandb_enabled and WANDB_AVAILABLE:
            logger.info("wandb 로깅 활성화")
        elif self.wandb_enabled and not WANDB_AVAILABLE:
            logger.warning("wandb가 설치되지 않았습니다. 로깅이 비활성화됩니다.")
    
    def _setup_criterion(self) -> nn.Module:
        """손실 함수 설정"""
        training_config = self.config.get('training', {})
        use_focal_loss = training_config.get('focal_loss', False)
        
        if use_focal_loss:
            # Focal Loss 사용
            alpha = training_config.get('focal_alpha', 0.25)
            gamma = training_config.get('focal_gamma', 2.0)
            
            if self.use_class_weights:
                # 클래스 가중치 계산
                class_weights = self._compute_class_weights()
                criterion = WeightedFocalLoss(
                    class_weights=class_weights,
                    alpha=alpha,
                    gamma=gamma
                )
                logger.info(f"클래스 가중치를 사용한 Focal Loss 설정 (α={alpha}, γ={gamma})")
            else:
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
                logger.info(f"Focal Loss 설정 (α={alpha}, γ={gamma})")
        else:
            # 표준 Cross Entropy 사용
            if self.use_class_weights:
                class_weights = self._compute_class_weights()
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info("클래스 가중치를 사용한 CrossEntropyLoss 설정")
            else:
                criterion = nn.CrossEntropyLoss()
                logger.info("표준 CrossEntropyLoss 설정")
        
        return criterion
    
    def _compute_class_weights(self) -> torch.Tensor:
        """클래스 가중치 계산"""
        # 학습 데이터에서 클래스 분포 계산
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.numpy())
        
        # sklearn을 사용한 클래스 가중치 계산
        unique_classes = np.unique(all_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=all_labels
        )
        
        # Tensor로 변환하여 device로 이동
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        logger.info(f"클래스 가중치: {class_weights.cpu().numpy()}")
        
        return class_weights
    
    def _param_groups(self):
        """백본과 헤드로 파라미터 그룹 분리"""
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': head_params, 'lr': self.learning_rate},
            {'params': backbone_params, 'lr': self.backbone_lr}
        ]
        
        logger.debug(f"파라미터 그룹 - 헤드: {len(head_params)}개, 백본: {len(backbone_params)}개")
        return param_groups
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """옵티마이저 설정 (파라미터 그룹별 학습률)"""
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adamw').lower()
        param_groups = self._param_groups()
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"지원하지 않는 옵티마이저: {optimizer_name}")
        
        logger.info(f"옵티마이저 설정: {optimizer_name.upper()}, 헤드 LR: {self.learning_rate}, 백본 LR: {self.backbone_lr}")
        return optimizer
    
    def _rebuild_optimizer_and_scheduler(self):
        """freeze/unfreeze 시 옵티마이저와 스케줄러 재구성"""
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        logger.info("옵티마이저와 스케줄러 재구성 완료")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """학습률 스케줄러 설정"""
        scheduler_name = self.config.get('training', {}).get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif scheduler_name == 'plateau':
            scheduler_patience = self.config.get('training', {}).get('scheduler_patience', 7)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.3, patience=scheduler_patience, verbose=True
            )
        elif scheduler_name == 'none':
            scheduler = None
        else:
            raise ValueError(f"지원하지 않는 스케줄러: {scheduler_name}")
        
        if scheduler:
            logger.info(f"스케줄러 설정: {scheduler_name.upper()}")
        else:
            logger.info("스케줄러 사용 안 함")
        
        return scheduler
    
    def train_epoch(self) -> Tuple[float, float]:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            # 통계 계산
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 진행률 업데이트
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation", leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self) -> Dict[str, Any]:
        """전체 학습 프로세스"""
        logger.info("=" * 60)
        logger.info("🚀 학습 시작")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # 초기 freeze 설정
        freeze_epochs = max(0, int(self.freeze_backbone_epochs))
        if freeze_epochs > 0:
            self.model.freeze_backbone()
            # BatchNorm 통계 고정: 백본만 eval 모드
            if hasattr(self.model, 'backbone'):
                self.model.backbone.eval()
            self._rebuild_optimizer_and_scheduler()
            logger.info(f"백본 고정: 처음 {freeze_epochs} 에포크 (BatchNorm 통계 고정)")
        
        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            
            # freeze 기간 종료 직후에 unfreeze
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                self.model.unfreeze_backbone()
                # BatchNorm 학습 재개
                if hasattr(self.model, 'backbone'):
                    self.model.backbone.train()
                self._rebuild_optimizer_and_scheduler()
                logger.info(f"에포크 {epoch}: 백본 고정 해제 및 옵티마이저 재구성")
            
            # 학습
            train_loss, train_acc = self.train_epoch()
            
            # 검증
            val_loss, val_acc = self.validate_epoch()
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 현재 학습률 기록
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # 결과 기록
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # 최고 성능 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # 에포크 시간 계산
            epoch_time = time.time() - epoch_start_time
            
            # 진행 상황 출력
            print(f"Epoch {epoch:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
                  f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # wandb 로깅
            if self.wandb_enabled and WANDB_AVAILABLE and wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time,
                    'best_val_loss': self.best_val_loss,
                    'best_val_acc': self.best_val_acc,
                    'epochs_without_improvement': self.epochs_without_improvement
                })
            
            # 조기 종료 확인
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"조기 종료: {self.patience} 에포크 동안 개선 없음")
                break
        
        # 최고 모델로 복원
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info("최고 성능 모델로 복원")
        
        total_time = time.time() - start_time
        
        # 학습 결과 요약
        training_summary = {
            'total_epochs': epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_acc,
            'training_time': total_time,
            'final_learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        logger.info("=" * 60)
        logger.info("✅ 학습 완료")
        logger.info(f"최고 검증 정확도: {self.best_val_acc:.2f}%")
        logger.info(f"최고 검증 손실: {self.best_val_loss:.4f}")
        logger.info(f"총 학습 시간: {total_time:.1f}초")
        logger.info("=" * 60)
        
        # 결과 저장
        self.save_training_results(training_summary)
        self.plot_training_curves()
        
        return training_summary
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        # 최신 체크포인트 저장
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 모델 저장
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"최고 모델 저장: {best_path}")
    
    def save_training_results(self, summary: Dict[str, Any]):
        """학습 결과 저장"""
        results = {
            'training_summary': summary,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        results_path = os.path.join(self.log_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"학습 결과 저장: {results_path}")
    
    def plot_training_curves(self):
        """학습 곡선 그래프 생성"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 손실 곡선
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 정확도 곡선
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 학습률 곡선
        ax3.plot(epochs, self.learning_rates, 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # 검증 손실 vs 정확도
        ax4.scatter(self.val_losses, self.val_accuracies, alpha=0.6)
        ax4.set_title('Validation Loss vs Accuracy')
        ax4.set_xlabel('Validation Loss')
        ax4.set_ylabel('Validation Accuracy (%)')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 그래프 저장
        plot_path = os.path.join(self.log_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"학습 곡선 저장: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        epoch = checkpoint['epoch']
        logger.info(f"체크포인트 로드: 에포크 {epoch}")
        
        return epoch