# image_classification/evaluator.py
"""
이미지 분류 모델 평가 클래스

ModelEvaluator:
- 모델 성능 평가 (정확도, F1-score, 정밀도, 재현율)
- 혼동 행렬 생성
- 클래스별 성능 분석
- 예측 결과 저장
- 시각화
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, 
                 model: nn.Module,
                 test_loader: DataLoader,
                 class_names: List[str],
                 device: torch.device = None,
                 save_dir: str = 'results',
                 class_thresholds: Optional[Dict[str, float]] = None):
        """
        Args:
            model: 평가할 모델
            test_loader: 테스트 데이터 로더
            class_names: 클래스 이름 목록
            device: 디바이스
            save_dir: 결과 저장 디렉토리
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        # 디바이스 관리자 사용 (import는 필요시 추가)
        if device is not None:
            self.device = device
        else:
            try:
                from ..utils.device_manager import DeviceManager
                self.device = DeviceManager.get_device()
            except ImportError:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.num_classes = len(class_names)
        self.class_thresholds = self._normalize_class_thresholds(class_thresholds)
        
        # 결과 저장 디렉토리 생성 (안전성 체크)
        if save_dir and isinstance(save_dir, str):
            os.makedirs(save_dir, exist_ok=True)
        else:
            logger.warning(f"유효하지 않은 저장 디렉토리: {save_dir}")
            self.save_dir = 'results'  # 기본값으로 설정
            os.makedirs(self.save_dir, exist_ok=True)
        
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        logger.info(f"평가기 초기화 완료 - 디바이스: {self.device}")
        logger.info(f"클래스 수: {self.num_classes}, 테스트 샘플: {len(test_loader.dataset)}")
        if self.class_thresholds is not None:
            logger.info("클래스별 threshold 적용 평가 활성화")

    def _normalize_class_thresholds(self, class_thresholds: Optional[Dict[str, float]]):
        """클래스별 threshold를 클래스 순서에 맞게 정렬"""
        if not class_thresholds:
            return None

        if isinstance(class_thresholds, dict):
            missing = [name for name in self.class_names if name not in class_thresholds]
            if missing:
                missing_str = ", ".join(missing)
                logger.warning(f"class_thresholds에 누락된 클래스가 있습니다: {missing_str}")
                return None
            return [float(class_thresholds[name]) for name in self.class_names]

        if isinstance(class_thresholds, (list, tuple, np.ndarray)):
            if len(class_thresholds) != self.num_classes:
                logger.warning("class_thresholds 길이가 클래스 수와 일치하지 않습니다.")
                return None
            return [float(value) for value in class_thresholds]

        logger.warning("class_thresholds 형식이 올바르지 않아 무시합니다.")
        return None

    def _select_label_with_thresholds(self, probabilities: np.ndarray) -> int:
        """클래스별 threshold를 반영해 예측 라벨 선택"""
        if self.class_thresholds is None:
            return int(np.argmax(probabilities))

        thresholds = np.array(self.class_thresholds, dtype=float)
        if thresholds.shape[0] != probabilities.shape[0]:
            return int(np.argmax(probabilities))

        passed = probabilities >= thresholds
        if np.any(passed):
            masked = np.where(passed, probabilities, -np.inf)
            return int(np.argmax(masked))

        return int(np.argmax(probabilities))
    
    def predict(self, return_probabilities: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        모델 예측 수행
        
        Args:
            return_probabilities: 확률값 반환 여부
            
        Returns:
            (true_labels, predicted_labels, probabilities)
        """
        self.model.eval()
        
        all_true_labels = []
        all_predicted_labels = []
        all_probabilities = []
        
        logger.info("예측 수행 중...")
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Predicting"):
                data, target = data.to(self.device), target.to(self.device)
                
                # 예측
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 결과 저장
                all_true_labels.extend(target.cpu().numpy())
                probabilities_np = probabilities.cpu().numpy()
                if self.class_thresholds is None:
                    _, predicted = torch.max(outputs, 1)
                    all_predicted_labels.extend(predicted.cpu().numpy())
                else:
                    predicted_labels = [
                        self._select_label_with_thresholds(probs)
                        for probs in probabilities_np
                    ]
                    all_predicted_labels.extend(predicted_labels)

                if return_probabilities:
                    all_probabilities.extend(probabilities_np)
        
        true_labels = np.array(all_true_labels)
        predicted_labels = np.array(all_predicted_labels)
        probabilities = np.array(all_probabilities) if return_probabilities else None
        
        logger.info(f"예측 완료: {len(true_labels)}개 샘플")
        
        return true_labels, predicted_labels, probabilities
    
    def evaluate(self, save_results: bool = True) -> Dict[str, Any]:
        """
        전체 평가 수행
        
        Args:
            save_results: 결과 저장 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        logger.info("=" * 60)
        logger.info("📊 모델 평가 시작")
        logger.info("=" * 60)
        
        # 예측 수행
        true_labels, predicted_labels, probabilities = self.predict(return_probabilities=True)
        
        # 기본 메트릭 계산
        metrics = self._calculate_metrics(true_labels, predicted_labels, probabilities)
        
        # 클래스별 성능 분석
        class_report = self._generate_classification_report(true_labels, predicted_labels)
        
        # 혼동 행렬 생성
        confusion_mat = self._generate_confusion_matrix(true_labels, predicted_labels)
        
        # 전체 평가 결과
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(true_labels),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': confusion_mat.tolist(),
            'predictions': {
                'true_labels': true_labels.tolist(),
                'predicted_labels': predicted_labels.tolist(),
                'probabilities': probabilities.tolist() if probabilities is not None else None
            }
        }
        
        # 결과 출력
        self._print_evaluation_results(evaluation_results)
        
        # 시각화
        if save_results:
            self._save_results(evaluation_results)
            self._plot_confusion_matrix(confusion_mat)
            self._plot_classification_report(class_report)
            self._plot_probability_distribution(probabilities, true_labels, predicted_labels)
        
        logger.info("=" * 60)
        logger.info("✅ 모델 평가 완료")
        logger.info("=" * 60)
        
        return evaluation_results
    
    def _calculate_metrics(self, true_labels: np.ndarray, predicted_labels: np.ndarray, 
                          probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """성능 메트릭 계산"""
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision_macro': precision_score(true_labels, predicted_labels, average='macro', zero_division=0),
            'precision_micro': precision_score(true_labels, predicted_labels, average='micro', zero_division=0),
            'precision_weighted': precision_score(true_labels, predicted_labels, average='weighted', zero_division=0),
            'recall_macro': recall_score(true_labels, predicted_labels, average='macro', zero_division=0),
            'recall_micro': recall_score(true_labels, predicted_labels, average='micro', zero_division=0),
            'recall_weighted': recall_score(true_labels, predicted_labels, average='weighted', zero_division=0),
            'f1_macro': f1_score(true_labels, predicted_labels, average='macro', zero_division=0),
            'f1_micro': f1_score(true_labels, predicted_labels, average='micro', zero_division=0),
            'f1_weighted': f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        }
        
        # AUC 계산 (다중 클래스)
        if probabilities is not None and self.num_classes > 2:
            try:
                # 원-핫 인코딩
                true_labels_bin = label_binarize(true_labels, classes=range(self.num_classes))
                auc_scores = []
                
                for i in range(self.num_classes):
                    if len(np.unique(true_labels_bin[:, i])) > 1:  # 클래스가 실제로 존재하는 경우만
                        auc = roc_auc_score(true_labels_bin[:, i], probabilities[:, i])
                        auc_scores.append(auc)
                
                if auc_scores:
                    metrics['auc_macro'] = np.mean(auc_scores)
                    metrics['auc_weighted'] = np.average(auc_scores, weights=[
                        np.sum(true_labels == i) for i in range(self.num_classes)
                        if np.sum(true_labels == i) > 0
                    ])
                    
            except Exception as e:
                logger.warning(f"AUC 계산 실패: {e}")
        
        return metrics
    
    def _generate_classification_report(self, true_labels: np.ndarray, 
                                      predicted_labels: np.ndarray) -> Dict[str, Any]:
        """분류 리포트 생성"""
        report = classification_report(
            true_labels, predicted_labels,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        return report
    
    def _generate_confusion_matrix(self, true_labels: np.ndarray, 
                                 predicted_labels: np.ndarray) -> np.ndarray:
        """혼동 행렬 생성"""
        cm = confusion_matrix(true_labels, predicted_labels)
        return cm
    
    def _print_evaluation_results(self, results: Dict[str, Any]):
        """평가 결과 출력"""
        metrics = results['metrics']
        
        print(f"\n📈 전체 성능 메트릭:")
        print(f"  정확도 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"  정밀도 (Precision):")
        print(f"    - Macro: {metrics['precision_macro']:.4f}")
        print(f"    - Weighted: {metrics['precision_weighted']:.4f}")
        print(f"  재현율 (Recall):")
        print(f"    - Macro: {metrics['recall_macro']:.4f}")
        print(f"    - Weighted: {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:")
        print(f"    - Macro: {metrics['f1_macro']:.4f}")
        print(f"    - Weighted: {metrics['f1_weighted']:.4f}")
        
        if 'auc_macro' in metrics:
            print(f"  AUC:")
            print(f"    - Macro: {metrics['auc_macro']:.4f}")
            print(f"    - Weighted: {metrics['auc_weighted']:.4f}")
        
        # 클래스별 성능
        print(f"\n📊 클래스별 성능:")
        class_report = results['classification_report']
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"  {class_name}:")
                print(f"    - Precision: {metrics['precision']:.4f}")
                print(f"    - Recall: {metrics['recall']:.4f}")
                print(f"    - F1-Score: {metrics['f1-score']:.4f}")
                print(f"    - Support: {metrics['support']}")
    
    def _save_results(self, results: Dict[str, Any]):
        """평가 결과 저장"""
        # JSON으로 저장
        results_path = os.path.join(self.save_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 요약 CSV 저장
        summary_data = []
        class_report = results['classification_report']
        
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                summary_data.append({
                    'class': class_name,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1-score'],
                    'support': metrics['support']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.save_dir, 'class_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        
        logger.info(f"평가 결과 저장: {results_path}")
        logger.info(f"클래스별 성능 요약: {summary_path}")
    
    def _plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        """혼동 행렬 시각화"""
        plt.figure(figsize=(10, 8))
        
        # 정규화된 혼동 행렬
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 저장
        cm_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"혼동 행렬 저장: {cm_path}")
    
    def _plot_classification_report(self, class_report: Dict[str, Any]):
        """분류 리포트 시각화"""
        # 클래스별 메트릭 추출
        metrics_data = []
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                metrics_data.append({
                    'Class': class_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score']
                })
        
        df = pd.DataFrame(metrics_data)
        
        # 그래프 생성
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 메트릭별 막대 그래프
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax1.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax1.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax1.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 히트맵
        metrics_matrix = df[['Precision', 'Recall', 'F1-Score']].values.T
        sns.heatmap(
            metrics_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=self.class_names,
            yticklabels=['Precision', 'Recall', 'F1-Score'],
            ax=ax2,
            cbar_kws={'label': 'Score'}
        )
        ax2.set_title('Performance Heatmap')
        ax2.set_xlabel('Classes')
        
        plt.tight_layout()
        
        # 저장
        report_path = os.path.join(self.save_dir, 'classification_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"분류 리포트 저장: {report_path}")
    
    def _plot_probability_distribution(self, probabilities: np.ndarray, 
                                     true_labels: np.ndarray, 
                                     predicted_labels: np.ndarray):
        """예측 확률 분포 시각화"""
        if probabilities is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # 최대 확률 분포
        max_probs = np.max(probabilities, axis=1)
        correct_mask = (true_labels == predicted_labels)
        
        axes[0].hist([max_probs[correct_mask], max_probs[~correct_mask]], 
                    bins=30, alpha=0.7, label=['Correct', 'Incorrect'])
        axes[0].set_xlabel('Maximum Probability')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Distribution of Maximum Probabilities')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 클래스별 평균 확률
        class_avg_probs = []
        for i, class_name in enumerate(self.class_names):
            class_mask = (true_labels == i)
            if np.sum(class_mask) > 0:
                avg_prob = np.mean(probabilities[class_mask, i])
                class_avg_probs.append(avg_prob)
            else:
                class_avg_probs.append(0)
        
        axes[1].bar(range(len(self.class_names)), class_avg_probs)
        axes[1].set_xlabel('Classes')
        axes[1].set_ylabel('Average Probability')
        axes[1].set_title('Average Probability for True Class')
        axes[1].set_xticks(range(len(self.class_names)))
        axes[1].set_xticklabels(self.class_names, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 확률 엔트로피 분포
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
        axes[2].hist([entropy[correct_mask], entropy[~correct_mask]], 
                    bins=30, alpha=0.7, label=['Correct', 'Incorrect'])
        axes[2].set_xlabel('Entropy')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Prediction Entropy Distribution')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 확신도별 정확도
        prob_bins = np.arange(0, 1.1, 0.1)
        accuracies = []
        for i in range(len(prob_bins) - 1):
            mask = (max_probs >= prob_bins[i]) & (max_probs < prob_bins[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(correct_mask[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        axes[3].plot(prob_bins[:-1], accuracies, marker='o')
        axes[3].set_xlabel('Probability Bin')
        axes[3].set_ylabel('Accuracy')
        axes[3].set_title('Accuracy vs Confidence')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        prob_path = os.path.join(self.save_dir, 'probability_analysis.png')
        plt.savefig(prob_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"확률 분석 저장: {prob_path}")
    
    def evaluate_single_image(self, image_tensor: torch.Tensor, 
                            return_top_k: int = 3) -> Dict[str, Any]:
        """단일 이미지 예측"""
        self.model.eval()
        
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
            
            image_tensor = image_tensor.to(self.device)
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            
            # Top-K 예측
            top_indices = np.argsort(probabilities)[::-1][:return_top_k]
            
            predictions = []
            for idx in top_indices:
                predictions.append({
                    'class': self.class_names[idx],
                    'class_idx': int(idx),
                    'probability': float(probabilities[idx])
                })
            
            result = {
                'top_predictions': predictions,
                'all_probabilities': probabilities.tolist()
            }
            
            return result
