#!/usr/bin/env python3
# main.py
"""
이미지 타입 분류 메인 실행 스크립트

CSV 파일 기반 이미지 분류 시스템의 전체 파이프라인을 실행:
1. CSV 데이터 로드 및 전처리
2. 데이터 분할 (train/val/test)
3. 모델 생성 및 학습
4. 평가 및 결과 저장

사용법:
    python main.py                          # 기본 설정으로 실행
    python main.py --config custom.json    # 커스텀 설정으로 실행
    python main.py --mode inference        # 추론 모드
    python main.py --quick-test             # 빠른 테스트
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import torch
import pandas as pd

from typing import Dict, Any, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager
from utils.device_manager import DeviceManager

from utils.logging_utils import setup_project_logging, log_execution_time, handle_exceptions
from utils.env_loader import load_env_once
from image_classification.cnn_model import create_image_classifier, print_model_summary
from image_classification.dataset import create_data_loaders, get_dataset_statistics
from image_classification.trainer import ImageClassifierTrainer
from image_classification.evaluator import ModelEvaluator

# 환경변수 로드 (한 번만)
load_env_once()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageTypeClassificationPipeline:
    """이미지 타입 분류 파이프라인 메인 클래스"""

    def __init__(self, config_path: str = "config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config_path = config_path
        self.start_time = datetime.now()
        
        # 설정 관리자를 통한 설정 로드 및 검증
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 클래스 정보 설정 (config에서 로드)
        self.CLASS_NAMES = self.config.get('data', {}).get('class_names', 
            ['care_label', 'detail_shot', 'full_shot', 'neck_label'])
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        
        # 고유 실행 디렉토리 생성
        base_results_dir = self.config.get('paths', {}).get('result_dir', 'results')
        
        # 고유한 run 폴더 생성
        import uuid
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # 8자리 고유 ID
        self.run_id = f"run_{timestamp}_{unique_id}"
        
        # run 폴더 경로 설정
        run_dir = os.path.join(base_results_dir, self.run_id)
        self.run_paths = {
            'run_dir': run_dir,
            'result_dir': run_dir,
            'model_dir': os.path.join(run_dir, 'model'),
            'log_dir': os.path.join(run_dir, 'logs'),
            'checkpoint_dir': os.path.join(run_dir, 'checkpoints')
        }
        
        # 필요한 디렉토리들 생성
        for path in self.run_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # 설정의 경로들을 새로운 구조에 맞게 업데이트
        self._update_config_paths()
        
        # 실행 시점의 설정 저장
        config_path = os.path.join(self.run_paths['run_dir'], 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        logger.info(f"실행 설정 저장: {config_path}")
        
        # 프로젝트 로깅 설정 (새로운 log 디렉토리 사용)
        setup_project_logging(self.config)
        
        # 디바이스 관리자를 통한 디바이스 설정
        device_config = self.config.get('system', {}).get('device', 'auto')
        self.device = DeviceManager.get_device(device_config)
        
        logger.info("=" * 60)
        logger.info(f"🚀 새로운 실행 시작 - ID: {self.run_id}")
        logger.info(f"📁 실행 디렉토리: {self.run_paths['run_dir']}")
        logger.info(f"💻 디바이스: {self.device}")
        logger.info("=" * 60)
        
        # 디바이스 정보 출력
        device_info = DeviceManager.get_device_info()
        logger.info(f"디바이스 정보: {device_info['name']} ({device_info['memory_gb']})")
        
        # 실행 메타데이터 생성
        metadata = {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'config_file': self.config_path,
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'device_info': DeviceManager.get_device_info()
        }
        metadata_path = os.path.join(self.run_paths['run_dir'], 'run_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"실행 메타데이터 저장: {metadata_path}")
    
    def _update_config_paths(self):
        """설정의 경로들을 새로운 실행 디렉토리 구조에 맞게 업데이트"""
        paths_config = self.config.get('paths', {})
        
        # 기존 경로를 새로운 구조로 매핑
        paths_config['result_dir'] = self.run_paths['result_dir']
        paths_config['model_dir'] = self.run_paths['model_dir']
        paths_config['log_dir'] = self.run_paths['log_dir']
        paths_config['checkpoint_dir'] = self.run_paths['checkpoint_dir']
        
        logger.info(f"경로 업데이트 완료:")
        logger.info(f"  결과: {self.run_paths['result_dir']}")
        logger.info(f"  모델: {self.run_paths['model_dir']}")
        logger.info(f"  로그: {self.run_paths['log_dir']}")

    @log_execution_time()
    @handle_exceptions()
    def load_data(self, data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """미리 분할된 데이터 로드"""
        logger.info("=" * 60)
        logger.info("📂 미리 분할된 데이터 로드")
        logger.info("=" * 60)
        
        # 분할된 데이터 파일 경로
        train_path = os.path.join(data_dir, 'train_data.csv')
        val_path = os.path.join(data_dir, 'validation_data.csv')
        test_path = os.path.join(data_dir, 'test_data.csv')
        
        # 파일 존재 확인
        missing_files = []
        for name, path in [('train', train_path), ('validation', val_path), ('test', test_path)]:
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            logger.error("다음 분할된 데이터 파일들을 찾을 수 없습니다:")
            for missing in missing_files:
                logger.error(f"  - {missing}")
            logger.error("먼저 'python divide_data.py'를 실행하여 데이터를 분할하세요.")
            raise FileNotFoundError("분할된 데이터 파일을 찾을 수 없습니다")
        
        # 데이터 로드 및 셔플링
        logger.info(f"Train 데이터 로드: {train_path}")
        train_df = pd.read_csv(train_path, encoding='utf-8')
        # Train 데이터 셔플링 (같은 제품 이미지들이 연속으로 나오는 것 방지)
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("Train 데이터 셔플링 완료 (제품별 순서 제거)")
        
        logger.info(f"Validation 데이터 로드: {val_path}")
        val_df = pd.read_csv(val_path, encoding='utf-8')
        # Validation은 재현성을 위해 셔플링하지 않음
        
        logger.info(f"Test 데이터 로드: {test_path}")
        test_df = pd.read_csv(test_path, encoding='utf-8')
        # Test도 재현성을 위해 셔플링하지 않음
        
        # 데이터 요약 출력 (4클래스 분류)
        total_samples = len(train_df) + len(val_df) + len(test_df)
        
        # 이미지 타입 분포 확인
        train_type_dist = train_df['image_type'].value_counts()
        val_type_dist = val_df['image_type'].value_counts()
        test_type_dist = test_df['image_type'].value_counts()
        
        logger.info(f"데이터 요약 (4클래스 이미지 타입 분류):")
        logger.info(f"  전체: {total_samples:,}개 이미지")
        logger.info(f"  Train: {len(train_df):,}개 ({len(train_df)/total_samples*100:.1f}%)")
        logger.info(f"  Validation: {len(val_df):,}개 ({len(val_df)/total_samples*100:.1f}%)")
        logger.info(f"  Test: {len(test_df):,}개 ({len(test_df)/total_samples*100:.1f}%)")
        logger.info(f"  클래스: {', '.join(self.CLASS_NAMES)}")
        
        # 이미지 타입 분포 출력
        logger.info(f"이미지 타입 분포:")
        for img_type in sorted(train_type_dist.index):
            logger.info(f"  {img_type}: Train {train_type_dist.get(img_type, 0):,}, Val {val_type_dist.get(img_type, 0):,}, Test {test_type_dist.get(img_type, 0):,}")
        
        # 제품 정보
        train_products = train_df['product_id'].nunique()
        val_products = val_df['product_id'].nunique()
        test_products = test_df['product_id'].nunique()
        total_products = train_products + val_products + test_products
        
        logger.info(f"제품 분할:")
        logger.info(f"  전체: {total_products:,}개 제품")
        logger.info(f"  Train: {train_products:,}개, Val: {val_products:,}개, Test: {test_products:,}개")
        
        # 분할 요약 파일이 있으면 정보 출력
        summary_path = os.path.join(data_dir, 'data_split_summary.json')
        if os.path.exists(summary_path):
            try:
                import json
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    logger.info(f"데이터 분할 정보: {summary['split_info']['created_at']}에 생성")
                    logger.info(f"랜덤 시드: {summary['split_info']['random_state']}")
            except Exception as e:
                logger.warning(f"분할 요약 파일 읽기 실패: {e}")
        
        return train_df, val_df, test_df
    
    @log_execution_time()
    @handle_exceptions()
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Tuple:
        """데이터 로더 생성"""
        logger.info("=" * 60)
        logger.info("🔄 데이터 로더 생성")
        logger.info("=" * 60)
        
        train_loader, val_loader, test_loader, label_encoder = create_data_loaders(
            train_df, val_df, test_df, self.config
        )
        
        # 데이터셋 통계
        train_stats = get_dataset_statistics(train_loader)
        logger.info(f"학습 데이터셋 통계: {train_stats}")
        
        return train_loader, val_loader, test_loader, label_encoder
    
    def create_model(self, num_classes: int):
        """모델 생성"""
        logger.info("=" * 60)
        logger.info("🧠 모델 생성")
        logger.info("=" * 60)
        
        model = create_image_classifier(self.config, num_classes)
        
        # 모델 요약 출력
        print_model_summary(model)
        
        return model
    
    @log_execution_time()
    @handle_exceptions()
    def train_model(self, model, train_loader, val_loader) -> Dict[str, Any]:
        """모델 학습 (향상된 트레이너 사용)"""
        logger.info("=" * 60)
        logger.info("🏋️ 향상된 모델 학습 시작")
        logger.info("=" * 60)
        
        # 향상된 트레이너 생성
        trainer = ImageClassifierTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=self.device
        )
        
        # 학습 실행
        training_results = trainer.train()
        
        return training_results
    
    @log_execution_time()
    @handle_exceptions()
    def evaluate_model(self, model, test_loader, class_names: list) -> Dict[str, Any]:
        """모델 평가"""
        logger.info("=" * 60)
        logger.info("📊 모델 평가")
        logger.info("=" * 60)
        
        # 평가기 생성 (새로운 결과 디렉토리 사용)
        evaluator = ModelEvaluator(
            model=model,
            test_loader=test_loader,
            class_names=class_names,
            device=self.device,
            save_dir=self.run_paths['result_dir']
        )
        
        # 평가 실행
        evaluation_results = evaluator.evaluate(save_results=True)
        
        return evaluation_results
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """전체 학습 파이프라인 실행"""
        logger.info("🚀 이미지 타입 분류 학습 파이프라인 시작")
        
        # wandb 초기화
        wandb_run = None
        if self.config.get('logging', {}).get('use_wandb', False) and WANDB_AVAILABLE:
            try:
                wandb_run = wandb.init(
                    project=self.config.get('logging', {}).get('wandb_project', 'image-classification'),
                    entity=self.config.get('logging', {}).get('wandb_entity', None),
                    config=self.config,
                    name=f"enhanced_{self.run_id}",
                    tags=['image-classification', 'pytorch', 'enhanced', 'discriminative-lr', 'ema']
                )
                logger.info(f"wandb 실행 시작: {wandb.run.url}")
            except Exception as e:
                logger.warning(f"wandb 초기화 실패: {e}")
                wandb_run = None
        
        try:
            # 1. 미리 분할된 데이터 로드
            train_df, val_df, test_df = self.load_data()
            
            # 2. 데이터 로더 생성
            train_loader, val_loader, test_loader, label_encoder = self.create_data_loaders(
                train_df, val_df, test_df
            )
            
            # 3. 모델 생성 (이진 분류)
            model = self.create_model(self.NUM_CLASSES)
            
            # 모델 정보를 메타데이터에 추가
            model_info = {
                'backbone': self.config.get('model', {}).get('backbone', 'unknown'),
                'num_classes': self.NUM_CLASSES,
                'class_names': self.CLASS_NAMES,
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            
            # 4. 모델 학습 (향상된 트레이너)
            training_results = self.train_model(model, train_loader, val_loader)
            
            # 5. 모델 평가
            evaluation_results = self.evaluate_model(model, test_loader, self.CLASS_NAMES)
            
            # 최종 결과 요약
            end_time = datetime.now()
            total_time = (end_time - self.start_time).total_seconds()
            
            final_results = {
                'run_info': {
                    'run_id': self.run_id,
                    'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time,
                    'run_directory': self.run_paths['run_dir']
                },
                'config': self.config,
                'data_summary': {
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'test_samples': len(test_df),
                    'num_classes': self.NUM_CLASSES,
                    'class_names': self.CLASS_NAMES
                },
                'model_info': model_info,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'enhancements_used': {
                    'discriminative_lr': True,
                    'label_smoothing': self.config.get('training', {}).get('label_smoothing', 0) > 0,
                    'ema': self.config.get('training', {}).get('use_ema', False),
                    'warmup': self.config.get('training', {}).get('warmup_epochs', 0) > 0,
                    'mixed_precision': self.config.get('system', {}).get('mixed_precision', False)
                }
            }
            
            # 최종 결과 저장
            self._save_final_results(final_results)
            
            # wandb 최종 결과 로깅
            if wandb_run:
                try:
                    wandb.log({
                        'final/total_time_seconds': total_time,
                        'final/best_val_accuracy': training_results['best_val_accuracy'],
                        'final/test_accuracy': evaluation_results['metrics']['accuracy'],
                        'final/num_classes': self.NUM_CLASSES,
                        'final/train_samples': len(train_df),
                        'final/val_samples': len(val_df),
                        'final/test_samples': len(test_df),
                        'final/label_smoothing_used': final_results['enhancements_used']['label_smoothing'],
                        'final/ema_used': final_results['enhancements_used']['ema'],
                        'final/run_id': self.run_id
                    })
                    # 학습 곡선 이미지가 있다면 wandb에 업로드
                    curve_path = os.path.join(self.run_paths['log_dir'], 'training_curves.png')
                    if os.path.exists(curve_path):
                        wandb.log({"training_curves": wandb.Image(curve_path)})
                except Exception as e:
                    logger.warning(f"wandb 최종 로깅 실패: {e}")
            
            logger.info("=" * 60)
            logger.info("✅ 향상된 파이프라인 완료")
            logger.info(f"🔖 실행 ID: {self.run_id}")
            logger.info(f"📁 결과 폴더: {self.run_paths['run_dir']}")
            logger.info(f"⏱️ 총 실행 시간: {total_time:.1f}초")
            logger.info(f"🎯 최고 검증 정확도: {training_results['best_val_accuracy']:.2f}%")
            logger.info(f"📊 테스트 정확도: {evaluation_results['metrics']['accuracy']:.4f}")
            logger.info("=" * 60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류 발생: {e}")
            raise
        finally:
            # wandb 종료
            if wandb_run:
                try:
                    wandb.finish()
                    logger.info("wandb 실행 종료")
                except Exception as e:
                    logger.warning(f"wandb 종료 실패: {e}")
    
    # run_inference 메서드 제거됨 (test.py에서 대체)
 
    def _save_final_results(self, results: Dict[str, Any]):
        """최종 결과 저장"""
        results_path = os.path.join(self.run_paths['result_dir'], 'final_results.json')
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"최종 결과 저장: {results_path}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='향상된 이미지 타입 분류 시스템 (구조화된 결과 저장)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='설정 파일 경로 (기본값: config.json)')
    parser.add_argument('--quick-test', action='store_true',
                       help='빠른 테스트 (적은 에포크)')
    
    args = parser.parse_args()

    try:
        # 파이프라인 생성
        pipeline = ImageTypeClassificationPipeline(config_path=args.config)
        
        # 빠른 테스트 설정
        if args.quick_test:
            pipeline.config['training']['epochs'] = 5
            pipeline.config['training']['patience'] = 3
            pipeline.config['model']['freeze_backbone_epochs'] = 2
            logger.info("빠른 테스트 모드: 에포크 수 조정")
        
        # 학습 실행
        results = pipeline.run_training_pipeline()
        print(f"\n🎉 향상된 학습 완료!")
        print(f"📁 결과 폴더: {pipeline.run_paths['run_dir']}")
        print(f"🔖 실행 ID: {pipeline.run_id}")
        print(f"📊 최종 정확도: {results['evaluation_results']['metrics']['accuracy']:.4f}")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
        sys.exit(1)
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
