#!/usr/bin/env python3
# test.py
"""
이미지 타입 분류 테스트 실행 스크립트

기존 main.py의 inference 기능과 동일하지만 test_result 폴더를 사용합니다.

사용법:
    python test.py --model-path <모델경로>                    # 기본 추론
    python test.py --model-path <모델경로> --image-path <이미지경로>  # 단일 이미지
    python test.py --model-path <모델경로> --csv-path <CSV경로>      # 배치 추론
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import time
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import uuid

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
from database.csv_connector import CSVConnector
from image_classification.cnn_model import create_image_classifier, get_model_summary, print_model_summary
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

class ImageTypeClassificationTest:
    """이미지 타입 분류 테스트 클래스 (단순 CSV 결과 생성)"""
    
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
        
        # 디바이스 관리자를 통한 디바이스 설정
        self.device = DeviceManager.get_device()
        
        # test_result 폴더에 run 디렉토리 생성
        base_results_dir = "test_result"
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.run_id = f"run_{timestamp}_{unique_id}"
        
        self.run_dir = os.path.join(base_results_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 디바이스 정보 출력
        device_info = DeviceManager.get_device_info()
        logger.info(f"디바이스 정보: {device_info['name']} ({device_info['memory_gb']})")
        logger.info(f"결과 저장 폴더: {self.run_dir}")

    def run_inference(self, image_path: str = None, 
                      csv_path: str = None,
                      output_path: str = None,
                      model_path: str = None) -> str:
        """추론 실행 - 단순 CSV 결과 생성"""
        
        from image_classification.inference import ImageClassifierInference
        
        # 모델 경로 확인
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        logger.info(f"사용할 모델: {model_path}")
        
        # 추론기 생성
        inference_config = self.config.get('inference', {})
        batch_size = inference_config.get('batch_size', 64)
        
        inference_engine = ImageClassifierInference(
            model_path=model_path,
            device=self.device
        )
        
        # 결과를 담을 리스트
        all_results = []
        
        if image_path:
            # 단일 이미지 예측
            logger.info(f"단일 이미지 추론: {image_path}")
            prediction = inference_engine.predict_single(
                image_path=image_path,
                return_probabilities=True,
                top_k=1
            )
            
            if 'error' not in prediction:
                all_results.append({
                    'image_path': image_path,
                    'predicted_class': prediction['predicted_class'],
                    'confidence': round(prediction['confidence'], 3),  # 소숫점 셋째자리
                    'true_class': 'N/A'  # 단일 이미지의 경우 실제 클래스 알 수 없음
                })
            else:
                all_results.append({
                    'image_path': image_path,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'true_class': 'N/A'
                })
                
        elif csv_path:
            # CSV 파일에서 배치 예측
            logger.info(f"배치 추론: {csv_path}")
            df = pd.read_csv(csv_path)
            if 'image_path' not in df.columns:
                raise ValueError("CSV 파일에 'image_path' 컬럼이 필요합니다")
            
            image_paths = df['image_path'].tolist()
            
            start_time = time.time()
            predictions = inference_engine.predict_batch(
                image_paths=image_paths,
                batch_size=batch_size,
                return_probabilities=True,
                top_k=1
            )
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_image = total_time / len(image_paths) if len(image_paths) > 0 else 0
            logger.info(f"배치 추론 시간: 총 {total_time:.2f}초, 개당 평균 {avg_time_per_image:.3f}초")
            
            # true_class 컬럼이 있는지 확인
            has_true_class = 'image_type' in df.columns
            
            for i, pred in enumerate(predictions):
                true_class = df.iloc[i]['image_type'] if has_true_class and i < len(df) else 'N/A'
                
                if 'error' not in pred:
                    all_results.append({
                        'image_path': pred['image_path'],
                        'predicted_class': pred['predicted_class'],
                        'confidence': round(pred['confidence'], 3),  # 소숫점 셋째자리
                        'true_class': true_class
                    })
                else:
                    all_results.append({
                        'image_path': pred['image_path'],
                        'predicted_class': 'ERROR',
                        'confidence': 0.0,
                        'true_class': true_class
                    })
                    
        else:
            # 전체 데이터셋 추론 (train/validation/test 모두)
            logger.info("📊 전체 데이터셋 추론 시작")
            
            data_dir = self.config.get('paths', {}).get('data_dir', 'data')
            splits = ['train', 'validation', 'test']
            
            for split_name in splits:
                split_file = os.path.join(data_dir, f'{split_name}_data.csv')
                
                if not os.path.exists(split_file):
                    logger.warning(f"{split_name} 데이터 파일이 없습니다: {split_file}")
                    continue
                
                logger.info(f"📊 {split_name.upper()} 데이터셋 추론 중...")
                
                try:
                    split_df = pd.read_csv(split_file)
                    image_paths = split_df['image_path'].tolist()
                    
                    start_time = time.time()
                    predictions = inference_engine.predict_batch(
                        image_paths=image_paths,
                        batch_size=batch_size,
                        return_probabilities=True,
                        top_k=1
                    )
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    avg_time_per_image = total_time / len(image_paths) if len(image_paths) > 0 else 0
                    logger.info(f"{split_name.upper()} 추론 시간: 총 {total_time:.2f}초, 개당 평균 {avg_time_per_image:.3f}초")
                    
                    # 정확도 계산
                    if 'image_type' in split_df.columns:
                        correct = 0
                        total = 0
                        for i, pred in enumerate(predictions):
                            if 'error' not in pred and i < len(split_df):
                                actual = split_df.iloc[i]['image_type']
                                predicted = pred['predicted_class']
                                if actual == predicted:
                                    correct += 1
                                total += 1
                        
                        if total > 0:
                            accuracy = correct / total
                            logger.info(f"{split_name.upper()} 정확도: {accuracy:.4f} ({correct}/{total})")
                    
                    # 결과 추가
                    for i, pred in enumerate(predictions):
                        true_class = split_df.iloc[i]['image_type'] if i < len(split_df) else 'N/A'
                        
                        if 'error' not in pred:
                            all_results.append({
                                'image_path': pred['image_path'],
                                'predicted_class': pred['predicted_class'],
                                'confidence': round(pred['confidence'], 3),  # 소숫점 셋째자리
                                'true_class': true_class
                            })
                        else:
                            all_results.append({
                                'image_path': pred['image_path'],
                                'predicted_class': 'ERROR',
                                'confidence': 0.0,
                                'true_class': true_class
                            })
                            
                except Exception as e:
                    logger.error(f"{split_name} 추론 실패: {e}")
        
        # 결과를 DataFrame으로 변환
        result_df = pd.DataFrame(all_results)
        
        # 메인 결과 파일 저장 (run 폴더 안에)
        if output_path:
            output_file = os.path.join(self.run_dir, os.path.basename(output_path))
        else:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.run_dir, f"inference_results_{timestamp}.csv")
        
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 틀린 결과만 모아서 wrong_*.csv 파일 생성
        self._create_wrong_result_csvs(result_df)
        
        logger.info("=" * 60)
        logger.info("✅ 추론 완료")
        logger.info(f"📁 결과 파일: {output_file}")
        logger.info(f"📊 총 {len(all_results)}개 결과")
        logger.info("=" * 60)
        
        return output_file
    
    def _create_wrong_result_csvs(self, result_df: pd.DataFrame):
        """틀린 결과만 모아서 wrong_*.csv 파일들 생성"""
        
        # 전체 결과에서 틀린 것만 추출 (ERROR와 N/A는 제외)
        wrong_results = result_df[
            (result_df['true_class'] != 'N/A') & 
            (result_df['predicted_class'] != 'ERROR') &
            (result_df['predicted_class'] != result_df['true_class'])
        ].copy()
        
        if len(wrong_results) == 0:
            logger.info("🎉 모든 예측이 정확합니다! wrong_*.csv 파일이 생성되지 않습니다.")
            return
        
        # confidence 큰 순으로 정렬
        wrong_results = wrong_results.sort_values('confidence', ascending=False)
        
        # 각 split별로 분리하여 저장
        splits = ['train', 'valid', 'test']
        
        for split in splits:
            # 해당 split의 이미지 경로들을 확인
            if split == 'valid':
                split_file = "data/validation_data.csv"
            else:
                split_file = f"data/{split}_data.csv"
            
            if os.path.exists(split_file):
                try:
                    split_df = pd.read_csv(split_file)
                    split_image_paths = set(split_df['image_path'].tolist())
                    
                    # 해당 split의 틀린 결과만 필터링
                    split_wrong = wrong_results[wrong_results['image_path'].isin(split_image_paths)]
                    
                    if len(split_wrong) > 0:
                        wrong_file = os.path.join(self.run_dir, f"wrong_{split}.csv")
                        split_wrong.to_csv(wrong_file, index=False, encoding='utf-8')
                        logger.info(f"📄 {split.upper()} 틀린 결과: {wrong_file} ({len(split_wrong)}개)")
                    else:
                        logger.info(f"🎉 {split.upper()} 데이터셋: 모든 예측이 정확합니다!")
                        
                except Exception as e:
                    logger.warning(f"⚠️ {split} wrong 결과 생성 실패: {e}")
            else:
                logger.warning(f"⚠️ {split} 데이터 파일이 없습니다: {split_file}")
        
        logger.info(f"📊 전체 틀린 결과: {len(wrong_results)}개 (confidence 높은 순 정렬)")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 타입 분류 테스트 - 단순 CSV 결과 생성')
    parser.add_argument('--config', type=str, default='config.json',
                       help='설정 파일 경로 (기본값: config.json)')
    parser.add_argument('--model-path', type=str, required=True,help='사용할 모델 파일 경로 (필수)')
    parser.add_argument('--image-path', type=str,
                       help='추론할 이미지 경로 (단일 이미지용)')
    parser.add_argument('--csv-path', type=str,
                       help='배치 추론할 CSV 파일 경로')
    parser.add_argument('--output-path', type=str,
                       help='추론 결과 저장 경로 (CSV 파일)')
    
    args = parser.parse_args()
    
    try:
        # 테스트 파이프라인 생성
        test_pipeline = ImageTypeClassificationTest(config_path=args.config)
        
        # 추론 실행
        output_file = test_pipeline.run_inference(
            image_path=args.image_path,
            csv_path=args.csv_path,
            output_path=args.output_path,
            model_path=args.model_path
        )
        
        print(f"\n🔍 추론 완료!")
        print(f"📁 결과 파일: {output_file}")
            
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
