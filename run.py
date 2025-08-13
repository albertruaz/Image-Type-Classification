#!/usr/bin/env python3
# run.py
"""
백엔드용 이미지 분류 추론 API

이미지 경로 배열을 받아서 결과 배열을 반환하는 간단한 구조
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any
import torch

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.config_manager import ConfigManager
from utils.device_manager import DeviceManager
from utils.env_loader import load_env_once

# 환경변수 로드
load_env_once()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageClassificationAPI:
    """백엔드용 이미지 분류 API"""
    
    def __init__(self, model_path: str, config_path: str = "config.json"):
        """
        Args:
            model_path: 학습된 모델 파일 경로
            config_path: 설정 파일 경로
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # 설정 로드
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # 디바이스 설정
        self.device = DeviceManager.get_device()
        device_info = DeviceManager.get_device_info()
        logger.info(f"디바이스 정보: {device_info['name']} ({device_info['memory_gb']})")
        
        # 추론 엔진 초기화
        self._initialize_inference_engine()
        
        logger.info("🚀 ImageClassificationAPI 초기화 완료")
    
    def _initialize_inference_engine(self):
        """추론 엔진 초기화"""
        from image_classification.inference import ImageClassifierInference
        
        try:
            self.inference_engine = ImageClassifierInference(
                model_path=self.model_path,
                device=self.device
            )
            logger.info(f"✅ 모델 로드 완료: {self.model_path}")
            
            # 모델 정보 출력
            model_info = self.inference_engine.get_model_info()
            logger.info(f"📊 모델 정보:")
            logger.info(f"  - 클래스 수: {model_info['num_classes']}")
            logger.info(f"  - 클래스 목록: {model_info['class_names']}")
            logger.info(f"  - 디바이스: {model_info['device']}")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            raise
    
    def predict(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        이미지 경로 배열에 대한 분류 결과 반환
        
        Args:
            image_paths: 이미지 경로 리스트
            
        Returns:
            예측 결과 리스트 [{'image_path': str, 'predicted_class': str, 'confidence': float}, ...]
        """
        if not image_paths:
            logger.warning("⚠️ 빈 이미지 경로 배열이 전달됨")
            return []
        
        logger.info(f"🔍 추론 시작: {len(image_paths)}개 이미지")
        
        start_time = time.time()
        
        try:
            # 배치 크기 설정
            batch_size = self.config.get('inference', {}).get('batch_size', 64)
            
            # 배치 예측 실행
            predictions = self.inference_engine.predict_batch(
                image_paths=image_paths,
                batch_size=batch_size,
                return_probabilities=True,
                top_k=1
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_image = total_time / len(image_paths) if len(image_paths) > 0 else 0
            
            # 결과 정리
            results = []
            success_count = 0
            error_count = 0
            
            for pred in predictions:
                if 'error' not in pred:
                    results.append({
                        'image_path': pred['image_path'],
                        'predicted_class': pred['predicted_class'],
                        'confidence': round(pred['confidence'], 4)
                    })
                    success_count += 1
                else:
                    results.append({
                        'image_path': pred['image_path'],
                        'predicted_class': 'ERROR',
                        'confidence': 0.0
                    })
                    error_count += 1
                    logger.warning(f"⚠️ 예측 실패: {pred['image_path']} - {pred.get('error', 'Unknown error')}")
            
            # 결과 로깅
            logger.info(f"✅ 추론 완료:")
            logger.info(f"  - 총 이미지: {len(image_paths)}개")
            logger.info(f"  - 성공: {success_count}개")
            logger.info(f"  - 실패: {error_count}개")
            logger.info(f"  - 총 시간: {total_time:.2f}초")
            logger.info(f"  - 개당 평균 시간: {avg_time_per_image:.3f}초")
            logger.info(f"  - 처리 속도: {len(image_paths)/total_time:.1f} images/sec")
            
            # 클래스별 분포 로깅
            if success_count > 0:
                class_counts = {}
                for result in results:
                    if result['predicted_class'] != 'ERROR':
                        class_name = result['predicted_class']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                logger.info(f"📊 예측 클래스 분포:")
                for class_name, count in class_counts.items():
                    percentage = (count / success_count) * 100
                    logger.info(f"  - {class_name}: {count}개 ({percentage:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 추론 중 오류 발생: {e}")
            # 에러 발생 시 전체 결과를 ERROR로 반환
            return [
                {
                    'image_path': path,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0
                }
                for path in image_paths
            ]


def main():
    """테스트용 메인 함수"""
    
    # 하드코딩된 테스트 데이터
    model_path = "results/run_20250812_163937_0f5fe933/model/best_model.pth"
    
    test_image_paths = [
        "product/unhashed/4250e3b9-113d-4bf8-aa98-cc9e8b3f080a-978451651",
        "product/unhashed/922c1440-348e-42fa-9a51-07da43260a44--1169411216", 
        "product/unhashed/31baef89-329d-485a-b9a0-76989c0ebc2d-1254761449",
        "product/unhashed/b2300cc8-a353-448c-8225-c648dac8f3b6--787488915",
        "product/unhashed/f97c42d5-d9a8-49d0-9d9a-505119d8f290--1494358730"
    ]
    
    try:
        # API 초기화
        api = ImageClassificationAPI(model_path=model_path)
        
        # 예측 실행
        results = api.predict(test_image_paths)
        
        # 결과 출력 (로그로만)
        logger.info("=" * 60)
        logger.info("🎯 최종 결과:")
        logger.info("=" * 60)
        
        for i, result in enumerate(results, 1):
            logger.info(f"{i:2d}. {result['image_path'][:50]}... -> {result['predicted_class']} ({result['confidence']:.3f})")
        
        logger.info("=" * 60)
        logger.info("✅ 테스트 완료")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 테스트 실행 중 오류: {e}")
        return []


if __name__ == "__main__":
    main()
