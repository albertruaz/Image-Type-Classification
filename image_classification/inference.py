# image_classification/inference.py
"""
이미지 분류 추론 클래스

ImageClassifierInference:
- 학습된 모델로 단일/배치 이미지 분류
- 체크포인트에서 모델 및 설정 로드
- 확률 분포 및 top-k 예측 제공
- 배치 처리 지원
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
import os
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms

from .cnn_model import create_image_classifier
from .dataset import ImageTransforms

logger = logging.getLogger(__name__)

class InferenceDataset(Dataset):
    """추론용 데이터셋"""
    
    def __init__(self, image_paths: List[str], transforms=None, base_image_path: str = None, base_image_url: str = None, is_remote: bool = False):
        self.image_paths = image_paths
        self.transforms = transforms
        self.base_image_path = base_image_path.rstrip('/') if base_image_path else None
        self.base_image_url = base_image_url.rstrip('/') if base_image_url else None
        self.is_remote = is_remote
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # 이미지 로드
            # 이미지 경로 처리
            if image_path.startswith(('http://', 'https://')):
                # 이미 URL인 경우 그대로 사용
                full_path = image_path
            elif self.base_image_path:
                # 로컬 기본 경로가 있는 경우
                full_path = os.path.join(self.base_image_path, image_path)
                logger.debug(f"로컬 경로 생성: {image_path} -> {full_path}")
            else:
                # CloudFront URL 구성
                from utils.env_loader import get_env_var
                cloudfront_domain = get_env_var('S3_CLOUDFRONT_DOMAIN')
                full_path = f"https://{cloudfront_domain}/{image_path}"
                logger.debug(f"CloudFront URL 생성: {image_path} -> {full_path}")

            try:
                if full_path.startswith(('http://', 'https://')):
                    import requests
                    from io import BytesIO
                    response = requests.get(full_path, timeout=10)
                    response.raise_for_status()  # HTTP 에러 체크
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {full_path}")
                    image = Image.open(full_path).convert('RGB')
            except Exception as e:
                logger.error(f"이미지 로드 실패 {full_path}: {e}")
                # 에러 발생 시 검은색 이미지 반환
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            
            # 변환 적용
            if self.transforms:
                image = self.transforms(image)
            
            return image, image_path
            
        except Exception as e:
            logger.error(f"이미지 로드 실패 {image_path}: {e}")
            # 기본 이미지 반환 (검은색)
            if self.transforms:
                dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transforms(dummy_image), image_path
            else:
                return torch.zeros(3, 224, 224), image_path


class ImageClassifierInference:
    """이미지 분류 추론 클래스"""
    
    def __init__(self, 
                 model_path: str,
                 device: torch.device = None):
        """
        Args:
            model_path: 모델 체크포인트 경로
            device: 추론 디바이스
        """
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 및 설정 로드
        self.model = None
        self.config = None
        self.class_names = None
        self.num_classes = None
        self.transforms = None
        self.base_image_path = None
        self.base_image_url = None
        self.is_remote = False
        
        self._load_model()
        self._setup_transforms()
        
        logger.info(f"추론기 초기화 완료 - 디바이스: {self.device}")
    
    def _load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 설정 및 메타데이터 추출
            self.config = checkpoint.get('config', {})
            self.class_names = checkpoint.get('class_names', [])
            self.num_classes = len(self.class_names)
            
            # 이미지 경로 설정
            data_config = self.config.get('data', {})
            self.base_image_path = data_config.get('base_image_path', '')
            self.base_image_url = data_config.get('base_image_url', '')
            self.is_remote = data_config.get('is_remote', False)
            
            if self.is_remote:
                if not self.base_image_url:
                    logger.warning("원격 이미지 모드이지만 base_image_url이 설정되지 않았습니다.")
            elif not self.base_image_path:
                logger.warning("로컬 이미지 모드이지만 base_image_path가 설정되지 않았습니다.")
            
            # 클래스 정보가 없으면 원본 데이터에서 추출
            if self.num_classes == 0:
                logger.warning("체크포인트에 클래스 정보가 없습니다. 원본 데이터에서 추출합니다.")
                self.class_names = self._extract_class_names_from_data()
                self.num_classes = len(self.class_names)
                
                if self.num_classes == 0:
                    raise ValueError("클래스 정보를 찾을 수 없습니다")
            
            # 모델 생성
            self.model = create_image_classifier(self.config, self.num_classes)
            
            # 가중치 로드
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # 전체 체크포인트가 state_dict인 경우
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"모델 로드 완료: {self.num_classes}개 클래스")
            logger.info(f"클래스 목록: {self.class_names}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise
    
    def _setup_transforms(self):
        """추론용 변환 설정"""
        augmentation_config = self.config.get('augmentation', {})
        image_size = augmentation_config.get('image_size', 224)
        
        # 추론시에는 augmentation 없이 정규화만 적용
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_single(self, 
                      image_path: str,
                      return_probabilities: bool = True,
                      top_k: int = None) -> Dict[str, Any]:
        """
        단일 이미지 예측
        
        Args:
            image_path: 이미지 경로 (로컬 또는 URL)
            return_probabilities: 확률 반환 여부
            top_k: 상위 k개 예측 반환
            
        Returns:
            예측 결과
        """
        try:
            # 이미지 경로 처리
            if image_path.startswith(('http://', 'https://')):
                # 이미 URL인 경우 그대로 사용
                full_path = image_path
            elif self.base_image_path:
                # 로컬 기본 경로가 있는 경우
                full_path = os.path.join(self.base_image_path, image_path)
                logger.debug(f"로컬 경로 생성: {image_path} -> {full_path}")
            else:
                # CloudFront URL 구성
                from utils.env_loader import get_env_var
                cloudfront_domain = get_env_var('S3_CLOUDFRONT_DOMAIN')
                full_path = f"https://{cloudfront_domain}/{image_path}"
                logger.debug(f"CloudFront URL 생성: {image_path} -> {full_path}")

            # 이미지 로드
            if full_path.startswith(('http://', 'https://')):
                import requests
                from io import BytesIO
                response = requests.get(full_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {full_path}")
                image = Image.open(full_path).convert('RGB')
            
            # 변환 적용
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                
                # CPU로 이동
                probabilities = probabilities.cpu().numpy()[0]
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = self.class_names[predicted_class_idx]
                confidence = float(probabilities[predicted_class_idx])
            
            # 결과 구성
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'predicted_class_idx': int(predicted_class_idx),
                'confidence': confidence
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, probabilities)
                }
            
            if top_k:
                top_k = min(top_k, self.num_classes)
                top_indices = np.argsort(probabilities)[-top_k:][::-1]
                result['top_k_predictions'] = [
                    {
                        'class': self.class_names[idx],
                        'class_idx': int(idx),
                        'confidence': float(probabilities[idx])
                    }
                    for idx in top_indices
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"예측 실패 {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, 
                     image_paths: List[str],
                     batch_size: int = 32,
                     return_probabilities: bool = True,
                     top_k: int = None) -> List[Dict[str, Any]]:
        """
        배치 이미지 예측
        
        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기
            return_probabilities: 확률 반환 여부
            top_k: 상위 k개 예측 반환
            
        Returns:
            예측 결과 리스트
        """
        # 데이터셋 및 로더 생성
        dataset = InferenceDataset(
            image_paths=image_paths,
            transforms=self.transforms,
            base_image_path=self.base_image_path,
            base_image_url=self.base_image_url,
            is_remote=self.is_remote
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        results = []
        
        with torch.no_grad():
            for batch_images, batch_paths in dataloader:
                try:
                    batch_images = batch_images.to(self.device)
                    
                    # 예측
                    logits = self.model(batch_images)
                    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    # 배치 결과 처리
                    for i, image_path in enumerate(batch_paths):
                        probs = probabilities[i]
                        predicted_class_idx = np.argmax(probs)
                        predicted_class = self.class_names[predicted_class_idx]
                        confidence = float(probs[predicted_class_idx])
                        
                        result = {
                            'image_path': image_path,
                            'predicted_class': predicted_class,
                            'predicted_class_idx': int(predicted_class_idx),
                            'confidence': confidence
                        }
                        
                        if return_probabilities:
                            result['probabilities'] = {
                                class_name: float(prob) 
                                for class_name, prob in zip(self.class_names, probs)
                            }
                        
                        if top_k:
                            top_k_actual = min(top_k, self.num_classes)
                            top_indices = np.argsort(probs)[-top_k_actual:][::-1]
                            result['top_k_predictions'] = [
                                {
                                    'class': self.class_names[idx],
                                    'class_idx': int(idx),
                                    'confidence': float(probs[idx])
                                }
                                for idx in top_indices
                            ]
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.error(f"배치 예측 실패: {e}")
                    # 실패한 배치의 각 이미지에 대해 에러 결과 추가
                    for image_path in batch_paths:
                        results.append({
                            'image_path': image_path,
                            'error': str(e),
                            'predicted_class': None,
                            'confidence': 0.0
                        })
        
        return results
    
    def predict_from_dataframe(self, 
                              df: pd.DataFrame, 
                              image_path_column: str = 'image_path',
                              batch_size: int = 32,
                              save_results: bool = True,
                              output_path: str = None) -> pd.DataFrame:
        """
        DataFrame의 이미지들에 대해 예측
        
        Args:
            df: 이미지 경로가 포함된 DataFrame
            image_path_column: 이미지 경로 컬럼명
            batch_size: 배치 크기
            save_results: 결과 저장 여부
            output_path: 결과 저장 경로
            
        Returns:
            예측 결과가 추가된 DataFrame
        """
        image_paths = df[image_path_column].tolist()
        
        logger.info(f"{len(image_paths)}개 이미지에 대해 예측 시작")
        
        # 배치 예측
        results = self.predict_batch(
            image_paths, 
            batch_size=batch_size,
            return_probabilities=True,
            top_k=3
        )
        
        # 결과를 DataFrame에 추가
        result_df = df.copy()
        result_df['predicted_class'] = [r.get('predicted_class') for r in results]
        result_df['confidence'] = [r.get('confidence', 0.0) for r in results]
        result_df['has_error'] = [r.get('error') is not None for r in results]
        
        # 확률 정보 추가
        for class_name in self.class_names:
            result_df[f'prob_{class_name}'] = [
                r.get('probabilities', {}).get(class_name, 0.0) for r in results
            ]
        
        # 결과 저장
        if save_results:
            if output_path is None:
                output_path = 'inference_results.csv'
            
            result_df.to_csv(output_path, index=False)
            logger.info(f"예측 결과 저장: {output_path}")
        
        logger.info(f"예측 완료: {len(results)}개 결과")
        
        return result_df
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            'model_path': self.model_path,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'device': str(self.device),
            'config': self.config
        }
    
    def _extract_class_names_from_data(self) -> List[str]:
        """원본 데이터에서 클래스 정보 추출"""
        try:
            # config에서 데이터 정보 추출
            data_config = self.config.get('data', {})
            target_column = data_config.get('target_column', 'is_text_tag')
            
            # 분할된 데이터 파일들을 우선적으로 사용
            possible_paths = [
                'data/train_data.csv',
                'data/validation_data.csv', 
                'data/test_data.csv',
                data_config.get('csv_path', 'image_data.csv')  # 마지막으로 원본 파일
            ]
            
            df = None
            used_path = None
            
            # 존재하는 파일을 찾아서 로드
            for csv_path in possible_paths:
                if os.path.exists(csv_path):
                    logger.info(f"클래스 정보 추출 시도: {csv_path}의 {target_column} 컬럼에서")
                    
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    
                    if target_column in df.columns:
                        used_path = csv_path
                        logger.info(f"클래스 정보 추출 성공: {csv_path} 사용")
                        break
                    else:
                        logger.warning(f"{csv_path}에 {target_column} 컬럼이 없음")
                        df = None
            
            if df is None:
                raise FileNotFoundError(f"타겟 컬럼 '{target_column}'이 있는 데이터 파일을 찾을 수 없습니다")
            
            # 클래스 이름 추출 (정렬하여 일관성 유지)
            class_names = sorted(df[target_column].unique().tolist())
            logger.info(f"추출된 클래스 ({used_path}): {class_names}")
            
            return class_names
            
        except Exception as e:
            logger.error(f"클래스 정보 추출 실패: {e}")
            raise

