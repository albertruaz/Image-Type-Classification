# image_classification/dataset.py
"""
이미지 분류를 위한 데이터셋 및 전처리

ImageDataset: CSV 기반 이미지 데이터셋 클래스
- 이미지 로드 및 전처리
- 카테고리 레이블 인코딩
- 데이터 증강

ImageTransforms: 이미지 전처리 파이프라인
- 학습/검증/테스트용 변환
- 정규화 및 크기 조정
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Tuple, Optional, List
import logging
from sklearn.preprocessing import LabelEncoder
import requests
from io import BytesIO
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

class ImageDataset(Dataset):
    """
    CSV 기반 이미지 데이터셋
    
    CSV 파일의 이미지 경로와 카테고리 정보를 사용하여
    이미지를 로드하고 전처리하는 데이터셋
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 transform: Optional[transforms.Compose] = None,
                 label_encoder: Optional[LabelEncoder] = None,
                 base_image_path: str = "",
                 target_column: str = 'is_text_tag',
                 is_remote: bool = False):
        """
        Args:
            data: 이미지 정보가 담긴 DataFrame
            transform: 이미지 전처리 변환
            label_encoder: 레이블 인코더
            base_image_path: 이미지 기본 경로
            target_column: 타겟 컬럼명 ('is_text_tag')
            is_remote: 원격 이미지 여부 (URL로 접근)
        """
        self.data = data.copy()
        self.transform = transform
        self.base_image_path = base_image_path
        self.target_column = target_column
        self.is_remote = is_remote
        
        # 유효한 이미지가 있는 데이터만 필터링
        self.data = self._filter_valid_images()
        
        # 레이블 인코더 설정
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            # is_text_tag가 이미 정수형인 경우 직접 사용
            if target_column == 'is_text_tag':
                self.labels = self.data[target_column].values
                # 0과 1만 있는지 확인
                unique_labels = np.unique(self.labels)
                self.label_encoder.classes_ = np.array(['others', 'tag_images'])
            else:
                self.labels = self.label_encoder.fit_transform(self.data[target_column])
        else:
            self.label_encoder = label_encoder
            if target_column == 'is_text_tag':
                self.labels = self.data[target_column].values
            else:
                self.labels = self.label_encoder.transform(self.data[target_column])
        
        self.num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"데이터셋 초기화 완료: {len(self.data)}개 샘플, {self.num_classes}개 클래스")
        self._print_class_distribution()
    
    def _filter_valid_images(self) -> pd.DataFrame:
        """유효한 이미지가 있는 데이터만 필터링"""
        # image_path가 있는 데이터만 선택 (새로운 데이터 구조)
        valid_mask = (self.data['image_path'].notna() & 
                     (self.data['image_path'] != '') & 
                     (self.data['image_path'].astype(str).str.strip() != ''))
        
        filtered_data = self.data[valid_mask].copy()
        
        if len(filtered_data) == 0:
            raise ValueError("유효한 이미지 데이터가 없습니다.")
        
        logger.info(f"유효한 이미지 데이터: {len(filtered_data)}/{len(self.data)}개")
        return filtered_data
    
    def _print_class_distribution(self):
        """클래스 분포 출력"""
        if self.target_column == 'is_text_tag':
            # 이진 분류용 특별 처리
            class_counts = pd.Series(self.labels).value_counts().sort_index()
            print("📊 클래스 분포:")
            print(f"  0 (others): {class_counts.get(0, 0):,}개")
            print(f"  1 (tag_images): {class_counts.get(1, 0):,}개")
        else:
            class_counts = pd.Series(self.labels).value_counts().sort_index()
            print("📊 클래스 분포:")
            for class_idx, count in class_counts.items():
                class_name = self.label_encoder.classes_[class_idx]
                print(f"  {class_idx}: {class_name} - {count:,}개")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        데이터셋에서 하나의 샘플 반환
        
        Returns:
            (image_tensor, label): 전처리된 이미지와 레이블
        """
        try:
            # 이미지 로드
            image = self._load_image(idx)
            
            # 변환 적용
            if self.transform:
                image = self.transform(image)
            
            # 레이블
            label = self.labels[idx]
            
            return image, label
            
        except Exception as e:
            logger.warning(f"이미지 로드 실패 (idx={idx}): {e}")
            # 기본 이미지 반환 (검은색 이미지)
            default_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                default_image = self.transform(default_image)
            return default_image, self.labels[idx]
    
    def _load_image(self, idx: int) -> Image.Image:
        """이미지 로드"""
        row = self.data.iloc[idx]
        image_path = str(row['image_path']).strip()
        
        # 이미지 경로 구성 (CloudFront URL 포함)
        full_image_url = self._build_image_path(image_path)
        
        try:
            if full_image_url.startswith(('http://', 'https://')):
                # 원격 이미지 로드 (CloudFront 등)
                response = requests.get(full_image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # 로컬 이미지 로드
                if not os.path.exists(full_image_url):
                    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {full_image_url}")
                
                image = Image.open(full_image_url)
            
            # RGB로 변환
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            logger.warning(f"이미지 로드 실패: {full_image_url} - {e}")
            # 기본 검은색 이미지 반환
            return Image.new('RGB', (224, 224), color='black')
    
    def _build_image_path(self, image_path: str) -> str:
        """이미지 경로 구성 (CloudFront URL 포함)"""
        if image_path.startswith(('http://', 'https://')):
            # 이미 완전한 URL인 경우
            return image_path
        elif self.base_image_path:
            # 로컬 기본 경로가 있는 경우
            return os.path.join(self.base_image_path, image_path)
        else:
            # CloudFront URL 구성
            from utils.env_loader import get_env_var
            cloudfront_domain = get_env_var('S3_CLOUDFRONT_DOMAIN')
            return f"https://{cloudfront_domain}/{image_path}"
    
    def get_class_names(self) -> List[str]:
        """클래스 이름 목록 반환"""
        return list(self.label_encoder.classes_)
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 가중치 계산 (불균형 데이터 처리용)"""
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


class ImageTransforms:
    """이미지 전처리 변환 클래스"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 224, augmentation_strength: str = 'medium') -> transforms.Compose:
        """학습용 이미지 변환 (데이터 증강 포함)"""
        transform_list = [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if augmentation_strength in ['medium', 'strong']:
            transform_list.extend([
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        
        if augmentation_strength == 'strong':
            transform_list.extend([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_val_transforms(image_size: int = 224) -> transforms.Compose:
        """검증/테스트용 이미지 변환"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, 
                       test_df: pd.DataFrame,
                       config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, LabelEncoder]:
    """
    데이터 로더 생성
    
    Args:
        train_df: 학습 데이터
        val_df: 검증 데이터  
        test_df: 테스트 데이터
        config: 설정 딕셔너리
        
    Returns:
        (train_loader, val_loader, test_loader, label_encoder)
    """
    # 설정 추출
    batch_size = config.get('training', {}).get('batch_size', 32)
    image_size = config.get('augmentation', {}).get('image_size', 224)
    augmentation_strength = config.get('augmentation', {}).get('strength', 'medium')
    num_workers = config.get('system', {}).get('num_workers', 4)
    pin_memory = config.get('system', {}).get('pin_memory', True)
    base_image_path = config.get('data', {}).get('base_image_path', '')
    target_column = config.get('data', {}).get('target_column', 'is_text_tag')
    is_remote = config.get('data', {}).get('is_remote', False)
    
    # 변환 생성
    train_transforms = ImageTransforms.get_train_transforms(image_size, augmentation_strength)
    val_transforms = ImageTransforms.get_val_transforms(image_size)
    
    # 레이블 인코더 생성
    if target_column == 'is_text_tag':
        # 이진 분류의 경우 미리 정의된 클래스 사용
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['others', 'tag_images'])
    else:
        # 다중 클래스 분류의 경우 전체 데이터의 카테고리 기준
        all_categories = pd.concat([train_df, val_df, test_df])[target_column].unique()
        label_encoder = LabelEncoder()
        label_encoder.fit(all_categories)
    
    # 데이터셋 생성
    train_dataset = ImageDataset(
        train_df, train_transforms, label_encoder, 
        base_image_path, target_column, is_remote
    )
    val_dataset = ImageDataset(
        val_df, val_transforms, label_encoder, 
        base_image_path, target_column, is_remote
    )
    test_dataset = ImageDataset(
        test_df, val_transforms, label_encoder, 
        base_image_path, target_column, is_remote
    )
    
    # 데이터 로더 생성 (BatchNorm 에러 방지를 위해 drop_last=True 추가)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    
    logger.info(f"데이터 로더 생성 완료:")
    logger.info(f"  - 학습: {len(train_dataset)}개 샘플, {len(train_loader)}개 배치")
    logger.info(f"  - 검증: {len(val_dataset)}개 샘플, {len(val_loader)}개 배치")
    logger.info(f"  - 테스트: {len(test_dataset)}개 샘플, {len(test_loader)}개 배치")
    
    return train_loader, val_loader, test_loader, label_encoder


def get_dataset_statistics(data_loader: DataLoader) -> Dict[str, Any]:
    """데이터셋 통계 정보 계산"""
    dataset = data_loader.dataset
    
    # 클래스 분포
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    
    statistics = {
        'total_samples': len(dataset),
        'num_classes': dataset.num_classes,
        'class_names': dataset.get_class_names(),
        'class_counts': class_counts.tolist(),
        'class_distribution': {
            name: count for name, count in zip(dataset.get_class_names(), class_counts)
        },
        'batch_size': data_loader.batch_size,
        'num_batches': len(data_loader)
    }
    
    return statistics
