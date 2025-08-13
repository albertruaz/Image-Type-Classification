import os
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_connector import BaseConnector
from utils.env_loader import load_env_once, get_env_var

class CSVConnector(BaseConnector):
    """이진 분류용 CSV 데이터 커넥터 (tag_images vs others)"""
    
    def __init__(self, csv_path: str = None, base_image_path: str = None, base_image_url: str = None):
        """
        Args:
            csv_path: CSV 파일 경로
            base_image_path: 이미지 기본 경로 (로컬)
            base_image_url: 이미지 기본 URL (원격)
        """
        super().__init__()
        
        # 환경변수 로드 (한 번만)
        load_env_once()
        
        # 기본값 설정
        self.csv_path = csv_path or 'image_data.csv'
        self.base_image_path = base_image_path or ''
        
        # CloudFront URL 설정 (환경변수에서 가져오기)
        if base_image_url is None:
            cloudfront_domain = get_env_var('S3_CLOUDFRONT_DOMAIN')
            self.base_image_url = f"https://{cloudfront_domain}"
        else:
            self.base_image_url = base_image_url
        
        self.df = None
        self.target_column = 'is_text_tag'
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """CSV 데이터 로드"""
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ CSV 데이터 로드 완료: {len(self.df)}개 행")
            
            # 기본 정보 출력
            if self.target_column in self.df.columns:
                value_counts = self.df[self.target_column].value_counts()
                print(f"📊 클래스 분포: {dict(value_counts)}")
            else:
                print(f"⚠️ 타겟 컬럼 '{self.target_column}'이 존재하지 않습니다.")
                print(f"사용 가능한 컬럼: {list(self.df.columns)}")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")
        except Exception as e:
            raise ValueError(f"CSV 파일 로드 중 오류 발생: {e}")

    def get_data_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                       random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터를 train/val/test로 분할"""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        from sklearn.model_selection import train_test_split
        
        # 테스트 비율 계산
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio는 1.0 이하여야 합니다.")
        
        # stratify를 위한 타겟 컬럼 확인
        if self.target_column not in self.df.columns:
            raise ValueError(f"타겟 컬럼 '{self.target_column}'이 존재하지 않습니다.")
        
        # 1차 분할: train + val / test
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=test_ratio,
            stratify=self.df[self.target_column],
            random_state=random_state
        )
        
        # 2차 분할: train / val
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            stratify=train_val_df[self.target_column],
            random_state=random_state
        )
        
        print(f"📊 데이터 분할 완료:")
        print(f"  Train: {len(train_df)}개 ({len(train_df)/len(self.df)*100:.1f}%)")
        print(f"  Val: {len(val_df)}개 ({len(val_df)/len(self.df)*100:.1f}%)")
        print(f"  Test: {len(test_df)}개 ({len(test_df)/len(self.df)*100:.1f}%)")
        
        return train_df, val_df, test_df

    def get_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보 반환"""
        if self.df is None:
            return {}
        
        summary = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'target_column': self.target_column
        }
        
        if self.target_column in self.df.columns:
            value_counts = self.df[self.target_column].value_counts()
            summary['classes'] = list(value_counts.index)
            summary['class_distribution'] = dict(value_counts)
        
        return summary
    
    def _build_image_path(self, image_path: str) -> str:
        """이미지 경로 구성"""
        if image_path.startswith(('http://', 'https://')):
            return image_path
        elif self.base_image_path:
            return os.path.join(self.base_image_path, image_path)
        else:
            # CloudFront URL 사용
            return f"{self.base_image_url}/{image_path}"
    
    def get_data(self) -> pd.DataFrame:
        """전체 데이터 반환 (경로 변환 포함)"""
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다.")
        
        # 데이터 복사
        result_df = self.df.copy()
        
        # 이미지 경로 변환
        if 'image_path' in result_df.columns:
            result_df['image_path'] = result_df['image_path'].apply(self._build_image_path)
        elif 'image_url' in result_df.columns:
            result_df['image_path'] = result_df['image_url'].apply(self._build_image_path)
        else:
            raise ValueError("이미지 경로 컬럼 (image_path 또는 image_url)을 찾을 수 없습니다.")
        
        return result_df
    
    def get_filtered_data(self, categories: List[str] = None, 
                         min_samples: int = None, 
                         max_samples: int = None) -> pd.DataFrame:
        """필터링된 데이터 반환"""
        df = self.get_data()
        
        if categories:
            df = df[df[self.target_column].isin(categories)]
        
        if min_samples or max_samples:
            # 클래스별 샘플 수 조정
            grouped = df.groupby(self.target_column)
            filtered_groups = []
            
            for name, group in grouped:
                if min_samples and len(group) < min_samples:
                    continue
                if max_samples and len(group) > max_samples:
                    group = group.sample(n=max_samples, random_state=42)
                filtered_groups.append(group)
            
            if filtered_groups:
                df = pd.concat(filtered_groups, ignore_index=True)
            else:
                df = pd.DataFrame()
        
        return df
    
    def validate_data(self) -> Dict[str, Any]:
        """데이터 유효성 검사"""
        if self.df is None:
            return {'valid': False, 'error': '데이터가 로드되지 않았습니다.'}
        
        validation_result = {
            'valid': True,
            'warnings': [],
            'stats': {}
        }
        
        # 필수 컬럼 확인
        required_columns = ['image_path', 'image_url']
        image_column = None
        for col in required_columns:
            if col in self.df.columns:
                image_column = col
                break
        
        if not image_column:
            validation_result['valid'] = False
            validation_result['error'] = f"이미지 경로 컬럼을 찾을 수 없습니다. 필요: {required_columns}"
            return validation_result
        
        # 타겟 컬럼 확인
        if self.target_column not in self.df.columns:
            validation_result['valid'] = False
            validation_result['error'] = f"타겟 컬럼 '{self.target_column}'이 존재하지 않습니다."
            return validation_result
        
        # 결측값 확인
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            validation_result['warnings'].append(f"결측값 발견: {dict(null_counts[null_counts > 0])}")
        
        # 클래스 분포 확인
        class_counts = self.df[self.target_column].value_counts()
        validation_result['stats']['class_distribution'] = dict(class_counts)
        
        # 클래스 불균형 경고
        min_count = class_counts.min()
        max_count = class_counts.max()
        if max_count / min_count > 10:
            validation_result['warnings'].append(f"심각한 클래스 불균형: {max_count}/{min_count} = {max_count/min_count:.1f}")
        
        return validation_result