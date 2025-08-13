#!/usr/bin/env python3
# divide_data.py
"""
이미지 태그 분류 데이터 분할 스크립트

image_data.csv 파일을 이미지별 행으로 변환하고
tag 이미지(text_tag_image, neck_tag_image) vs 일반 이미지로 분류하여
train/validation/test로 분할하여 data/ 폴더에 저장

사용법:
    python divide_data.py                    # 기본 설정으로 분할
    python divide_data.py --train-ratio 0.7 # 커스텀 비율
    python divide_data.py --random-state 123 # 시드 변경
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def expand_images_to_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    제품별 데이터를 이미지별 행으로 확장
    
    Args:
        df: 원본 데이터프레임 (제품별)
        
    Returns:
        확장된 데이터프레임 (이미지별)
    """
    logger.info("🔄 이미지별 행으로 데이터 확장 중...")
    
    # 결과를 저장할 리스트
    expanded_rows = []
    
    # 이미지 타입별 컬럼 정의
    image_type_columns = {
        'main_image': {'is_text_tag': 0, 'image_type': 'main_image'},
        'back_image': {'is_text_tag': 0, 'image_type': 'back_image'},
        'text_tag_image': {'is_text_tag': 1, 'image_type': 'text_tag_image'},
        'neck_tag_image': {'is_text_tag': 1, 'image_type': 'neck_tag_image'},
        'other': {'is_text_tag': 0, 'image_type': 'other'}
    }
    
    for idx, row in df.iterrows():
        product_id = row['id']
        
        # 각 이미지 타입별로 처리
        for column_name, label_info in image_type_columns.items():
            if column_name not in df.columns:
                continue
                
            image_paths_str = row[column_name]
            
            # 빈 값이면 건너뛰기
            if pd.isna(image_paths_str) or image_paths_str == '':
                continue
            
            # 이미지 경로들 분리 (& 또는 , 로 구분되어 있을 수 있음)
            if '&' in str(image_paths_str):
                image_paths = str(image_paths_str).split('&')
            elif ',' in str(image_paths_str):
                image_paths = str(image_paths_str).split(',')
            else:
                image_paths = [str(image_paths_str)]
            
            # 각 이미지 경로에 대해 행 생성
            for image_path in image_paths:
                image_path = image_path.strip()
                if image_path == '' or image_path == 'nan':
                    continue
                
                # 새로운 행 생성 (이진 분류에 필요한 컬럼만)
                new_row = {
                    'product_id': product_id,
                    'image_path': image_path,
                    'image_type': label_info['image_type'],
                    'is_text_tag': label_info['is_text_tag'],
                    # 분할 시에만 사용할 임시 컬럼 (최종 저장 시 제거)
                    '_temp_category_name': row['category_name']
                }
                
                expanded_rows.append(new_row)
    
    # 데이터프레임 생성
    expanded_df = pd.DataFrame(expanded_rows)
    
    logger.info(f"확장 완료:")
    logger.info(f"  원본: {len(df)}개 제품")
    logger.info(f"  확장후: {len(expanded_df)}개 이미지")
    
    # 태그별 분포 확인
    tag_distribution = expanded_df['is_text_tag'].value_counts()
    logger.info(f"태그 분포:")
    logger.info(f"  일반 이미지 (0): {tag_distribution.get(0, 0):,}개")
    logger.info(f"  태그 이미지 (1): {tag_distribution.get(1, 0):,}개")
    
    # 이미지 타입별 분포
    type_distribution = expanded_df['image_type'].value_counts()
    logger.info(f"이미지 타입별 분포:")
    for img_type, count in type_distribution.items():
        logger.info(f"  {img_type}: {count:,}개")
    
    return expanded_df

def analyze_expanded_data(df: pd.DataFrame) -> dict:
    """확장된 데이터 분석"""
    analysis = {
        'total_images': len(df),
        'total_products': df['product_id'].nunique(),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'tag_distribution': df['is_text_tag'].value_counts().to_dict(),
        'image_type_distribution': df['image_type'].value_counts().to_dict(),
        'category_distribution': df['_temp_category_name'].value_counts().to_dict(),
        'categories': df['_temp_category_name'].unique().tolist(),
        'num_categories': df['_temp_category_name'].nunique()
    }
    
    return analysis

def stratified_split_by_product(df: pd.DataFrame, 
                               train_ratio: float = 0.8,
                               val_ratio: float = 0.1,
                               test_ratio: float = 0.1,
                               random_state: int = 42) -> tuple:
    """
    제품 기준 계층화 분할 (같은 제품의 이미지들이 다른 세트에 섞이지 않도록)
    
    Args:
        df: 확장된 데이터프레임 (이미지별)
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        random_state: 랜덤 시드
        
    Returns:
        train_df, val_df, test_df
    """
    # 비율 검증
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"비율의 합이 1.0이 아닙니다: {total_ratio}")
    
    logger.info("🔄 제품 기준 계층화 분할 중...")
    
    # 제품별 집계 (카테고리와 태그 분포 계산)
    product_summary = df.groupby('product_id').agg({
        '_temp_category_name': 'first',  # 제품의 카테고리 (임시)
        'is_text_tag': ['sum', 'count']  # 태그 이미지 수, 전체 이미지 수
    }).reset_index()
    
    # 컬럼명 정리
    product_summary.columns = ['product_id', '_temp_category_name', 'tag_count', 'total_count']
    product_summary['has_tag'] = (product_summary['tag_count'] > 0).astype(int)
    
    logger.info(f"제품 분포:")
    logger.info(f"  전체 제품: {len(product_summary):,}개")
    logger.info(f"  태그 이미지 있는 제품: {product_summary['has_tag'].sum():,}개")
    logger.info(f"  태그 이미지 없는 제품: {(product_summary['has_tag'] == 0).sum():,}개")
    
    # 카테고리별 분포
    category_dist = product_summary['_temp_category_name'].value_counts()
    logger.info(f"제품 카테고리별 분포:")
    for category, count in category_dist.items():
        logger.info(f"  {category}: {count:,}개")
    
    # 1차 분할: train vs (val + test)
    train_products, temp_products = train_test_split(
        product_summary,
        test_size=(val_ratio + test_ratio),
        stratify=product_summary['_temp_category_name'],
        random_state=random_state
    )
    
    # 2차 분할: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_products, test_products = train_test_split(
        temp_products,
        test_size=(1 - val_test_ratio),
        stratify=temp_products['_temp_category_name'],
        random_state=random_state
    )
    
    logger.info(f"제품 분할 완료:")
    logger.info(f"  Train: {len(train_products):,}개 제품")
    logger.info(f"  Validation: {len(val_products):,}개 제품")
    logger.info(f"  Test: {len(test_products):,}개 제품")
    
    # 제품 ID를 기반으로 이미지 데이터 분할
    train_product_ids = set(train_products['product_id'])
    val_product_ids = set(val_products['product_id'])
    test_product_ids = set(test_products['product_id'])
    
    train_df = df[df['product_id'].isin(train_product_ids)].copy()
    val_df = df[df['product_id'].isin(val_product_ids)].copy()
    test_df = df[df['product_id'].isin(test_product_ids)].copy()
    
    logger.info(f"이미지 분할 결과:")
    logger.info(f"  Train: {len(train_df):,}개 이미지 ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df):,}개 이미지 ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df):,}개 이미지 ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def verify_split_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """분할 무결성 검증"""
    logger.info("=" * 60)
    logger.info("🔍 분할 무결성 검증")
    logger.info("=" * 60)
    
    # 제품 ID 중복 확인
    train_products = set(train_df['product_id'])
    val_products = set(val_df['product_id'])
    test_products = set(test_df['product_id'])
    
    overlap_train_val = train_products & val_products
    overlap_train_test = train_products & test_products
    overlap_val_test = val_products & test_products
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.error("❌ 제품 ID가 여러 세트에 중복됨!")
        logger.error(f"  Train-Val 중복: {len(overlap_train_val)}개")
        logger.error(f"  Train-Test 중복: {len(overlap_train_test)}개")
        logger.error(f"  Val-Test 중복: {len(overlap_val_test)}개")
        raise ValueError("데이터 누수: 같은 제품이 여러 세트에 포함됨")
    else:
        logger.info("✅ 제품 ID 중복 없음 - 데이터 누수 방지 완료")
    
    # 태그 분포 확인
    logger.info("태그 분포 비교:")
    for name, df_split in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        tag_dist = df_split['is_text_tag'].value_counts()
        total = len(df_split)
        tag_ratio = tag_dist.get(1, 0) / total * 100 if total > 0 else 0
        logger.info(f"  {name}: 태그 {tag_dist.get(1, 0):,}개 / 전체 {total:,}개 ({tag_ratio:.1f}%)")
    
    # 카테고리 분포 확인
    logger.info("카테고리 분포 비교:")
    all_categories = set(train_df['_temp_category_name']) | set(val_df['_temp_category_name']) | set(test_df['_temp_category_name'])
    
    comparison_data = []
    for category in sorted(all_categories):
        train_count = len(train_df[train_df['_temp_category_name'] == category])
        val_count = len(val_df[val_df['_temp_category_name'] == category])
        test_count = len(test_df[test_df['_temp_category_name'] == category])
        total_count = train_count + val_count + test_count
        
        comparison_data.append({
            'Category': category,
            'Train': train_count,
            'Val': val_count,
            'Test': test_count,
            'Total': total_count,
            'Train_%': round(train_count / total_count * 100, 1) if total_count > 0 else 0,
            'Val_%': round(val_count / total_count * 100, 1) if total_count > 0 else 0,
            'Test_%': round(test_count / total_count * 100, 1) if total_count > 0 else 0
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 카테고리별 분할 분포:")
    print(comparison_df.to_string(index=False))

def save_splits(train_df: pd.DataFrame, 
                val_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                output_dir: str = "data") -> dict:
    """분할된 데이터 저장 (필요한 컬럼만)"""
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(exist_ok=True)
    
    # 파일 경로 설정
    file_paths = {
        'train': os.path.join(output_dir, 'train_data.csv'),
        'validation': os.path.join(output_dir, 'validation_data.csv'),
        'test': os.path.join(output_dir, 'test_data.csv')
    }
    
    # 최종 저장할 컬럼 선택 (임시 컬럼 제거)
    final_columns = ['product_id', 'image_path', 'image_type', 'is_text_tag']
    
    # 데이터프레임 저장 (필요한 컬럼만)
    train_df[final_columns].to_csv(file_paths['train'], index=False, encoding='utf-8')
    val_df[final_columns].to_csv(file_paths['validation'], index=False, encoding='utf-8')
    test_df[final_columns].to_csv(file_paths['test'], index=False, encoding='utf-8')
    
    logger.info("=" * 60)
    logger.info("💾 분할된 데이터 저장 완료 (필요한 컬럼만)")
    logger.info("=" * 60)
    logger.info(f"저장된 컬럼: {', '.join(final_columns)}")
    for split_name, file_path in file_paths.items():
        logger.info(f"  {split_name.upper()}: {file_path}")
    
    return file_paths

def create_summary(train_df: pd.DataFrame, 
                  val_df: pd.DataFrame, 
                  test_df: pd.DataFrame,
                  file_paths: dict,
                  config: dict,
                  output_dir: str = "data") -> str:
    """분할 요약 정보 생성"""
    from datetime import datetime
    
    summary = {
        'split_info': {
            'created_at': datetime.now().isoformat(),
            'source_file': config['input_file'],
            'random_state': config['random_state'],
            'train_ratio': config['train_ratio'],
            'val_ratio': config['val_ratio'],
            'test_ratio': config['test_ratio'],
            'split_method': 'product_based_stratified'
        },
        'data_counts': {
            'total_images': len(train_df) + len(val_df) + len(test_df),
            'train_images': len(train_df),
            'validation_images': len(val_df),
            'test_images': len(test_df),
            'total_products': len(set(train_df['product_id']) | set(val_df['product_id']) | set(test_df['product_id'])),
            'train_products': len(set(train_df['product_id'])),
            'validation_products': len(set(val_df['product_id'])),
            'test_products': len(set(test_df['product_id']))
        },
        'file_paths': file_paths,
        'categories': {
            'total_categories': len(set(train_df['_temp_category_name']) | set(val_df['_temp_category_name']) | set(test_df['_temp_category_name'])),
            'category_list': sorted(list(set(train_df['_temp_category_name']) | set(val_df['_temp_category_name']) | set(test_df['_temp_category_name'])))
        },
        'tag_distribution': {
            'train': train_df['is_text_tag'].value_counts().to_dict(),
            'validation': val_df['is_text_tag'].value_counts().to_dict(),
            'test': test_df['is_text_tag'].value_counts().to_dict()
        },
        'image_type_distribution': {
            'train': train_df['image_type'].value_counts().to_dict(),
            'validation': val_df['image_type'].value_counts().to_dict(),
            'test': test_df['image_type'].value_counts().to_dict()
        },
        'category_distribution': {
            'train': train_df['_temp_category_name'].value_counts().to_dict(),
            'validation': val_df['_temp_category_name'].value_counts().to_dict(),
            'test': test_df['_temp_category_name'].value_counts().to_dict()
        }
    }
    
    # JSON으로 저장
    import json
    summary_path = os.path.join(output_dir, 'data_split_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📋 분할 요약 정보 저장: {summary_path}")
    
    return summary_path

def validate_image_paths(df: pd.DataFrame, base_path: str = "") -> dict:
    """이미지 경로 유효성 검사"""
    logger.info("🔍 이미지 경로 유효성 검사...")
    
    image_column = 'image_path'
    if image_column not in df.columns:
        logger.warning(f"이미지 컬럼 '{image_column}'을 찾을 수 없습니다")
        return {'valid_count': 0, 'invalid_count': len(df), 'missing_count': len(df)}
    
    valid_count = 0
    invalid_count = 0
    missing_count = 0
    
    for idx, row in df.iterrows():
        image_path = row[image_column]
        
        if pd.isna(image_path) or image_path == '':
            missing_count += 1
            continue
            
        full_path = os.path.join(base_path, image_path) if base_path else image_path
        
        if os.path.exists(full_path):
            valid_count += 1
        else:
            invalid_count += 1
    
    result = {
        'total_count': len(df),
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'missing_count': missing_count,
        'valid_ratio': valid_count / len(df) if len(df) > 0 else 0
    }
    
    logger.info(f"  총 이미지: {result['total_count']:,}")
    logger.info(f"  유효한 경로: {result['valid_count']:,} ({result['valid_ratio']*100:.1f}%)")
    logger.info(f"  잘못된 경로: {result['invalid_count']:,}")
    logger.info(f"  누락된 경로: {result['missing_count']:,}")
    
    return result

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 태그 분류 데이터 분할 스크립트')
    parser.add_argument('--input-file', type=str, default='data/original_data/image_data.csv',
                       help='입력 CSV 파일 경로 (기본값: image_data.csv)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='출력 디렉토리 (기본값: data)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='학습 데이터 비율 (기본값: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='검증 데이터 비율 (기본값: 0.1)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='테스트 데이터 비율 (기본값: 0.1)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    parser.add_argument('--validate-images', action='store_true',
                       help='이미지 경로 유효성 검사 수행')
    parser.add_argument('--image-base-path', type=str, default='',
                       help='이미지 기본 경로 (유효성 검사용)')
    
    args = parser.parse_args()
    
    # 설정 정보
    config = {
        'input_file': args.input_file,
        'output_dir': args.output_dir,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'random_state': args.random_state
    }
    
    try:
        logger.info("=" * 60)
        logger.info("🏷️ 이미지 태그 분류 데이터 분할 시작")
        logger.info("=" * 60)
        logger.info(f"입력 파일: {args.input_file}")
        logger.info(f"출력 디렉토리: {args.output_dir}")
        logger.info(f"분할 비율 - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
        logger.info(f"랜덤 시드: {args.random_state}")
        
        # 1. 데이터 로드
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input_file}")
        
        logger.info(f"📂 데이터 로드 중: {args.input_file}")
        df = pd.read_csv(args.input_file, encoding='utf-8')
        logger.info(f"로드 완료: {len(df):,}개 제품")
        
        # 필수 컬럼 확인
        required_columns = ['id', 'category_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")
        
        logger.info(f"사용 가능한 컬럼: {', '.join(df.columns)}")
        
        # 2. 제품별 데이터를 이미지별 행으로 확장
        expanded_df = expand_images_to_rows(df)
        
        if len(expanded_df) == 0:
            raise ValueError("확장된 데이터가 비어있습니다. 이미지 경로를 확인하세요.")
        
        # 3. 확장된 데이터 분석
        analysis = analyze_expanded_data(expanded_df)
        logger.info("=" * 60)
        logger.info("📊 확장된 데이터 분석")
        logger.info("=" * 60)
        logger.info(f"총 이미지: {analysis['total_images']:,}개")
        logger.info(f"총 제품: {analysis['total_products']:,}개")
        logger.info(f"카테고리 수: {analysis['num_categories']}개")
        
        logger.info("태그 분포:")
        for tag_value, count in analysis['tag_distribution'].items():
            tag_name = "태그 이미지" if tag_value == 1 else "일반 이미지"
            pct = count / analysis['total_images'] * 100
            logger.info(f"  {tag_name} ({tag_value}): {count:,}개 ({pct:.1f}%)")
        
        # 4. 누락값 체크 및 정리
        missing_image_path = analysis['missing_values'].get('image_path', 0)
        missing_category = analysis['missing_values'].get('_temp_category_name', 0)
        
        if missing_image_path > 0:
            logger.warning(f"image_path 누락: {missing_image_path}개")
        if missing_category > 0:
            logger.warning(f"카테고리 누락: {missing_category}개")
            
        # 누락값이 있는 행 제거
        original_len = len(expanded_df)
        expanded_df = expanded_df.dropna(subset=['image_path', '_temp_category_name'])
        if len(expanded_df) < original_len:
            logger.info(f"누락값 제거: {original_len - len(expanded_df)}개 행 제거됨")
        
        # 5. 이미지 경로 유효성 검사 (선택적)
        if args.validate_images:
            validate_image_paths(expanded_df, args.image_base_path)
        
        # 6. 제품 기준 계층화 분할
        train_df, val_df, test_df = stratified_split_by_product(
            expanded_df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state
        )
        
        # 7. 분할 무결성 검증
        verify_split_integrity(train_df, val_df, test_df)
        
        # 8. 분할된 데이터 저장
        file_paths = save_splits(train_df, val_df, test_df, args.output_dir)
        
        # 9. 요약 정보 생성
        summary_path = create_summary(train_df, val_df, test_df, file_paths, config, args.output_dir)
        
        logger.info("=" * 60)
        logger.info("✅ 이미지 태그 분류 데이터 분할 완료")
        logger.info("=" * 60)
        logger.info("📁 생성된 파일:")
        for split_name, file_path in file_paths.items():
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"  {file_path} ({size_mb:.1f}MB)")
        logger.info(f"  {summary_path}")
        
        # 최종 요약
        print(f"\n🎉 이미지 태그 분류 데이터 분할이 완료되었습니다!")
        print(f"📂 출력 폴더: {args.output_dir}")
        print(f"📊 분할 결과:")
        print(f"  Train: {len(train_df):,}개 이미지 ({len(set(train_df['product_id'])):,}개 제품)")
        print(f"  Val: {len(val_df):,}개 이미지 ({len(set(val_df['product_id'])):,}개 제품)")
        print(f"  Test: {len(test_df):,}개 이미지 ({len(set(test_df['product_id'])):,}개 제품)")
        
        # 태그 분포 요약
        train_tag_ratio = train_df['is_text_tag'].mean() * 100
        val_tag_ratio = val_df['is_text_tag'].mean() * 100
        test_tag_ratio = test_df['is_text_tag'].mean() * 100
        print(f"🏷️ 태그 이미지 비율:")
        print(f"  Train: {train_tag_ratio:.1f}%, Val: {val_tag_ratio:.1f}%, Test: {test_tag_ratio:.1f}%")
        
    except Exception as e:
        logger.error(f"데이터 분할 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()