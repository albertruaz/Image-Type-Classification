#!/usr/bin/env python3
"""
오답 분석 스크립트

Test 데이터에서 오답을 찾아서 각 클래스별로 
잘못 예측한 confidence가 높은 순으로 정렬하여 출력
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.device_manager import DeviceManager
from utils.env_loader import load_env_once
from image_classification.inference import ImageClassifierInference

# 환경변수 로드
load_env_once()


def analyze_errors(model_path: str, test_data_path: str, top_k: int = 10, output_path: str = None):
    """
    오답 분석 실행
    
    Args:
        model_path: 모델 경로
        test_data_path: 테스트 데이터 CSV 경로
        top_k: 각 클래스별 상위 k개 출력
        output_path: 결과 저장 경로 (선택)
    """
    # 테스트 데이터 로드
    print(f"📂 테스트 데이터 로드: {test_data_path}")
    test_df = pd.read_csv(test_data_path)
    print(f"   총 {len(test_df)}개 샘플")
    
    # 추론 엔진 초기화
    print(f"\n🔧 모델 로드: {model_path}")
    device = DeviceManager.get_device()
    inference_engine = ImageClassifierInference(model_path=model_path, device=device)
    
    class_names = inference_engine.class_names
    print(f"   클래스: {class_names}")
    
    # 이미지 경로 컬럼 확인
    image_col = 'image_path' if 'image_path' in test_df.columns else 'path'
    label_col = 'image_type' if 'image_type' in test_df.columns else 'label'
    
    image_paths = test_df[image_col].tolist()
    true_labels = test_df[label_col].tolist()
    
    # 배치 예측
    print(f"\n🔮 예측 실행 중...")
    predictions = inference_engine.predict_batch(
        image_paths=image_paths,
        batch_size=64,
        return_probabilities=True
    )
    
    # 결과 DataFrame 생성
    # product_id 컬럼이 있으면 포함
    has_product_id = 'product_id' in test_df.columns
    results = []
    for i, pred in enumerate(predictions):
        true_label = true_labels[i]
        pred_label = pred.get('predicted_class', 'ERROR')
        confidence = pred.get('confidence', 0.0)
        probs = pred.get('probabilities', {})

        is_correct = (str(true_label) == str(pred_label))

        row = {
            'image_path': image_paths[i],
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': confidence,
            'is_correct': is_correct,
            **{f'prob_{c}': probs.get(c, 0.0) for c in class_names}
        }
        if has_product_id:
            row['product_id'] = test_df.iloc[i]['product_id']
        results.append(row)

    results_df = pd.DataFrame(results)
    
    # 전체 정확도
    accuracy = results_df['is_correct'].mean() * 100
    total_errors = (~results_df['is_correct']).sum()
    print(f"\n📊 전체 정확도: {accuracy:.2f}% (오답: {total_errors}개)")
    
    # 오답만 필터링
    errors_df = results_df[~results_df['is_correct']].copy()
    
    if len(errors_df) == 0:
        print("\n✅ 오답이 없습니다!")
        return results_df
    
    # 각 클래스별 오답 분석
    print("\n" + "=" * 80)
    print("🔍 클래스별 오답 분석 (잘못 예측한 confidence가 높은 순)")
    print("=" * 80)
    
    all_class_errors = {}
    
    for true_class in class_names:
        # 해당 클래스의 샘플 중 오답인 것들
        class_errors = errors_df[errors_df['true_label'] == true_class].copy()
        
        if len(class_errors) == 0:
            print(f"\n📌 [{true_class}] 오답 없음")
            continue
        
        # confidence(잘못 예측한 클래스의 확률) 높은 순으로 정렬
        class_errors = class_errors.sort_values('confidence', ascending=False)
        
        total_class_samples = len(results_df[results_df['true_label'] == true_class])
        error_rate = len(class_errors) / total_class_samples * 100
        
        print(f"\n📌 [{true_class}] 오답: {len(class_errors)}개 / {total_class_samples}개 (오답률: {error_rate:.1f}%)")
        print("-" * 80)
        
        # 상위 k개 출력
        for idx, row in class_errors.head(top_k).iterrows():
            print(f"  이미지: {row['image_path']}")
            print(f"    정답: {row['true_label']} → 예측: {row['predicted_label']} (conf: {row['confidence']:.4f})")
            
            # 각 클래스 확률 표시
            prob_str = " | ".join([f"{c}: {row[f'prob_{c}']:.3f}" for c in class_names])
            print(f"    확률: {prob_str}")
            print()
        
        all_class_errors[true_class] = class_errors
    
    # 예측 클래스별 분석 (어떤 클래스로 잘못 예측했는지)
    print("\n" + "=" * 80)
    print("🎯 잘못 예측된 클래스별 분석 (이 클래스로 잘못 예측한 것들)")
    print("=" * 80)
    
    for pred_class in class_names:
        # 이 클래스로 잘못 예측된 것들
        wrong_as_this = errors_df[errors_df['predicted_label'] == pred_class].copy()
        
        if len(wrong_as_this) == 0:
            print(f"\n🎯 [{pred_class}]로 잘못 예측된 샘플 없음")
            continue
        
        wrong_as_this = wrong_as_this.sort_values('confidence', ascending=False)
        
        print(f"\n🎯 [{pred_class}]로 잘못 예측된 샘플: {len(wrong_as_this)}개")
        print("-" * 80)
        
        # 실제 클래스 분포
        true_dist = wrong_as_this['true_label'].value_counts()
        print(f"  실제 클래스 분포: {dict(true_dist)}")
        
        # 상위 k개 출력
        for idx, row in wrong_as_this.head(top_k).iterrows():
            print(f"  이미지: {row['image_path']}")
            print(f"    정답: {row['true_label']} → 예측: {row['predicted_label']} (conf: {row['confidence']:.4f})")
            print()
    
    # Confusion Matrix 요약
    print("\n" + "=" * 80)
    print("📈 Confusion Matrix 요약")
    print("=" * 80)
    
    confusion = pd.crosstab(
        errors_df['true_label'], 
        errors_df['predicted_label'],
        margins=True
    )
    print(confusion)
    
    # 결과 저장 (단순화된 형식)
    if output_path:
        # 필요한 컬럼만 선택 (product_id가 있으면 포함)
        cols = ['true_label', 'predicted_label', 'confidence', 'image_path']
        if has_product_id:
            cols.insert(0, 'product_id')
        simple_errors = errors_df[cols].copy()

        # 1차: true_label, 2차: confidence 내림차순 정렬
        simple_errors = simple_errors.sort_values(
            ['true_label', 'confidence'], 
            ascending=[True, False]
        )

        # true_label이 바뀔 때마다 빈 줄 추가
        rows_with_separator = []
        prev_label = None
        for _, row in simple_errors.iterrows():
            if prev_label is not None and row['true_label'] != prev_label:
                # 빈 줄 추가
                empty_row = {col: '' for col in cols}
                rows_with_separator.append(empty_row)
            rows_with_separator.append(row.to_dict())
            prev_label = row['true_label']

        final_df = pd.DataFrame(rows_with_separator)
        
        # 디렉토리 생성
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        final_df.to_csv(output_path, index=False)
        print(f"\n💾 오답 데이터 저장: {output_path}")
    
    return results_df, errors_df


def main():
    parser = argparse.ArgumentParser(description='오답 분석')
    parser.add_argument('--model', type=str, 
                       default='results/run_20260106_140031_32d4e2c3/model/best_model.pth',
                       help='모델 경로')
    parser.add_argument('--test-data', type=str,
                       default='data/test_data.csv',
                       help='테스트 데이터 경로')
    parser.add_argument('--top-k', type=int, default=40,
                       help='각 클래스별 상위 k개 출력')
    parser.add_argument('--output', type=str, default='error_results/error_analysis_test.csv',
                       help='오답 데이터 저장 경로')
    
    args = parser.parse_args()
    
    analyze_errors(
        model_path=args.model,
        test_data_path=args.test_data,
        top_k=args.top_k,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
