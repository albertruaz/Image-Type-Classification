"""
Error Analysis Visualization Script

각 true_label별로 k개의 오분류 샘플을 선택하고,
4개씩 하나의 이미지(grid)로 만들어 저장합니다.
각 셀에는 실제 이미지, true label, predicted label, confidence가 표시됩니다.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import requests
from io import BytesIO
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

from utils.env_loader import get_env_var


def load_image_from_path(image_path: str, cloudfront_domain: str = None) -> Optional[Image.Image]:
    """이미지 경로에서 이미지 로드 (URL 또는 로컬)"""
    try:
        if image_path.startswith(('http://', 'https://')):
            full_url = image_path
        elif cloudfront_domain:
            full_url = f"https://{cloudfront_domain}/{image_path}"
        else:
            # 로컬 파일
            if os.path.exists(image_path):
                return Image.open(image_path).convert('RGB')
            return None
        
        response = requests.get(full_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"이미지 로드 실패: {image_path} - {e}")
        return None


def create_error_grid(samples: pd.DataFrame, 
                      cloudfront_domain: str,
                      title: str,
                      save_path: str,
                      grid_size: int = 4) -> None:
    """
    4개의 샘플을 2x2 그리드로 시각화하여 저장
    
    Args:
        samples: 샘플 데이터프레임 (최대 4개)
        cloudfront_domain: CloudFront 도메인
        title: 이미지 제목
        save_path: 저장 경로
        grid_size: 한 이미지당 샘플 수 (기본 4)
    """
    n_samples = len(samples)
    if n_samples == 0:
        return
    
    # 2x2 그리드 설정
    n_cols = 2
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for idx in range(grid_size):
        ax = axes[idx]

        if idx < n_samples:
            row = samples.iloc[idx]
            true_label = row['true_label']
            pred_label = row['predicted_label']
            confidence = row['confidence']
            image_path = row['image_path']
            # id 또는 product_id 추출
            product_id = row['product_id'] if 'product_id' in row else (row['id'] if 'id' in row else None)

            # 이미지 로드
            image = load_image_from_path(image_path, cloudfront_domain)

            if image is not None:
                ax.imshow(image)
            else:
                # 이미지 로드 실패 시 검은색 배경
                ax.imshow(Image.new('RGB', (224, 224), color='gray'))
                ax.text(0.5, 0.5, 'Image\nLoad\nFailed', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, color='white')

            # 정보 텍스트 (이미지 아래)
            if product_id is not None:
                info_text = f"ID: {product_id}\nTrue: {true_label}\nPred: {pred_label}\nConf: {confidence:.4f}"
            else:
                info_text = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.4f}"
            ax.set_xlabel(info_text, fontsize=10, fontweight='bold')
        else:
            # 빈 셀
            ax.axis('off')
            continue

        ax.set_xticks([])
        ax.set_yticks([])

        # 예측이 틀린 경우 빨간색 테두리
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"저장 완료: {save_path}")


def visualize_errors_by_label(csv_path: str, 
                               output_dir: str,
                               k: int = 12,
                               samples_per_image: int = 4) -> None:
    """
    각 true_label별로 k개의 오분류 샘플을 선택하고,
    samples_per_image개씩 하나의 이미지로 만들어 저장
    
    Args:
        csv_path: error_analysis.csv 경로
        output_dir: 출력 디렉토리
        k: 각 true_label별 샘플 수 (기본 12)
        samples_per_image: 한 이미지당 샘플 수 (기본 4)
    """
    # CSV 로드
    df = pd.read_csv(csv_path)
    
    # 빈 행 제거
    df = df.dropna(subset=['true_label', 'predicted_label', 'image_path'])
    df = df[df['true_label'] != '']
    
    print(f"총 오분류 샘플 수: {len(df)}")
    print(f"True Label 종류: {df['true_label'].unique()}")
    
    # CloudFront 도메인 가져오기
    try:
        cloudfront_domain = get_env_var('S3_CLOUDFRONT_DOMAIN')
    except:
        cloudfront_domain = None
        print("CloudFront 도메인을 찾을 수 없습니다. 환경 변수를 확인하세요.")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 true_label별로 처리
    for true_label in df['true_label'].unique():
        print(f"\n=== Processing: {true_label} ===")
        
        # 해당 true_label의 샘플 필터링 (상위 k개, confidence 높은 순)
        label_df = df[df['true_label'] == true_label].copy()
        label_df = label_df.sort_values('confidence', ascending=False).head(k)
        
        n_samples = len(label_df)
        print(f"  샘플 수: {n_samples}")
        
        if n_samples == 0:
            continue
            
        # 클래스별 폴더 생성 (출력 디렉토리/클래스명)
        safe_label = str(true_label).replace('/', '_').replace(' ', '_')
        label_dir = os.path.join(output_dir, safe_label)
        os.makedirs(label_dir, exist_ok=True)
        
        # samples_per_image개씩 나누어 이미지 생성
        n_images = (n_samples + samples_per_image - 1) // samples_per_image
        
        for img_idx in range(n_images):
            start_idx = img_idx * samples_per_image
            end_idx = min(start_idx + samples_per_image, n_samples)
            
            samples = label_df.iloc[start_idx:end_idx]
            
            # 이미지 제목
            title = f"True Label: {true_label} (Part {img_idx + 1}/{n_images})"
            
            # 저장 경로 (클래스 폴더 내부)
            save_path = os.path.join(label_dir, f"part{img_idx + 1}.png")
            
            # 그리드 이미지 생성
            create_error_grid(samples, cloudfront_domain, title, save_path, samples_per_image)
    
    print(f"\n모든 이미지가 '{output_dir}' 디렉토리 내 각 클래스 폴더에 저장되었습니다.")


def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Error Analysis Visualization')
    parser.add_argument('--csv', type=str, default=None,
                        help='error_analysis.csv 파일 경로 (지정하지 않으면 error_results 폴더의 모든 csv 처리)')
    parser.add_argument('--input-dir', type=str, default='error_results',
                        help='CSV 파일을 찾을 디렉토리')
    parser.add_argument('--output', type=str, default='error_results',
                        help='출력 디렉토리 (기본값: error_results)')
    parser.add_argument('--k', type=int, default=40,
                        help='각 true_label별 샘플 수')
    parser.add_argument('--per-image', type=int, default=4,
                        help='한 이미지당 샘플 수')
    
    args = parser.parse_args()
    
    # 처리할 CSV 파일 목록 결정
    csv_files = []
    if args.csv:
        csv_files.append(args.csv)
    elif os.path.exists(args.input_dir):
        # input_dir에서 csv 파일 찾기
        found_files = glob.glob(os.path.join(args.input_dir, '*.csv'))
        csv_files.extend(found_files)
        if not csv_files:
            print(f"'{args.input_dir}' 디렉토리에서 CSV 파일을 찾을 수 없습니다.")
    else:
        # 호환성을 위해 현재 디렉토리의 error_analysis.csv 확인
        if os.path.exists('error_analysis.csv'):
            csv_files.append('error_analysis.csv')
            
    if not csv_files:
        print("처리할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 CSV 파일을 처리합니다: {csv_files}")
    
    for csv_path in csv_files:
        print(f"\nProcessing CSV: {csv_path}")
        
        # 출력 디렉토리 결정: 지정된 output dir / csv 파일명 (확장자 제외)
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        # error_results 폴더 안에 csv 이름으로 폴더를 만들고 그 안에 labeling
        # 사용자가 "내부의 폴더별로" 라고 했으므로, csv별로 폴더를 나누는 게 안전함.
        # 하지만 error_results/error_analysis_train.csv -> error_results/error_analysis_train/label_name/... 가 적절함.
        
        # 만약 args.output이 csv가 있는 폴더와 같다면 (error_results), 하위 폴더 생성
        if os.path.abspath(args.output) == os.path.abspath(os.path.dirname(csv_path)):
             current_output_dir = os.path.join(args.output, csv_name)
        else:
             # 다른 output dir이 지정된 경우
             current_output_dir = os.path.join(args.output, csv_name)
             
        visualize_errors_by_label(
            csv_path=csv_path,
            output_dir=current_output_dir,
            k=args.k,
            samples_per_image=args.per_image
        )

if __name__ == '__main__':
    main()
