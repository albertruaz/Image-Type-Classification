# Image Type Classification

EfficientNet 기반 이미지 타입 분류 시스템

## 환경 설정

### Conda 환경 생성
```bash
conda create -n image_classification python=3.9
conda activate image_classification
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 주요 패키지
- PyTorch 2.0+
- torchvision 0.15+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- wandb (실험 추적)

## 실행 방법

### 기본 학습
```bash
python main.py
```

### 빠른 테스트 (적은 에포크)
```bash
python main.py --quick-test
```

### 추론 모드
```bash
# 단일 이미지 예측
python main.py --mode inference --image-path path/to/image.jpg

# 배치 예측 (CSV 파일)
python main.py --mode inference --csv-path data.csv --output-path results.csv

# 모델 정보 확인
python main.py --mode inference
```

### 커스텀 설정
```bash
python main.py --config custom_config.json
```

## 프로젝트 구조

```
image_type_classification/
├── main.py                      # 메인 실행 스크립트
├── config.json                  # 설정 파일
├── requirements.txt             # 의존성 패키지
├── image_data.csv              # 학습 데이터 (이미지 경로 및 라벨)
│
├── utils/                       # 유틸리티 모듈
│   ├── __init__.py
│   ├── config_manager.py        # 설정 관리 (검증, 기본값)
│   ├── device_manager.py        # 디바이스 관리 (CUDA/MPS/CPU)
│   ├── resource_manager.py      # 리소스 관리 (디렉토리, 디스크)
│   └── logging_utils.py         # 로깅 시스템
│
├── image_classification/        # 이미지 분류 모듈
│   ├── __init__.py
│   ├── cnn_model.py            # CNN 모델 (EfficientNet 기반)
│   ├── dataset.py              # 데이터셋 및 변환
│   ├── trainer.py              # 모델 학습
│   ├── evaluator.py            # 모델 평가
│   ├── inference.py            # 추론 엔진
│   └── losses.py               # 손실 함수 (Focal Loss 등)
│
├── database/                    # 데이터베이스 모듈
│   ├── __init__.py
│   ├── base_connector.py        # 기본 커넥터
│   └── csv_connector.py         # CSV 데이터 처리
│
├── models/                      # 저장된 모델
│   └── best_model.pth          # 최고 성능 모델
│
├── results/                     # 평가 결과
│   ├── confusion_matrix.png     # 혼동 행렬
│   ├── classification_report.png # 분류 리포트
│   ├── evaluation_results.json  # 평가 지표
│   └── final_results.json      # 최종 결과
│
├── logs/                        # 로그 파일
│   ├── application.log          # 애플리케이션 로그
│   ├── error.log               # 에러 로그
│   └── training_curves.png     # 학습 곡선
│
├── checkpoints/                 # 체크포인트
│   └── latest_checkpoint.pth   # 최신 체크포인트
│
└── wandb/                       # Weights & Biases 실험 추적
    └── run-*/                  # 실험 기록들
```

## 모델 구조

### 아키텍처
- **백본**: EfficientNet-B0 (사전 훈련된 모델)
- **분류 헤드**: Linear layers with LayerNorm and Dropout
- **출력**: 다중 클래스 분류 (소프트맥스)

### 주요 특징
- 전이 학습: ImageNet 사전 훈련 가중치 사용
- 점진적 해동: 초기 에포크는 백본 고정, 이후 전체 학습
- 클래스 불균형 처리: 가중치 기반 손실 함수
- 데이터 증강: 회전, 색상 변화, 좌우 반전

### 학습 과정
1. CSV 데이터 로드 및 분할 (train/val/test)
2. 데이터 로더 생성 (배치 처리, 증강)
3. 모델 생성 및 초기화
4. 학습 실행 (조기 종료, 학습률 스케줄링)
5. 모델 평가 및 결과 저장

### 평가 지표
- 정확도 (Accuracy)
- F1-score (Macro/Weighted)
- 정밀도 (Precision)
- 재현율 (Recall)
- 혼동 행렬 (Confusion Matrix)

## 설정 파일 (config.json)

주요 설정 섹션:
- `data`: 데이터 경로, 분할 비율
- `model`: 모델 아키텍처 설정
- `training`: 학습 하이퍼파라미터
- `augmentation`: 데이터 증강 설정
- `logging`: 로깅 및 wandb 설정
- `paths`: 출력 디렉토리 경로# Image-Type-Classification
