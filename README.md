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


### 데이터 분할

```bash
python divide_data.py --random-state 2025
```

### 기본 학습

```bash
nohup python main.py 1>log.log 2>&1 &
```

### 추론 실행

#### 테스트 실행 (test.py)

```bash
# 기본 추론 (전체 데이터셋)
python test.py --model-path results/run_xxx/model/best_model.pth

# 단일 이미지 예측
python test.py --model-path results/run_xxx/model/best_model.pth --image-path path/to/image.jpg

# 배치 예측 (CSV 파일)
python test.py --model-path results/run_xxx/model/best_model.pth --csv-path data.csv --output-path results.csv
```

**특징:**

- 결과가 `test_result/run_YYYYMMDD_HHMMSS_UUID/` 폴더에 저장됨
- 메인 결과 CSV에 `true_class` 컬럼 포함
- 틀린 예측만 모아서 `wrong_train.csv`, `wrong_valid.csv`, `wrong_test.csv` 생성
- 틀린 결과는 confidence 높은 순으로 정렬됨
- 개당 추론 시간 계산 및 로깅

#### 백엔드용 API (run.py)

```python
from run import ImageClassificationAPI

# API 초기화
api = ImageClassificationAPI(model_path="results/run_xxx/model/best_model.pth")

# 이미지 배열 예측
image_paths = [
    "product/unhashed/image1.jpg",
    "product/unhashed/image2.jpg"
]
results = api.predict(image_paths)
```

**특징:**

- 배열 입력 → 배열 출력 구조
- 결과를 파일로 저장하지 않음 (로그만 출력)
- 상세한 로그 (처리 시간, 성공/실패 수, 클래스 분포)
- 에러 처리 및 실패 시 ERROR 반환

### 커스텀 설정

```bash
python main.py --config custom_config.json
```

## 프로젝트 구조

```
image_type_classification/
├── main.py                      # 메인 실행 스크립트 (학습)
├── test.py                     # 테스트 실행 스크립트 (추론 및 분석)
├── run.py                      # 백엔드용 추론 API
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
├── results/                     # 학습 결과
│   └── run_YYYYMMDD_HHMMSS_UUID/ # 실행별 결과 폴더
│       ├── model/              # 모델 파일
│       ├── logs/              # 로그 파일
│       ├── config.json        # 실행 설정
│       └── run_metadata.json  # 실행 메타데이터
│
├── test_result/                # 테스트 결과
│   └── run_YYYYMMDD_HHMMSS_UUID/ # 테스트 실행별 폴더
│       ├── inference_results.csv  # 전체 예측 결과
│       ├── wrong_train.csv       # 훈련 데이터 틀린 예측
│       ├── wrong_valid.csv       # 검증 데이터 틀린 예측
│       └── wrong_test.csv        # 테스트 데이터 틀린 예측
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
