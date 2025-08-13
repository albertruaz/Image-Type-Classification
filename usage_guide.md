# 📚 데이터 분할 및 실행 가이드

## 🔄 새로운 워크플로우

### 1️⃣ 데이터 분할 (한 번만 실행)

```bash
# 기본 설정으로 분할 (8:1:1 비율)
python divide_data.py

# 커스텀 비율로 분할
python divide_data.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

# 다른 랜덤 시드 사용
python divide_data.py --random-state 123

# 이미지 경로 유효성 검사와 함께
python divide_data.py --validate-images --image-base-path /path/to/images
```

**결과물:**

```
data/
├── train_data.csv          # 학습용 데이터
├── validation_data.csv     # 검증용 데이터
├── test_data.csv           # 테스트용 데이터
└── data_split_summary.json # 분할 정보 요약
```

### 2️⃣ 모델 학습

```bash
# 분할된 데이터로 학습 (train + validation 사용)
python main.py

# 빠른 테스트
python main.py --quick-test
```

**학습 시 사용되는 데이터:**

- **Train**: `data/train_data.csv` - 모델 학습
- **Validation**: `data/validation_data.csv` - 모델 검증 및 조기 종료
- **Test**: 학습 중에는 사용하지 않음 (추론 시에만 사용)

### 3️⃣ 추론 실행

```bash
# 확장된 추론 모드 (train/validation/test 모두 실행)
python main.py --mode inference

# 단일 이미지 추론
python main.py --mode inference --image-path test.jpg

# CSV 파일 배치 추론
python main.py --mode inference --csv-path custom_data.csv
```

**추론 시 동작:**

1. **Test 데이터**: 메인 추론 (실제 테스트)
2. **Train 데이터**: 추가 분석용 추론 (과적합 체크)
3. **Validation 데이터**: 추가 분석용 추론 (일관성 체크)

## 📊 출력 결과

### 학습 결과

```
results/run_20250812_154323_abc123/
├── model/
│   ├── best_model.pth
│   └── latest_checkpoint.pth
├── logs/
│   ├── training_curves.png
│   └── training_results.json
├── config.json
├── final_results.json
└── evaluation_results.json
```

### 추론 결과 (확장 모드)

```
results/run_20250812_154323_xyz789/
├── train_inference_results.csv      # Train 세트 추론 결과
├── validation_inference_results.csv # Validation 세트 추론 결과
├── test_inference_results.csv       # Test 세트 추론 결과
└── final_results.json               # 전체 추론 요약
```

## 🎯 주요 장점

### 1. **일관된 데이터 분할**

- 모든 실험에서 동일한 train/val/test 분할 사용
- 결과 비교 시 데이터 편향 제거

### 2. **완전한 성능 분석**

- Test: 실제 성능 측정
- Train: 과적합 정도 확인
- Validation: 학습 과정 검증

### 3. **재현성 보장**

- 분할 시점의 랜덤 시드 저장
- 실행별 독립적인 결과 폴더

## 🔧 설정 옵션

### divide_data.py 옵션

```bash
--input-file        # 입력 CSV 파일 (기본: image_data.csv)
--output-dir        # 출력 디렉토리 (기본: data)
--train-ratio       # 학습 비율 (기본: 0.8)
--val-ratio         # 검증 비율 (기본: 0.1)
--test-ratio        # 테스트 비율 (기본: 0.1)
--random-state      # 랜덤 시드 (기본: 42)
--validate-images   # 이미지 경로 유효성 검사
--image-base-path   # 이미지 기본 경로
```

### main.py 옵션

```bash
--config           # 설정 파일 (기본: config.json)
--mode             # 실행 모드 (train/inference)
--quick-test       # 빠른 테스트 모드
--image-path       # 단일 이미지 추론
--csv-path         # 배치 추론 CSV
--output-path      # 결과 저장 경로
--list-runs        # 실행 목록 확인
```

## ⚠️ 주의사항

1. **첫 실행 전 데이터 분할 필수**

   ```bash
   # 먼저 이것을 실행하세요
   python divide_data.py
   ```

2. **데이터 일관성**

   - 한 번 분할한 후에는 동일한 분할 유지
   - 새로운 분할이 필요하면 기존 data/ 폴더 백업

3. **추론 모드 특징**
   - 기본적으로 train/validation/test 모두 실행
   - 단일 이미지 추론 시에는 전체 세트 실행 안 함

## 🎉 완성된 기능

✅ **데이터 분할 자동화**  
✅ **일관된 데이터 사용**  
✅ **구조화된 결과 저장**  
✅ **확장된 추론 분석**  
✅ **완전한 재현성**

