#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.device_manager import DeviceManager
from image_classification.inference import ImageClassifierInference

def predict_images(model_path: str, image_paths: List[str]) -> List[Dict[str, Any]]:
    """이미지 경로 리스트를 받아서 예측 결과 리스트 반환"""
    if not image_paths:
        return []
    
    # 추론 엔진 초기화
    device = DeviceManager.get_device()
    inference_engine = ImageClassifierInference(model_path=model_path, device=device)
    
    # 배치 예측 실행
    predictions = inference_engine.predict_batch(
        image_paths=image_paths,
        batch_size=64,
        return_probabilities=True,
        top_k=1
    )
    
    # 결과 정리
    results = []
    for pred in predictions:
        if 'error' not in pred:
            results.append({
                'image_path': pred['image_path'],
                'predicted_class': pred['predicted_class'],
                'confidence': round(pred['confidence'], 4)
            })
        else:
            results.append({
                'image_path': pred['image_path'],
                'predicted_class': 'ERROR',
                'confidence': 0.0
            })
    
    return results

def main():
    model_path = "results/run_20250812_163937_0f5fe933/model/best_model.pth"
    
    test_image_paths = [
        "product/unhashed/4250e3b9-113d-4bf8-aa98-cc9e8b3f080a-978451651",
        "product/unhashed/922c1440-348e-42fa-9a51-07da43260a44--1169411216", 
        "product/unhashed/31baef89-329d-485a-b9a0-76989c0ebc2d-1254761449",
        "product/unhashed/b2300cc8-a353-448c-8225-c648dac8f3b6--787488915",
        "product/unhashed/f97c42d5-d9a8-49d0-9d9a-505119d8f290--1494358730"
    ]
    
    results = predict_images(model_path, test_image_paths)
    return results

if __name__ == "__main__":
    main()
