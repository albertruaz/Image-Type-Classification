#!/usr/bin/env python3
# tune_hyperparameters.py
"""
이미지 타입 분류 하이퍼파라미터 튜닝 스크립트

Validation 데이터를 기반으로 다양한 하이퍼파라미터 조합을 테스트하고
최적의 설정을 찾습니다. main.py의 ImageTypeClassificationPipeline을 재사용합니다.

사용법:
    python tune_hyperparameters.py                      # 권장 설정 테스트
    python tune_hyperparameters.py --quick-test         # 빠른 테스트 (3 에폭)
    python tune_hyperparameters.py --mode grid          # 그리드 서치
"""

import argparse
import json
import os
import sys
import logging
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import itertools
import random

import torch
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# main.py에서 파이프라인 재사용
from main import ImageTypeClassificationPipeline
from utils.device_manager import DeviceManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 튜닝할 하이퍼파라미터 정의
# ============================================================================

# 최종 기준 세팅 + 추가로 만질 만한 포인트만 구성:
# - 기준: focal_gamma=1.0, class_weight 미사용, dropout 0.2, cosine + warmup 1
# - 실험: class-wise threshold, full_shot만 낮춘 per-class gamma

RECOMMENDED_CONFIGS = [
    {
        'name': 'baseline_fixed',
        'description': '최종 기준 세팅 재현',
        'overrides': {
            'training.focal_loss': True,
            'training.focal_gamma': 1.0,
            'training.focal_alpha': 0.25,
            'training.use_class_weights': False,
            'training.learning_rate': 1e-4,
            'training.scheduler': 'cosine',
            'training.warmup_epochs': 1,
            'augmentation.strength': 'medium',
            'model.dropout_rate': 0.2,
        }
    },
    {
        'name': 'classwise_thresholds',
        'description': '클래스별 threshold 후처리로 recall 보정',
        'overrides': {
            'training.focal_loss': True,
            'training.focal_gamma': 1.0,
            'training.focal_alpha': 0.25,
            'training.use_class_weights': False,
            'training.learning_rate': 1e-4,
            'training.scheduler': 'cosine',
            'training.warmup_epochs': 1,
            'augmentation.strength': 'medium',
            'model.dropout_rate': 0.2,
            'inference.class_thresholds': {
                'care_label': 0.50,
                'detail_shot': 0.55,
                'full_shot': 0.35,
                'neck_label': 0.50
            }
        }
    },
    {
        'name': 'focal_gamma_full_shot_low',
        'description': 'full_shot만 약한 gamma로 recall 회복',
        'overrides': {
            'training.focal_loss': True,
            'training.focal_alpha': 0.25,
            'training.use_class_weights': False,
            'training.learning_rate': 1e-4,
            'training.scheduler': 'cosine',
            'training.warmup_epochs': 1,
            'augmentation.strength': 'medium',
            'model.dropout_rate': 0.2,
            'training.focal_gamma': {
                'care_label': 1.0,
                'detail_shot': 1.0,
                'full_shot': 0.5,
                'neck_label': 1.0
            }
        }
    }
]


def apply_overrides(config: Dict, overrides: Dict[str, Any]) -> Dict:
    """
    설정에 오버라이드 적용
    
    Args:
        config: 기본 설정 딕셔너리
        overrides: 'section.key': value 형태의 오버라이드
        
    Returns:
        수정된 설정 딕셔너리
    """
    config = copy.deepcopy(config)
    
    for key_path, value in overrides.items():
        parts = key_path.split('.')
        target = config
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value
    
    return config


def run_single_experiment(
    base_config_path: str,
    experiment_name: str,
    overrides: Dict[str, Any],
    output_dir: str,
    epochs: int,
    use_wandb: bool = False
) -> Dict[str, Any]:
    """
    단일 튜닝 실험 실행 (main.py의 파이프라인 재사용)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🧪 실험 시작: {experiment_name}")
    logger.info(f"   오버라이드: {overrides}")
    logger.info(f"{'='*60}")
    
    wandb_run = None
    try:
        # 파이프라인 생성 (기본 설정 로드)
        pipeline = ImageTypeClassificationPipeline(config_path=base_config_path)
        
        # 오버라이드 적용
        pipeline.config = apply_overrides(pipeline.config, overrides)
        
        # 튜닝용 설정 조정
        pipeline.config['training']['epochs'] = epochs
        pipeline.config['training']['patience'] = max(epochs // 2, 5)
        pipeline.config['logging']['use_wandb'] = use_wandb
        pipeline.config['logging']['wandb_prefix'] = 'tuning_'
        
        # 실험명 기반으로 run_id 수정
        pipeline.run_id = f"tune_{experiment_name}_{datetime.now().strftime('%H%M%S')}"
        
        # 결과 디렉토리 수정
        experiment_dir = os.path.join(output_dir, experiment_name)
        pipeline.run_paths = {
            'run_dir': experiment_dir,
            'result_dir': experiment_dir,
            'model_dir': os.path.join(experiment_dir, 'model'),
            'log_dir': os.path.join(experiment_dir, 'logs'),
            'checkpoint_dir': os.path.join(experiment_dir, 'checkpoints')
        }
        for path in pipeline.run_paths.values():
            os.makedirs(path, exist_ok=True)
        
        # config의 paths도 업데이트 (trainer가 config에서 경로를 읽음)
        pipeline.config['paths']['result_dir'] = experiment_dir
        pipeline.config['paths']['model_dir'] = pipeline.run_paths['model_dir']
        pipeline.config['paths']['log_dir'] = pipeline.run_paths['log_dir']
        pipeline.config['paths']['checkpoint_dir'] = pipeline.run_paths['checkpoint_dir']
        
        # 설정 저장
        config_save_path = os.path.join(experiment_dir, 'config.json')
        with open(config_save_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline.config, f, indent=2, ensure_ascii=False)

        # wandb 초기화 (튜닝 실험 단위)
        if use_wandb and WANDB_AVAILABLE:
            try:
                wandb_run = wandb.init(
                    project=pipeline.config.get('logging', {}).get('wandb_project', 'image-classification'),
                    entity=pipeline.config.get('logging', {}).get('wandb_entity', None),
                    config=pipeline.config,
                    name=f"tuning_{pipeline.run_id}",
                    tags=['image-classification', 'pytorch', 'tuning', experiment_name]
                )
                logger.info(f"wandb 실행 시작: {wandb.run.url}")
            except Exception as e:
                logger.warning(f"wandb 초기화 실패: {e}")
                wandb_run = None

        # 학습 파이프라인 실행 (main.py의 메서드 재사용)
        # 데이터 로드
        train_df, val_df, test_df = pipeline.load_data()
        
        # 데이터 로더 생성
        train_loader, val_loader, test_loader, label_encoder = pipeline.create_data_loaders(
            train_df, val_df, test_df
        )
        
        # 모델 생성 및 학습
        model = pipeline.create_model(pipeline.NUM_CLASSES)
        training_results = pipeline.train_model(model, train_loader, val_loader)
        
        # Validation으로 평가 (test는 최종 평가용으로 보존)
        from image_classification.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(
            model=model,
            test_loader=val_loader,  # validation 사용
            class_names=pipeline.CLASS_NAMES,
            device=pipeline.device,
            save_dir=experiment_dir,
            class_thresholds=pipeline.config.get('inference', {}).get('class_thresholds')
        )
        val_eval_results = evaluator.evaluate(save_results=False)
        
        # 결과 정리
        results = {
            'experiment_name': experiment_name,
            'overrides': overrides,
            'status': 'success',
            'best_val_loss': training_results.get('best_val_loss', float('inf')),
            'best_val_accuracy': training_results.get('best_val_accuracy', 0),
            'val_f1_weighted': val_eval_results['metrics'].get('f1_weighted', 0),
            'val_f1_macro': val_eval_results['metrics'].get('f1_macro', 0),
            'val_accuracy': val_eval_results['metrics'].get('accuracy', 0),
            'training_time': training_results.get('training_time', 0),
        }
        
        # 클래스별 성능
        for cls in pipeline.CLASS_NAMES:
            cls_report = val_eval_results['classification_report'].get(cls, {})
            results[f'{cls}_precision'] = cls_report.get('precision', 0)
            results[f'{cls}_recall'] = cls_report.get('recall', 0)
            results[f'{cls}_f1'] = cls_report.get('f1-score', 0)

        if wandb_run:
            try:
                prefix = pipeline.config.get('logging', {}).get('wandb_prefix', '')
                summary_metrics = {
                    'final/experiment_name': experiment_name,
                    'final/val_accuracy': results['val_accuracy'],
                    'final/val_f1_weighted': results['val_f1_weighted'],
                    'final/val_f1_macro': results['val_f1_macro'],
                    'final/best_val_loss': results['best_val_loss'],
                    'final/best_val_accuracy': results['best_val_accuracy']
                }
                if prefix:
                    summary_metrics = {f"{prefix}{key}": value for key, value in summary_metrics.items()}
                wandb.log(summary_metrics)
            except Exception as e:
                logger.warning(f"wandb 요약 로깅 실패: {e}")

        # 결과 저장
        results_path = os.path.join(experiment_dir, 'tuning_result.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 실험 완료: {experiment_name}")
        logger.info(f"   Val Accuracy: {results['val_accuracy']:.4f}")
        logger.info(f"   Val F1 (weighted): {results['val_f1_weighted']:.4f}")
        logger.info(f"   detail_shot F1: {results.get('detail_shot_f1', 0):.4f}")
        logger.info(f"   full_shot F1: {results.get('full_shot_f1', 0):.4f}")
        
        # 메모리 정리
        del model, pipeline, evaluator
        torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 실험 실패: {experiment_name} - {e}")
        import traceback
        traceback.print_exc()
        return {
            'experiment_name': experiment_name,
            'overrides': overrides,
            'status': 'failed',
            'error': str(e)
        }
    finally:
        if wandb_run:
            try:
                wandb.finish()
                logger.info("wandb 실행 종료")
            except Exception as e:
                logger.warning(f"wandb 종료 실패: {e}")


def run_tuning(
    base_config_path: str,
    output_dir: str,
    configs: List[Dict],
    epochs: int,
    use_wandb: bool = False
) -> pd.DataFrame:
    """
    여러 설정으로 튜닝 실행
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"tuning_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"📁 튜닝 결과 디렉토리: {output_dir}")
    logger.info(f"📊 총 {len(configs)}개 실험 예정")
    
    all_results = []
    
    for i, cfg in enumerate(configs):
        logger.info(f"\n[{i+1}/{len(configs)}] {cfg['name']}: {cfg.get('description', '')}")
        
        results = run_single_experiment(
            base_config_path=base_config_path,
            experiment_name=cfg['name'],
            overrides=cfg['overrides'],
            output_dir=output_dir,
            epochs=epochs,
            use_wandb=use_wandb
        )
        
        results['description'] = cfg.get('description', '')
        all_results.append(results)
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame(all_results)
    
    # 성공한 실험만 정렬
    success_df = results_df[results_df['status'] == 'success']
    if len(success_df) > 0:
        success_df = success_df.sort_values('val_f1_weighted', ascending=False)
    
    # 결과 저장
    results_df.to_csv(os.path.join(output_dir, 'tuning_results.csv'), index=False)
    
    with open(os.path.join(output_dir, 'tuning_results.json'), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 최종 요약
    logger.info("\n" + "=" * 60)
    logger.info("📊 튜닝 결과 요약")
    logger.info("=" * 60)
    
    if len(success_df) > 0:
        print("\n🏆 Validation F1 (weighted) 기준 상위 결과:")
        display_cols = ['experiment_name', 'val_f1_weighted', 'val_accuracy', 
                        'detail_shot_f1', 'full_shot_f1']
        available_cols = [c for c in display_cols if c in success_df.columns]
        print(success_df[available_cols].to_string(index=False))
        
        # 최적 설정 저장
        best_name = success_df.iloc[0]['experiment_name']
        best_overrides = success_df.iloc[0].get('overrides', {})
        
        # 최적 config 생성
        from utils.config_manager import ConfigManager
        config_manager = ConfigManager(base_config_path)
        best_config = apply_overrides(config_manager.get_config(), best_overrides)
        
        best_config_path = os.path.join(output_dir, 'best_config.json')
        with open(best_config_path, 'w', encoding='utf-8') as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ 최적 설정 저장: {best_config_path}")
        logger.info(f"   최적 실험: {best_name}")
        logger.info(f"   Val F1 (weighted): {success_df.iloc[0]['val_f1_weighted']:.4f}")
    
    return results_df


def generate_grid_configs(max_experiments: int = 20) -> List[Dict]:
    """그리드 서치용 설정 생성"""
    grid_params = {
        'training.learning_rate': [5e-5, 1e-4, 2e-4],
        'model.dropout_rate': [0.3, 0.4],
        'training.focal_loss': [True],
        'training.focal_gamma': [2.0, 3.0],
    }
    
    keys = list(grid_params.keys())
    combinations = list(itertools.product(*[grid_params[k] for k in keys]))
    
    if len(combinations) > max_experiments:
        combinations = random.sample(combinations, max_experiments)
    
    configs = []
    for i, combo in enumerate(combinations):
        overrides = dict(zip(keys, combo))
        configs.append({
            'name': f'grid_{i+1:03d}',
            'description': f'Grid search #{i+1}',
            'overrides': overrides
        })
    
    return configs


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이미지 타입 분류 하이퍼파라미터 튜닝')
    parser.add_argument('--config', type=str, default='config.json',
                        help='기본 설정 파일 경로')
    parser.add_argument('--output-dir', type=str, default='tuning_results',
                        help='튜닝 결과 저장 디렉토리')
    parser.add_argument('--mode', type=str, default='recommended',
                        choices=['recommended', 'grid'],
                        help='튜닝 모드')
    parser.add_argument('--epochs', type=int, default=15,
                        help='각 실험당 에폭 수')
    parser.add_argument('--quick-test', action='store_true',
                        help='빠른 테스트 모드 (3 에폭)')
    parser.add_argument('--max-experiments', type=int, default=20,
                        help='최대 실험 수 (grid 모드)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='wandb 로깅 사용')
    parser.add_argument('--no-wandb', action='store_true',
                        help='wandb 로깅 비활성화')
    
    args = parser.parse_args()
    if args.use_wandb and args.no_wandb:
        logger.error("--use-wandb 와 --no-wandb 는 동시에 사용할 수 없습니다.")
        sys.exit(1)
    
    # 에폭 설정
    epochs = 3 if args.quick_test else args.epochs
    
    logger.info(f"🔧 튜닝 모드: {args.mode}")
    logger.info(f"📊 에폭 수: {epochs}")
    
    # 설정 목록 생성
    if args.mode == 'recommended':
        configs = RECOMMENDED_CONFIGS
    elif args.mode == 'grid':
        configs = generate_grid_configs(args.max_experiments)
    else:
        logger.error(f"지원하지 않는 모드: {args.mode}")
        sys.exit(1)
    
    # wandb 설정 (명시적 플래그가 없으면 config를 따름)
    if args.use_wandb:
        use_wandb = True
    elif args.no_wandb:
        use_wandb = False
    else:
        from utils.config_manager import ConfigManager
        config_manager = ConfigManager(args.config)
        use_wandb = config_manager.get_config().get('logging', {}).get('use_wandb', False)
    
    # 튜닝 실행
    results_df = run_tuning(
        base_config_path=args.config,
        output_dir=args.output_dir,
        configs=configs,
        epochs=epochs,
        use_wandb=use_wandb
    )
    
    logger.info("\n🎉 하이퍼파라미터 튜닝 완료!")
    logger.info(f"📁 결과 저장 위치: {args.output_dir}")


if __name__ == "__main__":
    main()
