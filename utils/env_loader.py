# utils/env_loader.py
"""
환경변수 로더 - 중복 로드 방지
"""

import os
import logging

logger = logging.getLogger(__name__)

# 전역 플래그
_env_loaded = False

def load_env_once():
    """환경변수를 한 번만 로드"""
    global _env_loaded
    if _env_loaded:
        return
    
    # 현재 디렉토리와 상위 디렉토리에서 .env 파일 찾기
    env_paths = ['.env', '../.env', '../../.env']
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            logger.info(f"환경변수 파일 발견: {os.path.abspath(env_path)}")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and '=' in line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")  # 따옴표 제거
                            os.environ[key] = value
                            logger.debug(f"환경변수 로드: {key}={value}")
                _env_loaded = True
                logger.info("환경변수 로드 완료")
                break
            except Exception as e:
                logger.error(f"환경변수 파일 읽기 실패 {env_path}: {e}")
    else:
        logger.warning("환경변수 파일을 찾을 수 없습니다")
        _env_loaded = True

def get_env_var(key: str, default: str = None) -> str:
    """환경변수 가져오기 (필요시 로드)"""
    if not _env_loaded:
        load_env_once()
    return os.getenv(key, default)

