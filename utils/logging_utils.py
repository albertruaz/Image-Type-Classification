# utils/logging_utils.py
"""
로깅 유틸리티

LoggingManager:
- 통합 로깅 설정
- 파일 및 콘솔 로깅
- 에러 핸들링
- 성능 로깅
"""

import logging
import sys
import os
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import functools
import time

class ColoredFormatter(logging.Formatter):
    """컬러 로그 포맷터"""
    
    # ANSI 색상 코드
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # 로그 레벨에 따른 색상 적용
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # 기본 포맷팅
        formatted = super().format(record)
        
        # 터미널 환경에서만 색상 적용
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            return f"{color}{formatted}{reset}"
        else:
            return formatted

class LoggingManager:
    """로깅 관리자 클래스"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loggers = {}
            self.log_dir = None
            LoggingManager._initialized = True
    
    def setup_logging(self, 
                     log_dir: str = 'logs',
                     log_level: str = 'INFO',
                     console_output: bool = True,
                     file_output: bool = True,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5) -> logging.Logger:
        """
        로깅 설정
        
        Args:
            log_dir: 로그 파일 디렉토리
            log_level: 로그 레벨
            console_output: 콘솔 출력 여부
            file_output: 파일 출력 여부
            max_file_size: 최대 파일 크기 (bytes)
            backup_count: 백업 파일 개수
            
        Returns:
            설정된 로거
        """
        self.log_dir = log_dir
        
        # 로그 디렉토리 생성
        if file_output:
            os.makedirs(log_dir, exist_ok=True)
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 포맷터 설정
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 콘솔 핸들러
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        # 파일 핸들러들
        if file_output:
            from logging.handlers import RotatingFileHandler
            
            # 일반 로그 파일
            log_file = os.path.join(log_dir, 'application.log')
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # 에러 전용 로그 파일
            error_log_file = os.path.join(log_dir, 'error.log')
            error_handler = RotatingFileHandler(
                error_log_file, maxBytes=max_file_size, backupCount=backup_count
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
        
        logger = logging.getLogger(__name__)
        logger.info("로깅 시스템 초기화 완료")
        
        return root_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """특정 이름의 로거 반환"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def log_exception(self, logger: logging.Logger, 
                     exception: Exception, 
                     context: str = None):
        """예외 로깅"""
        error_msg = f"예외 발생: {type(exception).__name__}: {str(exception)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        logger.error(error_msg)
        logger.debug("상세 스택 트레이스:", exc_info=True)
    
    def log_performance(self, logger: logging.Logger, 
                       operation: str, 
                       duration: float, 
                       **kwargs):
        """성능 로깅"""
        perf_msg = f"성능 측정 - {operation}: {duration:.4f}초"
        
        if kwargs:
            details = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            perf_msg += f" ({details})"
        
        logger.info(perf_msg)
    
    def create_session_log(self, session_name: str) -> str:
        """세션별 로그 파일 생성"""
        if not self.log_dir:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_log_file = os.path.join(
            self.log_dir, f"{session_name}_{timestamp}.log"
        )
        
        return session_log_file

def setup_project_logging(config: Dict[str, Any]) -> logging.Logger:
    """프로젝트 전체 로깅 설정"""
    logging_config = config.get('logging', {})
    
    log_dir = config.get('paths', {}).get('log_dir', 'logs')
    log_level = logging_config.get('level', 'INFO')
    
    manager = LoggingManager()
    return manager.setup_logging(
        log_dir=log_dir,
        log_level=log_level,
        console_output=True,
        file_output=True
    )

def log_execution_time(logger: logging.Logger = None):
    """실행 시간 측정 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # 로거가 지정되지 않으면 함수명으로 로거 생성
            actual_logger = logger or logging.getLogger(func.__module__)
            
            try:
                actual_logger.info(f"{func.__name__} 실행 시작")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                actual_logger.info(f"{func.__name__} 실행 완료: {duration:.4f}초")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                actual_logger.error(f"{func.__name__} 실행 실패 ({duration:.4f}초): {e}")
                raise
        
        return wrapper
    return decorator

def handle_exceptions(logger: logging.Logger = None, 
                     reraise: bool = True,
                     default_return: Any = None):
    """예외 처리 데코레이터"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            actual_logger = logger or logging.getLogger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                manager = LoggingManager()
                manager.log_exception(actual_logger, e, f"함수 {func.__name__}")
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator

class ProgressLogger:
    """진행상황 로깅 클래스"""
    
    def __init__(self, logger: logging.Logger, total: int, 
                 description: str = "Processing"):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 10  # 10초마다 로깅
    
    def update(self, count: int = 1):
        """진행상황 업데이트"""
        self.current += count
        current_time = time.time()
        
        # 로그 간격 체크
        if current_time - self.last_log_time >= self.log_interval:
            self._log_progress()
            self.last_log_time = current_time
        
        # 완료시 최종 로깅
        if self.current >= self.total:
            self._log_progress(final=True)
    
    def _log_progress(self, final: bool = False):
        """진행상황 로깅"""
        elapsed = time.time() - self.start_time
        progress_percent = (self.current / self.total) * 100
        
        if final:
            self.logger.info(
                f"{self.description} 완료: {self.current}/{self.total} "
                f"(100.0%) - 총 시간: {elapsed:.1f}초"
            )
        else:
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({progress_percent:.1f}%) - "
                f"속도: {rate:.1f}/s, ETA: {eta:.0f}초"
            )

