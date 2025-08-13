# utils/device_manager.py
"""
디바이스 관리자 싱글톤 클래스

DeviceManager:
- 시스템에서 사용 가능한 디바이스 자동 감지
- 싱글톤 패턴으로 중복 디바이스 설정 방지
- CUDA, MPS, CPU 지원
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """디바이스 관리자 싱글톤 클래스"""
    
    _instance = None
    _device = None
    _device_info = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._device is None:
            self._initialize_device()
    
    def _initialize_device(self):
        """디바이스 초기화"""
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self._device_info = {
                'type': 'cuda',
                'name': device_name,
                'memory_gb': f"{memory_gb:.1f}GB",
                'device_count': torch.cuda.device_count()
            }
            logger.info(f"CUDA 디바이스 사용 - {device_name} ({memory_gb:.1f}GB)")
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device('mps')
            self._device_info = {
                'type': 'mps',
                'name': 'Apple Silicon GPU',
                'memory_gb': 'Unified Memory',
                'device_count': 1
            }
            logger.info("MPS (Apple Silicon) 디바이스 사용")
            
        else:
            self._device = torch.device('cpu')
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / 1024**3
            self._device_info = {
                'type': 'cpu',
                'name': f"CPU ({cpu_count} cores)",
                'memory_gb': f"{memory_gb:.1f}GB",
                'device_count': 1
            }
            logger.info(f"CPU 디바이스 사용 - {cpu_count}개 코어, {memory_gb:.1f}GB RAM")
    
    @classmethod
    def get_device(cls, device_config: str = 'auto') -> torch.device:
        """
        디바이스 반환
        
        Args:
            device_config: 디바이스 설정 ('auto', 'cpu', 'cuda', 'mps')
            
        Returns:
            torch.device
        """
        instance = cls()
        
        if device_config == 'auto':
            return instance._device
        else:
            # 수동 디바이스 설정
            requested_device = torch.device(device_config)
            
            # 요청된 디바이스가 사용 가능한지 확인
            if device_config == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA가 요청되었지만 사용할 수 없습니다. CPU를 사용합니다.")
                return torch.device('cpu')
            elif device_config == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                logger.warning("MPS가 요청되었지만 사용할 수 없습니다. CPU를 사용합니다.")
                return torch.device('cpu')
            
            logger.info(f"수동 디바이스 설정: {requested_device}")
            return requested_device
    
    @classmethod
    def get_device_info(cls) -> dict:
        """디바이스 정보 반환"""
        instance = cls()
        return instance._device_info.copy()
    
    @classmethod
    def is_cuda_available(cls) -> bool:
        """CUDA 사용 가능 여부"""
        return torch.cuda.is_available()
    
    @classmethod
    def is_mps_available(cls) -> bool:
        """MPS 사용 가능 여부"""
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    @classmethod
    def get_memory_info(cls) -> dict:
        """메모리 정보 반환 (CUDA인 경우)"""
        instance = cls()
        
        if instance._device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated_gb': f"{allocated:.2f}GB",
                'cached_gb': f"{cached:.2f}GB", 
                'total_gb': f"{total:.1f}GB",
                'free_gb': f"{total - allocated:.2f}GB"
            }
        else:
            return {'message': 'Memory info only available for CUDA devices'}
    
    @classmethod
    def clear_cache(cls):
        """GPU 메모리 캐시 정리 (CUDA인 경우)"""
        instance = cls()
        
        if instance._device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("CUDA 메모리 캐시 정리됨")
        else:
            logger.info("CPU 디바이스에서는 캐시 정리가 필요하지 않습니다")
    
    @classmethod
    def reset_peak_memory_stats(cls):
        """메모리 통계 리셋 (CUDA인 경우)"""
        instance = cls()
        
        if instance._device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            logger.info("CUDA 메모리 통계 리셋됨")
    
    @classmethod
    def set_cuda_device(cls, device_id: int):
        """CUDA 디바이스 ID 설정"""
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            logger.info(f"CUDA 디바이스 {device_id} 설정됨")
        else:
            logger.warning(f"CUDA 디바이스 {device_id}를 설정할 수 없습니다")

