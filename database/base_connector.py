# database/base_connector.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os

class BaseConnector(ABC):
    """
    모든 데이터 커넥터의 기본 클래스
    """
    
    def __init__(self):
        self.engine = None
        self.tunnel = None
        self._is_connected = False
    
    @abstractmethod
    def connect(self):
        """데이터베이스 연결"""
        pass
    
    @abstractmethod
    def close(self):
        """데이터베이스 연결 종료"""
        pass
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._is_connected
    
    @abstractmethod
    def get_all_data(self):
        """전체 데이터 반환"""
        pass
    
    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """데이터 요약 정보 반환"""
        pass