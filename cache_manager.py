"""
缓存管理器模块
负责管理API响应的缓存
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, Tuple
from config import ENABLE_CACHE, CACHE_EXPIRY

logger = logging.getLogger(__name__)

class CacheManager:
    _instance: Optional['CacheManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._expiry = CACHE_EXPIRY
        self._enabled = ENABLE_CACHE
        
        logger.info(f"缓存管理器初始化完成，缓存{'启用' if self._enabled else '禁用'}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存的值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，如果不存在或已过期则返回None
        """
        if not self._enabled:
            return None
            
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._expiry):
                logger.debug(f"缓存命中: {key}")
                return value
            logger.debug(f"缓存过期: {key}")
            del self._cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        if not self._enabled:
            return
            
        self._cache[key] = (datetime.now(), value)
        logger.debug(f"设置缓存: {key}")
    
    def clear(self) -> None:
        """清除所有缓存"""
        self._cache.clear()
        logger.info("缓存已清除")
    
    def remove_expired(self) -> None:
        """移除所有过期的缓存项"""
        now = datetime.now()
        expired_keys = [
            key for key, (timestamp, _) in self._cache.items()
            if now - timestamp >= timedelta(seconds=self._expiry)
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.info(f"已移除 {len(expired_keys)} 个过期缓存项")
    
    @property
    def enabled(self) -> bool:
        """获取缓存是否启用"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """设置缓存是否启用"""
        self._enabled = value
        if not value:
            self.clear()
        logger.info(f"缓存{'启用' if value else '禁用'}")
    
    @property
    def size(self) -> int:
        """获取当前缓存项数量"""
        return len(self._cache) 