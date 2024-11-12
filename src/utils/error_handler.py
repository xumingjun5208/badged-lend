import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """处理错误并返回统一的错误响应"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 记录错误
        self.logger.error(
            f"Error occurred: {error_type} - {error_msg}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        
        # 返回统一的错误响应
        return {
            "success": False,
            "error": {
                "type": error_type,
                "message": error_msg
            },
            "data": None
        }
    
    def catch_error(self, func: Callable) -> Callable:
        """错误捕获装饰器"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            try:
                result = func(*args, **kwargs)
                return {
                    "success": True,
                    "error": None,
                    "data": result
                }
            except Exception as e:
                return self.handle_error(e)
        return wrapper

# 导出类和函数
__all__ = [
    'ErrorHandler'
] 