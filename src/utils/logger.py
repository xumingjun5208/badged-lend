import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    """日志管理器"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()
        
    def _setup_logger(self) -> None:
        """设置日志记录器"""
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志记录器
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # 创建日志文件处理器
        today = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(self.log_dir, f"{self.name}_{today}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.logger = logger
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.logger 