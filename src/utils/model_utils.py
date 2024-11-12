import torch
import numpy as np
import shap
from typing import Optional, Any, Dict, List, Union
import logging

class ModelWrapper:
    """模型包装器基类"""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.explainer = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_model(self) -> None:
        """加载模型"""
        raise NotImplementedError
        
    def predict(self, inputs: Any) -> Dict[str, Any]:
        """预测函数"""
        raise NotImplementedError

    def get_shap_values(self, data: Any) -> np.ndarray:
        """获取SHAP值"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
            
        try:
            shap_values = self.explainer.shap_values(data)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 获取正类的SHAP值
            return shap_values
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {e}")
            return np.array([])

    def get_expected_value(self) -> float:
        """获取基准值"""
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
            
        try:
            if hasattr(self.explainer, 'expected_value'):
                if isinstance(self.explainer.expected_value, list):
                    return self.explainer.expected_value[1]  # 获取正类的基准值
                return self.explainer.expected_value
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting expected value: {e}")
            return 0.0

    @property
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None 