import unittest
import os
import sys
import torch
import numpy as np

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.ischemic.ischemic_text_model import predict_ischemic_text
from src.hemorrhagic.hemorrhagic_text_model import predict_hemorrhagic_text

class TestModels(unittest.TestCase):
    """模型测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_text = "患者，男，68岁，因突发意识障碍4小时入院。"
    
    def test_ischemic_text_model(self):
        """测试缺血性卒中文本模型"""
        try:
            html, prob = predict_ischemic_text(self.test_text)
            self.assertIsInstance(html, str)
            self.assertIsInstance(prob, float)
            self.assertTrue(0 <= prob <= 1)
        except Exception as e:
            self.fail(f"预测失败: {str(e)}")
    
    def test_hemorrhagic_text_model(self):
        """测试出血性卒中文本模型"""
        try:
            html, prob = predict_hemorrhagic_text(self.test_text)
            self.assertIsInstance(html, str)
            self.assertIsInstance(prob, float)
            self.assertTrue(0 <= prob <= 1)
        except Exception as e:
            self.fail(f"预测失败: {str(e)}")

if __name__ == '__main__':
    unittest.main() 