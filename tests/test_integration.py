import unittest
import gradio.components as gr
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app import demo
from tests.test_data import ISCHEMIC_TEST_CASES, HEMORRHAGIC_TEST_CASES
from src.ischemic.ischemic_structured_model import ISCHEMIC_FEATURE_NAMES
from src.hemorrhagic.hemorrhagic_structured_model import HEMORRHAGIC_FEATURE_NAMES

class TestIntegration(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.test_client = gr.testing.Client()
    
    def test_ischemic_prediction(self):
        """测试缺血性卒中预测流程"""
        test_case = ISCHEMIC_TEST_CASES[0]
        text = test_case["text"]
        features = test_case["features"]
        
        # 直接调用预测函数
        from src.ischemic.ischemic_combined_model import predict_ischemic_combined
        results = predict_ischemic_combined(text, features)
        
        self.assertIsNotNone(results)
        self.assertIn("probability", results)
        self.assertIsInstance(results["probability"], float)
        self.assertTrue(0 <= results["probability"] <= 1)
    
    def test_hemorrhagic_prediction(self):
        """测试出血性卒中预测流程"""
        test_case = HEMORRHAGIC_TEST_CASES[0]
        text = test_case["text"]
        features = test_case["features"]
        
        # 直接调用预测函数
        from src.hemorrhagic.hemorrhagic_combined_model import predict_hemorrhagic_combined
        results = predict_hemorrhagic_combined(text, features)
        
        self.assertIsNotNone(results)
        self.assertIn("probability", results)
        self.assertIsInstance(results["probability"], float)
        self.assertTrue(0 <= results["probability"] <= 1)