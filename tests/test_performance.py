import unittest
import time
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.test_data import ISCHEMIC_TEST_CASES, HEMORRHAGIC_TEST_CASES
from src.ischemic.ischemic_text_model import predict_ischemic_text
from src.hemorrhagic.hemorrhagic_text_model import predict_hemorrhagic_text

class TestPerformance(unittest.TestCase):
    def test_response_time(self):
        """测试响应时间"""
        # 测试缺血性卒中文本模型
        times = []
        for case in ISCHEMIC_TEST_CASES:
            start = time.time()
            predict_ischemic_text(case["text"])
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        self.assertLess(avg_time, 2.0, "缺血性卒中文本模型平均响应时间超过2秒")
        
        # 测试出血性卒中文本模型
        times = []
        for case in HEMORRHAGIC_TEST_CASES:
            start = time.time()
            predict_hemorrhagic_text(case["text"])
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        self.assertLess(avg_time, 2.0, "出血性卒中文本模型平均响应时间超过2秒") 