import unittest
import sys
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
import json

class TestReport:
    def __init__(self):
        self.start_time = time.time()
        self.results: Dict[str, Any] = {
            "summary": {
                "total": 0,
                "success": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "duration": 0
            },
            "tests": [],
            "errors": [],
            "failures": []
        }
        
    def generate_report(self, result: unittest.TestResult) -> Dict[str, Any]:
        """生成测试报告"""
        duration = time.time() - self.start_time
        
        # 更新汇总信息
        self.results["summary"].update({
            "total": result.testsRun,
            "success": result.testsRun - len(result.failures) - len(result.errors),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "duration": f"{duration:.2f}s"
        })
        
        # 记录失败的测试
        for test, trace in result.failures:
            self.results["failures"].append({
                "test": str(test),
                "trace": trace
            })
        
        # 记录错误的测试
        for test, trace in result.errors:
            self.results["errors"].append({
                "test": str(test),
                "trace": trace
            })
        
        return self.results
    
    def save_report(self, filename: str = "test_report.json"):
        """保存测试报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
    
    def print_report(self):
        """打印测试报告"""
        print("\n" + "="*50)
        print("测试报告")
        print("="*50)
        print(f"总计运行测试: {self.results['summary']['total']}")
        print(f"成功: {self.results['summary']['success']}")
        print(f"失败: {self.results['summary']['failures']}")
        print(f"错误: {self.results['summary']['errors']}")
        print(f"跳过: {self.results['summary']['skipped']}")
        print(f"总耗时: {self.results['summary']['duration']}")
        
        if self.results["failures"]:
            print("\n失败的测试:")
            for failure in self.results["failures"]:
                print(f"\n{failure['test']}")
                print(failure['trace'])
        
        if self.results["errors"]:
            print("\n错误的测试:")
            for error in self.results["errors"]:
                print(f"\n{error['test']}")
                print(error['trace'])

def run_tests():
    """运行所有测试并生成报告"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # 创建测试报告实例
    report = TestReport()
    
    try:
        # 检查模型文件
        from app import check_model_files
        check_model_files()
        
        # 发现并运行测试
        loader = unittest.TestLoader()
        suite = loader.discover(os.path.join(project_root, 'tests'), pattern='test_*.py')
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # 生成报告
        report.generate_report(result)
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(project_root, 'output', 'test_reports', f"test_report_{timestamp}.json")
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        report.save_report(report_file)
        
        # 打印报告
        report.print_report()
        
        # 返回测试是否全部通过
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"运行测试时发生错误: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 