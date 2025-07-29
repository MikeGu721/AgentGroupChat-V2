import signal
import re
from contextlib import contextmanager
from typing import Tuple, List

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)

class CodeEvaluator:
    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = timeout_seconds
    
    def prepare_test_environment(self, namespace):
        """准备测试环境"""
        import_statements = [
            "import math",
            "import collections", 
            "import itertools",
            "import string",
            "import random",
            "import time",
            "import functools",
            "import re",
            "import json",
            "import copy",
            "import datetime",
            "from typing import *",
            "from collections import Counter, defaultdict, deque",
        ]

        for import_stmt in import_statements:
            try:
                exec(import_stmt, namespace)
            except ImportError:
                continue
    
    def run_humaneval_test(self, code_str: str, test_case_str: str) -> Tuple[bool, List[str]]:
        """运行HumanEval格式的测试用例"""
        namespace = {}
        failed_tests = []

        try:
            with timeout(self.timeout_seconds):
                self.prepare_test_environment(namespace)

            with timeout(self.timeout_seconds):
                exec(code_str, namespace)

            func_name = self._extract_function_name(code_str)
            if func_name:
                namespace["candidate"] = namespace[func_name]

            test_lines = test_case_str.strip().split("\n")
            start_idx = 1 if test_lines[0].startswith("METADATA") else 0

            for line in test_lines[start_idx + 1:]:
                line = line.strip()
                if line.startswith("assert"):
                    try:
                        with timeout(self.timeout_seconds):
                            exec(line, namespace)
                    except AssertionError:
                        failed_tests.append(f"Failed test: {line}")
                    except Exception as e:
                        failed_tests.append(f"Error in test {line}: {str(e)}")

            return len(failed_tests) == 0, failed_tests

        except Exception as e:
            return False, [f"Execution error: {str(e)}"]
    
    def run_mbpp_test(self, code_str: str, test_cases: List[str]) -> Tuple[bool, List[str]]:
        """运行MBPP格式的测试用例"""
        failed_tests = []
        
        for test in test_cases:
            if not self._run_single_test(code_str, test):
                failed_tests.append(f"Failed test: {test}")
        
        return len(failed_tests) == 0, failed_tests
    
    def _run_single_test(self, code_str: str, test_case: str) -> bool:
        """运行单个测试用例"""
        try:
            namespace = {}
            
            with timeout(self.timeout_seconds):
                exec(code_str, namespace)
            
            with timeout(self.timeout_seconds):
                exec(test_case, namespace)
            
            return True
        except:
            return False
    
    def _extract_function_name(self, code_str: str) -> str:
        """从代码字符串中提取主函数名"""
        match = re.search(r"def\s+(\w+)\s*\(", code_str)
        if match:
            return match.group(1)
        return None
