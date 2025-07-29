import os
import sys
from typing import Dict, Any, Tuple, List, Optional
from logger import Logger
from src.environment.managers.event_manager import EventManager
from src.environment.managers.group_manager import GroupManager
from src.environment.managers.task_manager import TaskManager
from src.environment.managers.actions import (
    run_extract_math_answer, 
    run_extract_code_answer,
    run_extract_struct_answer
)
from src.character_manager import CharacterManager
from code_evaluator import CodeEvaluator
from utils import TASK_CONTEXT
import numpy as np

class AgentGroupChatSolver:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.character_manager = CharacterManager(logger)
        self.code_evaluator = CodeEvaluator()
        
        # 设置环境变量和路径
        self._setup_environment()
        
        # 加载或生成角色
        self.characters = self._load_characters()
        
    def _setup_environment(self):
        """设置环境变量和模块路径"""
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        os.environ["HF_HOME"] = self.config.hf_home
        os.chdir(self.config.module_path)
        
        if self.config.module_path not in sys.path:
            sys.path.append(self.config.module_path)
    
    def _load_characters(self):
        """加载或生成角色"""
        return self.character_manager.load_characters(
            character_path=self.config.character_dir,
            engine=self.config.engine if self.config.auto_generate_characters else self.config.engine,
            task_type=self.config.task_name,
            task_description=self.config.task_name,
            problem_domain=self.config.problem_domain,
            num_characters=self.config.num_characters
        )
    
    def solve_problem(self, task_type: str, question: str, additional_info: Any = None, 
                     ground_truth: str = None, sample_num: int = 1) -> Tuple[Any, Dict[str, Any]]:
        """解决问题的通用方法"""
        
        # 使用预加载的角色
        all_character_dict = self.characters
        
        # 初始化管理器
        event_manager = EventManager(engine=self.config.engine, logger=self.logger)
        group_manager = GroupManager(event_manager, engine=self.config.engine, logger=self.logger)
        
        # 根据任务类型设置任务上下文
        task_context = None
        task_decompose = True
        if task_type in ["humaneval", "mbpp"]:
            task_context = TASK_CONTEXT.copy()
            if additional_info:
                task_context["test_setup_code"] = additional_info.get("test_setup_code", "")
            task_decompose = False
        
        task_manager = TaskManager(
            self.config.task_name,
            question,
            all_character_dict,
            event_manager,
            task_decompose=task_decompose,
            engine=self.config.engine,
            logger=self.logger,
            task_context=task_context,
        )

        # 根据任务类型执行不同的求解逻辑
        if task_type in ["humaneval", "mbpp"]:
            return self._solve_code_problem(task_type, question, additional_info, 
                                          group_manager, sample_num)
        else:
            return self._solve_standard_problem(task_type, question, ground_truth, 
                                              group_manager)
    
    def _solve_standard_problem(self, task_type: str, question: str, ground_truth: str,
                               group_manager) -> Tuple[bool, Dict[str, Any]]:
        """解决标准问题（数学、文本等）"""
        
        # 选择合适的答案提取方法
        if task_type == "structext":
            model_response, answer = run_extract_struct_answer(
                question, group_manager.groups[self.config.main_group_id].messages, 
                self.config.engine, self.logger
            )
        else:
            model_response, answer = run_extract_math_answer(
                question, group_manager.groups[self.config.main_group_id].messages,
                self.config.engine, self.logger
            )
        
        success = True if ground_truth == answer else False
        
        self.logger.gprint(
            "Answer INFO",
            question=question,
            message=model_response,
            answer=answer,
            ground_truth=ground_truth,
            success=success,
        )
        
        result = {
            "success": success,
            "answer": answer,
            "ground_truth": ground_truth
        }
        
        return success, result
    
    def _solve_code_problem(self, task_type: str, question: str, additional_info: Dict[str, Any],
                           group_manager, sample_num: int) -> Tuple[int, List[Dict[str, Any]]]:
        """解决编程问题，支持pass@k评估"""
        
        correct = 0
        all_answers = []
        
        for _ in range(sample_num):
            model_response, answer = run_extract_code_answer(
                question, group_manager.groups[self.config.main_group_id].messages,
                self.config.engine, self.logger
            )
            
            # 评估代码
            if task_type == "humaneval":
                all_tests_pass, failed_tests = self.code_evaluator.run_humaneval_test(
                    answer, additional_info["test_case"]
                )
            else:  # mbpp
                all_tests_pass, failed_tests = self.code_evaluator.run_mbpp_test(
                    answer, additional_info["test_case"]
                )
            
            if all_tests_pass:
                correct += 1
            
            answer_info = {
                "answer": answer,
                "success": all_tests_pass,
                "failed_tests": failed_tests if not all_tests_pass else []
            }
            all_answers.append(answer_info)
            
            self.logger.gprint(
                "Answer INFO",
                question=question,
                message=model_response,
                answer=answer,
                success=all_tests_pass,
                failed_tests=failed_tests if not all_tests_pass else [],
            )
        
        return correct, all_answers
