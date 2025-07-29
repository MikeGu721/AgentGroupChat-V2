from typing import Dict, Any, Tuple, Optional, List
import regex
import re

class ProblemParser:
    @staticmethod
    def parse_problem(task_type: str, problem: Dict[str, Any]) -> Tuple[str, Any, str]:
        """根据任务类型解析问题，返回(问题描述, 附加信息, 正确答案)"""
        
        if task_type == "aime":
            return problem["Problem"], None, str(problem["Answer"])
        
        elif task_type == "finance":
            return problem["problem"], None, problem["ground_truth"]
        
        elif task_type == "gsm8k":
            question = problem["question"]
            answer = problem["answer"].split("####")[-1].strip()
            return question, None, answer
        
        elif task_type == "hellaswag":
            question = problem["ctx"]
            endings = problem["endings"]
            ground_truth = str(problem["label"])
            return question, endings, ground_truth
        
        elif task_type in ["humaneval", "mbpp"]:
            if task_type == "humaneval":
                question = problem["prompt"]
                test_info = {
                    "test_case": problem["test"],
                    "test_setup_code": ""
                }
            else:  # mbpp
                question = problem["text"]
                test_info = {
                    "test_case": problem["test_list"],
                    "test_setup_code": problem["test_setup_code"]
                }
            return question, test_info, None  # 编程题没有直接的ground_truth
        
        elif task_type == "structext":
            return problem["q"], None, problem["a"]
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    @staticmethod
    def format_question(task_type: str, question: str, additional_info: Any = None) -> str:
        """根据任务类型格式化问题描述"""
        
        if task_type == "hellaswag" and additional_info:
            question_prompt = f'Story: {question}'
            for index, ending in enumerate(additional_info):
                question_prompt += f'\nEnding {index}: {ending}'
            return question_prompt
        
        elif task_type in ["humaneval", "mbpp"] and additional_info:
            if task_type == "mbpp":
                # 为MBPP添加函数签名提示
                test_case = additional_info["test_case"]
                if test_case:
                    analysis = ProblemParser._analyze_test_case(test_case[0])
                    if analysis:
                        signature_hint = f"\nFunction name should be: {analysis['function_name']}\nExample usage: {analysis['function_name']}({analysis['example_args']})"
                        return f"Problem:{question}\n{signature_hint}"
            return f"Problem:\n{question}"
        
        else:
            return question
    
    @staticmethod
    def _analyze_test_case(test_case: str) -> Optional[Dict[str, str]]:
        """分析测试用例以提取函数名和参数格式"""
        test_case = test_case.replace("assert ", "")
        match = re.match(r"(\w+)\((.*?)\)", test_case)
        if not match:
            return None
        
        func_name = match.group(1)
        args_str = match.group(2)
        
        return {"function_name": func_name, "example_args": args_str}
