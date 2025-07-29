import json
import os
import sys
import argparse
import numpy as np
from typing import Dict, Any

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__)))

from config import TASK_CONFIGS
from data_loader import DataLoader
from problem_parser import ProblemParser
from solver import AgentGroupChatSolver
from logger import Logger

def pass_at_k(n, c, k):
    """计算pass@k指标"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def run_task(task_type: str, start_idx: int = 0, sample_num: int = 1):
    """运行指定任务"""
    
    if task_type not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type}")
    
    config = TASK_CONFIGS[task_type]
    logger = Logger(config.log_dir)
    logger.gprint("========== AgentGroupChat Start ==========")
    
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    save_path = os.path.join(config.save_dir, config.save_filename)
    
    # 加载数据
    test_data = DataLoader.load_data(config.test_file)
    num_problems = len(test_data)
    
    # 初始化求解器
    solver = AgentGroupChatSolver(config, logger)
    
    # 初始化统计变量
    successful = 0
    if task_type in ["humaneval", "mbpp"]:
        sample_k = [1, 3, 5]
        passk_lists = [[] for _ in sample_k]
    
    # 处理每个问题
    for i in range(start_idx, min(start_idx + 10, num_problems)):  # 限制只处理10个问题用于测试
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        print(f"=======Test File {config.test_file}==========")
        
        problem = test_data[i]
        
        try:
            # 解析问题
            question, additional_info, ground_truth = ProblemParser.parse_problem(task_type, problem)
            formatted_question = ProblemParser.format_question(task_type, question, additional_info)
            
            # 求解问题
            if task_type in ["humaneval", "mbpp"]:
                correct, all_answers = solver.solve_problem(
                    task_type, formatted_question, additional_info, 
                    ground_truth, sample_num
                )
                
                # 计算pass@k
                passk = []
                for k_i, k in enumerate(sample_k):
                    passk_val = pass_at_k(sample_num, correct, k)
                    passk.append(passk_val)
                    passk_lists[k_i].append(passk_val)
                
                result = {
                    "idx": i,
                    "passk": passk,
                    "problem": problem,
                    "answer": all_answers,
                }
            else:
                success, result = solver.solve_problem(
                    task_type, formatted_question, additional_info, ground_truth
                )
                
                result.update({
                    "idx": i,
                    "question": question,
                    "success": success
                })
                
                if success:
                    successful += 1
            
            # 保存结果
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
                
        except Exception as e:
            # 处理错误
            print(f"Error processing problem {i}: {str(e)}")
            if task_type in ["humaneval", "mbpp"]:
                for k_i in range(len(sample_k)):
                    passk_lists[k_i].append(0)
                result = {"idx": i, "problem": problem, "error": str(e)}
            else:
                result = {"idx": i, "question": str(problem), "success": False, "error": str(e)}
            
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue
    
    # 输出最终结果
    if task_type in ["humaneval", "mbpp"]:
        for i, passk_list in enumerate(passk_lists):
            print(f"pass@{sample_k[i]}: {np.average(passk_list):.4f}")
    else:
        processed_count = min(10, num_problems - start_idx)
        print(f"Successful: {successful}")
        print(f"Accuracy: {successful / processed_count:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run AgentGroupChat tasks")
    parser.add_argument("--task", type=str, required=True, 
                       choices=list(TASK_CONFIGS.keys()),
                       help="Task type to run")
    parser.add_argument("--start", type=int, default=0, 
                       help="Starting index (default: 0)")
    parser.add_argument("--sample_num", type=int, default=10,
                       help="Number of samples for pass@k evaluation (default: 10)")
    parser.add_argument("--character_dir", type=str, default=None,
                       help="Custom character directory (overrides config)")
    parser.add_argument("--auto_generate", action="store_true",
                       help="Auto generate characters if directory doesn't exist")
    parser.add_argument("--num_characters", type=int, default=5,
                       help="Number of characters to generate (default: 5)")
    
    args = parser.parse_args()
    
    # 如果指定了自定义角色目录，更新配置
    if args.character_dir:
        TASK_CONFIGS[args.task].character_dir = args.character_dir
        TASK_CONFIGS[args.task].auto_generate_characters = args.auto_generate
        TASK_CONFIGS[args.task].num_characters = args.num_characters
    
    run_task(args.task, args.start, args.sample_num)

if __name__ == "__main__":
    main()
