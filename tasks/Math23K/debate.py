import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/Users/zhuxiaoxuan/Project/hf_cache"
MODULE_PATH = "/Users/zhuxiaoxuan/Project/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import time
from logger import Logger
import requests
import json
from datasets import load_dataset


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
SAVE_PATH = "result_debate.json"

LOG_DIR = "tasks/Math23K/logs_qwen_72b"
SAVE_DIR = "tasks/Math23K/results_qwen_72b"
ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

# LOG_DIR = "tasks/Math23K/logs_llama_70b"
# SAVE_DIR = "tasks/Math23K/results_llama_70b"
# ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)

def qwen2_by_api(prompt):
    api_key = API_KEY

    headers = {
        "Authorization": "Bearer " + api_key,
    }

    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": ENGINE_NAME,
    }
    response = requests.post(
        "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False,
    )
    res = response.json()
    message = res["choices"][0]["message"]["content"]
    usage = res["usage"]

    return message, usage


class Agent:
    def __init__(self, role: str, engine: str = "qwen2.5:7b"):
        self.role = role
        self.engine = engine

    def qwen2_by_api(self, prompt):
        api_key = API_KEY

        headers = {
            "Authorization": "Bearer " + api_key,
        }

        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": ENGINE_NAME,
        }
        response = requests.post(
            "https://aigptx.top/v1/chat/completions",
            headers=headers,
            json=params,
            stream=False,
        )
        res = response.json()
        message = res["choices"][0]["message"]["content"]
        usage = res["usage"]

        return message, usage


class Expert(Agent):
    def __init__(self):
        super().__init__("Expert")

    def generate_prompt(self, problem_text, feedback=None) -> str:
        prompt = f"""
你是一个数学专家，你的任务是解决下面的问题：

{problem_text}

你需要一步步思考，先输出你的思考过程，再输出你的最终答案。

请按如下格式输出：
#### Thought: [你的思考过程]
#### Answer: [你的最终答案，只输出答案的数值，不要包含任何单位或解释]

这是一个示例：
#### Thought: xxx
#### Answer: 72
"""
        if feedback:
            prompt += f"\n这是其他人的解决方案：\n{feedback}\n请加入讨论并输出你的解决方案。"
        return prompt

    def generate(self, prompt):
        retry = 0
        while retry < MAX_RETRY:
            message = ""
            try:
                message, usage = self.qwen2_by_api(prompt)
                feedback = message
                logger.gprint(
                    "Prompt INFO", prompt=prompt, message=message, usage=usage
                )
                return message, feedback
            except Exception as e:
                logger.gprint(
                    "### ERROR: Failed in generate_with_response_parser!",
                    prompt=prompt,
                    model_response=message,
                    error=str(e),
                )
                print(e)

            # retry
            print(f"Retrying ({retry + 1}/{MAX_RETRY})...")
            time.sleep(5)
            retry += 1
        # Raise exception after exceeding retries
        raise Exception(f"[Error]: Exceeded Max Retry Times ({MAX_RETRY}).")


class MultiAgentSystem:
    def __init__(self):
        self.expert1 = Expert()
        self.expert2 = Expert()

        dataset = load_dataset("Gxg/Math23K")
        test_data = dataset["test"]
        test_data = test_data.shuffle(seed=42).select(range(200))
        self.dataset = test_data
    
    def check_answer(self, problem, ground_truth, answer) -> bool:
        prompt = f"""
给定一道数学题，请将学生答案与标准答案进行对比，判断学生答案是否正确。

注意：
1. 学生答案与标准答案可能等价但形式不同（例如：0.5 = 1/2 = 50%）
2. 忽略以下差异：
    - 空格和格式
    - 小数与分数表示
    - 百分比与小数表示
    - 可交换运算中的项的顺序
3. 你需要考虑数学等价性而非精确的字符串匹配

要求：
如果答案正确，输出True；如果答案错误，输出False。
不要输出任何解释或其他内容。

输入：
问题：{problem}
标准答案：{ground_truth}
学生答案：{answer}

请按照如下格式输出：
### Answer: [True/False]
"""

        retry = 0
        while retry < MAX_RETRY:
            try:
                response, _ = qwen2_by_api(prompt)
                response = response.split("### Answer:")[-1].strip()
                if "False" in response or "false" in response:
                    return False
                elif "True" in response or "true" in response:
                    return True
            except Exception as e:
                print(str(e))
            retry += 1
            time.sleep(5)

        raise Exception(
            f"[Error]: Failed to Extract Answer. Exceeded Max Retry Times ({MAX_RETRY}).")


    def solve_problem(self, problem_idx: int, max_iterations: int = 3):
        problem = self.dataset[problem_idx]
        question, ground_truth = problem["original_text"], problem["answer"]

        iterations = []
        feedback = None
        res = None

        for i in range(max_iterations):
            print("========== Expert 1 ==========")
            expert1_prompt = self.expert1.generate_prompt(question, feedback)
            message1, feedback = self.expert1.generate(expert1_prompt)

            print("========== Expert 2 ==========")
            expert2_prompt = self.expert2.generate_prompt(question, feedback)
            message2, feedback = self.expert2.generate(expert2_prompt)

            iterations.append(
                {
                    "iteration": i + 1,
                    "expert1": message1,
                    "expert2": message2,
                }
            )

            expert1_answer = message1.split("#### Answer:")[-1].strip()
            expert2_answer = message2.split("#### Answer:")[-1].strip()
            res = expert2_answer

            if expert1_answer == expert2_answer:
                break
        
        success = self.check_answer(question, ground_truth, res)

        return {
            "idx": problem_idx,
            "success": success,
            "answer": res,
            "ground_truth": ground_truth,
            "question": question,
            "iterations": iterations,
        }


if __name__ == "__main__":
    system = MultiAgentSystem()

    logger.gprint("========== Debate Start ==========")
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = 200
    successful = 0

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        try:
            result = system.solve_problem(i)
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)

            if result["success"]:
                successful += 1
        except Exception as e:
            result = {"idx": i, "success": False, "error": str(e)}
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue

    print(successful)
    print(successful / num_problems)
