import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/Users/zhuxiaoxuan/Project/hf_cache"
MODULE_PATH = "/Users/zhuxiaoxuan/Project/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import json
import requests
from logger import Logger
import time
from typing import List, Dict, Any
from dataclasses import dataclass
from datasets import load_dataset
import re


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 5
SAVE_PATH = "result_react.json"

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


@dataclass
class Action:
    name: str
    args: Dict[str, Any]

    def __str__(self) -> str:
        # 将参数字典转换成字符串形式
        args_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({args_str})"


class ReActAgent:
    def __init__(self):
        self.actions = {
            "calculate": self._calculate,
            "extract_numbers": self._extract_numbers,
            "form_expression": self._form_expression,
            "verify": self._verify_answer,
        }
        self.current_question = ""
        self.last_numbers = []
        self.expression = ""

    def _format_react_prompt(
        self,
        question: str,
        thoughts: List[str],
        actions: List[Action],
        observations: List[str],
    ) -> str:
        """格式化ReAct提示"""
        prompt = f"""你是一个数学专家，你的任务是解决下面的问题：
{question}\n\n

你可以采取这些行动：
- extract_numbers：从问题文本中提取所有数字
  - 示例：对于"2个苹果和3个橙子"，返回[2, 3]
- form_expression：构建表达式
  - 示例：返回"2+3"
  - 如果你选择form_expression，你应该在"Action:"后按如下格式输出：
    - Action: form_expression, [你的表达式]
- verify：验证计算结果是否正确

之前的步骤：
"""
        for i, (thought, action, obs) in enumerate(
            zip(thoughts, actions, observations)
        ):
            prompt += f"\nStep {i+1}:"
            prompt += f"\nThought: {thought}"
            prompt += f"\nAction: {action}"
            prompt += f"\nObservation: {obs}\n"

        prompt += "\n下一步你想采取什么行动？按照以下格式回答："
        prompt += "\nThought: [your reasoning]"
        prompt += "\nAction: [action name] or '#### [answer]'（如果问题已解决，你应该只包含最终的数值答案，不要包含任何单位或解释。）"

        return prompt

    def solve(self, question: str) -> str:
        """
        使用ReAct方法解决问题

        Args:
            question: 输入问题
        Returns:
            str: 最终答案
        """
        self.current_question = question
        thoughts: List[str] = []
        actions: List[Action] = []
        observations: List[str] = []

        while len(thoughts) < MAX_ITER:
            # 生成下一步的思考
            prompt = self._format_react_prompt(
                question, thoughts, actions, observations
            )
            print(prompt)

            retry = 0
            while retry < MAX_RETRY:
                response = ""
                try:
                    response, usage = qwen2_by_api(prompt)
                    print(response)
                    print(usage)
                    logger.gprint(
                        "Prompt INFO",
                        prompt=prompt,
                        model_response=response,
                        usage=usage,
                    )

                    thought, action = self._parse_response(response)
                    print("Thought:\n", thought)
                    print("Action:\n", action)
                    thoughts.append(thought)

                    # 执行动作
                    action_obj = self._parse_action(action)
                    actions.append(
                        str(action_obj)
                    )  # 很多答案其实写在action里，所以后面response能识别出####，但是因为action被parse过了，所以在log的action里看不到####，但实际是有的

                    # 获取观察结果
                    observation = self._execute_action(action_obj)
                    print("observation:\n", observation)
                    observations.append(observation)

                    # 检查是否得到最终答案
                    if "####" in response:
                        return self._extract_final_answer(response), {
                            "thoughts": thoughts,
                            "actions": actions,
                            "observations": observations,
                            "response": response,
                        }

                    break
                except Exception as e:
                    logger.gprint(
                        "### ERROR: Failed in generate_with_response_parser!",
                        prompt=prompt,
                        model_response=response,
                        error=str(e),
                    )
                    print(e)
                # retry
                print(f"Retrying ({retry + 1}/{MAX_RETRY})...")
                time.sleep(5)
                retry += 1
            if retry == MAX_RETRY:
                # Raise exception after exceeding retries
                raise Exception(f"[Error]: Exceeded Max Retry Times ({MAX_RETRY}).")

        return "达到最大步骤限制，未找到答案", {
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
        }

    def _parse_response(self, response: str) -> tuple[str, str]:
        """解析LLM响应，提取思考和行动"""
        thought_match = re.search(r"Thought: (.*?)(?=\nAction:|$)", response, re.DOTALL)
        action_match = re.search(r"Action: (.*?)(?=\n|$)", response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""

        return thought, action

    def _parse_action(self, action_str: str) -> Action:
        """解析动作字符串为Action对象"""
        if "calculate" in action_str.lower():
            return Action("calculate", {"expression": self.expression})
        elif "extract" in action_str.lower():
            return Action("extract_numbers", {})
        elif "form" in action_str.lower():
            expression = action_str.split(",")[-1].strip()
            self.expression = expression
            return Action("form_expression", {"expression": self.expression})
        else:
            return Action("verify", {})

    def _execute_action(self, action: Action) -> str:
        """执行具体动作"""
        try:
            return self.actions[action.name](**action.args)
        except Exception as e:
            return f"Action failed: {str(e)}"

    def _calculate(self, expression: str) -> str:
        """执行数学计算"""
        res = ""
        try:
            result = eval(expression)
            res = result
        except Exception as e:
            print(str(e))
            res = str(e)
        return res

    def _extract_numbers(self) -> str:
        """从问题中提取数字"""
        numbers = re.findall(r"\d+", self.current_question)
        self.last_numbers = [int(n) for n in numbers]
        return str(self.last_numbers)

    def _form_expression(self, expression) -> str:
        self.expression = expression
        return expression

    def _verify_answer(self) -> str:
        """验证答案"""
        return "验证完成"

    def _extract_final_answer(self, action: str) -> str:
        """从最终思考中提取答案"""
        answer = action.split("####")[-1].strip()
        return answer


def check_answer(problem, ground_truth, answer) -> bool:
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


def main():
    logger.gprint("========== ReAct Start ==========")

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    dataset = load_dataset("Gxg/Math23K")
    test_data = dataset["test"]
    test_data = test_data.shuffle(seed=42).select(range(200))

    num_problems = len(test_data)
    successful = 0

    agent = ReActAgent()

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["original_text"], problem["answer"]

        try:
            answer, result = agent.solve(question)
            success = check_answer(question, ground_truth, answer)
            result["idx"] = i
            result["question"] = question
            result["ground_truth"] = ground_truth
            result["success"] = success

            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)

            if success:
                successful += 1
        except Exception as e:
            result = {"idx": i, "question": question, "success": False, "error": str(e)}
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue

    print(successful)
    print(successful / num_problems)


if __name__ == "__main__":
    main()
