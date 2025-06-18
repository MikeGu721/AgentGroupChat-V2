import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/home/guzhouhong/zxx/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import json
import requests
from datasets import load_dataset
from logger import Logger
import time

from typing import List, Dict, Any
import re
from dataclasses import dataclass


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 10
SAVE_PATH = "result_react.json"

# LOG_DIR = "tasks/GSM8K/logs_qwen_72b"
# SAVE_DIR = "tasks/GSM8K/results_qwen_72b"
# ENGINE_NAME = "TA/Qwen/Qwen2-72B-Instruct"

LOG_DIR = "tasks/GSM8K/logs_llama_70b"
SAVE_DIR = "tasks/GSM8K/results_llama_70b"
ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)


def generate(prompt, engine=ENGINE):
    """
    Utilize open source models with ollama server.
    """
    messages = [{"role": "user", "content": prompt}]
    url = "http://localhost:11434/api/chat"
    payload = {"model": engine, "messages": messages, "stream": False}
    payload_json = json.dumps(payload)
    response_json = requests.request("POST", url, data=payload_json).json()
    return response_json["message"]["content"]


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
        prompt = f"""You are a Math expert. Your task is to solve the following problem:
{question}\n\n

You have access to these actions:
- extract_numbers: Extract all numbers from the question text
  Example: For "2 apples and 3 oranges", returns [2, 3]
- form_expression: Form expression
  Example: returns "2+3"
  If you choose form_expression, you should output after "Action:" with this format:
    Action: form_expression, [your expression]
- calculate(expression: str): Perform expression computation, you can only calculate after form expression
  Example: calculate(expression="2+3") returns "5"
- verify: Verify if the calculation result makes sense

Previous steps:"""

        for i, (thought, action, obs) in enumerate(
            zip(thoughts, actions, observations)
        ):
            prompt += f"\nStep {i+1}:"
            prompt += f"\nThought: {thought}"
            prompt += f"\nAction: {action}"
            prompt += f"\nObservation: {obs}\n"

        prompt += "\nWhat should be the next step? Respond in the following format:"
        prompt += "\nThought: [your reasoning]"
        prompt += "\nAction: [action name] or '#### [answer]' if solved, and you should ONLY include the final numerical answer without any units or explanations."

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


def main():
    logger.gprint("========== ReAct Start ==========")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = len(test_data)
    successful = 0

    agent = ReActAgent()

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["question"], problem["answer"]
        ground_truth = ground_truth.split("####")[-1].strip()

        try:
            answer, result = agent.solve(question)
            if answer == ground_truth:
                success = True
            else:
                success = False
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
