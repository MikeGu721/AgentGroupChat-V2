import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/shenhao/AgentGroupChat_v2/hf_cache"
MODULE_PATH = "/data/shenhao/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import json
import requests
import regex
from typing import Dict
from logger import Logger
import time

ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
SAVE_PATH = "result_debate.json"

# LOG_DIR = "tasks/Finance/logs_qwen_72b"
# SAVE_DIR = "tasks/Finance/results_qwen_72b"
# ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

# OLLAMA_ENGINE = "qwen2.5:72b"
OLLAMA_ENGINE = "llama3.1:70b"
LOG_DIR = "tasks/Finance/logs_llama_70b"
SAVE_DIR = "tasks/Finance/results_llama_70b"
# ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)


class Agent:
    def __init__(self, role: str, engine: str = "qwen2.5:7b"):
        self.role = role
        self.engine = engine

    def ollama_generate(self, prompt):
        """
        Utilize open source models with ollama server.
        """
        messages = [{"role": "user", "content": prompt}]
        url = "http://localhost:11434/api/chat"
        payload = {"model": OLLAMA_ENGINE, "messages": messages, "stream": False}
        payload_json = json.dumps(payload)
        response_json = requests.request("POST", url, data=payload_json).json()
        usage = {"prompt_tokens": response_json["prompt_eval_count"], "completion_tokens": response_json["eval_count"], "total_tokens": response_json["prompt_eval_count"] + response_json["eval_count"]}
        return response_json["message"]["content"], usage

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
        prompt = f"""You are a finance expert. Your task is to solve the following problem:

{problem_text}

You should think step-by-step.
Output your thought first, and then output your final answer.

Format your response as:
#### Thought: [Your thought]
#### Answer: [Your final answer, Always return ONLY the final choices without any units or explanations.]

Here is an example:
#### Thought: xxx
#### Answer: ABCD
"""
        if feedback:
            prompt += f"\nHere is other's solution:\n{feedback}\nPlease join the discussion and provide your solution."
        return prompt

    def generate(self, prompt):
        retry = 0
        while retry < MAX_RETRY:
            message = ""
            try:
                message, usage = self.ollama_generate(prompt)
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
        # read test data
        test_file = "tasks/Finance/dataset/AgentGroupChat_Finance.json"
        # test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.loads(f.read())
        # with open(test_file, "r", encoding="utf-8") as f:
        #     for line in f.readlines():
        #         sample = json.loads(line.strip())
        #         test_data.append(sample)
        self.dataset = test_data

    def solve_problem(self, problem_idx: int, max_iterations: int = 3) -> Dict:
        problem = self.dataset[problem_idx]
        question, ground_truth = problem["problem"], problem["solution"]
        # 提取ground truth
        # pattern = r"\\boxed\{(.*)\}"
        # match = regex.search(pattern, ground_truth, flags=regex.DOTALL)
        # ground_truth = match.group(1)
        ground_truth = problem['ground_truth']

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

        if res == ground_truth:
            success = True
        else:
            success = False

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

    # read test data
    test_file = "tasks/Finance/dataset/AgentGroupChat_Finance.json"
    # test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.loads(f.read())
    # with open(test_file, "r", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         sample = json.loads(line.strip())
    #         test_data.append(sample)

    num_problems = len(test_data)
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
