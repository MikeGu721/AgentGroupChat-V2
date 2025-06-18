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
from typing import Dict
import re
from logger import Logger
import time
import numpy as np
import signal
from contextlib import contextmanager

ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
SAMPLE_NUM = 10
SAVE_PATH = "result_debate.json"

LOG_DIR = "tasks/MBPP/logs_qwen_72b"
SAVE_DIR = "tasks/MBPP/results_qwen_72b"
ENGINE_NAME = "TA/Qwen/Qwen2-72B-Instruct"

# LOG_DIR = "tasks/MBPP/logs_llama_70b"
# SAVE_DIR = "tasks/MBPP/results_llama_70b"
# ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)


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


def extract_code(text):
    # 去掉前面的 ```python 或 ``` 标记
    text = re.sub(r"^```(?:python)?\n?", "", text)
    # 去掉后面的 ``` 标记
    text = re.sub(r"\n?```$", "", text)
    # 去掉前后多余的空白和换行
    return text.strip()


def execute_code(message: str, test_setup_code: str = "", timeout_seconds=5) -> str:
    """
    代码生成任务，编译代码作为反馈
    """
    # parse message
    message = message.split("### Answer:")[-1].strip()
    func_code = extract_code(message)

    namespace = {}
    # First execute setup code if any
    if test_setup_code:
        try:
            with timeout(timeout_seconds):
                exec(test_setup_code, namespace)
        except Exception as e:
            return f"Setup code error: {str(e)}"
    # Then execute function code
    try:
        with timeout(timeout_seconds):
            exec(func_code, namespace)
    except Exception as e:
        return f"Code execution error: {str(e)}"

    return "Code compiled successfully."


def run_test_case(code_str, test_case, timeout_seconds=5):
    """Run a single test case and return if it passes."""
    try:
        # Create a namespace for execution
        namespace = {}

        # Execute the function definition with timeout
        with timeout(timeout_seconds):
            exec(code_str, namespace)

        # Execute the test case with timeout
        with timeout(timeout_seconds):
            exec(test_case, namespace)

        return True
    except TimeoutException:
        print("Test failed: Execution timed out")
        return False
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


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
            "temperature": 0.7,
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

    @staticmethod
    def analyze_test_case(test_case: str) -> dict:
        """Analyze test case to extract function name and argument format"""
        # Remove assert keyword
        test_case = test_case.replace("assert ", "")

        # Extract function name and arguments
        match = re.match(r"(\w+)\((.*?)\)", test_case)
        if not match:
            return None

        func_name = match.group(1)
        args_str = match.group(2)

        # Return analysis
        return {"function_name": func_name, "example_args": args_str}


class Coder(Agent):
    def __init__(self):
        super().__init__("Coder")

    def generate_prompt(self, task: str, test_case: str, feedback: str = "") -> str:
        # Analyze first test case to understand function signature
        analysis = self.analyze_test_case(test_case)
        if analysis:
            signature_hint = f"\nFunction name should be: {analysis['function_name']}\nExample usage: {analysis['function_name']}({analysis['example_args']})"
        else:
            signature_hint = ""

        base_prompt = f"""You are a Python programmer. 
You should leverage professional expertise to engage in collaborative discussions and write python code for the given programming problems, deriving accurate solutions.

Problem:
{task}
{signature_hint}
"""
        if feedback:
            base_prompt += f"\nHere is other's solution:\n{feedback}\n\n"

        base_prompt += """
You should think carefully, and output your discussion and answer (ONLY your final code for the problem).

Note for your answer:
1. Write only the function code without any explanations. Do not include any test cases or example usage.
2. The code should handle the input format shown in the example usage.
3. The code should follow Markdown format, such as '```python\n[your code]```'.

Your output should follow this format with no additional content:
### Discussion: xxx
### Answer: xxx
"""

        return base_prompt

    def generate(self, prompt, test_setup_code):
        retry = 0
        while retry < MAX_RETRY:
            message = ""
            try:
                message, usage = self.qwen2_by_api(prompt)
                compile_res = execute_code(message, test_setup_code)
                logger.gprint(
                    "Prompt INFO",
                    prompt=prompt,
                    message=message,
                    compile_res=compile_res,
                    usage=usage,
                )
                return message, compile_res
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
        self.coder1 = Coder()
        self.coder2 = Coder()
        self.dataset = load_dataset("google-research-datasets/mbpp")

    def solve_problem(self, problem_idx: int, max_iterations: int = 2) -> Dict:
        example = self.dataset["test"][problem_idx]
        task = example["text"]
        test_list = example["test_list"]
        test_setup_code = example.get("test_setup_code", "")
        # 使用第一个测试用例来推断函数签名
        first_test = test_list[0]

        iterations = []
        feedback = ""
        for i in range(max_iterations):
            # Coder generates code with signature information
            coder1_prompt = self.coder1.generate_prompt(task, first_test, feedback)
            print("Coder1 Prompt:")
            print(coder1_prompt)
            coder1_response, code1_compile_res = self.coder1.generate(
                coder1_prompt, test_setup_code
            )
            feedback = f"{coder1_response}\n{code1_compile_res}"

            coder2_prompt = self.coder2.generate_prompt(task, first_test, feedback)
            print("Coder2 Prompt:")
            print(coder2_prompt)
            coder2_response, code2_compile_res = self.coder2.generate(
                coder2_prompt, test_setup_code
            )
            feedback = f"{coder2_response}\n{code2_compile_res}"

            answer = coder2_response.split("### Answer:")[-1].strip()
            answer = extract_code(answer)
            print("==========")
            print(answer)
            print("==========")

            # run test cases
            all_tests_pass = True
            for test in test_list:
                if not run_test_case(answer, test):
                    all_tests_pass = False
                    break
            print("==========")

            iterations.append(
                {
                    "iteration": i + 1,
                    "coder1": coder1_response,
                    "coder2": coder2_response,
                    "test_results": all_tests_pass,
                }
            )

            if all_tests_pass:
                break

        return all_tests_pass, iterations


# Example usage
if __name__ == "__main__":
    system = MultiAgentSystem()

    logger.gprint("========== Debate 4.0 Start ==========")
    dataset = load_dataset("google-research-datasets/mbpp")
    test_data = dataset["test"]

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = len(test_data)

    sample_k = [1, 3, 5]
    passk_lists = [[], [], []]

    for i in range(312, num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        try:
            # pass@k，要生成和测试SAMPLE_NUM次
            correct = 0
            all_answers = []
            for _ in range(SAMPLE_NUM):
                all_tests_pass, iterations = system.solve_problem(i)

                if all_tests_pass:
                    correct += 1

                all_answers.append({"answer": iterations, "success": all_tests_pass})

                logger.gprint(
                    "Answer INFO",
                    question_idx=i,
                    answer=iterations,
                    success=all_tests_pass,
                )
            passk = []
            for k_i, k in enumerate(sample_k):
                passk_val = pass_at_k(SAMPLE_NUM, correct, k)
                passk.append(passk_val)
                passk_lists[k_i].append(passk_val)

            result = {
                "idx": i,
                "passk": passk,
                "answer": all_answers[-1],
            }

            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
        except Exception as e:
            for k_i, k in enumerate(sample_k):
                passk_lists[k_i].append(0)
            result = {"idx": i, "error": str(e)}
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue

    for i, passk_list in enumerate(passk_lists):
        print(i, np.average(passk_list))


# class Interpreter(Agent):
#     def __init__(self):
#         super().__init__("Interpreter")

#     def execute_test_case(
#         self, func_code: str, test_case: str, test_setup_code: str = ""
#     ) -> tuple[bool, str]:
#         """Execute a single test case with setup code"""
#         namespace = {}

#         # First execute setup code if any
#         if test_setup_code:
#             try:
#                 exec(test_setup_code, namespace)
#             except Exception as e:
#                 return False, f"Setup code error: {str(e)}"

#         # Then execute function code
#         try:
#             exec(func_code, namespace)
#         except Exception as e:
#             return False, f"Code execution error: {str(e)}"

#         # Finally execute test case
#         output = StringIO()
#         with contextlib.redirect_stdout(output):
#             try:
#                 exec(test_case, namespace)
#                 return True, "Test passed"
#             except AssertionError:
#                 return False, f"Test failed: {test_case}"
#             except Exception as e:
#                 return False, f"Test execution error: {str(e)}"

#     def generate_prompt(
#         self, code: str, task: str, test_case: str, test_results: List[tuple[bool, str]]
#     ) -> str:
#         analysis = self.analyze_test_case(test_case)
#         if analysis:
#             signature_hint = f"\nFunction name should be: {analysis['function_name']}\nExample usage: {analysis['function_name']}({analysis['example_args']})"
#         else:
#             signature_hint = ""

#         results_str = "\n".join(
#             f"Test {i+1}: {'Success' if res[0] else 'Failed'} - {res[1]}"
#             for i, res in enumerate(test_results)
#         )

#         return f"""As an Interpreter, analyze this code and its execution results for the task:

# Task:
# {task}
# {signature_hint}

# Code:
# {code}

# Test Results:
# {results_str}

# Provide specific feedback on:
# 1. Code correctness
# 2. Any syntax or runtime errors
# 3. Logic issues
# 4. Suggestions for improvement

# Be concise and specific."""
