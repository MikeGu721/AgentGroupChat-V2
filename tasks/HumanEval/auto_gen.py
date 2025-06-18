import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/home/guzhouhong/zxx/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
from autogen import AssistantAgent
import json
import requests
from datasets import load_dataset
import re
from logger import Logger
import numpy as np
import time
import signal
from contextlib import contextmanager

ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 2
SAMPLE_NUM = 10
SAVE_PATH = "result_autogen.json"

LOG_DIR = "tasks/HumanEval/logs_qwen_72b"
SAVE_DIR = "tasks/HumanEval/results_qwen_72b"
ENGINE_NAME = "TA/Qwen/Qwen2-72B-Instruct"

# LOG_DIR = "tasks/HumanEval/logs_llama_70b"
# SAVE_DIR = "tasks/HumanEval/results_llama_70b"
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


def prepare_test_environment(namespace):
    """Prepare the test environment with commonly used imports."""
    # Standard library imports
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
        "import numpy as np",  # 如果确定测试环境有numpy的话
        "import pandas as pd",  # 如果确定测试环境有pandas的话
    ]

    # Execute all imports in the namespace
    for import_stmt in import_statements:
        try:
            exec(import_stmt, namespace)
        except ImportError:
            # Skip if a package is not available
            continue


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
            prepare_test_environment(namespace)
            exec(func_code, namespace)
    except Exception as e:
        return f"Code execution error: {str(e)}"

    return "Code compiled successfully."


def extract_function_name(code_str):
    """Extract the main function name from the code string."""
    match = re.search(r"def\s+(\w+)\s*\(", code_str)
    if match:
        return match.group(1)
    return None


def run_test_case(code_str, test_case_str, timeout_seconds=5):
    """
    Run HumanEval format test cases and return if all tests pass.

    Args:
        code_str: String containing the function implementation
        test_case_str: String containing the test cases in HumanEval format

    Returns:
        bool: True if all tests pass, False otherwise
        list: List of failed test messages if any tests fail
    """
    # Create namespace for execution
    namespace = {}
    failed_tests = []

    try:
        # Prepare the environment with necessary imports
        with timeout(timeout_seconds):
            prepare_test_environment(namespace)

        # Execute the function definition
        with timeout(timeout_seconds):
            exec(code_str, namespace)

        func_name = extract_function_name(code_str)
        if func_name:
            namespace["candidate"] = namespace[func_name]

        # Parse and execute the test case
        # First split the test case string to separate metadata and checks
        test_lines = test_case_str.strip().split("\n")

        # Skip metadata line if present
        start_idx = 0
        if test_lines[0].startswith("METADATA"):
            start_idx = 1

        # Execute each assert statement
        for line in test_lines[start_idx + 1 :]:
            line = line.strip()
            if line.startswith("assert"):
                try:
                    with timeout(timeout_seconds):
                        exec(line, namespace)
                except AssertionError:
                    failed_tests.append(f"Failed test: {line}")
                except Exception as e:
                    failed_tests.append(f"Error in test {line}: {str(e)}")

        return len(failed_tests) == 0, failed_tests

    except Exception as e:
        return False, [f"Execution error: {str(e)}"]


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


format_requirement = """
You should think carefully, and output your discussion and answer (ONLY your final code for the problem).

Note for your answer:
1. Write only the function code without any explanations. Do not include any test cases or example usage.
2. The code should handle the input format shown in the example usage.
3. The code should follow Markdown format, such as '```python\n[your code]```'.

Your output should follow this format with no additional content:
### Discussion: xxx
### Answer: xxx
"""


def qwen2_by_api(prompt):
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


class CustomAssistant(AssistantAgent):
    def generate_reply(self, messages, sender=None):
        last_message = messages[-1]["content"]
        if len(messages) > 1:
            global test_setup_code
            compile_res = execute_code(last_message, test_setup_code)
            prompt = f"Here is other's solution:\n{last_message}\n{compile_res}\n\n{format_requirement}"
        else:
            prompt = last_message

        message, usage = qwen2_by_api(prompt)
        logger.gprint("Prompt INFO", prompt=prompt, message=message, usage=usage)

        return message


expert1 = CustomAssistant(
    name="Expert1",
    system_message=f"""You are a Python programmer. 
You should leverage professional expertise to engage in collaborative discussions and write python code for the given programming problems, deriving accurate solutions.
{format_requirement}
""",
)

expert2 = CustomAssistant(
    name="Expert2",
    system_message=f"""You are a Python programmer. 
You should leverage professional expertise to engage in collaborative discussions and write python code for the given programming problems, deriving accurate solutions.
{format_requirement}
""",
)

test_setup_code = ""


def solve_problem(example):
    global test_setup_code
    question, test_case_str, test_setup_code = (example["prompt"], example["test"], "")

    prompt = f"""You are a Python programmer. 
You should leverage professional expertise to engage in collaborative discussions and write python code for the given programming problems, deriving accurate solutions.

Problem:
{question}

{format_requirement}
"""

    retry = 0
    while retry < MAX_RETRY:
        message = ""
        try:
            # Initialize conversation between agents
            chat_response = expert1.initiate_chat(
                expert2, message=prompt, max_turns=MAX_ITER
            )

            # Extract the code from the response
            chat_history = chat_response.chat_history
            message = chat_history[-1]["content"]

            answer = message.split("### Answer:")[-1].strip()
            answer = extract_code(answer)
            print("==========")
            print(answer)
            print("==========")

            # run test cases
            all_tests_pass, failed_tests = run_test_case(answer, test_case_str)
            print("==========")

            return all_tests_pass, failed_tests, chat_history
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


def main():
    logger.gprint("========== Autogen Start ==========")
    dataset = load_dataset("openai/openai_humaneval")
    test_data = dataset["test"]

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = len(test_data)

    sample_k = [1, 3, 5]
    passk_lists = [[], [], []]

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        try:
            # pass@k，要生成和测试SAMPLE_NUM次
            correct = 0
            all_answers = []
            for _ in range(SAMPLE_NUM):
                all_tests_pass, failed_tests, iterations = solve_problem(test_data[i])

                if all_tests_pass:
                    correct += 1

                all_answers.append(
                    {
                        "answer": iterations,
                        "success": all_tests_pass,
                        "failed_tests": failed_tests,
                    }
                )

                logger.gprint(
                    "Answer INFO",
                    question_idx=i,
                    answer=iterations,
                    success=all_tests_pass,
                    failed_tests=failed_tests,
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


if __name__ == "__main__":
    main()
