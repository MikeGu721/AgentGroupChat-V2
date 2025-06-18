import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/home/guzhouhong/zxx/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

import json
from logger import Logger
from environment.managers.event_manager import EventManager
from environment.managers.group_manager import GroupManager
from environment.managers.task_manager import TaskManager
from environment.characters.character import TaskCharacter
from environment.managers.actions import run_extract_code_answer
from datasets import load_dataset
from utils import TASK_CONTEXT
import numpy as np
import signal
from contextlib import contextmanager
import re

MAX_RETRY = 10
SAMPLE_NUM = 10

LOG_DIR = "tasks/HumanEval/logs_qwen_72b"
SAVE_DIR = "tasks/HumanEval/results_qwen_72b"
ENGINE = "TA/Qwen/Qwen2-72B-Instruct#0.7"

# LOG_DIR = "tasks/HumanEval/logs_llama_70b"
# SAVE_DIR = "tasks/HumanEval/results_llama_70b"
# ENGINE = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo#0.7"

SAVE_PATH = "result_agentgroupchat.json"
CHARACTER_DIR = "tasks/HumanEval/characters"
MAIN_GROUP_ID = "Group_Main"


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


def solve_problem(problem, logger):
    question, test_case_str, test_setup_code = (problem["prompt"], problem["test"], "")

    # initialize main task
    task_name = "Solving programming problem"
    main_task_desc = f"Problem:\n{question}"

    # initialize characters
    all_character_dict = {}
    for file in os.listdir(CHARACTER_DIR):
        character_file = os.path.join(CHARACTER_DIR, file)
        character = TaskCharacter(character_file, ENGINE, logger)
        all_character_dict[character.id] = character

    event_manager = EventManager(engine=ENGINE, logger=logger)
    group_manager = GroupManager(event_manager, engine=ENGINE, logger=logger)
    task_context = TASK_CONTEXT
    task_context["test_setup_code"] = test_setup_code
    task_manager = TaskManager(
        task_name,
        main_task_desc,
        all_character_dict,
        event_manager,
        task_decompose=False,
        engine=ENGINE,
        logger=logger,
        task_context=task_context,
    )

    # pass@k，要生成和测试SAMPLE_NUM次
    correct = 0
    all_answers = []
    for _ in range(SAMPLE_NUM):
        model_response, answer = run_extract_code_answer(
            main_task_desc, group_manager.groups[MAIN_GROUP_ID].messages, ENGINE, logger
        )

        all_tests_pass, failed_tests = run_test_case(answer, test_case_str)
        if all_tests_pass:
            correct += 1

        all_answers.append(
            {"answer": answer, "success": all_tests_pass, "failed_tests": failed_tests}
        )

        logger.gprint(
            "Answer INFO",
            question=main_task_desc,
            message=model_response,
            answer=answer,
            success=all_tests_pass,
            failed_tests=failed_tests,
        )

    return correct, all_answers


def main():
    logger = Logger(LOG_DIR)
    logger.gprint("========== AgentGroupChat Start ==========")

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
        problem = test_data[i]
        try:
            correct, all_answers = solve_problem(problem, logger)
            passk = []
            for k_i, k in enumerate(sample_k):
                passk_val = pass_at_k(SAMPLE_NUM, correct, k)
                passk.append(passk_val)
                passk_lists[k_i].append(passk_val)

            result = {
                "idx": i,
                "passk": passk,
                "problem": problem,
                "answer": all_answers,
            }

            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
        except Exception as e:
            for k_i, k in enumerate(sample_k):
                passk_lists[k_i].append(0)
            result = {"idx": i, "problem": problem, "error": str(e)}
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue

    for i, passk_list in enumerate(passk_lists):
        print(i, np.average(passk_list))


if __name__ == "__main__":
    main()
