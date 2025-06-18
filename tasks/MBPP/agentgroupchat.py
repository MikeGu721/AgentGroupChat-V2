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
import re
import numpy as np
import signal
from contextlib import contextmanager

MAX_RETRY = 10
SAMPLE_NUM = 10

LOG_DIR = "tasks/MBPP/logs_qwen_72b"
SAVE_DIR = "tasks/MBPP/results_qwen_72b"
ENGINE = "TA/Qwen/Qwen2-72B-Instruct#0.7"

# LOG_DIR = "tasks/MBPP/logs_llama_70b"
# SAVE_DIR = "tasks/MBPP/results_llama_70b"
# ENGINE = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo#0.7"

SAVE_PATH = "result_agentgroupchat.json"
CHARACTER_DIR = "tasks/MBPP/characters"
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


def solve_problem(problem, logger):
    question, test_case, test_setup_code = (
        problem["text"],
        problem["test_list"],
        problem["test_setup_code"],
    )

    analysis = analyze_test_case(test_case[0])
    if analysis:
        signature_hint = f"\nFunction name should be: {analysis['function_name']}\nExample usage: {analysis['function_name']}({analysis['example_args']})"
    else:
        signature_hint = ""

    # initialize main task
    task_name = "Solving programming problem"
    main_task_desc = f"Problem:{question}\n{signature_hint}"

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

        all_tests_pass = True
        for test in test_case:
            if not run_test_case(answer, test):
                all_tests_pass = False
                break

        if all_tests_pass:
            correct += 1

        all_answers.append({"answer": answer, "success": all_tests_pass})

        logger.gprint(
            "Answer INFO",
            question=main_task_desc,
            message=model_response,
            answer=answer,
            success=all_tests_pass,
        )

    return correct, all_answers


def main():
    logger = Logger(LOG_DIR)
    logger.gprint("========== AgentGroupChat 2.0 Start ==========")

    dataset = load_dataset("google-research-datasets/mbpp")
    test_data = dataset["test"]

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = len(test_data)

    sample_k = [1, 3, 5]
    passk_lists = [[], [], []]

    for i in range(249, num_problems):
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
