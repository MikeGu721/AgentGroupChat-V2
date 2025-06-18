import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/Users/guzhouhong1/Work/AgentGroupChat_v2/"
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
from environment.managers.actions import run_extract_math_answer
import regex


MAX_RETRY = 10


LOG_DIR = "tasks/AIME/logs_qwen2.5-72b"
SAVE_DIR = "tasks/AIME/results_qwen2.5-72b"
ENGINE = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

SAVE_PATH = "result_agentgroupchat.jsonl"
CHARACTER_DIR = "tasks/AIME/characters"
MAIN_GROUP_ID = "Group_Main"


def solve_problem(question, ground_truth, logger):
    # initialize main task
    task_name = "Output the answer of the given question, the answer should be an int value."
    question_prompt = question
    main_task_desc = question_prompt

    # initialize characters
    all_character_dict = {}
    for file in os.listdir(CHARACTER_DIR):
        character_file = os.path.join(CHARACTER_DIR, file)
        character = TaskCharacter(character_file, ENGINE, logger)
        all_character_dict[character.id] = character

    event_manager = EventManager(engine=ENGINE, logger=logger)
    group_manager = GroupManager(event_manager, engine=ENGINE, logger=logger)
    task_manager = TaskManager(
        task_name,
        main_task_desc,
        all_character_dict,
        event_manager,
        engine=ENGINE,
        logger=logger,
    )

    model_response, answer = run_extract_math_answer(
        question, group_manager.groups[MAIN_GROUP_ID].messages, ENGINE, logger
    )

    success = True if ground_truth == answer else False
    logger.gprint(
        "Answer INFO",
        question=question,
        message=model_response,
        answer=answer,
        ground_truth=ground_truth,
        success=success,
    )
    result = {"success": success, "answer": answer, "ground_truth": ground_truth}

    return success, result


def main():
    logger = Logger(LOG_DIR)
    logger.gprint("========== AgentGroupChat Start ==========")
    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    # read test data
    test_file = "tasks/AIME/dataset/aime_2024.jsonl"
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            test_data.append(sample)

    num_problems = len(test_data)
    successful = 0

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        print(f"=======Test File {test_file}==========")
        problem = test_data[i]
        question, ground_truth = problem["Problem"], str(problem["Answer"])

        try:
            success, result = solve_problem(question, ground_truth, logger)
            result["idx"] = i

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
