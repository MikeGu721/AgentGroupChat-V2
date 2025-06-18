import regex
from config import *
from typing import Any
from models.generator import generate_prompt, generate_with_response_parser
from environment.characters.character import Character
from utils import extract_code


def run_main_task_decompose(
    main_task_desc, all_character_dict: dict[str, Character], engine, logger
):
    # generate prompt
    all_character_desc = "\n".join(
        [
            f"{character_id}: {character.scratch}"
            for character_id, character in all_character_dict.items()
        ]
    )
    prompt_template = PROMPT_TASK_DECOMPOSE
    prompt_inputs = [main_task_desc, all_character_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)
    # requirements
    valid_character_ids = all_character_dict.keys()
    requirements = {"### Choose Space": valid_character_ids}

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        valid_character_ids = requirements["### Choose Space"]
        try:
            results = []
            for line in model_response.strip().split("\n"):
                if not line:
                    continue  # empty line
                subtask_title = (
                    line.split("### Title:")[-1].split("<DELIMITER>")[0].strip()
                )
                subtask_desc = (
                    line.split("### Description:")[-1].split("<DELIMITER>")[0].strip()
                )
                character_ids = [
                    character_id.strip()
                    for character_id in line.split("### Character:")[-1]
                    .strip()
                    .split(",")
                ]
                # check validation
                for idx in character_ids:
                    if idx not in valid_character_ids:
                        raise Exception("[Error]: Invalid Character Error.")
                # 子任务名称，子任务描述，主群聊/子群聊，成员行动顺序
                results.append((subtask_title, subtask_desc, character_ids))
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return results

    subtasks_and_characters = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_main_task_decompose",
    )

    return subtasks_and_characters


def run_task_prioritize(main_task, sub_tasks: list, engine, logger) -> list[int]:
    # generate prompt
    main_task_desc = main_task.desc
    sub_task_desc = "\n".join(
        [f"{idx + 1}. {task.name}: {task.desc}" for idx, task in enumerate(sub_tasks)]
    )
    prompt_template = PROMPT_TASK_PRIORITIZE
    prompt_inputs = [main_task_desc, sub_task_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)
    # requirements
    valid_sub_task_idx = [i + 1 for i in range(len(sub_tasks))]
    requirements = {"### Sort Space": valid_sub_task_idx}

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        valid_sub_task_idx = requirements["### Sort Space"]
        try:
            priorities = model_response.split("### Priorities:")[-1].strip().split(",")
            priorities = [int(priority.strip()) for priority in priorities]
            # check validation
            if len(priorities) < len(valid_sub_task_idx):
                raise Exception("[Error]: Invalid Priority Length Error.")
            for idx in priorities:
                if idx not in valid_sub_task_idx:
                    raise Exception("[Error]: Invalid Priority Idx Error.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return priorities

    priorities = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_task_prioritize",
    )

    return priorities


def run_check_message(
    main_task_desc, all_subtask_desc, curr_task_desc, group_message_desc, engine, logger
):
    prompt_template = PROMPT_CHECK_MESSAGE
    prompt_inputs = [
        main_task_desc,
        all_subtask_desc,
        curr_task_desc,
        group_message_desc,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            is_completed = (
                model_response.split("### Completed:")[-1]
                .split("### NewTask:")[0]
                .strip()
            )
            if_new_task, new_task = (
                model_response.split("### NewTask:")[-1].strip().split("<DELIMITER>")
            )
            if_new_task, new_task = if_new_task.strip(), new_task.strip()
            # check validation
            requirements = ["Yes", "No"]
            if is_completed not in requirements or if_new_task not in requirements:
                raise Exception("[Error]: Invalid Model Response.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return is_completed, if_new_task, new_task

    is_completed, if_new_task, new_task = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        None,
        MAX_RETRY,
        logger,
        func_name="run_check_message",
    )
    is_completed = True if is_completed == "Yes" else False
    if_new_task = True if if_new_task == "Yes" else False

    return is_completed, if_new_task, new_task


def run_process_new_task(
    main_task_desc,
    all_subtask_desc,
    all_character_desc,
    new_subtask_desc,
    engine,
    logger,
):
    prompt_template = PROMPT_PROCESS_NEW_TASK
    prompt_inputs = [
        main_task_desc,
        all_subtask_desc,
        all_character_desc,
        new_subtask_desc,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            is_new_task = (
                model_response.split("### NewTask:")[-1].split("<DELIMITER>")[0].strip()
            )
            if is_new_task not in ["Yes", "No"]:
                raise Exception("[Error]: Invalid Model Response.")
            is_new_task = True if is_new_task == "Yes" else False
            new_task = None
            if is_new_task:
                subtask_title = (
                    model_response.split("### Title:")[-1]
                    .split("<DELIMITER>")[0]
                    .strip()
                )
                subtask_desc = (
                    model_response.split("### Description:")[-1]
                    .split("<DELIMITER>")[0]
                    .strip()
                )
                character_ids = [
                    character_id.strip()
                    for character_id in model_response.split("### Character:")[-1]
                    .strip()
                    .split(",")
                ]
                new_task = (subtask_title, subtask_desc, character_ids)
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return is_new_task, new_task

    is_new_task, new_task = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        None,
        MAX_RETRY,
        logger,
        func_name="run_process_new_task",
    )
    return is_new_task, new_task


def run_summarize_group_message(task_context, group_messages, engine, logger):
    group_message_desc = "\n".join(
        [message.get_messages_desc() for message in group_messages]
    )
    prompt_template = PROMPT_SUMMARIZE_GROUP_MESSAGE
    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        group_message_desc,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    summary = generate_with_response_parser(
        prompt, engine, logger=logger, func_name="run_summarize_group_message"
    )
    return summary


def run_extract_math_answer(problem, main_group_messages, engine, logger):
    group_message_desc = "\n".join(
        [message.get_messages_desc() for message in main_group_messages]
    )
    prompt_template = PROMPT_EXTRACT_MATH_ANSWER
    prompt_inputs = [problem, group_message_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            answer = model_response.split("####")[-1].strip()
            return model_response, answer
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")

    model_response, answer = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        None,
        MAX_RETRY,
        logger,
        func_name="run_extract_math_answer",
    )
    return model_response, answer

def run_extract_struct_answer(problem, main_group_messages, engine, logger):
    group_message_desc = "\n".join(
        [message.get_messages_desc() for message in main_group_messages]
    )
    prompt_template = PROMPT_EXTRACT_STRUCT_ANSWER
    prompt_inputs = [problem, group_message_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            answer = model_response.split("####")[-1].strip()
            return model_response, answer
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")

    model_response, answer = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        None,
        MAX_RETRY,
        logger,
        func_name="run_extract_struct_answer",
    )
    return model_response, answer


def run_extract_code_answer(problem, main_group_messages, engine, logger):
    group_message_desc = "\n".join(
        [message.get_messages_desc() for message in main_group_messages]
    )
    prompt_template = PROMPT_EXTRACT_CODE_ANSWER
    prompt_inputs = [problem, group_message_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            answer = extract_code(model_response)
            if not answer:
                raise Exception("[Error]: Empty Response Error.")
            return model_response, answer
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")

    model_response, answer = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        None,
        MAX_RETRY,
        logger,
        func_name="run_extract_code_answer",
    )
    return model_response, answer

def run_extract_medical_answer(problem, main_group_messages, engine, logger):
    group_message_desc = "\n".join(
        [message.get_messages_desc() for message in main_group_messages]
    )
    prompt_template = PROMPT_EXTRACT_MEDICAL_ANSWER
    prompt_inputs = [problem, group_message_desc]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        try:
            answer = model_response.split("####")[-1].strip()
            return model_response, answer
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")