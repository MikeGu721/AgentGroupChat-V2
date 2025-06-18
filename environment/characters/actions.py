import re
from config import *
from utils import *
from typing import Any
from models.generator import generate_prompt, generate_with_response_parser


def run_perceive(character, task_context: dict, engine, logger) -> str:
    # 检索记忆
    short_memory = "\n".join(character.memory.retrieve_memory("short"))
    long_memory = "\n".join(character.memory.retrieve_memory("long"))

    prompt_template = PROMPT_PERCEIVE
    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        character.id,
        character.get_self_description(),
        task_context["all_curr_group_member_desc"],
        long_memory,
        short_memory,
        task_context["max_act_turn"],
        task_context["curr_act_turn"],
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    environment_desc = generate_with_response_parser(
        prompt, engine, logger=logger, func_name="run_perceive"
    )

    return environment_desc


def run_decide_action(
    character,
    environment_desc,
    task_context: dict,
    action_map: dict[str, dict],
    engine,
    logger,
):
    action_desc = "\n".join(
        [
            f"{action_name}: {action_detail['desc']}"
            for action_name, action_detail in action_map.items()
        ]
    )

    prompt_template = PROMPT_DECIDE_ACTION
    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        character.id,
        character.get_self_description(),
        task_context["all_curr_group_member_desc"],
        environment_desc,
        task_context["max_act_turn"],
        task_context["curr_act_turn"],
        action_desc,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    valid_actions = action_map.keys()
    requirements = {"### Action Space": valid_actions}

    def parse_output_action(model_response: str, requirements: dict[str, Any] = None):
        valid_actions = requirements["### Action Space"]
        try:
            action_name = model_response.split("### Action:")[-1].strip()
            if action_name not in valid_actions:
                raise Exception("[Error]: Invalid Action Error.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return action_name

    action_name = generate_with_response_parser(
        prompt,
        engine,
        parse_output_action,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_decide_action",
    )

    return action_name


def run_decide_target(
    character,
    environment_desc,
    task_context: dict,
    action_map: dict[str, dict],
    action_name,
    engine,
    logger,
):
    action = action_map[action_name]
    action_desc = f"{action_name}: {action['desc']}"

    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        character.id,
        character.get_self_description(),
        task_context["all_curr_group_member_desc"],
        environment_desc,
        task_context["max_act_turn"],
        task_context["curr_act_turn"],
        action_desc,
    ]

    prompt_template = PROMPT_DECIDE_TARGET
    prompt = generate_prompt(prompt_inputs, prompt_template)
    all_curr_group_member_ids = list(task_context["all_curr_group_members"].keys())
    all_curr_group_member_ids.extend(["All", "None"])
    requirements = {
        "### Choose Space": all_curr_group_member_ids,
    }

    def parse_output_target(model_response: str, requirements: dict[str, Any] = None):
        valid_targets = requirements["### Choose Space"]
        try:
            target_id = model_response.split("### Target:")[-1].strip()
            if target_id not in valid_targets:
                raise Exception("[Error]: Invalid Target Error.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return target_id

    target_id = generate_with_response_parser(
        prompt,
        engine,
        parse_output_target,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_decide_action",
    )

    return target_id


def run_chat(
    character,
    environment_desc,
    task_context: dict,
    chat_type: str,
    target_id,
    engine,
    logger,
):
    chat_type_desc = ALL_CHAT_TYPES[chat_type]["desc"]

    message_format_desc = character.message_format_desc
    message_format_field = character.message_format_field.split(",")

    message_format = f"{message_format_desc}\n"
    if message_format_field:
        message_format += (
            "Do NOT response with json format. Your output should follow this format with no additional content:\n"
        )
        for field in message_format_field:
            message_format += f"### {field}: xxx\n"

    prompt_template = PROMPT_CHAT
    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        character.id,
        character.get_self_description(),
        task_context["all_curr_group_member_desc"],
        environment_desc,
        chat_type_desc,
        target_id,
        message_format,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    # 输出格式要求
    requirements = None
    if message_format_field:
        requirements = {"### Response Format": message_format_field}

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        valid_formats: list = (
            requirements["### Response Format"] if requirements else []
        )
        try:
            for valid_format in valid_formats:
                if f"### {valid_format}:" not in model_response:
                    raise Exception("[Error]: Invalid Response Format Error.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return model_response

    message = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_chat",
    )

    return message


def run_respond(
    character, sender, task_context: dict, chat_type, chat_history: list, engine, logger
):
    chat_name = ALL_CHAT_TYPES[chat_type]["name"]
    chat_history = "\n".join(
        [
            f"{sender} say to {receiver}: {message}"
            for sender, receiver, message in chat_history
        ]
    )

    message_format_desc = character.message_format_desc
    message_format_field = character.message_format_field.split(",")

    message_format = f"{message_format_desc}\n"
    if message_format_field:
        message_format += (
            "Do NOT response with json format. Your output should follow this format with no additional content:\n"
        )
        for field in message_format_field:
            message_format += f"### {field}: xxx\n"

    prompt_template = PROMPT_RESPOND
    prompt_inputs = [
        task_context["main_task_desc"],
        task_context["all_subtask_desc"],
        task_context["curr_task_desc"],
        character.id,
        character.get_self_description(),
        sender.id,
        chat_name,
        sender.scratch,
        chat_history,
        message_format,
    ]
    prompt = generate_prompt(prompt_inputs, prompt_template)

    # 输出格式要求
    requirements = None
    if message_format_field:
        requirements = {"### Response Format": message_format_field}

    def parse_output(model_response: str, requirements: dict[str, Any] = None):
        valid_formats: list = (
            requirements["### Response Format"] if requirements else []
        )
        try:
            if "None" in model_response:  # 不回复
                return None
            for valid_format in valid_formats:
                if f"### {valid_format}:" not in model_response:
                    raise Exception("[Error]: Invalid Response Format Error.")
        except Exception as e:
            print(
                "==================== MODEL RESPONSE PARSE ERROR ===================="
            )
            print(str(e))
            print(model_response)
            raise Exception("[Error]: Model Response Parse Error.")
        return model_response

    message = generate_with_response_parser(
        prompt,
        engine,
        parse_output,
        requirements,
        MAX_RETRY,
        logger,
        func_name="run_respond",
    )

    return message


def run_summarize(character, task_context: dict, content: str, engine, logger) -> str:
    # TODO: summarize
    pass
