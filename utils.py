import re
import time
import requests

TASK_CONTEXT = {
    "main_task_desc": "",
    "all_subtask_desc": "",
    "all_character_desc": "",
    "curr_task_desc": "",
    "curr_group_actor_ids": "",
    "all_curr_group_members": None,
    "all_curr_group_member_desc": "",
    "max_act_turn": 0,
    "curr_act_turn": 0,
}

MAIN_GROUP_ID = "Group_Main"

# ALL_CHAT_TYPES = {
#     "Skip": {"name": "不采取行动", "desc": "不采取任何行动，先观察一下"},
#     "Private": {
#         "name": "私聊",
#         "desc": "选择当前群组中的某个角色私聊，你们的对话不会公开，其他人也不会知道你们进行过对话。",
#     },
#     "Meeting": {
#         "name": "秘密会晤",
#         "desc": "选择当前群组中的某个角色秘密会晤，你们的对话不会公开，其他人将仅知道你们对话过，不会知道你们的对话内容。",
#     },
#     "GroupChat": {
#         "name": "群内聊天",
#         "desc": "和你当前所在群组中的成员群聊，你可以对所有人发言，也可以选择某个对象发言。你可以将需要同步的内容发送到群聊中。",
#     },
# }

ALL_CHAT_TYPES = {
    "Skip": {"name": "No Action", "desc": "Take no action, observe the situation"},
    "Private": {
        "name": "Private Chat",
        "desc": "Select a character from current group for private chat. Your conversation will not be public, others won't know about it.",
    },
    "Meeting": {
        "name": "Secret Meeting",
        "desc": "Select a character from current group for secret meeting. Your conversation will not be public, others will only know you talked but not the content.",
    },
    "GroupChat": {
        "name": "Group Chat",
        "desc": "Select a character or everyone from current group for group chat. You can share content that needs synchronization in group chat.",
    },
}


def extract_code(text):
    # 去掉前面的 ```python 或 ``` 标记
    text = re.sub(r"^```(?:python)?\n?", "", text)
    # 去掉后面的 ``` 标记
    text = re.sub(r"\n?```$", "", text)
    # 去掉前后多余的空白和换行
    return text.strip()


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

def qwen2_by_api(prompt):
    api_key = "sk-5FMt1WOBeb2B8203e06CT3BlbKFJ16F4c8b84f1f497fAE1D"

    headers = {
        "Authorization": "Bearer " + api_key,
    }

    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "TA/Qwen/Qwen2.5-72B-Instruct-Turbo",
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
    while retry < 10:
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
        f"[Error]: Failed to Extract Answer. Exceeded Max Retry Times ({10}).")
