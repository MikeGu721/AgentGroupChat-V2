我现在对于模型的调用也比较乱，有直接装载模型的，有使用api的。下面是两份代码，请你修改generator.py的代码，让其变成api式交互。然后修改api_models.py里的代码，其中一个是用ollama调用，一个是调用外部api，你再帮我写一个调用generator.py的api，再帮我写一个完整的接口，这个接口的名字你自拟，但我希望，后面的代码全都使用这个接口去调用大模型的回复，然后这个接口可以对接ollama，可以对接外部api，可以对接generator.py的api都行，反正用户指定好就行，同时，你把需要修改的其他代码也一并修改了，下面两个代码的位置是：./models/generator.py和./models/api_models.py。现在的代码文件很多了，你修改好之后，再帮我生成一个代码目录给我

generator.py

import time
from typing import Any
from .api_models import *
from config import *


def read_prompt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        prompt = f.read()

    return prompt


def generate_prompt(prompt_inputs: list, prompt_file):
    """
    Generate prompt with prompt template and input variables

    Args:
        prompt_inputs: variables to feed in template
        prompt_file: prompt template file

    Returns:
        str: new prompt
    """
    prompt_inputs = [str(i) for i in prompt_inputs]
    prompt = read_prompt_file(prompt_file)

    for idx, value in enumerate(prompt_inputs):
        prompt = prompt.replace(f"!<INPUT {idx}>!", value)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

    return prompt.strip()


def generate(prompt, engine: str) -> dict:
    response_json = {}
    try:
        if ":" in engine:  # ollama format
            message, token_usage, time_costed = ollama_request_by_url(prompt, engine)
            response_json = {
                "message": message,
                "token_usage": token_usage,
                "time_costed": time_costed,
            }
            return response_json
        elif engine.startswith("TA"):
            message, usage = qwen2_by_api(prompt, engine)
            response_json = {"message": message, "usage": usage}
            return response_json
        else:
            raise Exception(
                f"[Error]: Engine {engine} Not Implemented Error. Only ollama-based models are available."
            )
    except:
        print("==================== MODEL RESPONSE ERROR ====================")
        print(response_json)
        raise Exception(f"[Error]: Engine {engine} Request Error.")


def non_parse_fn(response_message: str, requirements=None):
    if requirements is not None:
        print("Requirements are provided but ignored in non_parse_fn.")

    return response_message


def generate_with_response_parser(
    prompt,
    engine,
    parser_fn=non_parse_fn,
    requirements: dict[str, Any] = None,
    max_retry=MAX_RETRY,
    logger=None,
    func_name=None,
):
    """
    Generate output with retry logic and response parsing.

    Args:
    requirements: generation space limitation
    """
    curr_retry = 0

    while curr_retry < max_retry:
        output, response_json = "", None
        try:
            response_json = generate(prompt, engine)
            output = parser_fn(response_json["message"], requirements)

            if logger:
                logger.gprint(
                    "Prompt INFO",
                    prompt=prompt,
                    model_response=response_json,
                    output=output,
                    func_name=func_name,
                )

            if DEBUG and func_name:
                print("==================== PROMPT DEBUG START ====================")
                print("Function Name: ", func_name)
                print("Prompt:\n", prompt)
                print("Model Response:\n", response_json)
                print("Output:\n", output)
                print("==================== PROMPT DEBUG END ======================\n")

            return output
        except Exception as e:
            if logger:
                logger.gprint(
                    "### ERROR: Failed in generate_with_response_parser!",
                    prompt=prompt,
                    model_response=response_json,
                    output=output,
                    error=str(e),
                )
            print(e)

        # retry
        print(f"Retrying ({curr_retry + 1}/{max_retry})...")
        time.sleep(5)
        curr_retry += 1

    # Raise exception after exceeding retries
    raise Exception(f"[Error]: Exceeded Max Retry Times ({max_retry}).")


api_models.py

import json
import requests
from config import *


def ollama_request_by_url(prompt, engine):
    """
    Utilize open source models with ollama server.
    """
    messages = [{"role": "user", "content": prompt}]
    url = "http://localhost:11434/api/chat"
    payload = {"model": engine, "messages": messages, "stream": False}
    payload_json = json.dumps(payload)
    response_json = requests.request("POST", url, data=payload_json).json()

    message = response_json["message"]["content"]
    token_usage = response_json["prompt_eval_count"] + response_json["eval_count"]
    time_costed = response_json["prompt_eval_duration"] + response_json["eval_duration"]

    return message, token_usage, time_costed


def qwen2_by_api(prompt, engine_name):
    api_key = API_KEY
    if "#" in engine_name:
        temp = engine_name.split("#")
        engine_name = temp[0]
        temperature = float(temp[1])
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
            "temperature": temperature,
        }
    else:
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
        }

    headers = {
        "Authorization": "Bearer " + api_key,
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