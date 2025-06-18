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
