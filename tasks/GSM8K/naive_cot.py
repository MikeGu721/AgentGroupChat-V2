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
from logger import Logger
import time


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
SAVE_PATH = "result_naive_cot.json"

LOG_DIR = "tasks/GSM8K/logs_qwen_72b"
SAVE_DIR = "tasks/GSM8K/results_qwen_72b"
ENGINE_NAME = "TA/Qwen/Qwen2-72B-Instruct"

# LOG_DIR = "tasks/GSM8K/logs_llama_70b"
# SAVE_DIR = "tasks/GSM8K/results_llama_70b"
# ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)


def generate_prompt(task: str):
    prompt = f"""You are a Math expert. Your task is to solve the following problem:

{task}

You should think step-by-step.
Output your thought first, and then output your final answer.

Format your response as:
#### Thought: [Your thought]
#### Answer: [Your final answer, Always return ONLY the final numerical answer without any units or explanations.]

Here is an example:
#### Thought: xxx
#### Answer: 72
"""
    return prompt


def generate(prompt, engine=ENGINE):
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


def qwen2_by_api(prompt):
    api_key = API_KEY

    headers = {
        "Authorization": "Bearer " + api_key,
    }

    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": ENGINE_NAME,
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


def solve_problem(problem, ground_truth):
    prompt = generate_prompt(problem)
    retry = 0
    while retry < MAX_RETRY:
        message = ""
        try:
            message, usage = qwen2_by_api(prompt)
            answer = message.split("#### Answer:")[-1].strip()
            ground_truth = ground_truth.split("####")[-1].strip()

            if answer == ground_truth:
                success = True
            else:
                success = False

            result = {
                "message": message,
                "answer": answer,
                "ground_truth": ground_truth,
                "usage": usage,
            }
            logger.gprint(
                "Prompt INFO", success=success, prompt=prompt, model_response=result
            )
            return success, result
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


if __name__ == "__main__":
    logger.gprint("========== Naive-CoT Start ==========")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    num_problems = len(test_data)
    successful = 0

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["question"], problem["answer"]

        try:
            success, result = solve_problem(question, ground_truth)
            result["idx"] = i
            result["question"] = question
            result["success"] = success

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
