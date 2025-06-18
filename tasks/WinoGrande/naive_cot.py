import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/Users/guzhouhong1/Work/AgentGroupChat_v2/"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
import json
import requests
import regex
from logger import Logger
import time


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 3
SAVE_PATH = "result_naive_cot.jsonl"


LOG_DIR = "tasks/WinoGrande/logs_llama3.1-70b"
SAVE_DIR = "tasks/WinoGrande/results_llama3.1-70b"
ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


logger = Logger(LOG_DIR)


def generate_prompt(task: str, question: str):
    prompt = f"""You are a Commonsense Reasoning expert. Your task is to solve the following problem:

Task Description:
{task}

Question:
{question}

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



def solve_problem(question, endings, ground_truth):
    # initialize main task
    task_name = "Choose the best option to fill in the blank of the sentence. Direct output the index number of the option."
    question_prompt = '''Story: %s'''%question
    for index, ending in enumerate(endings):
        question_prompt += '\nOption %d: %s'%(index + 1, ending)
    main_task_desc = question_prompt
    prompt = generate_prompt(task_name, question_prompt)
    retry = 0
    while retry < MAX_RETRY:
        message = ""
        try:
            message, usage = qwen2_by_api(prompt)
            answer = message.split("#### Answer:")[-1].strip()

            if answer == ground_truth:
                success = True
            else:
                success = False

            result = {
                "answer": answer,
                "ground_truth": ground_truth,
                "success": success,
                "message": message,
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
    logger.gprint("========== Naive Start ==========")

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    # read test data
    test_file = "tasks/WinoGrande/dataset/WinoGrande_sample_val_anti_cheat.jsonl"
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            sample = json.loads(line.strip())
            test_data.append(sample)

    num_problems = len(test_data)
    successful = 0

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["sentence"], str(problem["answer"])
        endings = [problem['option1'], problem['option2']]

        try:
            success, result = solve_problem(question, endings, ground_truth)
            result["idx"] = i
            result["question"] = question
            result['endings'] = endings

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
