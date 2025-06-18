import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "F:\\AgentGroupChat_v2"
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
SAVE_PATH = "result_naive_cot.json"

LOG_DIR = "tasks/StructText/logs_llama_70b"
SAVE_DIR = "tasks/StructText/results_llama_70b"
ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
# LOG_DIR = "tasks/StructText/logs_qwen_72b"
# SAVE_DIR = "tasks/StructText/results_qwen_72b"
# ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

logger = Logger(LOG_DIR)


def generate_prompt(task: str):
    prompt = f"""You are a Structured Data Analysis expert. Your task is to analyze and process structured data including CSV/JSON/YAML/XML/Markdown/Latex/Self-defined formats. Follow these steps:
1. Data Identification:
Identify data format and structure in the Reference section
2. Data Parsing:
- For CSV: Understand table relationships using headers
- For JSON/YAML: Analyze nested structure and key paths
- For XML: Identify node hierarchy and attributes
- Else Please self-thinking
3. Operation Planning:
- If multiple tables: Determine join keys
- If nested data: Plan traversal path
- For retrieval tasks: Identify target identifiers
4. Execution Steps:
Show data processing steps with intermediate results
5. Final Verification:
Check result consistency with data schema
Format your response as:
#### Thought: [Your analysis steps]
#### Answer: [Final answer in original data format]
Example(CSV):
Question: How many people in GA company with height > 155?
Reference: 
  Table1: id,height
  Table2: id,company
#### Thought: 
1. Join tables on id
2. Filter company=GA and height>155
3. Count matching records
#### Answer: 0
{task}"""
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

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    # read test data
    test_file = "tasks/StructText/dataset/strucText_test.jsonl"
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
        question, ground_truth = problem["q"], problem["a"]
        # 提取ground truth
        # pattern = r"\\boxed\{(.*)\}"
        # match = regex.search(pattern, ground_truth, flags=regex.DOTALL)
        # ground_truth = match.group(1)

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
