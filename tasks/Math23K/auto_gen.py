import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/Users/zhuxiaoxuan/Project/hf_cache"
MODULE_PATH = "/Users/zhuxiaoxuan/Project/AgentGroupChat_v2"
os.chdir(MODULE_PATH)
import sys
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
from autogen import AssistantAgent
import json
import requests
from logger import Logger
import time
from datasets import load_dataset


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 3
SAVE_PATH = "result_autogen.json"

LOG_DIR = "tasks/Math23K/logs_qwen_72b"
SAVE_DIR = "tasks/Math23K/results_qwen_72b"
ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

# LOG_DIR = "tasks/Math23K/logs_llama_70b"
# SAVE_DIR = "tasks/Math23K/results_llama_70b"
# ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)

format_requirement = """
请按如下格式输出：
#### Discussion: [你的详细讨论]
#### Answer: [你的最终答案，仅包含最终数值答案，不包含任何单位或解释。]
"""


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


class CustomAssistant(AssistantAgent):
    def generate_reply(self, messages, sender=None):
        last_message = messages[-1]["content"]
        prompt = f"这是其他人的解决方案：\n{last_message}\n\n请对此提出一些意见或你的解决方案。\n{format_requirement}"

        message, usage = qwen2_by_api(prompt)
        logger.gprint("Prompt INFO", prompt=prompt, message=message, usage=usage)

        return message


expert1 = CustomAssistant(
    name="Expert1",
    system_message=f"""你是一个数学专家，你的任务是解决下面的数学问题。
{format_requirement}
""",
)

expert2 = CustomAssistant(
    name="Expert2",
    system_message=f"""你是一个数学专家，你的任务是解决下面的数学问题。
{format_requirement}
""",
)


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
    while retry < MAX_RETRY:
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
        f"[Error]: Failed to Extract Answer. Exceeded Max Retry Times ({MAX_RETRY}).")


def solve_problem(problem_text, ground_truth):
    prompt = f"""你是一个数学专家，你的任务是解决下面的数学问题：

{problem_text}

你需要一步步思考，先输出你的思考过程，再输出你的最终答案。
"""
    retry = 0
    while retry < MAX_RETRY:
        message = ""
        try:
            # Initialize conversation between agents
            chat_response = expert1.initiate_chat(
                expert2, message=prompt, max_turns=MAX_ITER
            )

            # Extract the code from the response
            chat_history = chat_response.chat_history
            message = chat_history[-1]["content"]

            answer = message.split("#### Answer:")[-1].strip()

            success = check_answer(problem_text, ground_truth, answer)

            return success, chat_history
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


def main():
    logger.gprint("========== AutoGen Start ==========")

    save_dir = SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, SAVE_PATH)

    dataset = load_dataset("Gxg/Math23K")
    test_data = dataset["test"]
    test_data = test_data.shuffle(seed=42).select(range(200))

    num_problems = len(test_data)
    successful = 0

    for i in range(num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["original_text"], problem["answer"]

        try:
            success, chat_history = solve_problem(question, ground_truth)

            result = {
                "idx": i,
                "task": question,
                "iterations": chat_history,
                "ground_truth": ground_truth,
                "final_success": success,
            }
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)

            if success:
                successful += 1
                print(f"Problem {i+1} solved successfully!")
            else:
                print(f"Problem {i+1} failed.")
        except Exception as e:
            result = {"idx": i, "question": question, "success": False, "error": str(e)}
            with open(save_path, "a", encoding="utf-8") as f:
                json_str = json.dumps(result, ensure_ascii=False) + "\n"
                f.write(json_str)
            continue

    print(f"\nFinal Results: {successful}/{num_problems} problems solved successfully")
    print(successful / num_problems)


if __name__ == "__main__":
    main()
