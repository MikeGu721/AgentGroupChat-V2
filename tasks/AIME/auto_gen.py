import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/home/sharedata/zxx/hf_cache"
MODULE_PATH = "/Users/guzhouhong1/Work/AgentGroupChat_v2/"
os.chdir(MODULE_PATH)
import sys

if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)
from autogen import AssistantAgent
import json
import requests
import regex
from logger import Logger
import time


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 3
SAVE_PATH = "result_autogen.jsonl"


LOG_DIR = "tasks/AIME/logs_qwen2.5-72b"
SAVE_DIR = "tasks/AIME/results_qwen2.5-72b"
ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

logger = Logger(LOG_DIR)

format_requirement = """
Format your response as:
#### Discussion: [Your detailed discussion]
#### Answer: [Your final answer, Always return ONLY the final numerical answer without any units or explanations.]
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
        prompt = f"Here is other's solution:\n{last_message}\n\nPlease provide some suggestions or your solution.\n{format_requirement}"

        message, usage = qwen2_by_api(prompt)
        logger.gprint("Prompt INFO", prompt=prompt, message=message, usage=usage)

        return message


expert1 = CustomAssistant(
    name="Expert1",
    system_message=f"""You are an Math expert. Your task is to solve the Math problem.
You can provide some suggestions or your solution.
{format_requirement}
""",
)

expert2 = CustomAssistant(
    name="Expert2",
    system_message=f"""You are an Math expert. Your task is to solve the Math problem.
You can provide some suggestions or your solution.
{format_requirement}
""",
)

def solve_problem(question, ground_truth):
    task_name = "Output the answer of the given question, the answer should be an int value."
    
    question_prompt = question
    main_task_desc = question_prompt

    prompt = f"""You are a Math expert. Your task is to solve the following problem:

Task Description:
{task_name}

Question:
{question_prompt}

You can think first or privide some idea, and then output your final answer.
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
            if answer == ground_truth:
                success = True
            else:
                success = False

            return success, chat_history, answer
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
        problem = test_data[i]
        question, ground_truth = problem["Problem"], str(problem["Answer"])

        try:
            success, chat_history, res = solve_problem(question, ground_truth)

            result = {
                "idx": i,
                "success": success,
                "answer": res,
                "ground_truth": ground_truth,
                "task": question,
                "iterations": chat_history,
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


# # Custom LLM client for Ollama
# def ollama_request(prompt, engine=ENGINE):
#     """Utilize open source models with ollama server."""
#     messages = [{"role": "user", "content": prompt}]
#     url = "http://localhost:11434/api/chat"
#     payload = {"model": engine, "messages": messages, "stream": False}
#     payload_json = json.dumps(payload)
#     response_json = requests.request("POST", url, data=payload_json).json()
#     return response_json["message"]["content"]


# # Configure the agents
# config_list = [
#     {
#         "model": ENGINE,
#         "api_base": "http://localhost:11434/api/chat",
#         "api_type": "ollama",
#     }
# ]

# # Create the agents
# expert1 = AssistantAgent(
#     name="Expert1",
#     llm_config={
#         "config_list": config_list,
#         "max_consecutive_auto_reply": MAX_ITER,
#     },
#     system_message="""You are a Math expert. Your task is to solve the math problem.
#     You can provide some suggestions or your solution.""",
# )

# expert2 = AssistantAgent(
#     name="Expert2",
#     llm_config={
#         "config_list": config_list,
#         "max_consecutive_auto_reply": MAX_ITER,
#     },
#     system_message="""You are a Math expert. Your task is to solve the math problem.
#     You can provide some suggestions or your solution.""",
# )
