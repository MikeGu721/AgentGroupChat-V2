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
from logger import Logger
import time

from typing import List, Dict, Any
import re
from dataclasses import dataclass
import regex


ENGINE = "qwen2_api"
API_KEY = ""
MAX_RETRY = 10
MAX_ITER = 2
SAVE_PATH = "result_react.json"

# LOG_DIR = "tasks/StructText/logs_qwen_72b"
# SAVE_DIR = "tasks/StructText/results_qwen_72b"
# # ENGINE_NAME = "TA/Qwen/Qwen2-72B-Instruct"
# ENGINE_NAME = "TA/Qwen/Qwen2.5-72B-Instruct-Turbo"

LOG_DIR = "tasks/StructText/logs_llama_70b"
SAVE_DIR = "tasks/StructText/results_llama_70b"
ENGINE_NAME = "TA/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

logger = Logger(LOG_DIR)

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


@dataclass
class Action:
    name: str
    args: Dict[str, Any]

    def __str__(self) -> str:
        # 将参数字典转换成字符串形式
        args_str = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.name}({args_str})"


class ReActAgent:
    def __init__(self):
        self.actions = {
            "extract_conditions": self._extract_conditions,
            "query_from_reference": self._query_from_reference,
            "conclude_result": self._conclude_result,
        }
        self.current_question = ""
        self.reference_text = ""
    def _format_react_prompt(
        self,
        question: str,
        thoughts: List[str],
        actions: List[Action],
        observations: List[str],
    ) -> str:
        prompt = f"""You are a struct text analysis expert. Your task is to solve the following problem:
{question}

### Reference:
{self.reference_text}

You have access to these actions:
- extract_conditions: Extract key information or conditions from the question.
  Example1: For "What is the content of 1th section?", returns ["1th section"]
- query_from_reference: Query content that matches the condition(s) from the reference.
  Example: returns "* 1th Section\\nSome excerpt"
- conclude_result: Count or summarize the final result and return it.
  Example: conclude_result("matched content") → #### [result]

Previous steps:"""

        for i, (thought, action, obs) in enumerate(zip(thoughts, actions, observations)):
            prompt += f"\nStep {i+1}:"
            prompt += f"\nThought: {thought}"
            prompt += f"\nAction: {action}"
            prompt += f"\nObservation: {obs}\n"

        prompt += "\nWhat should be the next step? Respond in the following format:"
        prompt += "\nThought: [your reasoning]"
        prompt += "\nAction: [action name] or '#### [answer]' if solved, and you should ONLY include the final answer."

        return prompt

    def solve(self, question: str) -> str:
        self.current_question = question
        thoughts: List[str] = []
        actions: List[Action] = []
        observations: List[str] = []

        for step in range(MAX_ITER):
            prompt = self._format_react_prompt(question, thoughts, actions, observations)
            print(prompt)

            retry = 0
            while retry < MAX_RETRY:
                response = ""
                try:
                    response, usage = qwen2_by_api(prompt)
                    print(response)
                    print(usage)

                    logger.gprint("Prompt INFO", prompt=prompt, model_response=response, usage=usage)

                    thought, action_str = self._parse_response(response)
                    print("Thought:\n", thought)
                    print("Action:\n", action_str)
                    thoughts.append(thought)

                    action_obj = self._parse_action(action_str)
                    actions.append(action_obj)

                    observation = self._execute_action(action_obj)
                    print("observation:\n", observation)
                    observations.append(observation)

                    if "####" in response:
                        return self._extract_final_answer(response), {
                            "thoughts": thoughts,
                            "actions": [str(a) for a in actions],
                            "observations": observations,
                            "response": response,
                        }

                    break
                except Exception as e:
                    logger.gprint(
                        "### ERROR: Failed in generate_with_response_parser!",
                        prompt=prompt,
                        model_response=response,
                        error=str(e),
                    )
                    print(e)
                    time.sleep(5)
                    retry += 1

            if retry == MAX_RETRY:
                raise Exception(f"[Error]: Exceeded Max Retry Times ({MAX_RETRY}).")

        return "Max Trial, answer not found", {
            "thoughts": thoughts,
            "actions": [str(a) for a in actions],
            "observations": observations,
        }
    def set_reference(self, reference_text: str):
        self.reference_text = reference_text
        tables = reference_text.split('\n\n')
        parsed_tables = []
        
        # 解析所有CSV表格
        for table in tables:
            lines = [line.strip() for line in table.split('\n') if line.strip()]
            if not lines:
                continue
            headers = lines[0].split(',')
            data = [
                dict(zip(headers, line.split(','))) 
                for line in lines[1:]
            ]
            parsed_tables.append(data)

        # 合并关联表格
        self.merged_data = []
        if len(parsed_tables) >= 2:
            main_table = parsed_tables[0]
            aux_table = parsed_tables[1]
            for main_row in main_table:
                prime_key = main_row.get('primeKey')
                for aux_row in aux_table:
                    if aux_row.get('primeKey') == prime_key:
                        merged = {**main_row, **aux_row}
                        self.merged_data.append(merged)
        else:
            self.merged_data = parsed_tables[0] if parsed_tables else []

    def _extract_conditions(self) -> List[str]:
        patterns = [
            # 匹配数值比较（如 salary > 63659）
            (r'(\w+)\s+(more than|less than|greater than|>\s*|<\s*|=|equal to)\s*(\d+)', 
             lambda m: f"{m[1]}{'>' if m[2] in ['more than', 'greater than'] else '<' if m[2] in ['less than'] else '='}{m[3]}"),
            # 匹配枚举值（如 gender == female）
            (r'(are|is)\s+(male|female|other)', 
             lambda m: f"gender=={m[2].lower()}"),
            # 匹配存在性检查（如 unemployed）
            (r'(retired|unemployed)', 
             lambda m: f"status=={m[1].lower()}")
        ]
        
        conditions = []
        for pattern, processor in patterns:
            matches = re.finditer(pattern, self.current_question, re.IGNORECASE)
            for match in matches:
                conditions.append(processor(match))
        
        return conditions

    def _query_from_reference(self, conditions: List[str]) -> str:
        parsed_conditions = []
        for cond in conditions:
            # 解析条件表达式
            for op in ['>=', '<=', '!=', '==', '>', '<', '=']:
                if op in cond:
                    field, value = cond.split(op, 1)
                    parsed_conditions.append({
                        'field': field.strip(),
                        'op': op if op != '=' else '==',
                        'value': value.strip()
                    })
                    break
        
        matched = 0
        for record in self.merged_data:
            valid = True
            for cond in parsed_conditions:
                field = cond['field']
                op = cond['op']
                value = cond['value']
                
                # 获取记录中的值并转换类型
                record_value = record.get(field)
                if record_value is None:
                    valid = False
                    break
                
                # 数值类型转换
                if field in ['salary', 'age', 'height', 'weight']:
                    try:
                        record_value = float(record_value)
                        value = float(value)
                    except:
                        pass
                
                # 执行比较操作
                if op == '>' and not (record_value > value):
                    valid = False
                    break
                elif op == '<' and not (record_value < value):
                    valid = False
                    break
                elif op == '==' and str(record_value).lower() != str(value).lower():
                    valid = False
                    break
            
            if valid:
                matched += 1
        
        return str(matched)

    def _parse_response(self, response: str) -> tuple[str, str]:
        thought_match = re.search(r"Thought: (.*?)(?=\nAction:|$)", response, re.DOTALL)
        action_match = re.search(r"Action: (.*?)(?=\n|$)", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        return thought, action

    def _parse_action(self, action_str: str) -> Action:
        if "extract_conditions" in action_str:
            return Action("extract_conditions", {})
        elif "query_from_reference" in action_str:
            match = re.search(r"query_from_reference\s*,\s*\[(.*)\]", action_str)
            conditions = eval(f"[{match.group(1)}]") if match else []
            return Action("query_from_reference", {"conditions": conditions})
        elif "conclude_result" in action_str:
            match = re.search(r'conclude_result\s*,\s*"([^"]+)"', action_str)
            content = match.group(1) if match else ""
            return Action("conclude_result", {"queried_result": content})
        elif "####" in action_str:
            return Action("conclude_result", {"queried_result": action_str})
        else:
            raise ValueError(f"Unrecognized action: {action_str}")

    def _execute_action(self, action: Action) -> str:
        try:
            return self.actions[action.name](**action.args)
        except Exception as e:
            return f"Action failed: {str(e)}"
    # def _query_from_reference(self, conditions: List[str]) -> str:
    #     lines = self.reference_text.splitlines()
    #     matched_lines = []
    #     for line in lines:
    #         if any(cond.lower() in line.lower() for cond in conditions):
    #             matched_lines.append(line)
    #     return "\n".join(matched_lines)

    def _conclude_result(self, queried_result: str) -> str:
        return f"#### {queried_result.strip()}"

    def _extract_final_answer(self, action: str) -> str:
        return action.split("####")[-1].strip()

def main():
    logger.gprint("========== ReAct Start ==========")

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

    agent = ReActAgent()

    for i in range(155,num_problems):
        print(f"========== Solving {i}/{num_problems} Problem ==========")
        problem = test_data[i]
        question, ground_truth = problem["q"], problem["a"]
        # 提取ground truth
        pattern = r"\\boxed\{(.*)\}"

        try:
            answer, result = agent.solve(question)
            if answer == ground_truth:
                success = True
            else:
                success = False
            result["idx"] = i
            result["answer"] = answer
            result["question"] = question
            result["ground_truth"] = ground_truth
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


if __name__ == "__main__":
    main()
