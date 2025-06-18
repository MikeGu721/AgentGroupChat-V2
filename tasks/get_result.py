import os
import json


import json

import re
def extract_answers(data):
    # 存储所有找到的答案
    answers = []
    
    # 遍历所有对话轮次
    for iteration in data["iterations"]:
        content = iteration["content"]
        
        # 使用正则表达式匹配 #### Answer: 后的内容
        match = re.search(r'#### Answer:\s*([^\n]+)', content)
        if match:
            answer = match.group(1).strip()
    return {
        "ground_truth": data["ground_truth"],
        "final_success": data["final_success"],
        "answer": answer
    }

import json
import re
import os
from pathlib import Path

def extract_number_from_text(text):
    """Extract the numeric answer from a text response."""
    if not isinstance(text, str):
        return text
        
    # Try to find "Answer: X" pattern first
    answer_match = re.search(r'Answer:\s*([-+]?\d*\.?\d+)', text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)
    
    # Try to find any number in the text
    number_match = re.search(r'([-+]?\d*\.?\d+)', text)
    if number_match:
        return number_match.group(1)
    
    return text

def convert_to_float(value):
    """Convert a string value to float, handling percentage representations."""
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove any whitespace and handle string inputs
    value = str(value).strip()
    
    # Handle percentage values
    if value.endswith('%'):
        return float(value.rstrip('%')) / 100
    
    try:
        float_val = float(value)
        # If the value is between 0 and 1, check if it might be a decimal percentage
        if 0 < float_val < 1:
            return float_val
        # If value is between 1 and 100, it might be a percentage without % symbol
        elif 1 <= float_val <= 100:
            # Check if the other common format exists in ground truth/answer pairs
            return float_val / 100
        return float_val
    except ValueError:
        return value  # Return original value if not convertible to float

def is_equivalent(answer, ground_truth, tolerance=1e-1):
    """
    Compare two values for equivalence, handling:
    - Percentage representations (both with and without % symbol)
    - Different decimal precision
    - String/numeric comparisons
    - Values where one is in decimal form (0.1234) and another in percentage form (12.34)
    - Multiple choice questions (A, B, C or full text)
    - Numeric values with different formats (e.g. 47250188 vs €47,250,188)
    """
    # Extract numbers from text responses
    answer = extract_number_from_text(answer)
    ground_truth = extract_number_from_text(ground_truth)
    
    # First try to convert both values to floats
    ans_val = convert_to_float(answer)
    truth_val = convert_to_float(ground_truth)
    
    # If both are numeric, compare with tolerance
    if isinstance(ans_val, float) and isinstance(truth_val, float):
        # Try both direct comparison and percentage comparison
        direct_match = abs(ans_val - truth_val) < tolerance
        percentage_match = abs(ans_val * 100 - truth_val * 100) < tolerance
        return direct_match or percentage_match
    
    # Handle multiple choice questions
    if isinstance(ground_truth, str) and len(ground_truth) == 1 and ground_truth in ['A', 'B', 'C']:
        # Check if answer is just the letter
        if answer == ground_truth:
            return True
        # Check if answer contains the letter followed by colon or dot
        if re.match(f'^{ground_truth}[.:]', answer):
            return True
        # Check if answer is the full text after the letter
        if re.match(f'^{ground_truth}[.:]\\s*(.*)', answer):
            return True
    
    # Handle numeric values with different formats
    if isinstance(answer, str) and isinstance(ground_truth, str):
        # Remove currency symbols, commas, and whitespace
        clean_ans = re.sub(r'[€$,\s]', '', answer)
        clean_truth = re.sub(r'[€$,\s]', '', ground_truth)
        if clean_ans == clean_truth:
            return True
    
    # If conversion to float failed, compare strings directly
    return str(answer).strip().upper() == str(ground_truth).strip().upper()

def extract_answer_from_result(result, method):
    """Extract answer and ground truth from result based on method."""
    if method == "autogen":
        # For autogen, try to find answer in iterations
        if "iterations" in result:
            iterations = result["iterations"]
            if iterations:
                # Get the last iteration's content
                last_message = iterations[-1]["content"]
                # Extract answer after "#### Answer:"
                answer_match = re.search(r'#### Answer:\s*(.*?)(?=\n|$)', last_message, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                    return answer, result.get("ground_truth")
        return None, result.get("ground_truth")
    elif method == "react":
        # Extract answer from response field
        response = result.get("response", "")
        # First try to find answer after "####"
        answer_match = re.search(r'####\s*(.*?)(?=\n|$)', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return answer, result.get("ground_truth")
        
        # If no answer found in response, try to find in thoughts/actions
        if "thoughts" in result:
            # Check the last thought
            thoughts = result["thoughts"]
            if thoughts:
                last_thought = thoughts[-1]
                answer_match = re.search(r'####\s*(.*?)(?=\n|$)', last_thought, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                    return answer, result.get("ground_truth")
        
        # If still no answer found, return None
        return None, result.get("ground_truth")
    else:
        return result.get("answer"), result.get("ground_truth")

def compare_answer(result, method):
    """Compare answer with ground truth in a result dictionary."""
    if not isinstance(result, dict):
        return False
    
    answer, ground_truth = extract_answer_from_result(result, method)
    
    if answer is None or ground_truth is None:
        return False
        
    return is_equivalent(answer, ground_truth)

def calculate_accuracy(file_path):
    """Calculate accuracy for a single result file."""
    total = 0
    correct = 0
    
    # Extract method name from file path
    method = Path(file_path).stem.replace("result_", "")
    
    with open(file_path, "r") as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                total += 1
                if compare_answer(result, method):
                    correct += 1
            except json.JSONDecodeError:
                print(f"Error parsing line in file: {file_path}")
                continue
    
    return total, correct

def process_results_directory(base_dir):
    """Process all result files in a directory and print accuracies."""
    base_path = Path(base_dir)
    print(f"\nProcessing results in: {base_dir}")
    print("-" * 60)
    print(f"{'Method':<20} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    # Get all result files
    result_files = list(base_path.glob("result_*.json"))
    
    # Sort files by method name
    result_files.sort()
    
    # Process each file
    for file_path in result_files:
        method_name = file_path.stem.replace("result_", "")
        total, correct = calculate_accuracy(file_path)
        accuracy = correct / total if total > 0 else 0
        print(f"{method_name:<20} {total:<10} {correct:<10} {accuracy:.2%}")

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def remove_whitespace(s):
    return re.sub(r'\s+', '', s)
def remove_format(s):
    # 去除一些```yaml \n 等格式的内容
    s = re.sub(r'```yaml\n', '', s)
    s = re.sub(r'```\n', '', s)
    s = re.sub(r'```json\n', '\n', s)
    s = re.sub(r'```xml\n', '\n', s)
    s = re.sub(r'```', '\n', s)
    s = re.sub(r'```markdown\n', ' ', s)
    return s

name_dict = {
    'debate': "Multi-Agent Debate",
    'naive': 'Naive',
    'naive_cot': 'Naive-CoT',
    'react': 'ReAct',
    'autogen':'AutoGen',
    'agentgroupchat': 'AgentGroupChat'
}

def print_experiment_results(json_file_path):
    """
    读取实验结果JSON并以指定格式打印
    格式: Method: {}, Model: {}, Metric: {}, Result: {}
    """
    # 从文件加载JSON数据
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 遍历每个数据集/指标
    for dataset, models in data.items():
        # 遍历每个模型
        for model, methods in models.items():
            # 遍历每种方法
            for method, result in methods.items():
                print(f"Task: {dataset.split('_')[0]}, \tModel: {model}, \tMethod: {method}, \tMetric: {'Accuracy' if dataset in ['gsm8k', 'math23k'] else dataset.split('_')[-1]}, \tResult: {result}")


# if __name__ == "__main__":
#     # Process results for both models
#     process_results_directory("tasks/Finance/results_qwen_72b")
#     process_results_directory("tasks/Finance/results_llama_70b")


# 使用示例
print_experiment_results('existing_results.json')


for task in os.listdir():
    if not os.path.isdir(task): continue

    for dirr in os.listdir(task):
        if not dirr.startswith('results'): continue
        model_name = 'qwen2.5-72b' if 'qwen' in dirr else 'llama3.1-70b'

        for jsonlfile in os.listdir(os.path.join(task, dirr)):
            if not 'json' in jsonlfile: continue
            try:
                method_name = jsonlfile.split('result_')[-1].split('.json')[0]
                if 'agentgroupchat' in method_name: method_name = 'agentgroupchat'
                jsonldata = open(os.path.join(task, dirr, jsonlfile), encoding='utf-8')
                succ = count = 0

                for line in jsonldata:
                    jsonline = json.loads(line)
                    if 'error' in jsonline: continue

                    if 'answer' not in jsonline: 
                        if'iterations' in jsonline:
                            answer = jsonline["iterations"][-1]['content'].split('### Answer:')[-1].strip()
                            jsonline['answer'] = answer
                    score = 0
                    if task == 'WinoGrande':
                        if jsonline['answer'] in jsonline['endings']:
                            if str(jsonline['endings'].index(jsonline['answer'])) == jsonline['ground_truth']:
                                jsonline['success'] = True
                                jsonline['answer'] = jsonline['ground_truth']
                        if jsonline['answer'] not in ['0', '1', '2', '3']: continue
                    elif task == 'HellaSwag':
                        if jsonline['answer'] in jsonline['endings']:
                            if str(jsonline['endings'].index(jsonline['answer'])) == jsonline['ground_truth']:
                                jsonline['success'] = True
                                jsonline['answer'] = jsonline['ground_truth']
                        if jsonline['answer'] not in ['0', '1', '2', '3']: continue
                    elif task == 'Finance':
                        if 'ground_truth' in jsonline and 'answer' in jsonline:
                            if jsonline['ground_truth'].lower() in 'abcd':
                                if jsonline['answer'].lower() not in 'abcd': continue
                            ans_num = ''.join([i for i in jsonline['answer'] if i in '0123456789'])
                            gt_num = ''.join([i for i in jsonline['ground_truth'] if i in '0123456789'])
                            if len(ans_num)>=1 and len(gt_num) >=1 :
                                while ans_num.startswith('0') and len(ans_num) > 1:
                                    ans_num = ans_num[1:]
                                while gt_num.startswith('0') and len(gt_num) > 1:
                                    gt_num = gt_num[1:]
                                if ans_num in gt_num: score = 1
                                if gt_num in ans_num: score = 1
                        if score == 0 and compare_answer(jsonline, method_name):
                            score = 1
                    elif task == 'StructText':
                        try:
                            record = jsonline
                            answer = record['answer']
                            ground_truth = record['ground_truth']
                            if len(ground_truth) <= 8:
                                # 短文本直接比对
                                if answer == ground_truth:
                                    score = 1
                            else:
                                # 长文本处理
                                processed_ans = remove_whitespace(answer)
                                processed_ans = remove_format(processed_ans)
                                processed_ans = processed_ans.lower()
                                processed_gt = remove_whitespace(ground_truth)
                                processed_gt = processed_gt.lower()
                                distance = levenshtein_distance(processed_ans, processed_gt)
                                gt_length = len(processed_gt)
                                
                                if gt_length == 0:
                                    score = 1.0 if processed_ans == processed_gt else 0.0
                                else:
                                    score = max(0.0, 1.0 - (distance / gt_length))
                        except:
                            continue
                    if score!=0:
                        succ += score

                    else:
                        if 'success' in jsonline:
                            succ += 1 if jsonline['success'] else 0
                        elif 'final_success' in jsonline:
                            succ += 1 if jsonline['final_success'] else 0
                    count += 1

                print(f'Task: {task}, \tModel: {model_name}, \tMethod: {name_dict[method_name]}, \tMetric: Accuracy, \tResult: {100*succ/count}')
            except:
                cc = os.path.join(task, dirr, jsonlfile)
                print(f'error in {cc}')