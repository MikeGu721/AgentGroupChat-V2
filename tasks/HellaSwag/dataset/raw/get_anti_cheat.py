import json

def filter_anti_cheat(result_path, data_path, output_path):
    with open(result_path, 'r', encoding='utf-8') as result_file, \
         open(data_path, 'r', encoding='utf-8') as data_file, \
         open(output_path, 'w', encoding='utf-8') as output_file:

        for result_line, data_line in zip(result_file, data_file):
            try:
                # 解析 JSONL 行
                result_entry = json.loads(result_line.strip())
                data_entry = json.loads(data_line.strip())

                # 比对字段
                if result_entry.get('ground_truth') != result_entry.get('answer'):
                    output_file.write(json.dumps(data_entry) + '\n')
            except json.JSONDecodeError:
                print(f"解析错误，跳过当前行：{result_line} 或 {data_line}")
                continue

# 调用函数
filter_anti_cheat(
    result_path='result_cheat.jsonl',
    data_path='hellaswag_sample_val.jsonl',
    output_path='hellaswag_sample_val_anti_cheat.jsonl'
)