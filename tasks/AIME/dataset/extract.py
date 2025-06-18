import csv
import json

def extract_aime_data(csv_file_path, output_jsonl_path, years):
    """
    从CSV文件中提取指定年份的AIME数据，并保存为JSONL格式
    
    参数:
    csv_file_path (str): CSV文件路径
    output_jsonl_path (str): 输出JSONL文件路径
    years (list): 要提取数据的年份列表
    """
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file, open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            try:
                year = int(row['Year'])
            except (ValueError, TypeError):
                year = row['Year']
            
            if year in years:
                # 提取所需字段
                structured_data = {
                    "year": year,
                    "question": row['Question'],
                    "ground_truth": row['Answer'],
                    "other": {}
                }
                
                # 将所有其他字段添加到'other'字典中
                for key, value in row.items():
                    if key not in ['Year', 'Question', 'Answer']:
                        structured_data["other"][key] = value
                
                # 将结构化数据写入JSONL文件
                jsonl_file.write(json.dumps(structured_data) + '\n')

# 要提取的年份列表
years = [2024]  # 您可以根据需要修改这个列表

# 调用函数
extract_aime_data('aime_data.csv', 'aime.jsonl', years)
