# 请将此文件命名为calculate_accuracy.py或其他名称，但不要命名为math.py

# 从文件读取数据
def read_data_from_file(file_path):
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            # 解析每行数据
            parts = line.split(',')
            if len(parts) < 5:  # 如果最后一行没有Result值，跳过
                continue
                
            task = parts[0].split(':')[1].strip()
            model = parts[1].split(':')[1].strip()
            method = parts[2].split(':')[1].strip()
            if not 'level' in parts[3]: continue
            metric = parts[3].split(':')[1].strip()
            try:
                result = float(parts[4].split(':')[1].strip())
                data.append((task, model, method, metric, result))
            except (IndexError, ValueError):
                # 如果无法解析结果或结果不存在，跳过这一行
                continue
    
    return data

# 定义各级别题目数量
level_counts = {
    "level1": 437,
    "level2": 894,
    "level3": 1131,
    "level4": 1214,
    "level5": 1324
}

# 计算总题目数
total_questions = sum(level_counts.values())

# 计算每个级别的权重
level_weights = {level: count / total_questions for level, count in level_counts.items()}

# 读取数据
try:
    data = read_data_from_file('print_result.txt')  # 替换为您的文件路径
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit(1)

# 将数据组织为字典，键为(task, model, method, metric)
data_dict = {}
for task, model, method, metric, result in data:
    key = (task, model, method, metric)
    data_dict[key] = result

# 获取唯一的(task, model, method)组合
unique_combinations = set()
for task, model, method, _, _ in data:
    unique_combinations.add((task, model, method))

# 计算每个组合的加权平均
results = {}
for task, model, method in unique_combinations:
    weighted_sum = 0
    total_weight_used = 0
    
    for level, weight in level_weights.items():
        key = (task, model, method, level)
        if key in data_dict:
            weighted_sum += data_dict[key] * weight
            total_weight_used += weight
    
    if total_weight_used > 0:
        weighted_avg = weighted_sum / total_weight_used
        results[(task, model, method)] = weighted_avg

# 打印结果
for (task, model, method), result in results.items():
    print(f"Task: {task}, \tModel: {model}, \tMethod: {method}, \tMetric: Accuracy, \tResult: {result:.2f}")
