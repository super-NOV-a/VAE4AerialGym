import numpy as np

# 读取文本文件
with open('visualization/model_results.txt', 'r') as file:
    lines = file.readlines()

# 初始化字典来存储每个模型的最大值和最小值
model_stats = {}

# 提取每个模型的最大值和最小值
current_model = None
for line in lines:
    line = line.strip()
    if line.startswith('Model:'):
        current_model = line.split('Model: ')[1]
        model_stats[current_model] = {'max': [], 'min': []}
    elif line.startswith('Max:'):
        # 去掉方括号并分割数据
        max_str = line.split('Max: ')[1]
        max_str = max_str.replace('[', '').replace(']', '')
        max_values = list(map(float, max_str.split()))
        model_stats[current_model]['max'].extend(max_values)
    elif line.startswith('Min:'):
        # 去掉方括号并分割数据
        min_str = line.split('Min: ')[1]
        min_str = min_str.replace('[', '').replace(']', '')
        min_values = list(map(float, min_str.split()))
        model_stats[current_model]['min'].extend(min_values)

# 计算每个模型的最大值和最小值的均值
model_means = {}
for model, stats in model_stats.items():
    max_mean = np.mean(stats['max'])
    min_mean = np.mean(stats['min'])
    model_means[model] = {'max_mean': max_mean, 'min_mean': min_mean}

# 打印结果
for model, means in model_means.items():
    print(f"Model: {model}")
    print(f"  Max Mean: {means['max_mean']}")
    print(f"  Min Mean: {means['min_mean']}\n")

# 保存结果到文件
with open('visualization/model_comparison.txt', 'w') as file:
    for model, means in model_means.items():
        file.write(f"Model: {model}\n")
        file.write(f"  Max Mean: {means['max_mean']}\n")
        file.write(f"  Min Mean: {means['min_mean']}\n\n")