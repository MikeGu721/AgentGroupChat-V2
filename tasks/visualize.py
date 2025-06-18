import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict

# Define the category ordering with correct benchmark names
sorted_dic = {
    "CommonSense": ["HellaSwag", "WinoGrande"],
    "Domain": ["JEC-QA", "FinQual", "MedmcQA"],
    "Structural": ["StrucText-Eval"],
    "Math": ["MATH", "GSM8K", "AIME"],
    "Code": ["MBPP", "HumanEval"]
}

# 添加方法名称映射字典
method_display_mapping = {
    'AgentGroupChat': 'AgentGroupChat-V2'
}

def parse_data(data_string):
    """Parse the input string data into a DataFrame"""
    rows = []
    
    # Split the input string into lines
    lines = [line.strip() for line in data_string.strip().split('\n')]
    
    for line in lines:
        # Use regex to extract all components
        match = re.search(r'Task:\s*([^,]+),\s*Model:\s*([^,]+),\s*Method:\s*([^,]+),\s*Metric:\s*([^,]+),\s*Result:\s*([0-9.]+)', line)
        if match:
            task, model, method, metric, result = match.groups()
            if 'level' in metric: continue
            rows.append({
                'Task': task.strip(),
                'Model': model.strip(),
                'Method': method.strip(),
                'Metric': metric.strip(),
                'Result': float(result)
            })
    
    return pd.DataFrame(rows)

def get_sorted_tasks(df, sorted_dic):
    """Sort tasks according to the categories in sorted_dic"""
    ordered_tasks = []
    task_category_map = {}
    
    # Create a lowercase mapping for case-insensitive matching
    df_tasks_lower = {task.lower(): task for task in df['Task'].unique()}
    
    for category, tasks in sorted_dic.items():
        for task in tasks:
            # Case insensitive matching
            if task.lower() in df_tasks_lower:
                actual_task = df_tasks_lower[task.lower()]
                ordered_tasks.append(actual_task)
                task_category_map[actual_task] = category
    
    # Add any remaining tasks
    for task in df['Task'].unique():
        if task not in ordered_tasks:
            ordered_tasks.append(task)
            task_category_map[task] = "Other"
    
    return ordered_tasks, task_category_map

def create_deepseek_style_chart(df, model_name):
    """Create a bar chart with category separations"""
    # Filter data for the specified model
    model_df = df[df['Model'] == model_name]
    
    # Get tasks sorted according to sorted_dic
    tasks, task_to_category = get_sorted_tasks(model_df, sorted_dic)
    
    # Set the style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.5
    })
    
    # Create figure with reduced height to minimize whitespace
    fig, ax = plt.subplots(figsize=(22, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 修改colors字典，使用显示名称作为key
    colors = {
    'AgentGroupChat-V2': '#B01E3A',     # Blue-purple（不变）
    'Multi-Agent Debate': '#505050', 
    'ReAct': '#747474',              # 浅灰
    'AutoGen': '#929292',            # 深灰
    'Naive': '#B0B0B0',              # 中亮灰
    'Naive-CoT': '#D0D0D0',          # 更浅灰
    }
    
    
    all_methods = model_df['Method'].unique()
    methods = []
    display_methods = []  # 用于显示的方法名称列表

    # 按照colors字典的顺序添加存在的方法
    for display_method in colors.keys():
        # 检查原始方法名或映射后的方法名是否存在
        original_method = None
        for method in all_methods:
            mapped_name = method_display_mapping.get(method, method)
            if mapped_name == display_method:
                original_method = method
                break
        
        if original_method:
            methods.append(original_method)
            display_methods.append(display_method)

    # 添加不在colors中的其他方法
    for method in all_methods:
        if method not in methods:
            methods.append(method)
            display_method = method_display_mapping.get(method, method)
            display_methods.append(display_method)

    additional_colors = ['#AED6F1', '#F5CBA7', '#D7BDE2', '#A9DFBF', '#F7DC6F']
    
    for i, display_method in enumerate(display_methods):
        if display_method not in colors:
            colors[display_method] = additional_colors[i % len(additional_colors)]
    
    # Organize data by task and metric
    grouped_data = OrderedDict()
    task_metrics = []
    task_metric_labels = {}
    
    for task in tasks:
        metrics = model_df[model_df['Task'] == task]['Metric'].unique()
        for metric in metrics:
            key = f"{task}"
            task_metrics.append(key)
            
            # Format labels
            if metric.lower() == 'accuracy' and task == 'FinQual':
                label_suffix = "<Finance> (EM)"
            elif metric.lower() == 'accuracy' and task == 'JEC-QA':
                label_suffix = "<Law> (EM)"
            elif metric.lower() == 'accuracy':
                label_suffix = "(EM)"
            elif metric.lower() == 'pass@1':
                label_suffix = "(Pass@1)"
            else:
                label_suffix = f"({metric})"
                
            task_metric_labels[key] = [task, label_suffix]
                
            grouped_data[key] = {}
            for method in methods:
                result = model_df[(model_df['Task'] == task) & 
                                 (model_df['Metric'] == metric) & 
                                 (model_df['Method'] == method)]
                
                if not result.empty:
                    grouped_data[key][method] = result['Result'].iloc[0]
                else:
                    grouped_data[key][method] = 0
    
    # Define bar width and positions - increased group_width for better spacing
    n_groups = len(task_metrics)
    n_methods = len(methods)
    group_width = 0.9  # Wider group width to allow for wider bars
    bar_width = group_width / n_methods
    group_positions = np.arange(n_groups)
    
    # Plot bars for each method
    for i, (method, display_method) in enumerate(zip(methods, display_methods)):
        values = [grouped_data[task][method] for task in task_metrics]
        x_pos = group_positions - (group_width/2) + (i+0.5)*bar_width
        
        # Special style for AgentGroupChat (使用原始方法名判断)
        if method == 'AgentGroupChat':
            bars = ax.bar(x_pos, values, bar_width*1, 
                        label=display_method, color=colors[display_method],  # 使用显示名称
                        edgecolor='white', linewidth=0.5,
                        hatch='///')
        else:
            bars = ax.bar(x_pos, values, bar_width*0.9, 
                        label=display_method, color=colors[display_method],  # 使用显示名称
                        edgecolor='white', linewidth=0.5)
        
        # Add value labels with less overlap
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height >= 0:
                # Adjust vertical positioning to prevent overlap
                vertical_offset = 1 if method == 'AgentGroupChat' else 0.3
                font_size = 11 if method == 'AgentGroupChat' else 7
                font_weight = 'bold' if method == 'AgentGroupChat' else 'normal'
                
                ax.text(bar.get_x() + bar.get_width()/2.5 if method == 'AgentGroupChat' else bar.get_x() + bar.get_width()/2. , height + vertical_offset,
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=font_size, fontweight=font_weight)
    
    # Set y-axis limits with less whitespace
    ax.set_ylim(0, 103)
    ax.set_ylabel('Accuracy / Percentile (%)', fontsize=11)
    
    # Set x-tick positions
    ax.set_xticks(group_positions)
    ax.set_xticklabels([])  # Clear existing labels
    
    # Add task names and metrics with better spacing
    for i, task in enumerate(task_metrics):
        task_name = task_metric_labels[task][0]
        metric_suffix = task_metric_labels[task][1]
        
        ax.text(i, -5, task_name, ha='center', fontsize=10)
        ax.text(i, -9, metric_suffix, ha='center', fontsize=9, color='gray')
    
    # Find category boundaries
    current_category = None
    category_spans = []  # (category, start_idx, end_idx)
    start_idx = 0
    
    for i, task_key in enumerate(task_metrics):
        task_name = task_metric_labels[task_key][0]
        category = task_to_category.get(task_name, "Other")
        
        if category != current_category:
            if current_category:
                category_spans.append((current_category, start_idx, i-1))
            current_category = category
            start_idx = i
    
    # Add the last category
    if current_category:
        category_spans.append((current_category, start_idx, len(task_metrics)-1))
    
    # Draw category dividers and labels
    top_y = 101  # Position for category labels
    
    for i, (category, start, end) in enumerate(category_spans):
        # Add category label at the top
        mid_point = (start + end) / 2
        ax.text(mid_point, top_y, category, ha='center', fontsize=12, fontweight='bold')
        
        # Add dashed line after each category (except the last)
        if i < len(category_spans) - 1:
            boundary = end + 0.5
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # Grid settings
    ax.yaxis.grid(True, linestyle='-', alpha=0.2)
    ax.set_axisbelow(True)
    
    # Legend with less whitespace at top
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                      ncol=len(methods), frameon=False, fontsize=10)
    
    # Title at the bottom
    # plt.figtext(0.5, 0.01, f"Benchmark performance of {model_name}.",
    #           ha='center', fontsize=12, fontweight='bold')
    
    # Adjust layout to reduce whitespace
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.15, top=0.90)
    
    return fig

def visualize_data(data_string):
    """Parse data and create visualizations"""
    df = parse_data(data_string)
    
    # Create plots for each model
    models = df['Model'].unique()
    figures = {}
    
    for model in models:
        fig = create_deepseek_style_chart(df, model)
        figures[model] = fig
        
    return figures


# Example usage
data = open('print_result.txt', encoding='utf-8').read()

# Execute the visualization
figures = visualize_data(data)

for model in figures:
    figures[model].savefig(f"{model}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figure for {model}")
