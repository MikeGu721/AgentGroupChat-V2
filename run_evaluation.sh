#!/bin/bash

echo "Starting AgentGroupChat Evaluation for MBPP..."

# 检查必要的目录和文件
if [ ! -f "src/main.py" ]; then
    echo "Error: src/main.py not found!"
    exit 1
fi

# 创建必要的目录
mkdir -p mbpp_log
mkdir -p mbpp_result
mkdir -p characters/gen_mbpp
mkdir -p prompts/managers

# 检查prompt文件是否存在，如果不存在则创建一个基础版本
if [ ! -f "prompts/managers/prompt_character_generation.txt" ]; then
    echo "Creating basic character generation prompt..."
    cat > prompts/managers/prompt_character_generation.txt << 'EOF'
You are an expert system designer tasked with creating specialized agent characters for collaborative problem-solving in multi-agent systems. Your goal is to generate diverse, complementary agent roles that can work together effectively to solve complex problems.

Given the following task information:
- Task Type: {task_type}
- Task Description: {task_description}
- Problem Domain: {problem_domain}

Please generate {num_characters} distinct agent characters that would be most effective for solving problems in this domain.

For each character, provide:

```json
{
  "id": "C0001",
  "scratch": "[Brief description of character's expertise and role]",
  "objective": "[Clear statement of character's goal and purpose]",
  "message_format_desc": "[Instructions for how this character should structure responses]",
  "message_format_field": "[Comma-separated list of response fields]"
}
