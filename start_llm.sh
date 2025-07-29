#!/bin/bash

echo "Starting LLM API Services..."

# 检查ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo "Ollama is not installed. Please install ollama first."
    echo "Visit: https://ollama.ai"
    exit 1
fi

# 启动ollama服务
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# 等待ollama服务启动
echo "Waiting for Ollama to start..."
sleep 5

# 检查ollama是否正在运行
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Failed to start Ollama service"
    exit 1
fi

# 拉取需要的模型（如果还没有的话）
echo "Checking and pulling required models..."

# 检查llama3.1:70b模型
if ! ollama list | grep -q "llama3.1:70b"; then
    echo "Pulling llama3.1:70b model..."
    ollama pull llama3.1:70b
else
    echo "llama3.1:70b model already exists"
fi

# 可以添加其他需要的模型
# if ! ollama list | grep -q "qwen2.5:72b"; then
#     echo "Pulling qwen2.5:72b model..."
#     ollama pull qwen2.5:72b
# else
#     echo "qwen2.5:72b model already exists"
# fi

echo "LLM API Services started successfully!"
echo "Ollama is running on http://localhost:11434"
echo "Available models:"
ollama list

# 保存PID以便后续关闭
echo $OLLAMA_PID > .ollama.pid
echo "Ollama PID: $OLLAMA_PID saved to .ollama.pid"

echo "To stop the services, run: kill -9 \$(cat .ollama.pid)"
echo "Services are ready for AgentGroupChat!"
