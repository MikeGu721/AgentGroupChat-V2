# 添加模型相关配置
import os

# API配置
API_KEY = os.getenv("API_KEY", "your-api-key-here")

# 调试配置
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# 重试配置
MAX_RETRY = int(os.getenv("MAX_RETRY", "3"))

# 模型超时配置
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "300"))

# 支持的引擎格式说明
SUPPORTED_ENGINES = {
    "ollama": "Use format: model_name:version (e.g., llama3.1:70b)",
    "external_api": "Use format: TA/provider/model_name (e.g., TA/Qwen/Qwen2.5-72B-Instruct-Turbo)",
    "generator_api": "Use format: generator://model_name (e.g., generator://local_model)"
}

# ... 其他原有配置保持不变 ...


# 在现有配置基础上添加模型配置
MAX_RETRY = 3
DEBUG = True
API_KEY = "your_api_key_here"

# 模型配置示例
MODEL_CONFIGS = {
    "ollama_llama": {
        "backend": "ollama",
        "model": "llama2:7b",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "external_qwen": {
        "backend": "external_api", 
        "model": "TA-qwen-72b",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "local_generator": {
        "backend": "generator_api",
        "model": "local_model",
        "temperature": 0.7,
        "max_tokens": 1000
    }
}

# 默认模型配置
DEFAULT_MODEL_CONFIG = MODEL_CONFIGS["ollama_llama"]
