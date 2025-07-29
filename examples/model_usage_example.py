# ./examples/model_usage_example.py
from models.unified_model_client import model_client, generate_prompt
from config import MODEL_CONFIGS

def example_usage():
    # 示例1：使用ollama
    ollama_config = MODEL_CONFIGS["ollama_llama"]
    response1 = model_client.generate("Hello, how are you?", ollama_config)
    print("Ollama response:", response1["message"])
    
    # 示例2：使用外部API
    external_config = MODEL_CONFIGS["external_qwen"] 
    response2 = model_client.generate("Explain quantum computing", external_config)
    print("External API response:", response2["message"])
    
    # 示例3：使用本地generator API
    local_config = MODEL_CONFIGS["local_generator"]
    response3 = model_client.generate("Write a poem", local_config)
    print("Local generator response:", response3["message"])
    
    # 示例4：带重试的生成
    def simple_parser(text, requirements=None):
        return text.strip()
    
    result = model_client.generate_with_retry(
        prompt="Generate a short story",
        model_config=ollama_config,
        parser_fn=simple_parser,
        max_retry=3,
        func_name="story_generation"
    )
    print("Generated story:", result)

if __name__ == "__main__":
    example_usage()
