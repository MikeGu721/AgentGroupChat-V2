import json
import requests
from typing import Dict, Any, Tuple, Optional
from config import API_KEY


def ollama_request_by_url(prompt: str, engine: str, temperature: Optional[float] = None) -> Tuple[str, int, float]:
    """
    通过ollama服务器调用开源模型
    
    Args:
        prompt: 输入prompt
        engine: 模型名称
        temperature: 温度参数
        
    Returns:
        Tuple[str, int, float]: (响应消息, token使用量, 耗时)
    """
    messages = [{"role": "user", "content": prompt}]
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": engine,
        "messages": messages,
        "stream": False
    }
    
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        response_json = response.json()

        message = response_json["message"]["content"]
        token_usage = response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0)
        time_costed = response_json.get("prompt_eval_duration", 0) + response_json.get("eval_duration", 0)

        return message, token_usage, time_costed
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ollama API request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Unexpected ollama response format: {str(e)}")


def external_api_request(prompt: str, engine_name: str, temperature: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """
    调用外部API（如OpenAI兼容接口）
    
    Args:
        prompt: 输入prompt
        engine_name: 模型名称，可以包含温度参数（用#分隔）
        temperature: 温度参数（优先级高于engine_name中的参数）
        
    Returns:
        Tuple[str, Dict]: (响应消息, 使用情况)
    """
    api_key = API_KEY
    if not api_key:
        raise Exception("API_KEY not configured")
    
    # 解析engine_name中的温度参数
    if "#" in engine_name and temperature is None:
        temp_parts = engine_name.split("#")
        engine_name = temp_parts[0]
        temperature = float(temp_parts[1])
    
    # 构建请求参数
    params = {
        "messages": [{"role": "user", "content": prompt}],
        "model": engine_name,
    }
    
    if temperature is not None:
        params["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            "https://aigptx.top/v1/chat/completions",
            headers=headers,
            json=params,
            timeout=300
        )
        response.raise_for_status()
        res = response.json()
        
        message = res["choices"][0]["message"]["content"]
        usage = res.get("usage", {})

        return message, usage
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"External API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"Unexpected API response format: {str(e)}")


def generator_api_request(prompt: str, engine: str, temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    调用本地generator API
    
    Args:
        prompt: 输入prompt
        engine: 模型引擎
        temperature: 温度参数
        
    Returns:
        Dict: API响应结果
    """
    url = "http://localhost:8000/v1/generate"  # 假设generator API运行在8000端口
    
    payload = {
        "prompt": prompt,
        "engine": engine
    }
    
    if temperature is not None:
        payload["temperature"] = temperature
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Generator API request failed: {str(e)}")
