import json
import requests
from config import *


def ollama_request_by_url(prompt, engine):
    """
    Utilize open source models with ollama server.
    """
    messages = [{"role": "user", "content": prompt}]
    url = "http://localhost:11434/api/chat"
    payload = {"model": engine, "messages": messages, "stream": False}
    payload_json = json.dumps(payload)
    response_json = requests.request("POST", url, data=payload_json).json()

    message = response_json["message"]["content"]
    token_usage = response_json["prompt_eval_count"] + response_json["eval_count"]
    time_costed = response_json["prompt_eval_duration"] + response_json["eval_duration"]

    return message, token_usage, time_costed


def qwen2_by_api(prompt, engine_name):
    api_key = API_KEY
    if "#" in engine_name:
        temp = engine_name.split("#")
        engine_name = temp[0]
        temperature = float(temp[1])
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
            "temperature": temperature,
        }
    else:
        params = {
            "messages": [{"role": "user", "content": prompt}],
            "model": engine_name,
        }

    headers = {
        "Authorization": "Bearer " + api_key,
    }

    response = requests.post(
        "https://aigptx.top/v1/chat/completions",
        headers=headers,
        json=params,
        stream=False,
    )
    res = response.json()
    message = res["choices"][0]["message"]["content"]
    usage = res["usage"]

    return message, usage
