import time
from typing import Any, Dict, Optional
from .unified_model_client import UnifiedModelClient


def read_prompt_file(file_path: str) -> str:
    """读取prompt模板文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_prompt(prompt_inputs: list, prompt_file: str) -> str:
    """
    根据prompt模板和输入变量生成prompt

    Args:
        prompt_inputs: 要填入模板的变量
        prompt_file: prompt模板文件路径

    Returns:
        str: 生成的prompt
    """
    prompt_inputs = [str(i) for i in prompt_inputs]
    prompt = read_prompt_file(prompt_file)

    for idx, value in enumerate(prompt_inputs):
        prompt = prompt.replace(f"!<INPUT {idx}>!", value)
    
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

    return prompt.strip()


class ModelGenerator:
    """模型生成器API类"""
    
    def __init__(self, engine: str, max_retry: int = 3):
        self.engine = engine
        self.max_retry = max_retry
        self.client = UnifiedModelClient()
    
    def generate(self, prompt: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        生成响应的API接口
        
        Args:
            prompt: 输入prompt
            temperature: 温度参数（可选）
            
        Returns:
            Dict: 包含message, usage等信息的响应
        """
        try:
            return self.client.generate(
                prompt=prompt, 
                engine=self.engine, 
                temperature=temperature
            )
        except Exception as e:
            print(f"==================== MODEL RESPONSE ERROR ====================")
            print(f"Error: {str(e)}")
            raise Exception(f"[Error]: Engine {self.engine} Request Error: {str(e)}")
    
    def generate_with_retry(self, prompt: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        带重试机制的生成方法
        
        Args:
            prompt: 输入prompt
            temperature: 温度参数（可选）
            
        Returns:
            Dict: 响应结果
        """
        curr_retry = 0
        last_error = None
        
        while curr_retry < self.max_retry:
            try:
                return self.generate(prompt, temperature)
            except Exception as e:
                last_error = e
                print(f"Retrying ({curr_retry + 1}/{self.max_retry})... Error: {str(e)}")
                time.sleep(5)
                curr_retry += 1
        
        raise Exception(f"[Error]: Exceeded Max Retry Times ({self.max_retry}). Last error: {str(last_error)}")


def non_parse_fn(response_message: str, requirements: Optional[Dict[str, Any]] = None) -> str:
    """默认的响应解析函数"""
    if requirements is not None:
        print("Requirements are provided but ignored in non_parse_fn.")
    return response_message


def generate_with_response_parser(
    prompt: str,
    engine: str,
    parser_fn=non_parse_fn,
    requirements: Optional[Dict[str, Any]] = None,
    max_retry: int = 3,
    logger=None,
    func_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    带响应解析和重试逻辑的生成函数

    Args:
        prompt: 输入prompt
        engine: 模型引擎
        parser_fn: 响应解析函数
        requirements: 生成限制条件
        max_retry: 最大重试次数
        logger: 日志记录器
        func_name: 函数名（用于调试）
        temperature: 温度参数

    Returns:
        str: 解析后的输出
    """
    generator = ModelGenerator(engine, max_retry)
    curr_retry = 0

    while curr_retry < max_retry:
        output, response_json = "", None
        try:
            response_json = generator.generate(prompt, temperature)
            output = parser_fn(response_json["message"], requirements)

            if logger:
                logger.gprint(
                    "Prompt INFO",
                    prompt=prompt,
                    model_response=response_json,
                    output=output,
                    func_name=func_name,
                )

            # Debug输出（如果需要的话）
            from config import DEBUG
            if DEBUG and func_name:
                print("==================== PROMPT DEBUG START ====================")
                print("Function Name: ", func_name)
                print("Prompt:\n", prompt)
                print("Model Response:\n", response_json)
                print("Output:\n", output)
                print("==================== PROMPT DEBUG END ======================\n")

            return output
            
        except Exception as e:
            if logger:
                logger.gprint(
                    "### ERROR: Failed in generate_with_response_parser!",
                    prompt=prompt,
                    model_response=response_json,
                    output=output,
                    error=str(e),
                )
            print(f"Error in attempt {curr_retry + 1}: {e}")

        # 重试
        print(f"Retrying ({curr_retry + 1}/{max_retry})...")
        time.sleep(5)
        curr_retry += 1

    # 超过最大重试次数后抛出异常
    raise Exception(f"[Error]: Exceeded Max Retry Times ({max_retry}).")
