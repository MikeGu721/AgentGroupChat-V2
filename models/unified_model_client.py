from typing import Dict, Any, Optional
from .api_models import ollama_request_by_url, external_api_request, generator_api_request


class UnifiedModelClient:
    """统一的模型调用客户端"""
    
    def __init__(self):
        pass
    
    def generate(self, prompt: str, engine: str, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        统一的生成接口
        
        Args:
            prompt: 输入prompt
            engine: 模型引擎标识
            temperature: 温度参数
            
        Returns:
            Dict: 标准化的响应格式 {"message": str, "usage": dict, "metadata": dict}
        """
        try:
            if self._is_ollama_engine(engine):
                return self._handle_ollama(prompt, engine, temperature)
            elif self._is_external_api_engine(engine):
                return self._handle_external_api(prompt, engine, temperature)
            elif self._is_generator_api_engine(engine):
                return self._handle_generator_api(prompt, engine, temperature)
            else:
                raise Exception(f"Unsupported engine format: {engine}")
                
        except Exception as e:
            raise Exception(f"Model generation failed for engine {engine}: {str(e)}")
    
    def _is_ollama_engine(self, engine: str) -> bool:
        """判断是否为ollama格式的引擎"""
        return ":" in engine and not engine.startswith(("TA/", "http://", "https://", "generator://"))
    
    def _is_external_api_engine(self, engine: str) -> bool:
        """判断是否为外部API引擎"""
        return engine.startswith("TA/")
    
    def _is_generator_api_engine(self, engine: str) -> bool:
        """判断是否为generator API引擎"""
        return engine.startswith("generator://")
    
    def _handle_ollama(self, prompt: str, engine: str, temperature: Optional[float]) -> Dict[str, Any]:
        """处理ollama请求"""
        try:
            message, token_usage, time_costed = ollama_request_by_url(prompt, engine, temperature)
            
            return {
                "message": message,
                "usage": {
                    "total_tokens": token_usage,
                    "prompt_tokens": 0,  # ollama不单独返回prompt tokens
                    "completion_tokens": token_usage
                },
                "metadata": {
                    "time_costed": time_costed,
                    "engine": engine,
                    "provider": "ollama"
                }
            }
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def _handle_external_api(self, prompt: str, engine: str, temperature: Optional[float]) -> Dict[str, Any]:
        """处理外部API请求"""
        try:
            message, usage = external_api_request(prompt, engine, temperature)
            
            return {
                "message": message,
                "usage": usage,
                "metadata": {
                    "engine": engine,
                    "provider": "external_api"
                }
            }
        except Exception as e:
            raise Exception(f"External API request failed: {str(e)}")
    
    def _handle_generator_api(self, prompt: str, engine: str, temperature: Optional[float]) -> Dict[str, Any]:
        """处理generator API请求"""
        try:
            # 移除 "generator://" 前缀
            clean_engine = engine.replace("generator://", "")
            response = generator_api_request(prompt, clean_engine, temperature)
            
            # 假设generator API返回标准格式
            return {
                "message": response.get("message", ""),
                "usage": response.get("usage", {}),
                "metadata": {
                    "engine": clean_engine,
                    "provider": "generator_api",
                    **response.get("metadata", {})
                }
            }
        except Exception as e:
            raise Exception(f"Generator API request failed: {str(e)}")
    
    def get_supported_engines(self) -> Dict[str, str]:
        """获取支持的引擎格式说明"""
        return {
            "ollama": "model_name:version (e.g., llama3.1:70b, qwen2.5:72b)",
            "external_api": "TA/provider/model_name (e.g., TA/Qwen/Qwen2.5-72B-Instruct-Turbo)",
            "generator_api": "generator://model_name (e.g., generator://local_model)"
        }
