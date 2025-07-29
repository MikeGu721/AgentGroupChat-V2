import os
import json
import re
from typing import List, Dict, Any, Optional
from src.models.model_utils import call_llm
from logger import Logger

class CharacterGenerator:
    def __init__(self, engine: str, logger: Logger):
        self.engine = engine
        self.logger = logger
        self.prompt_file = "./prompts/managers/prompt_character_generation.txt"
    
    def load_prompt_template(self) -> str:
        """加载角色生成的prompt模板"""
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_characters(self, task_type: str, task_description: str, 
                          problem_domain: str, num_characters: int = 5,
                          save_dir: str = None) -> List[Dict[str, Any]]:
        """
        生成指定数量的角色
        
        Args:
            task_type: 任务类型
            task_description: 任务描述
            problem_domain: 问题领域
            num_characters: 要生成的角色数量
            save_dir: 保存目录，如果提供则保存到文件
        
        Returns:
            生成的角色列表
        """
        # 加载并填充prompt模板
        prompt_template = self.load_prompt_template()
        formatted_prompt = prompt_template.format(
            task_type=task_type,
            task_description=task_description,
            problem_domain=problem_domain,
            num_characters=num_characters
        )
        
        # 调用模型生成角色
        try:
            # 使用统一的LLM调用接口
            response = call_llm(formatted_prompt, self.engine, self.logger)
            
            # 解析生成的角色
            characters = self._parse_generated_characters(response, num_characters)
            
            # 如果指定了保存目录，则保存角色文件
            if save_dir:
                self._save_characters(characters, save_dir)
            
            return characters
            
        except Exception as e:
            self.logger.gprint("Character Generation Error", error=str(e))
            # 如果生成失败，返回默认角色
            return self._get_default_characters(task_type, num_characters, save_dir)
    
    def _parse_generated_characters(self, response: str, expected_num: int) -> List[Dict[str, Any]]:
        """解析模型生成的角色JSON"""
        characters = []
        
        # 提取JSON对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        for i, json_str in enumerate(json_matches[:expected_num]):
            try:
                # 清理JSON字符串
                cleaned_json = self._clean_json_string(json_str)
                character = json.loads(cleaned_json)
                
                # 验证必要字段
                if self._validate_character(character):
                    # 确保ID格式正确
                    character['id'] = f"C{i+1:04d}"
                    characters.append(character)
                else:
                    self.logger.gprint("Invalid character format", character=character)
                    
            except json.JSONDecodeError as e:
                self.logger.gprint("JSON parse error", json_str=json_str, error=str(e))
                continue
        
        # 如果解析的角色数量不够，生成默认角色补充
        if len(characters) < expected_num:
            default_chars = self._get_default_characters("generic", expected_num - len(characters))
            for i, char in enumerate(default_chars):
                char['id'] = f"C{len(characters) + i + 1:04d}"
                characters.append(char)
        
        return characters[:expected_num]
    
    def _clean_json_string(self, json_str: str) -> str:
        """清理JSON字符串，移除多余的字符"""
        # 移除代码块标记
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        
        # 移除多余的空白字符
        json_str = json_str.strip()
        
        return json_str
    
    def _validate_character(self, character: Dict[str, Any]) -> bool:
        """验证角色数据格式是否正确"""
        required_fields = ['id', 'scratch', 'objective', 'message_format_desc', 'message_format_field']
        return all(field in character and character[field] for field in required_fields)
    
    def _get_default_characters(self, task_type: str, num_characters: int, 
                               save_dir: str = None) -> List[Dict[str, Any]]:
        """获取默认角色配置"""
        default_templates = {
            "code": [
                {
                    "id": "C0001",
                    "scratch": "C0001 is a problem analysis specialist who breaks down programming requirements and designs solution architectures.",
                    "objective": "Analyze programming problems, understand requirements, and design appropriate solution architectures and algorithms.",
                    "message_format_desc": "You should analyze the problem requirements, design the solution architecture, and outline the algorithmic approach.",
                    "message_format_field": "Requirements Analysis,Solution Design,Algorithm Outline"
                },
                {
                    "id": "C0002",
                    "scratch": "C0002 is a code implementation expert who writes clean, efficient, and well-structured code solutions.",
                    "objective": "Implement clean, efficient, and well-documented code solutions based on the designed architecture.",
                    "message_format_desc": "You should implement the solution with clean code, proper structure, and clear documentation.",
                    "message_format_field": "Code Implementation,Code Structure,Documentation"
                },
                {
                    "id": "C0003",
                    "scratch": "C0003 is a testing and debugging specialist who ensures code correctness through comprehensive testing.",
                    "objective": "Design test cases, debug code issues, and ensure the solution works correctly for all edge cases.",
                    "message_format_desc": "You should design comprehensive test cases, identify potential bugs, and ensure code reliability.",
                    "message_format_field": "Test Case Design,Bug Analysis,Code Debugging"
                },
                {
                    "id": "C0004",
                    "scratch": "C0004 is an optimization specialist who improves code performance and efficiency.",
                    "objective": "Analyze code performance, identify bottlenecks, and suggest optimizations for better efficiency.",
                    "message_format_desc": "You should analyze code performance, identify optimization opportunities, and suggest improvements.",
                    "message_format_field": "Performance Analysis,Optimization Suggestions,Efficiency Improvements"
                },
                {
                    "id": "C0005",
                    "scratch": "C0005 is a solution validator who verifies the correctness and completeness of the implemented solution.",
                    "objective": "Validate the final solution against requirements and ensure it meets all specified criteria.",
                    "message_format_desc": "You should validate the solution comprehensively and provide final verification results.",
                    "message_format_field": "Solution Validation,Requirements Check,Final Verification"
                }
            ]
        }
        
        templates = default_templates.get("code", default_templates["code"])
        
        # 生成指定数量的角色
        characters = []
        for i in range(num_characters):
            template_idx = i % len(templates)
            char = templates[template_idx].copy()
            char['id'] = f"C{i+1:04d}"
            characters.append(char)
        
        # 如果指定了保存目录，保存默认角色
        if save_dir:
            self._save_characters(characters, save_dir)
        
        return characters
    
    def _save_characters(self, characters: List[Dict[str, Any]], save_dir: str):
        """将角色保存到指定目录"""
        os.makedirs(save_dir, exist_ok=True)
        
        for character in characters:
            filename = f"{character['id']}.json"
            filepath = os.path.join(save_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(character, f, ensure_ascii=False, indent=2)
        
        self.logger.gprint("Characters saved", 
                          save_dir=save_dir, 
                          num_characters=len(characters))
