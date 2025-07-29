import os
import json
from typing import List, Dict, Any, Optional
from logger import Logger
from .character_generator import CharacterGenerator

class CharacterManager:
    def __init__(self, logger: Logger):
        self.logger = logger
    
    def load_characters(self, character_path: str, engine: str = None, 
                       task_type: str = None, task_description: str = None,
                       problem_domain: str = None, num_characters: int = 5) -> Dict[str, Any]:
        """
        加载或生成角色
        
        Args:
            character_path: 角色路径，可以是已存在的目录或要生成的目录
            engine: 模型引擎（生成角色时需要）
            task_type: 任务类型（生成角色时需要）
            task_description: 任务描述（生成角色时需要）
            problem_domain: 问题领域（生成角色时需要）
            num_characters: 要生成的角色数量
        
        Returns:
            角色字典 {character_id: character_object}
        """
        
        if os.path.exists(character_path) and os.listdir(character_path):
            # 目录存在且不为空，加载已有角色
            self.logger.gprint("Loading existing characters", path=character_path)
            return self._load_existing_characters(character_path, engine)
        else:
            # 目录不存在或为空，生成新角色
            if not all([engine, task_type, task_description, problem_domain]):
                raise ValueError("To generate characters, engine, task_type, task_description, and problem_domain are required")
            
            self.logger.gprint("Generating new characters", 
                             path=character_path,
                             task_type=task_type,
                             num_characters=num_characters)
            return self._generate_and_load_characters(
                character_path, engine, task_type, task_description, 
                problem_domain, num_characters
            )
    
    def _load_existing_characters(self, character_dir: str, engine: str) -> Dict[str, Any]:
        """从指定目录加载已有的角色文件"""
        from src.environment.characters.character import TaskCharacter
        
        all_character_dict = {}
        
        # 获取所有JSON文件并排序
        json_files = [f for f in os.listdir(character_dir) if f.endswith('.json')]
        json_files.sort()  # 确保按照文件名顺序加载
        
        if not json_files:
            raise ValueError(f"No character files found in {character_dir}")
        
        for file in json_files:
            try:
                character_file = os.path.join(character_dir, file)
                # 验证文件格式
                with open(character_file, 'r', encoding='utf-8') as f:
                    char_data = json.load(f)
                
                # 验证必要字段
                if not self._validate_character_file(char_data):
                    self.logger.gprint("Invalid character file", file=character_file)
                    continue
                
                # 创建角色对象
                character = TaskCharacter(character_file, engine, self.logger)
                all_character_dict[character.id] = character
                
            except Exception as e:
                self.logger.gprint("Failed to load character", 
                                 file=file, 
                                 error=str(e))
                continue
        
        if not all_character_dict:
            raise ValueError(f"No valid character files loaded from {character_dir}")
        
        self.logger.gprint("Characters loaded successfully", 
                          count=len(all_character_dict),
                          characters=list(all_character_dict.keys()))
        
        return all_character_dict
    
    def _generate_and_load_characters(self, character_dir: str, engine: str,
                                    task_type: str, task_description: str,
                                    problem_domain: str, num_characters: int) -> Dict[str, Any]:
        """生成新角色并加载"""
        
        # 创建角色生成器
        generator = CharacterGenerator(engine, self.logger)
        
        # 生成角色并保存到指定目录
        characters = generator.generate_characters(
            task_type=task_type,
            task_description=task_description,
            problem_domain=problem_domain,
            num_characters=num_characters,
            save_dir=character_dir
        )
        
        # 加载生成的角色
        return self._load_existing_characters(character_dir, engine)
    
    def _validate_character_file(self, char_data: Dict[str, Any]) -> bool:
        """验证角色文件格式"""
        required_fields = ['id', 'scratch', 'objective', 'message_format_desc', 'message_format_field']
        return all(field in char_data and char_data[field] for field in required_fields)
    
    def get_character_info(self, character_dict: Dict[str, Any]) -> Dict[str, str]:
        """获取角色信息摘要"""
        info = {}
        for char_id, character in character_dict.items():
            info[char_id] = {
                "id": getattr(character, 'id', char_id),
                "scratch": getattr(character, 'scratch', 'No description available'),
                "objective": getattr(character, 'objective', 'No objective specified')
            }
        return info
