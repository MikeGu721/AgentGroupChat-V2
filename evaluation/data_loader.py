import json
from typing import List, Dict, Any
from datasets import load_dataset

class DataLoader:
    @staticmethod
    def load_data(test_file: str) -> List[Dict[str, Any]]:
        """根据文件路径或特殊标记加载数据"""
        if test_file.startswith("huggingface:"):
            return DataLoader._load_from_huggingface(test_file)
        elif test_file.endswith(".jsonl"):
            return DataLoader._load_jsonl(test_file)
        elif test_file.endswith(".json"):
            return DataLoader._load_json(test_file)
        else:
            raise ValueError(f"Unsupported file format: {test_file}")
    
    @staticmethod
    def _load_from_huggingface(test_file: str) -> List[Dict[str, Any]]:
        """从HuggingFace加载数据集"""
        parts = test_file.replace("huggingface:", "").split(":")
        if len(parts) == 2:
            dataset_name, split = parts
            subset = None
        elif len(parts) == 3:
            dataset_name, subset, split = parts
        else:
            raise ValueError(f"Invalid huggingface format: {test_file}")
        
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        return list(dataset[split])
    
    @staticmethod
    def _load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """加载JSONL文件"""
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def _load_json(file_path: str) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
