from datasets import load_dataset

# Load only the train split of the dataset
dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train", cache_dir='.')

# To see the first few examples
import json

fw = open('aime_2024.jsonl', 'w', encoding='utf-8')
for dataline in dataset:
    
    fw.write(json.dumps(dataline, ensure_ascii=False)+'\n')
