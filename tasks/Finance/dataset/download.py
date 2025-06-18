from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('TheFinAI/flare-cfa', subset_name='default', split='test', cache_dir='/data/shenhao/AgentGroupChat_v2/tasks/Finance/dataset/')

import json
import random
import os
from collections import defaultdict
from modelscope.msdatasets import MsDataset
from datasets import load_dataset

def convert_dataset_to_jsonl(src_file, tgt_file, dataset_type='modelscope', subset_name='default', split='test'):
    """Convert dataset to JSONL format
    
    Args:
        src_file: Source dataset path or name
        tgt_file: Target JSONL file path
        dataset_type: Dataset type, 'modelscope' or 'huggingface'
        subset_name: Subset name for ModelScope dataset
        split: Split name for HuggingFace dataset
    """
    # Load dataset
    ds = (MsDataset.load(src_file, subset_name=subset_name, split=split) if dataset_type == 'modelscope'
          else load_dataset(src_file)[split] if isinstance(load_dataset(src_file), dict)
          else load_dataset(src_file))

    # Convert and save to JSONL
    with open(tgt_file, 'w', encoding='utf-8') as f:
        for item in ds:
            item = dict(item) if not isinstance(item, dict) else item
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Dataset saved to: {tgt_file}")

if __name__ == '__main__':
    convert_dataset_to_jsonl('TheFinAI/flare-cfa', 'tasks/Finance/dataset/flare-cfa.jsonl')