import json
import os
import random

def convert_finqa_to_json():
    """
    Convert flare-finqa.jsonl format to json format.
    The output will be saved as AgentGroupChat_Finance.json with the following structure:
    [
        {
            "problem": "...",
            "solution": "",
            "ground_truth": "..."
        },
        ...
    ]
    Maximum 1000 examples will be included. If the input file has more examples,
    random sampling will be performed.
    """
    input_file = "tasks/Finance/dataset/flare-finqa.jsonl"
    output_file = "tasks/Finance/dataset/AgentGroupChat_Finance.json"
    
    # First, count total examples
    total_examples = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for _ in f:
            total_examples += 1
    
    # Determine if sampling is needed
    max_examples = 1000
    sampling_needed = total_examples > max_examples
    
    if sampling_needed:
        # Calculate sampling probability to get approximately max_examples
        sample_prob = max_examples / total_examples
        print(f"Dataset has {total_examples} examples, sampling to {max_examples}")
    
    data = []
    
    # Read the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip this example with probability (1-sample_prob) if sampling is needed
            if sampling_needed and random.random() > sample_prob:
                continue
                
            item = json.loads(line.strip())
            
            # Extract required fields
            new_item = {
                "problem": item.get("query", ""),
                "solution": "",  # Empty solution as required
                "ground_truth": item.get("answer", "")
            }
            
            data.append(new_item)
            
            # Stop if we reached the maximum number of examples
            if len(data) >= max_examples:
                break
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion complete. {len(data)} examples saved to {output_file}")
    return output_file

def combine_finqa_and_cfa(finqa_samples=500, cfa_samples=500):
    """
    Combine both flare-finqa.jsonl and flare-cfa.jsonl datasets, sampling from each
    to create a combined dataset with 1000 examples total.
    
    Args:
        finqa_samples: Number of samples to take from finqa dataset
        cfa_samples: Number of samples to take from cfa dataset
        
    Returns:
        Path to the output file
    """
    finqa_file = "tasks/Finance/dataset/flare-finqa.jsonl"
    cfa_file = "tasks/Finance/dataset/flare-cfa.jsonl"
    output_file = "tasks/Finance/dataset/AgentGroupChat_Finance.json"
    
    combined_data = []
    
    # Process FinQA dataset
    finqa_data = []
    with open(finqa_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            finqa_data.append({
                "problem": item.get("query", ""),
                "solution": "",
                "ground_truth": item.get("answer", ""),
                "source": "finqa"
            })
    
    print(f"Loaded {len(finqa_data)} examples from FinQA dataset")
    
    # Process CFA dataset
    cfa_data = []
    with open(cfa_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            cfa_data.append({
                "problem": item.get("query", ""),
                "solution": "",
                "ground_truth": item.get("answer", ""),
                "source": "cfa"
            })
    
    print(f"Loaded {len(cfa_data)} examples from CFA dataset")
    
    # Sample from each dataset
    sampled_finqa = random.sample(finqa_data, min(finqa_samples, len(finqa_data)))
    sampled_cfa = random.sample(cfa_data, min(cfa_samples, len(cfa_data)))
    
    # Combine datasets
    combined_data = sampled_finqa + sampled_cfa
    
    # Shuffle to mix the datasets
    random.shuffle(combined_data)
    
    # Remove source field from final output
    for item in combined_data:
        item.pop("source", None)
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    
    print(f"Conversion complete. Combined dataset with {len(combined_data)} examples saved to {output_file}")
    print(f"- FinQA: {len(sampled_finqa)} examples")
    print(f"- CFA: {len(sampled_cfa)} examples")
    
    return output_file

if __name__ == "__main__":
    combine_finqa_and_cfa()
    
    