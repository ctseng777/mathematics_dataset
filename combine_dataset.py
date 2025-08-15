#!/usr/bin/env python3
"""Convert street math dataset to Alpaca format for approximation/estimation training."""

import json
import os
import random
from pathlib import Path

def get_approximation_instruction(difficulty):
    """Generate instruction text based on difficulty level."""
    instructions = {
        'train-easy': "Estimate the following calculation using basic mental math and rounding techniques. Provide your approximation and briefly explain your rounding strategy.",
        'train-medium': "Estimate the following calculation using mental math approximation techniques. Show your rounding approach and provide a reasonable approximation.",
        'train-hard': "Use advanced estimation strategies to approximate the following calculation. Explain your method and provide a well-reasoned estimate.",
        'interpolate': "Estimate the following calculation using appropriate mental math techniques for the given complexity level.",
        'extrapolate': "Apply sophisticated estimation methods to approximate this challenging calculation. Demonstrate your reasoning process."
    }
    return instructions.get(difficulty, instructions['train-medium'])

def convert_to_alpaca_format(item, source_difficulty='train-medium'):
    """Convert original format to Alpaca format with approximation focus."""
    
    # Extract the solution reasoning and combine with approximation
    solution_text = item.get('solution', '')
    approximate_answer = item.get('approximate_answer', '')
    
    # Create a focused output that emphasizes the approximation process
    output_parts = []
    
    # Start with the approximation
    if approximate_answer:
        output_parts.append(f"**Approximation: {approximate_answer}**")
    
    # Extract key reasoning from the solution if available
    if solution_text:
        # Try to extract the key steps and reasoning
        if "Step 1:" in solution_text and "round" in solution_text.lower():
            # Extract rounding strategy
            lines = solution_text.split('\n')
            rounding_section = []
            reasoning_section = []
            
            in_step1 = False
            in_reasoning = False
            
            for line in lines:
                if "Step 1:" in line or "Simplify" in line:
                    in_step1 = True
                elif "Step 2:" in line or "Mental Math" in line:
                    in_step1 = False
                elif "Reasoning" in line or "method" in line.lower():
                    in_reasoning = True
                
                if in_step1 and ("round" in line.lower() or "**" in line):
                    rounding_section.append(line.strip())
                elif in_reasoning and line.strip():
                    reasoning_section.append(line.strip())
            
            if rounding_section:
                output_parts.append("\n**Method:**")
                output_parts.extend(rounding_section[:3])  # Limit to key points
            
            if reasoning_section:
                output_parts.append("\n**Reasoning:**")
                output_parts.append(reasoning_section[0] if reasoning_section else "This estimation method provides a quick, practical approximation.")
    
    # Fallback if no detailed solution available
    if len(output_parts) == 1:  # Only have the approximation
        output_parts.append("\n**Method:** Round the numbers to make mental calculation easier, then adjust the result based on the direction of rounding.")
    
    alpaca_item = {
        "instruction": get_approximation_instruction(source_difficulty),
        "input": item['input'],
        "output": '\n'.join(output_parts)
    }
    
    # Preserve important metadata
    if 'lower_bound' in item and 'upper_bound' in item:
        alpaca_item['metadata'] = {
            'exact_answer': item.get('exact_answer'),
            'lower_bound': item['lower_bound'],
            'upper_bound': item['upper_bound'],
            'difficulty': item.get('difficulty', source_difficulty),
            'module': item.get('module', 'street_math')
        }
    
    return alpaca_item

def combine_and_split_dataset(input_dir, output_dir):
    """Combine files and convert to Alpaca format with train/validation/test splits."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read all data from all files and convert to Alpaca format
    all_data = []
    
    # Read train-* files
    for difficulty in ['train-easy', 'train-medium', 'train-hard']:
        file_path = input_path / f"{difficulty}.jsonl"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                converted_data = [convert_to_alpaca_format(item, difficulty) for item in data]
                print(f"Loaded and converted {len(converted_data)} samples from {difficulty}.jsonl")
                all_data.extend(converted_data)
    
    # Read interpolate and extrapolate files
    for test_type in ['interpolate', 'extrapolate']:
        file_path = input_path / f"{test_type}.jsonl"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                converted_data = [convert_to_alpaca_format(item, test_type) for item in data]
                print(f"Loaded and converted {len(converted_data)} samples from {test_type}.jsonl")
                all_data.extend(converted_data)
    
    print(f"Total samples converted to Alpaca format: {len(all_data)}")
    
    # Shuffle all data together
    random.seed(42)  # For reproducibility
    random.shuffle(all_data)
    
    # Split ratios: 70% train, 15% validation, 15% test
    total = len(all_data)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    print(f"\nFinal splits:")
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Write split files
    def write_jsonl(data, filename):
        with open(output_path / filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"Written {len(data)} samples to {filename}")
    
    write_jsonl(train_data, 'train.jsonl')
    write_jsonl(val_data, 'validation.jsonl') 
    write_jsonl(test_data, 'test.jsonl')
    
    # Create a sample file to show the format
    sample_data = train_data[:3] if train_data else []
    write_jsonl(sample_data, 'sample.jsonl')
    
    print(f"\nDataset converted to Alpaca format and saved to {output_dir}/")
    print("Files created: train.jsonl, validation.jsonl, test.jsonl, sample.jsonl")

if __name__ == "__main__":
    combine_and_split_dataset(
        input_dir="street_math_dataset_o4_mini_viable",
        output_dir="street_math_hf_dataset"
    )
    print("Dataset preparation complete!")