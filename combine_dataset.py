#!/usr/bin/env python3
"""Combine and split street math JSONL files for Hugging Face upload."""

import json
import os
import random
from pathlib import Path

def combine_and_split_dataset(input_dir, output_dir):
    """Combine ALL files (train-*, interpolate, extrapolate) and create train/validation/test splits."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Read all data from all files
    all_data = []
    
    # Read train-* files
    for difficulty in ['train-easy', 'train-medium', 'train-hard']:
        file_path = input_path / f"{difficulty}.jsonl"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                print(f"Loaded {len(data)} samples from {difficulty}.jsonl")
                all_data.extend(data)
    
    # Read interpolate and extrapolate files
    for test_type in ['interpolate', 'extrapolate']:
        file_path = input_path / f"{test_type}.jsonl"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f]
                print(f"Loaded {len(data)} samples from {test_type}.jsonl")
                all_data.extend(data)
    
    print(f"Total samples from all files: {len(all_data)}")
    
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

if __name__ == "__main__":
    combine_and_split_dataset(
        input_dir="street_math_dataset_o4_mini_viable",
        output_dir="street_math_hf_dataset"
    )
    print("Dataset preparation complete!")