# Street Math Approximation Dataset

A dataset for training language models on mental math approximation and reasoning skills.

## Dataset Description

This dataset contains mental math problems designed to teach approximation strategies and reasoning. Each example includes:

- **Input**: A mental math question requiring approximation
- **Output**: The approximate answer using mental math techniques  
- **Exact Answer**: The precise mathematical result
- **Bounds**: Acceptable approximation range (±10% error margin)
- **Solution**: Step-by-step educational explanation of the mental math technique

## Usage

### For Training/Fine-tuning

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "test.jsonl"})

# Basic training format
def format_for_training(example):
    return {
        "text": f"Question: {example['input']}\nAnswer: {example['output']}"
    }

# Instruction-following format  
def format_for_instruction(example):
    return {
        "instruction": example['input'],
        "response": example['output'],
        "reasoning": example['solution']
    }

# Apply formatting
train_dataset = dataset["train"].map(format_for_instruction)
```

### Loading with Hugging Face

```python
from datasets import Dataset
import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

dataset = load_jsonl("train.jsonl")
```

## Dataset Statistics

- **Total Examples**: ~24,000
- **Difficulty Levels**: Easy, Medium, Hard
- **Operations**: Addition, Subtraction, Multiplication, Division
- **Number Ranges**: Optimized for realistic mental math scenarios

## File Structure for Hugging Face Upload

```
your-dataset/
├── README.md (this file)
├── dataset_info.json
├── train.jsonl
├── validation.jsonl (optional)
└── test.jsonl (optional)
```

## Citation

Based on the DeepMind Mathematics Dataset framework.

## License

Apache-2.0