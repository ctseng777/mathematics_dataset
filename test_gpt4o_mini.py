#!/usr/bin/env python3
"""Test GPT-4o-mini solution generation."""

import os
import sys
sys.path.append('/Users/moonshine/workspace/mathematics_dataset')

from mathematics_dataset.solution_generator import generate_solution

def test_gpt4o_mini():
    """Test GPT-4o-mini with sample problems."""
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ No OPENAI_API_KEY found. Set your API key to test GPT-4o-mini.")
        return
    
    test_cases = [
        ("Mental math: approximately 47 - 820?", -773.0, "-770"),
        ("Without a calculator, about what is 156 * 23?", 3588.0, "3600"),
        ("Quick estimate of -84 / 7:", -12.0, "-12")
    ]
    
    print("🧮 Testing GPT-4o-mini for educational math solutions:")
    print("=" * 60)
    
    for i, (question, exact_answer, approx_answer) in enumerate(test_cases, 1):
        print(f"\n**Example {i}:**")
        print(f"Question: {question}")
        print(f"Exact: {exact_answer}, Approximation: {approx_answer}")
        print("-" * 40)
        
        try:
            solution = generate_solution(question, exact_answer, approx_answer)
            print(f"Solution:\n{solution}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 60)

if __name__ == "__main__":
    test_gpt4o_mini()