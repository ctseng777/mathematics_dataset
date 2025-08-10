#!/usr/bin/env python3
"""Test script for solution generation functionality."""

import os
import sys
sys.path.append('/Users/moonshine/workspace/mathematics_dataset')

from mathematics_dataset.solution_generator import generate_solution

def test_basic_solution_generation():
    """Test basic solution generation without LLM."""
    print("=== Testing Basic Solution Generation ===")
    
    test_cases = [
        ("Mental math: approximately 47 - 820?", -773.0, "-770"),
        ("Without a calculator, about what is -6 / 188?", -0.031914, "-0.03"),
        ("Around what is 25 * -46?", -1150.0, "-1200"),
        ("Estimate 567 + 234.", 801.0, "800")
    ]
    
    for question, exact_answer, approx_answer in test_cases:
        print(f"\nQuestion: {question}")
        print(f"Exact: {exact_answer}, Approximate: {approx_answer}")
        
        solution = generate_solution(question, exact_answer, approx_answer)
        print(f"Solution:\n{solution}")
        print("-" * 50)

def test_with_llm():
    """Test with LLM if API key is available."""
    print("\n=== Testing with LLM (if API key available) ===")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("No OPENAI_API_KEY found. Skipping LLM test.")
        print("To test with LLM, set: export OPENAI_API_KEY=your_key")
        return
    
    question = "Mental math: approximately 47 - 820?"
    exact_answer = -773.0
    approx_answer = "-770"
    
    print(f"Question: {question}")
    print(f"Exact: {exact_answer}, Approximate: {approx_answer}")
    
    solution = generate_solution(question, exact_answer, approx_answer)
    print(f"LLM Solution:\n{solution}")

if __name__ == "__main__":
    test_basic_solution_generation()
    test_with_llm()