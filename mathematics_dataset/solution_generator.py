# Copyright 2018 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Solution generator for street math problems using LLM."""

import json
import os
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' library not found. LLM solution generation will be disabled.")


def generate_solution(question: str, exact_answer: float, approximate_answer: str) -> str:
    """Generate step-by-step solution using LLM.
    
    Args:
        question: The math question (e.g., "Mental math: approximately 47 - 820?")
        exact_answer: The precise mathematical result 
        approximate_answer: The approximated answer for mental math
        
    Returns:
        Step-by-step solution string
    """
    # Check for API key and requests library
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or not HAS_REQUESTS:
        return _generate_basic_solution(question, exact_answer, approximate_answer)
    
    # Prepare prompt for LLM
    prompt = f"""Generate a clear, educational solution for this mental math approximation problem.

Question: {question}
Exact answer: {exact_answer}
Mental math approximation: {approximate_answer}

Provide a step-by-step solution (2-4 steps) that teaches mental math techniques:

1. **Simplify numbers**: How to round numbers for easier mental calculation
2. **Mental math steps**: Show the actual mental arithmetic process
3. **Reasoning**: Explain why this approximation strategy is effective

Focus on teaching the mental math technique, not just getting the answer. Use simple language suitable for learning.
Keep the solution concise but complete."""

    try:
        # Call OpenAI API
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': 'gpt-4o-mini',  # Optimal for educational content
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful math tutor focused on mental math techniques and approximation strategies. Always provide complete, concise explanations.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 400,  # Increased for complete solutions
                'temperature': 0.2  # Lower for more consistent educational content
            },
            timeout=15  # Increased timeout for longer responses
        )
        
        if response.status_code == 200:
            result = response.json()
            solution = result['choices'][0]['message']['content'].strip()
            
            # Ensure solution is complete (not cut off)
            if solution and not solution.endswith(('.', '!', '?', ')')):
                # If solution appears truncated, try to complete it with basic ending
                solution += "... This mental math technique provides a quick and reasonable approximation."
                
            return solution
        else:
            print(f"OpenAI API error: {response.status_code}")
            return _generate_basic_solution(question, exact_answer, approximate_answer)
            
    except requests.exceptions.Timeout:
        print("OpenAI API timeout - falling back to basic solution")
        return _generate_basic_solution(question, exact_answer, approximate_answer)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return _generate_basic_solution(question, exact_answer, approximate_answer)


def _generate_basic_solution(question: str, exact_answer: float, approximate_answer: str) -> str:
    """Generate basic solution without LLM as fallback."""
    
    # Extract operation and numbers from question
    question_lower = question.lower()
    
    if '+' in question or 'add' in question_lower:
        operation = 'addition'
        op_verb = 'add'
    elif '-' in question or 'subtract' in question_lower:
        operation = 'subtraction'  
        op_verb = 'subtract'
    elif '*' in question or 'times' in question_lower or 'multiply' in question_lower:
        operation = 'multiplication'
        op_verb = 'multiply'
    elif '/' in question or 'divide' in question_lower:
        operation = 'division'
        op_verb = 'divide'
    else:
        operation = 'calculation'
        op_verb = 'calculate'
    
    # Basic solution template
    try:
        approx_float = float(str(approximate_answer))
        abs_approx = abs(approx_float)
    except:
        abs_approx = 100  # Default fallback
        
    if abs_approx >= 1000:
        magnitude = "thousands"
        rounding = "nearest thousand"
    elif abs_approx >= 100:
        magnitude = "hundreds"
        rounding = "nearest ten or hundred"
    elif abs_approx >= 10:
        magnitude = "tens"
        rounding = "nearest five or ten"
    else:
        magnitude = "units"
        rounding = "nearest whole number or half"
        
    # Extract numbers from question for more specific guidance
    import re
    numbers = re.findall(r'-?\d+\.?\d*', question)
    
    if len(numbers) >= 2:
        solution = f"""Step 1: To {op_verb} {numbers[0]} and {numbers[1]} mentally, round to simpler numbers.
Step 2: Focus on the {magnitude} place - round to the {rounding}.
Step 3: Perform the {operation} with the rounded numbers.
Step 4: The result {approximate_answer} is close to the exact answer {exact_answer:.1f}."""
    else:
        solution = f"""Step 1: For mental {operation}, focus on the dominant {magnitude}.
Step 2: Round numbers to the {rounding} for easier calculation.
Step 3: The approximation {approximate_answer} is close to the exact answer {exact_answer:.1f}.
Step 4: This rounding makes the mental math manageable while staying reasonably accurate."""
    
    return solution


def generate_solution_batch(problems: list) -> list:
    """Generate solutions for a batch of problems.
    
    Args:
        problems: List of (question, exact_answer, approximate_answer) tuples
        
    Returns:
        List of solution strings
    """
    solutions = []
    for question, exact_answer, approximate_answer in problems:
        solution = generate_solution(question, exact_answer, approximate_answer)
        solutions.append(solution)
    return solutions


# Configuration settings
def get_model_config():
    """Get current model configuration."""
    return {
        'model': 'gpt-4o-mini',
        'max_tokens': 400,
        'temperature': 0.2,
        'timeout': 15
    }


def set_max_tokens(tokens: int):
    """Override max tokens if needed for longer solutions."""
    global MAX_TOKENS
    MAX_TOKENS = tokens
    
    
# Default configuration
MAX_TOKENS = 400  # Can be overridden if needed