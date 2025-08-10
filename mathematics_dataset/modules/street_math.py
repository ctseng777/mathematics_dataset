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

"""Street math approximation problems for mental math reasoning training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import random

# Dependency imports
from mathematics_dataset import example
from mathematics_dataset.sample import number
from mathematics_dataset.util import composition
from mathematics_dataset.util import display
import sympy
import collections
from mathematics_dataset.solution_generator import generate_solution


# Extended Problem class for street_math with additional fields
StreetMathProblem = collections.namedtuple('StreetMathProblem', 
    ('question', 'answer', 'exact_answer', 'lower_bound', 'upper_bound', 'solution'))


def _calculate_bounds(exact_answer, target_percent=10):
  """Calculate upper and lower bounds within target percentage of exact answer."""
  exact_float = _safe_float_convert(exact_answer)
  margin = abs(exact_float) * (target_percent / 100.0)
  
  lower_bound = exact_float - margin
  upper_bound = exact_float + margin
  
  # Format bounds nicely 
  if abs(lower_bound) < 0.001:
    lower_bound = 0
  if abs(upper_bound) < 0.001:
    upper_bound = 0
    
  # Round to reasonable precision
  if abs(exact_float) >= 1:
    decimal_places = max(0, 2 - len(str(int(abs(exact_float)))))
    lower_bound = round(lower_bound, decimal_places)
    upper_bound = round(upper_bound, decimal_places)
  else:
    lower_bound = round(lower_bound, 4)
    upper_bound = round(upper_bound, 4)
    
  return lower_bound, upper_bound


# Lower entropy for mental-math-friendly numbers
_ENTROPY_TRAIN = (2, 5)
_ENTROPY_INTERPOLATE = (4, 6)  
_ENTROPY_EXTRAPOLATE = (5, 7)

_ADD_SUB_ENTROPY_TRAIN = (3, 6)
_ADD_SUB_ENTROPY_INTERPOLATE = (4, 7)
_ADD_SUB_ENTROPY_EXTRAPOLATE = (5, 8)

_INT = 'int'
_INT_OR_RATIONAL = 'rational'

# Control solution generation (can be set via environment variable)
import os
_GENERATE_SOLUTIONS = os.getenv('GENERATE_SOLUTIONS', 'false').lower() == 'true'


# ==============================================================================
# CORE APPROXIMATION ENGINE
# ==============================================================================

def _safe_float_convert(val):
  """Safely convert various number types to float."""
  if hasattr(val, 'value'):
    actual_val = val.value
  else:
    actual_val = val
    
  if hasattr(actual_val, 'evalf'):
    return float(actual_val.evalf())
  else:
    return float(str(actual_val))


def _approximate_to_nice_number(exact_value, target_error_percent=10):
  """Apply human-like approximation patterns for mental math.
  
  Args:
    exact_value: The precise mathematical result
    target_error_percent: Maximum acceptable error percentage (default 10%)
    
  Returns:
    Approximated value that's easy for mental math
  """
  if exact_value == 0:
    return 0
    
  abs_val = abs(exact_value)
  sign = 1 if exact_value >= 0 else -1
  
  # Calculate acceptable error range
  max_error = abs_val * (target_error_percent / 100.0)
  
  # Human approximation patterns by magnitude
  if abs_val >= 10000:
    # Large numbers: round to nearest thousand or ten-thousand
    if abs_val >= 100000:
      rounded = round(abs_val / 10000) * 10000
    else:
      rounded = round(abs_val / 1000) * 1000
  elif abs_val >= 1000:
    # Thousands: round to nearest 100 or 50
    if abs_val <= 2000:
      rounded = round(abs_val / 50) * 50
    else:
      rounded = round(abs_val / 100) * 100
  elif abs_val >= 100:
    # Hundreds: round to nearest 10 or 25
    rounded = round(abs_val / 10) * 10
  elif abs_val >= 10:
    # Tens: round to nearest 5 or integer
    if abs_val <= 30:
      rounded = round(abs_val / 5) * 5
    else:
      rounded = round(abs_val)
  elif abs_val >= 1:
    # Units: round to nearest 0.5 or integer
    if abs_val <= 3:
      rounded = round(abs_val * 2) / 2
    else:
      rounded = round(abs_val)
  else:
    # Decimals: round to 1-2 decimal places max
    if abs_val >= 0.1:
      rounded = round(abs_val, 1)
    elif abs_val >= 0.01:
      rounded = round(abs_val, 2)
    else:
      rounded = round(abs_val, 3)
  
  # Validate approximation is within error bounds
  error = abs(rounded - abs_val)
  if error > max_error and abs_val > 0.01:
    # Fallback: use percentage-based adjustment but keep it clean
    factor = 1 + random.uniform(-target_error_percent/200, target_error_percent/200)
    rounded = abs_val * factor
    
    # Clean up the fallback result
    if rounded >= 100:
      rounded = round(rounded / 10) * 10
    elif rounded >= 10:
      rounded = round(rounded)
    elif rounded >= 1:
      rounded = round(rounded, 1)
    else:
      rounded = round(rounded, 2)
  
  return sign * rounded


def _generate_solution_if_enabled(question_str, exact_answer, approx_answer):
  """Generate solution using LLM if enabled, otherwise return empty string."""
  if _GENERATE_SOLUTIONS:
    try:
      return generate_solution(question_str, exact_answer, str(approx_answer))
    except Exception as e:
      print(f"Warning: Failed to generate solution: {e}")
      return ""
  return ""


def _is_mental_math_friendly(p_val, q_val, operation):
  """Check if numbers are suitable for mental math approximation."""
  p_abs, q_abs = abs(p_val), abs(q_val)
  
  # Avoid trivial cases that don't need approximation
  if operation in ['add', 'sub']:
    # Avoid tiny changes (like 1000 + 1)
    if max(p_abs, q_abs) > 50 and min(p_abs, q_abs) < 2:
      return False
    # Avoid single digit operations  
    if p_abs < 10 and q_abs < 10 and p_val == int(p_val) and q_val == int(q_val):
      return False
      
  elif operation == 'mul':
    # Avoid multiplication by 0 or 1
    if p_abs <= 1 or q_abs <= 1:
      return False
    # Avoid basic multiplication table (2-10 � 2-10)
    if (2 <= p_abs <= 10 and 2 <= q_abs <= 10 and 
        p_val == int(p_val) and q_val == int(q_val)):
      return False
      
  elif operation == 'div':
    # Avoid division by 1
    if q_abs == 1:
      return False
    # Avoid simple exact divisions
    if p_val != 0 and q_val != 0 and abs(p_val % q_val) < 0.001:
      result = abs(p_val / q_val)
      if result == int(result) and result < 20:
        return False
  
  return True


def _generate_mental_math_number(entropy, signed=True):
  """Generate numbers optimized for mental math approximation."""
  max_attempts = 20
  for _ in range(max_attempts):
    # Prefer slightly larger entropy for more interesting problems
    if random.choice([True, False]):
      val = number.integer(min(entropy + 1, 6), signed=signed, min_abs=2)
    else:
      val = number.integer_or_decimal(min(entropy + 1, 6), signed=signed)
    
    # Check if number is reasonable for mental math
    float_val = _safe_float_convert(val)
    abs_val = abs(float_val)
    
    # Acceptable range for mental math
    if 2 <= abs_val <= 50000:
      # Prefer numbers that aren't too precise for decimals
      if '.' in str(float_val):
        decimal_places = len(str(float_val).split('.')[1].rstrip('0'))
        if decimal_places <= 2:  # At most 2 decimal places
          return val
      else:
        return val
  
  # Fallback: simple integer
  return number.integer(3, signed=signed, min_abs=2)


# ==============================================================================
# APPROXIMATION QUESTION TEMPLATES  
# ==============================================================================

def _get_approximation_templates(operation):
  """Get varied question templates for approximation problems."""
  
  base_templates = {
    'add': [
      "About how much is {p} + {q}?",
      "Roughly what's {p} + {q}?", 
      "Estimate {p} + {q}.",
      "Mental math: approximately {p} + {q}?",
      "Quick estimate of {p} + {q}:",
      "Around what is {p} + {q}?",
      "What's your rough calculation of {p} + {q}?",
      "Ballpark figure for {p} + {q}:",
      "Without a calculator, about what is {p} + {q}?",
      "In your head, roughly {p} + {q} =",
    ],
    'sub': [
      "About how much is {p} - {q}?",
      "Roughly what's {p} - {q}?",
      "Estimate {p} - {q}.",
      "Mental math: approximately {p} - {q}?", 
      "Quick estimate of {p} - {q}:",
      "Around what is {p} - {q}?",
      "What's your rough calculation of {p} - {q}?",
      "Ballpark figure for {p} - {q}:",
      "Without a calculator, about what is {p} - {q}?",
      "In your head, roughly {p} - {q} =",
    ],
    'mul': [
      "About how much is {p} * {q}?",
      "Roughly what's {p} * {q}?",
      "Estimate {p} * {q}.",
      "Mental math: approximately {p} times {q}?",
      "Quick estimate of {p} * {q}:", 
      "Around what is {p} * {q}?",
      "What's your rough calculation of {p} * {q}?",
      "Ballpark figure for {p} * {q}:",
      "Without a calculator, about what is {p} * {q}?",
      "In your head, roughly {p} * {q} =",
    ],
    'div': [
      "About how much is {p} / {q}?",
      "Roughly what's {p} / {q}?",
      "Estimate {p} / {q}.",
      "Mental math: approximately {p} divided by {q}?",
      "Quick estimate of {p} / {q}:",
      "Around what is {p} / {q}?", 
      "What's your rough calculation of {p} / {q}?",
      "Ballpark figure for {p} / {q}:",
      "Without a calculator, about what is {p} / {q}?",
      "In your head, roughly {p} / {q} =",
    ]
  }
  
  return base_templates.get(operation, base_templates['add'])


# ==============================================================================
# MODULE STRUCTURE (based on arithmetic.py)
# ==============================================================================

def _entropy_for_pair(entropy):
  """Split entropy between two numbers."""
  entropy_1 = max(1, random.uniform(0, entropy))  
  entropy_2 = max(1, entropy - entropy_1)
  return entropy_1, entropy_2


def _make_modules(entropy, add_sub_entropy):
  """Returns modules for street math approximation problems."""
  sample_args_pure = composition.PreSampleArgs(1, 1, *entropy)
  add_sub_sample_args_pure = composition.PreSampleArgs(1, 1, *add_sub_entropy)
  
  return {
      'add_or_sub': functools.partial(
          add_or_sub, None, add_sub_sample_args_pure),
      'mul': functools.partial(mul, None, sample_args_pure),
      'div': functools.partial(div, None, sample_args_pure),
  }


def train(entropy_fn):
  """Returns dict of training modules."""
  return _make_modules(
      entropy=entropy_fn(_ENTROPY_TRAIN),
      add_sub_entropy=entropy_fn(_ADD_SUB_ENTROPY_TRAIN))


def test():
  """Returns dict of testing modules."""
  return _make_modules(
      entropy=_ENTROPY_INTERPOLATE,
      add_sub_entropy=_ADD_SUB_ENTROPY_INTERPOLATE)


def test_extra():
  """Returns dict of extrapolation testing modules."""
  return _make_modules(
      entropy=_ENTROPY_EXTRAPOLATE,
      add_sub_entropy=_ADD_SUB_ENTROPY_EXTRAPOLATE)


# ==============================================================================
# APPROXIMATION PROBLEM GENERATORS
# ==============================================================================

@composition.module(number.is_integer_or_rational_or_decimal)
def add_or_sub(value, sample_args, context=None):
  """Generate approximation problems for addition or subtraction."""
  is_question = context is None
  if context is None:
    context = composition.Context()
    
  is_addition = random.choice([False, True])
  entropy, sample_args = sample_args.peel()
  
  # Generate mental-math-friendly numbers
  if value is None:
    entropy_p, entropy_q = _entropy_for_pair(entropy)
    
    # Try multiple times to get good numbers for mental math
    max_attempts = 15
    for _ in range(max_attempts):
      p = _generate_mental_math_number(entropy_p, signed=True)
      q = _generate_mental_math_number(entropy_q, signed=True)
      
      p_val = _safe_float_convert(p)
      q_val = _safe_float_convert(q)
      operation = 'add' if is_addition else 'sub'
      
      if _is_mental_math_friendly(p_val, q_val, operation):
        break
  else:
    # Handle constrained generation (less common case)
    entropy = max(entropy, number.entropy_of_value(value))
    p = _generate_mental_math_number(entropy, signed=True)
    if is_addition:
      q_val = _safe_float_convert(value) - _safe_float_convert(p)
      q = display.Decimal(q_val) if isinstance(p, display.Decimal) else q_val
    else:
      q_val = _safe_float_convert(p) - _safe_float_convert(value)
      q = display.Decimal(q_val) if isinstance(p, display.Decimal) else q_val
  
  p, q = context.sample(sample_args, [p, q])
  
  # Calculate exact and approximate answers
  p_float = _safe_float_convert(p)
  q_float = _safe_float_convert(q)
  
  if is_addition:
    exact_answer = p_float + q_float
    operation = 'add'
  else:
    exact_answer = p_float - q_float
    operation = 'sub'
    
  approx_answer = _approximate_to_nice_number(exact_answer)
  lower_bound, upper_bound = _calculate_bounds(exact_answer)
  
  if is_question:
    template = random.choice(_get_approximation_templates(operation))
    question_obj = example.question(context, template, p=p, q=q)
    solution = _generate_solution_if_enabled(str(question_obj), exact_answer, approx_answer)
    return StreetMathProblem(
        question=question_obj,
        answer=int(approx_answer) if approx_answer == int(approx_answer) else approx_answer,
        exact_answer=exact_answer,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        solution=solution)
  else:
    return composition.Entity(
        context=context,
        value=approx_answer,
        description='Let {self} H {p} + {q}.' if is_addition else 'Let {self} H {p} - {q}.',
        p=p, q=q)


@composition.module(number.is_integer_or_rational_or_decimal)  
def mul(value, sample_args, context=None):
  """Generate approximation problems for multiplication."""
  del value  # unused for now
  is_question = context is None
  if context is None:
    context = composition.Context()
    
  entropy, sample_args = sample_args.peel()
  entropy_p, entropy_q = _entropy_for_pair(entropy)
  
  # Generate mental-math-friendly numbers
  max_attempts = 15
  for _ in range(max_attempts):
    p = _generate_mental_math_number(entropy_p, signed=True)
    q = _generate_mental_math_number(entropy_q, signed=True)
    
    p_val = _safe_float_convert(p)
    q_val = _safe_float_convert(q)
    
    if _is_mental_math_friendly(p_val, q_val, 'mul'):
      break
  
  p, q = context.sample(sample_args, [p, q])
  
  # Calculate exact and approximate answers
  p_float = _safe_float_convert(p)
  q_float = _safe_float_convert(q)
  exact_answer = p_float * q_float
  approx_answer = _approximate_to_nice_number(exact_answer)
  lower_bound, upper_bound = _calculate_bounds(exact_answer)
  
  if is_question:
    template = random.choice(_get_approximation_templates('mul'))
    question_obj = example.question(context, template, p=p, q=q)
    solution = _generate_solution_if_enabled(str(question_obj), exact_answer, approx_answer)
    return StreetMathProblem(
        question=question_obj,
        answer=int(approx_answer) if approx_answer == int(approx_answer) else approx_answer,
        exact_answer=exact_answer,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        solution=solution)
  else:
    return composition.Entity(
        context=context,
        value=approx_answer,
        description='Let {self} H {p} * {q}.',
        p=p, q=q)


@composition.module(number.is_integer_or_rational_or_decimal)
def div(value, sample_args, context=None):
  """Generate approximation problems for division."""
  del value  # unused for now
  is_question = context is None
  if context is None:
    context = composition.Context()
    
  entropy, sample_args = sample_args.peel()
  entropy_1, entropy_q = _entropy_for_pair(entropy)
  
  # Generate mental-math-friendly numbers
  max_attempts = 15
  for _ in range(max_attempts):
    p = _generate_mental_math_number(entropy_1, signed=True)
    q = _generate_mental_math_number(entropy_q, signed=True)
    
    # Ensure q is not zero
    q_val = _safe_float_convert(q)
    if q_val == 0:
      continue
        
    p_val = _safe_float_convert(p)
    
    if _is_mental_math_friendly(p_val, q_val, 'div'):
      break
  
  p, q = context.sample(sample_args, [p, q])
  
  # Calculate exact and approximate answers
  p_float = _safe_float_convert(p)
  q_float = _safe_float_convert(q)
  exact_answer = p_float / q_float
  approx_answer = _approximate_to_nice_number(exact_answer)
  lower_bound, upper_bound = _calculate_bounds(exact_answer)
  
  if is_question:
    template = random.choice(_get_approximation_templates('div'))
    question_obj = example.question(context, template, p=p, q=q)
    solution = _generate_solution_if_enabled(str(question_obj), exact_answer, approx_answer)
    return StreetMathProblem(
        question=question_obj,
        answer=int(approx_answer) if approx_answer == int(approx_answer) else approx_answer,
        exact_answer=exact_answer,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        solution=solution)
  else:
    return composition.Entity(
        context=context,
        value=approx_answer,
        description='Let {self} H {p} / {q}.',
        p=p, q=q)