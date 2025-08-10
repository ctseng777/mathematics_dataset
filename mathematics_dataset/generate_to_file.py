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

"""Example of how to write generated questions to text or JSON files.

Given an output directory, this will create the following subdirectories:

*   train-easy
*   train-medium
*   train-hard
*   interpolate
*   extrapolate

For text format: populate each subdirectory with a text file for each module,
where the text file contains lines alternating between the question and answer.

For JSON format: create JSONL files compatible with Hugging Face datasets,
with each line containing question, answer, difficulty, and module information.

Passing --train_split=False will create a single output directory 'train' for
training data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# Dependency imports
from absl import app
from absl import flags
from absl import logging
from mathematics_dataset import generate
import six
from six.moves import range

FLAGS = flags.FLAGS

flags.DEFINE_string('output_dir', None, 'Where to write output files')
flags.DEFINE_boolean('train_split', True,
                     'Whether to split training data by difficulty')
flags.DEFINE_enum('format', 'txt', ['txt', 'json'],
                  'Output format: txt for text files, json for JSONL files')
flags.DEFINE_boolean('single_file', False,
                     'For JSON format: write all data to single file')
flags.mark_flag_as_required('output_dir')


def main(unused_argv):
  generate.init_modules(FLAGS.train_split)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.fatal('output dir %s already exists', output_dir)
  logging.info('Writing to %s', output_dir)
  os.makedirs(output_dir)

  if FLAGS.format == 'json':
    _write_json_format(output_dir)
  else:
    _write_text_format(output_dir)


def _write_text_format(output_dir):
  """Write data in the original text format."""
  for regime, flat_modules in six.iteritems(generate.filtered_modules):
    regime_dir = os.path.join(output_dir, regime)
    os.mkdir(regime_dir)
    per_module = generate.counts[regime]
    for module_name, module in six.iteritems(flat_modules):
      path = os.path.join(regime_dir, module_name + '.txt')
      with open(path, 'w') as text_file:
        for _ in range(per_module):
          problem, _ = generate.sample_from_module(module)
          text_file.write(str(problem.question) + '\n')
          text_file.write(str(problem.answer) + '\n')
      logging.info('Written %s', path)


def _write_json_format(output_dir):
  """Write data in JSON format compatible with Hugging Face."""
  if FLAGS.single_file:
    # Write all data to a single JSONL file
    output_path = os.path.join(output_dir, 'dataset.jsonl')
    with open(output_path, 'w') as json_file:
      for regime, flat_modules in six.iteritems(generate.filtered_modules):
        per_module = generate.counts[regime]
        for module_name, module in six.iteritems(flat_modules):
          for _ in range(per_module):
            problem, _ = generate.sample_from_module(module)
            
            # Check if it's a StreetMathProblem with additional fields
            if hasattr(problem, 'exact_answer'):
              example = {
                'input': str(problem.question),
                'output': str(problem.answer),
                'exact_answer': str(problem.exact_answer),
                'lower_bound': str(problem.lower_bound),
                'upper_bound': str(problem.upper_bound),
                'solution': problem.solution,
                'difficulty': regime,
                'module': module_name
              }
            else:
              example = {
                'input': str(problem.question),
                'approximate_answer': str(problem.answer),
                'difficulty': regime,
                'module': module_name
              }
            json_file.write(json.dumps(example) + '\n')
    logging.info('Written %s', output_path)
  else:
    # Write separate files for each regime
    for regime, flat_modules in six.iteritems(generate.filtered_modules):
      output_path = os.path.join(output_dir, regime + '.jsonl')
      with open(output_path, 'w') as json_file:
        per_module = generate.counts[regime]
        for module_name, module in six.iteritems(flat_modules):
          for _ in range(per_module):
            problem, _ = generate.sample_from_module(module)
            
            # Check if it's a StreetMathProblem with additional fields
            if hasattr(problem, 'exact_answer'):
              example = {
                'input': str(problem.question),
                'approximate_answer': str(problem.answer),
                'exact_answer': str(problem.exact_answer),
                'lower_bound': str(problem.lower_bound),
                'upper_bound': str(problem.upper_bound),
                'solution': problem.solution,
                'difficulty': regime,
                'module': module_name
              }
            else:
              example = {
                'input': str(problem.question),
                'output': str(problem.answer),
                'difficulty': regime,
                'module': module_name
              }
            json_file.write(json.dumps(example) + '\n')
      logging.info('Written %s', output_path)


if __name__ == '__main__':
  app.run(main)
