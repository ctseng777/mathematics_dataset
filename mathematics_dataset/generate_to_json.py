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

"""Example of how to write generated questions to JSON files.

Given an output directory, this will create JSON files compatible with
Hugging Face datasets library. The JSON files contain one example per line
(JSONL format) with each line containing question, answer, difficulty,
and module information.

Passing --train_split=False will create a single output file 'train.jsonl'
for all training data.
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

flags.DEFINE_string('output_dir', None, 'Where to write output JSON files')
flags.DEFINE_boolean('train_split', True,
                     'Whether to split training data by difficulty')
flags.DEFINE_boolean('single_file', False,
                     'Whether to write all data to a single JSON file')
flags.mark_flag_as_required('output_dir')


def main(unused_argv):
  generate.init_modules(FLAGS.train_split)

  output_dir = os.path.expanduser(FLAGS.output_dir)
  if os.path.exists(output_dir):
    logging.fatal('output dir %s already exists', output_dir)
  logging.info('Writing to %s', output_dir)
  os.makedirs(output_dir)

  if FLAGS.single_file:
    # Write all data to a single JSONL file
    output_path = os.path.join(output_dir, 'dataset.jsonl')
    with open(output_path, 'w') as json_file:
      for regime, flat_modules in six.iteritems(generate.filtered_modules):
        per_module = generate.counts[regime]
        for module_name, module in six.iteritems(flat_modules):
          for _ in range(per_module):
            problem, _ = generate.sample_from_module(module)
            example = {
              'input': str(problem.question),
              'output': str(problem.answer),
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