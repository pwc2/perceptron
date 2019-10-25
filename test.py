"""
    File name: test.py
    Author: Patrick Cummings
    Date created: 10/24/2019
    Date last modified: 10/24/2019
    Python Version: 3.7

    Used to test instance of Perceptron class.
"""

import json
import pprint
from pathlib import Path

from models.perceptron import Perceptron

pp = pprint.PrettyPrinter()

model = Perceptron(train='pa2_train_clean.csv',
                   validation='pa2_valid_clean.csv',
                   test='pa2_test_no_label_clean.csv',
                   label='label',
                   mod_type='average',
                   max_iter=15)

# learned_model = model.train(max_iter=15)
learned_model = model.train_model()

# Save output for learned model to .json file.
file_name = Path('model_output', 'test_avg.json')
file_path = Path(__file__).parent.resolve().joinpath(file_name)

# Create output directory if doesn't exist.
output_dir = file_path.parent.resolve()
if not Path(output_dir).exists():
    Path(output_dir).mkdir()
with open(file_path, 'w') as f:
    json.dump(learned_model, f, indent=4)

# import inspect
# pp.pprint(inspect.getmembers(Perceptron, lambda x: not (inspect.isroutine(x))))
# pp.pprint(model.__dict__.keys())
# print(model.train_features)
# print(model.validation_features)
