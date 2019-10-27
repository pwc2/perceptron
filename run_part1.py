"""
    File name: run_part1.py
    Author: Patrick Cummings
    Date created: 10/24/2019
    Date last modified: 10/26/2019
    Python Version: 3.7

    To run online perceptron.
    Output model results to /model_output folder.
"""

import csv
import json
from pathlib import Path

from models.perceptron import Perceptron

model = Perceptron(train='pa2_train_clean.csv',
                   validation='pa2_valid_clean.csv',
                   test='pa2_test_no_label_clean.csv',
                   label='label',
                   mod_type='online',
                   max_iter=15)

learned_model = model.train_model()

# Save output for learned model to .json file.
output_folder = Path('model_output')
output_path = Path(__file__).parent.resolve().joinpath(output_folder)
training_file = output_path.joinpath(Path('op_training.json'))

# Create output directory if doesn't exist.
if not Path(output_path).exists():
    Path(output_path).mkdir()
with open(training_file, 'w') as f:
    json.dump(learned_model, f, indent=4)

# Best validation accuracy with 14 iterations, calculate and save predictions.
test_predictions = model.predict_test(learned_model['weights'][13])
prediction_file = output_path.joinpath(Path('oplabel.csv'))
with open(prediction_file, 'w') as fp:
    writer = csv.writer(fp)
    for i in test_predictions:
        writer.writerows([[i]])
