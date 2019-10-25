"""
    File name: test.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7

    Used to test instance of LinearModel() class.
"""

import inspect
import pprint

from models.perceptron import Perceptron

pp = pprint.PrettyPrinter()

model = Perceptron(train='pa2_train_clean.csv',
                   validation='pa2_valid_clean.csv',
                   test='pa2_test_no_label_clean.csv',
                   label='label')

pp.pprint(inspect.getmembers(Perceptron, lambda x: not (inspect.isroutine(x))))
pp.pprint(model.__dict__.keys())
print(model.train_features)
print(model.validation_features)

# names = model.weight_labels
# learned_model = model.train_model(50000)
# val_predictions = model.predict_validation(learned_model['weights'])['predictions']
# test_predictions = model.predict_test((learned_model['weights']))['predictions']
#
# prediction_output = pathlib.Path('model_output/predictions.pkl')
# prediction_file = pathlib.Path('model_output/predictions.txt')
#
# pred_output_path = pathlib.Path(__file__).parent.resolve().joinpath(prediction_output)
# pred_file_path = pathlib.Path(__file__).parent.resolve().joinpath(prediction_file)
#
# # Save predictions
# with open(pred_output_path, 'wb') as fp:
#     pickle.dump(test_predictions, fp, pickle.HIGHEST_PROTOCOL)
#
# # Output predictions to text file
# with open(pred_file_path, 'w') as f:
#     for prediction in test_predictions:
#         f.write('%s\n' % prediction)

# import inspect
# pp.pprint(inspect.getmembers(LinearModel, lambda x:not(inspect.isroutine(x))))
# pp.pprint(model.__dict__.keys())

# print(learned_model)
# print(dict(zip(names, learned_model['weights'])))
# pp.pprint(val_predictions[:10])
# pp.pprint(test_predictions[:10])
