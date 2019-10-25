# ridge-regression

`ridge-regression` is an implementation of linear regression with L2 regularization that uses batch gradient descent to optimize weights.

## Requirements:

- `numpy 1.17.2`

- `pandas 0.25.1`

- `progressbar2 3.37.1`

## Usage:

```python
from models.perceptron import Perceptron

model = LinearModel(train='data/PA1_train.pkl', # Path to training set
                    validation='data/PA1_dev.pkl', # Path to validation set
                    test='data/PA1_test.pkl', # Path to test set
                    target='price', # Target for prediction
                    rate=1e-05, # Learning rate for gradient descent
                    lam=0, # Regularization penalty
                    eps=0.5, # Stopping condition for norm of gradient
                    normalize=True) # Indicate whether or not to normalize data

names = model.weight_labels # Extract labels for weights
learned_model = model.train_model(max_iter=10000) # Train model
val_predictions = model.predict_validation(learned_model['weights'])['predictions'] # Get predictions on validation set
test_predictions = model.predict_test((learned_model['weights']))['predictions'] # Get predictions on test set
```

## Data:

In `data/` folder:

- `PA1_train.csv` contains training set with targets.

- `PA1_dev.csv` contains validation set with targets.

- `PA1_test.csv` contains test set without targets.

## To run models:

In `run` module:

- `run_part0.py` takes care of cleaning training, validation, and test sets.

- `run_part1.py` normalizes data to [0, 1] scale, and runs linear regression *without* L2 regularization penalty for various learning rates.

- `run_part2.py` normalizes data to [0, 1] scale, runs linear regression *with* L2 regularization with fixed learning rate and various regularization penalties.

- `run_part3.py` no normalization, runs linear regression without L2 regularization penalty for given learning rates.

Use `python main.py` to run parts 0 through 3.
