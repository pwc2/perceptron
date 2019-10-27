# perceptron

`perceptron` has implementations of the perceptron with online learning, the average perceptron, and a polynomial kernel perceptron.
## Requirements:

- `numpy 1.17.2`

- `pandas 0.25.1`

## Usage:

```python
from models.perceptron import Perceptron

model = Perceptron(train='pa2_train_clean.csv',
                   validation='pa2_valid_clean.csv',
                   test='pa2_test_no_label_clean.csv',
                   label='label', # Specify target name
                   mod_type='online', # Choose model type
                   max_iter=15, # Set maximum iterations for training
                   p=None) # If using polynomial kernel, set degree

learned_model = model.train_model()
```

## Data:

In `data/` folder:

## To run models:

In `run` module:

