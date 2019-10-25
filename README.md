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
                   label='label')

learned_model = model.train_online_model(max_iter=14)
```

## Data:

In `data/` folder:

## To run models:

In `run` module:

Use `python main.py` to run parts 0 through 3.
