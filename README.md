# perceptron

`perceptron` contains implementations of the perceptron with online learning, the average perceptron, and a polynomial kernel perceptron.

### Requirements:

- `numpy 1.17.2`

- `pandas 0.25.1`

- `progressbar2 3.37.1`

### Usage:

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

### Data:

The `data/` folder contains .csv files with training, validation, and test sets.

### To run models:

- `run_part0.py` runs pre-processing.
- `run_part1.py` runs online perceptron.
- `run_part2.py` runs average perceptron.
- `run_part3.py` runs perceptron using a polynomial kernel.

`python main.py` will run all four parts in order, output will be saved in `model_output` folder.

