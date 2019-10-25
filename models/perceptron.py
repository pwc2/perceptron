"""
    File name: perceptron.py
    Author: Patrick Cummings
    Date created: 10/24/2019
    Date last modified: 10/24/2019
    Python Version: 3.7
"""

from pathlib import Path

import numpy as np
import pandas as pd

from models.metrics import calc_predictions, calc_accuracy
from models.algorithms import online_perceptron, average_perceptron


class Perceptron:
    """Class to construct perceptron object, including online perceptron, average perceptron, and polynomial kernel
    perceptron.
    """

    def __init__(self, train, validation, test, label, mod_type, max_iter):
        """Constructs a perceptron object...TODO

        Args:
            train (str): Name of normalized csv file in /data folder with training examples.
            validation (str): Name of normalized csv file in /data folder with validation examples.
            test (str): Name of normalized csv file in /data folder with testing examples.
            label (str): Name of column with labels in train and validation sets.
            mod_type (str): String specifying 'online', 'average', or 'kernel' for model to run.
            max_iter (int): Maximum number of iterations for training.
        """
        # Get path to data folder to read in .csv files.
        data_path = Path(__file__).parent.resolve().joinpath(Path('..', 'data'))
        with open(data_path.joinpath(Path(train))) as f:
            self.train = pd.read_csv(f)
        with open(data_path.joinpath(Path(validation))) as f:
            self.validation = pd.read_csv(f)
        with open(data_path.joinpath(Path(test))) as f:
            self.test = pd.read_csv(f)

        # Check that columns for train, validation, and test in same order.
        train_cols = self.train.drop(label, axis=1).columns.to_list()
        validation_cols = self.validation.drop(label, axis=1).columns.to_list()
        test_cols = self.test.columns.to_list()
        assert np.array_equal(train_cols, validation_cols), 'Train and validation columns not in same order.'
        assert np.array_equal(train_cols, test_cols), 'Train and test columns not in same order.'
        assert np.array_equal(validation_cols, test_cols), 'Validation and test columns not in same order.'

        # Name of target column in DataFrames
        self.label = label
        self.mod_type = mod_type
        self.max_iter = max_iter
        # Labels to associate with weights, if needed
        self.weight_labels = self.train.columns.to_list()
        # Extract features and labels from train, validation, and test sets.
        self.train_features = self.train.drop(label, axis=1)
        self.train_labels = self.train[label]
        self.validation_features = self.validation.drop(label, axis=1)
        self.validation_labels = self.validation[label]
        self.test_features = self.test

    def train_model(self):
        """Trains online perceptron, without random shuffling of training data.
        Calls online_perceptron() from algorithms.py.

        Args:
            max_iter (int): Maximum number of iterations

        Returns:
            results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
            list of validation accuracies for each iteration, and a list of weights from each iteration.
        """
        # Training and validation sets and labels.
        x_train = self.train_features.to_numpy(dtype=np.float64)
        y_train = self.train_labels.to_numpy(dtype=int)
        x_val = self.validation_features.to_numpy(dtype=np.float64)
        y_val = self.validation_labels.to_numpy(dtype=int)

        if self.mod_type is 'online':
            results = online_perceptron(x_train, y_train, x_val, y_val, self.max_iter)
            return results
        if self.mod_type is 'average':
            results = average_perceptron(x_train, y_train, x_val, y_val, self.max_iter)
            return results
        if self.mod_type is 'kernel':
            # TODO
            return None

    def predict_test(self, weights):
        """Generate predictions for unlabeled test data.

        Args:
            weights (ndarray): (1 x m) ndarray of m weights from learned model.

        Returns:
            predictions (list of int): List of predicted labels.
        """
        x_test = self.test_features.to_numpy(dtype=np.float64)
        predictions = calc_predictions(x_test, weights)
        return predictions
