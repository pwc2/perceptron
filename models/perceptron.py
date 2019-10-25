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


class Perceptron:
    """Class to construct perceptron object, including online perceptron, average perceptron, and polynomial kernel
    perceptron.
    """

    def __init__(self, train, validation, test, label):
        """Constructs a perceptron object...TODO

        Args:
            train (str): Name of normalized csv file in /data folder with training examples.
            validation (str): Name of normalized csv file in /data folder with validation examples.
            test (str): Name of normalized csv file in /data folder with testing examples.
            label (str): Name of column with labels in train and validation sets.
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
        # Labels to associate with weights, if needed
        self.weight_labels = self.train.columns.to_list()
        # Extract features and labels from train, validation, and test sets.
        self.train_features = self.train.drop(label, axis=1)
        self.train_labels = self.train[label]
        self.validation_features = self.validation.drop(label, axis=1)
        self.validation_labels = self.validation[label]
        self.test_features = self.test

    def train_online_model(self, max_iter):
        """Trains online perceptron, without random shuffling of training data.

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

        # Number of features and number of samples.
        samp_size = np.size(x_train, axis=0)
        feature_size = np.size(x_train, axis=1)

        # Initialize all weights as zero, create list for training and validation accuracy for each iteration.
        weights = np.zeros(feature_size, dtype=np.float64)
        weights_list = []
        train_acc_list = []
        val_acc_list = []

        for iteration in range(max_iter):
            print('Current iteration: ' + str(iteration))
            for sample in range(samp_size):
                loss = y_train[sample] * (weights.T.dot(x_train[sample]))
                if loss <= 0:
                    weights += (y_train[sample] * x_train[sample])

            # Calculate predictions and get accuracy for each iteration, append to lists.
            train_pred = calc_predictions(x_train, weights)
            val_pred = calc_predictions(x_val, weights)

            train_acc = calc_accuracy(train_pred, y_train)
            train_acc_list.append(train_acc)

            val_acc = calc_accuracy(val_pred, y_val)
            val_acc_list.append(val_acc)

            weights_list.append(weights.tolist())

        results = {'iterations': max_iter,
                   'train_acc': train_acc_list,
                   'val_acc': val_acc_list,
                   'weights': weights_list}
        return results

    def online_test_predictions(self, weights):
        """Generate predictions for unlabeled test data.

        Args:
            weights (ndarray): (1 x m) ndarray of m weights from learned model.

        Returns:
            predictions (list of int): List of predicted labels.
        """
        x_test = self.test_features.to_numpy(dtype=np.float64)
        predictions = calc_predictions(x_test, weights)
        return predictions
