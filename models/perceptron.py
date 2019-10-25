"""
    File name: perceptron.py
    Author: Patrick Cummings
    Date created: 10/13/2019
    Date last modified: 10/18/2019
    Python Version: 3.7
"""

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import progressbar

from models.metrics import calc_predictions, calc_accuracy


class Perceptron:
    """Class to construct perceptron object.

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
        """Trains online perceptron, with (almost) stochastic gradient descent (no random shuffling for training
        data).

        Args:
            max_iter (int): Maximum number of iterations

        Returns:
            results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
            and list of validation accuracies for each iteration.
        """
        # Training and validation sets and labels.
        x_train = self.train_features.to_numpy(dtype=np.float64)
        y_train = self.train_labels.to_numpy(dtype=int)
        x_val = self.validation_features.to_numpy(dtype=np.float64)
        y_val = self.validation_labels.to_numpy(dtype=int)

        # Number of features and number of samples.
        samp_size = np.size(x_train, axis=1)
        feature_size = np.size(x_train, axis=0)

        # Initialize all weights as zero, create list for training and validation accuracy for each iteration.
        weights = np.zeros(feature_size, dtype=np.float64)
        train_acc_list = []
        val_acc_list = []

        for iteration in range(max_iter):
            for sample in range(samp_size):
                # loss = np.dot(y_train[sample], weights.T.dot(x_train[sample]))
                loss = y_train[sample] * weights.T.dot(x_train[sample])
                if loss <= 0:
                    # weights += np.dot(y_train[sample], x_train[sample])
                    weights += (y_train[sample] * x_train[sample])

            # Calculate predictions and get accuracy for each iteration, append to lists.
            train_pred = calc_predictions(x_train, weights)
            val_pred = calc_predictions(x_val, weights)

            train_acc = calc_accuracy(train_pred, y_train)
            train_acc_list.append(train_acc)

            val_acc = calc_accuracy(val_pred, y_val)
            val_acc_list.append(val_acc)

        results = {'iterations': max_iter,
                   'train_acc': train_acc_list,
                   'val_acc': val_acc_list}
        return results

    # def train_model(self, max_iter):
    #     """Trains model with batch gradient descent to optimize weights.
    #
    #     Calls calc_sse(), calc_gradient(), gradient_descent() from models.metrics.py.
    #
    #     Args:
    #         max_iter (int): Maximum number of iterations before terminating training.
    #
    #     Returns:
    #         results (dict): Dictionary with lam (regularization parameter), learn_rate (learning rate),
    #         epsilon  (epsilon for convergence), iterations (number of iterations), convergence (T or F), exploding (
    #         if gradient
    #         explodes), labeled_weights (dict with {labels : weights}), weights (nd array of optimized weights),
    #         train_sse, validation_sse (list with SSE on each set for each iteration), and gradient_norm (list with
    #         norm of gradient after each iteration).
    #     """
    #     # Training and validation sets and labels
    #     x_train = self.train_features.to_numpy(dtype=np.float64)
    #     y_train = self.train_labels.to_numpy(dtype=np.float64)
    #     x_val = self.validation_features.to_numpy(dtype=np.float64)
    #     y_val = self.validation_labels.to_numpy(dtype=np.float64)
    #
    #     rate = self.rate
    #     lam = self.lam
    #     eps = self.eps
    #     weight_labels = self.train.drop(self.label, axis=1).columns.tolist()
    #
    #     print('Initializing training...')
    #
    #     # Initialize all weights as zero.
    #     weights = np.zeros(np.size(x_train, axis=1), dtype=np.float64)
    #
    #     print('Learning rate = ' + str(rate) + ', penalty = ' + str(lam) + ', epsilon = ' + str(eps) + '.')
    #
    #     # Shape of x[i] is (22, ), shape of y[i] is (1), shape of weights * x[i] - y[i] is (1), shape of grad is (
    #     22, ).
    #     print('Optimizing weights...')
    #
    #     # Included a progress bar for iteration progress, list for SSE's, indicators for gradient and convergence.
    #     bar = progressbar.ProgressBar()
    #     iter_count = 0
    #
    #     train_sse = []
    #     val_sse = []
    #     norm_list = []
    #
    #     exploding_grad = False
    #     converge = False
    #
    #     # Perform batch gradient descent to optimize weights.
    #     for iteration in bar(range(max_iter)):
    #         # Calculate gradient and update weights
    #         gradient = calc_gradient(x_train, y_train, weights)
    #         weights = gradient_descent(gradient, weights, rate, lam)
    #
    #         # Calculate sum of squared error for each iteration to store in list.
    #         train_sse.append(calc_sse(x_train, y_train, weights, lam))
    #         val_sse.append(calc_sse(x_val, y_val, weights, lam))
    #
    #         # Calculate norm of gradient to monitor for convergence.
    #         grad_norm = np.sqrt(gradient.dot(gradient))
    #         norm_list.append(grad_norm)
    #
    #         # Check for divergence with the norm of the gradient to see if exploding.
    #         if np.isinf(grad_norm):
    #             print('\nGradient exploding.\n')
    #             exploding_grad = True
    #             break
    #
    #         # Check for convergence using the norm of the gradient.
    #         if grad_norm <= eps:
    #             print('\nConvergence achieved with epsilon = ' + str(eps) + ' in ' + str(iteration) + ' iterations.')
    #             converge = True
    #             break
    #
    #         # Check that gradient is still decreasing sufficiently
    #         if iter_count > 10000 and abs(norm_list[-2] - norm_list[-1]) < 1e-06:
    #             print('\nGradient not decreasing significantly.')
    #             converge = True
    #             break
    #
    #     # If we haven't converged by this point might as well stop and figure out why.
    #     if iter_count == max_iter:
    #         print('Maximum iterations reached without convergence.\n')
    #
    #     labeled_weights = dict(zip(weight_labels, weights.tolist()))
    #
    #     results = {'lam': lam,
    #                'learn_rate': rate,
    #                'epsilon': eps,
    #                'iterations': iter_count,
    #                'convergence': converge,
    #                'exploding': exploding_grad,
    #                'labeled_weights': labeled_weights,
    #                'weights': weights.tolist(),
    #                'train_sse': train_sse,
    #                'validation_sse': val_sse,
    #                'gradient_norm': norm_list}
    #     return results
    #
    # def predict_validation(self, weights):
    #     """Generates predictions and sum-of-squared error calculations for validation data with labels.
    #
    #     Uses validation set provided when instance of class is created.
    #
    #     Args:
    #         weights (ndarray): (, n) ndarray of weights produced from training.
    #
    #     Returns:
    #         results (dict): Dictionary with lam (regularization parameter), predictions (list of predictions),
    #         and SSE (SSE for predictions).
    #     """
    #     lam = self.lam
    #     x_val = self.validation_features.to_numpy(dtype=np.float64)
    #     y_val = self.validation_labels.to_numpy(dtype=np.float64)
    #     predictions = calc_predictions(x_val, weights)
    #     sse = calc_sse(x_val, y_val, weights, lam)
    #     results = {'lam': lam,
    #                'SSE': sse,
    #                'predictions': predictions}
    #     return results
    #
    # def predict_test(self, weights):
    #     """Generates predictions for unlabeled test data, transformed back to original scale if necessary.
    #
    #     Args:
    #         weights (ndarray): (, n) ndarray of weights produced from training.
    #
    #     Returns:
    #         results (dict): Dictionary with lam (regularization parameter) and predictions (list of predictions).
    #     """
    #     lam = self.lam
    #     x_test = self.test_features.to_numpy(dtype=np.float64)
    #     predictions = calc_predictions(x_test, weights)
    #     if self.normalize is True:
    #         label_max = self.train_label_max
    #         label_min = self.train_label_min
    #         predictions = [pred * (label_max - label_min) + label_min for pred in predictions]
    #     results = {'lam': lam,
    #                'predictions': predictions}
    #     return results
