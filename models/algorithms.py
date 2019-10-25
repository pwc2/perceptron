"""
    File name: algorithms.py
    Author: Patrick Cummings
    Date created: 10/25/2019
    Date last modified: 10/25/2019
    Python Version: 3.7

    Contains functions to run different variations of perceptron algorithm.
    online_perceptron() to train model with online updates.
    average_perceptron() to train model with running update to average weights.
    kernel_perceptron() to train model using polynomial kernel.
"""

import numpy as np

from models.metrics import calc_predictions, calc_accuracy


def online_perceptron(x_train, y_train, x_val, y_val, max_iter):
    """Trains online perceptron, without random shuffling of training data.

    Args:
        x_train (ndarray): ndarray of training features.
        y_train (ndarray): ndarray of training labels.
        x_val (ndarray): ndarray of validation features.
        y_val (ndarray): ndarray of validation labels.
        max_iter (int): maximum number of iterations for training.

    Returns:
        results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
        list of validation accuracies for each iteration, and a list of weights from each iteration.
    """

    # Number of features and number of samples.
    samp_size = np.size(x_train, axis=0)
    feature_size = np.size(x_train, axis=1)

    # Initialize all weights as zero, create list for training and validation accuracy for each iteration.
    weights = np.zeros(feature_size, dtype=np.float64)
    weights_list = []
    train_acc_list = []
    val_acc_list = []

    for iteration in range(max_iter):
        print('Current iteration: ' + str(iteration + 1))
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

    results = {'model': 'online',
               'iterations': max_iter,
               'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'weights': weights_list}
    return results


def average_perceptron(x_train, y_train, x_val, y_val, max_iter):
    """Trains average perceptron, without random shuffling of training data.

    Args:
        x_train (ndarray): ndarray of training features.
        y_train (ndarray): ndarray of training labels.
        x_val (ndarray): ndarray of validation features.
        y_val (ndarray): ndarray of validation labels.
        max_iter (int): maximum number of iterations for training.

    Returns:
        results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
        list of validation accuracies for each iteration, and a list of average weights from each iteration.
    """

    # Number of features and number of samples.
    samp_size = np.size(x_train, axis=0)
    feature_size = np.size(x_train, axis=1)

    # Initialize all weights as zero, count to 1, create list for training and validation accuracy for each iteration.
    weights = np.zeros(feature_size, dtype=np.float64)
    avg_weights = np.zeros(feature_size, dtype=np.float64)
    count = 1
    avg_weights_list = []
    train_acc_list = []
    val_acc_list = []

    for iteration in range(max_iter):
        print('Current iteration: ' + str(iteration + 1))
        for sample in range(samp_size):
            loss = y_train[sample] * (weights.T.dot(x_train[sample]))
            if loss <= 0:
                weights += (y_train[sample] * x_train[sample])
            avg_weights = (count * avg_weights + weights) / (count + 1)
            count += 1

        # Calculate predictions and get accuracy for each iteration, append to lists.
        train_pred = calc_predictions(x_train, avg_weights)
        val_pred = calc_predictions(x_val, avg_weights)

        train_acc = calc_accuracy(train_pred, y_train)
        train_acc_list.append(train_acc)

        val_acc = calc_accuracy(val_pred, y_val)
        val_acc_list.append(val_acc)

        avg_weights_list.append(avg_weights.tolist())

    results = {'model': 'average',
               'iterations': max_iter,
               'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'avg_weights': avg_weights_list}
    return results
