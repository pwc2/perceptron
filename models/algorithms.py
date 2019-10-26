"""
    File name: algorithms.py
    Author: Patrick Cummings
    Date created: 10/25/2019
    Date last modified: 10/26/2019
    Python Version: 3.7

    Contains functions for training and evaluating perceptron model.
    calc_predictions() to calculate predictions on test or validation sets.
    calc_accuracy() to calculate accuracy of predictions generated by classifier.

    Also contains functions to run different variations of perceptron algorithm.
    online_perceptron() to train model with online updates.
    average_perceptron() to train model with running update to average weights.
    kernel_perceptron() to train model using polynomial kernel.
"""

import numpy as np


def predict(X, weights):
    """Calculates predicted values for perceptron classifier using given weights.

    Args:
        X (ndarray): (n x m) ndarray of n observations on m features.
        weights (ndarray): (1 x m) ndarray of m weights from learned model.

    Returns:
        predictions (list of float): list of predicted values in {-1, 1}.
    """
    wtx = X.dot(weights)
    predictions = [1 if x >= 0 else -1 for x in wtx]
    return predictions


def predict_kernel(alphas, y_train, K):
    """Calculates predicted values for polynomial kernel perceptron classifier using given alphas.

    Args:
        alphas (ndarray): (1 x n) ndarray of n alphas from learned model.
        y_train (ndarray): (n x 1) ndarray of labels from training set.
        K (ndarray): (k x m) ndarray of gram matrix, or similarity matrix.

    Returns:
        predictions (list of float): list of predicted values in {-1, 1}.
    """
    u = (alphas * y_train).dot(K)
    predictions = [1 if x >= 0 else -1 for x in u]
    return predictions


def accuracy(predictions, labels):
    """Calculate accuracy of perceptron classifier.

    Args:
        predictions (list or ndarray): list or ndarray of class predictions in {-1, 1}
        labels (list or ndarray): list or ndarray of true labels.

    Returns:
        accuracy (float): calculated accuracy.
    """
    # Ensure lists are coerced to ndarrays of integers.
    predictions = np.array(predictions, dtype=int)
    labels = np.array(labels, dtype=int)
    correct = (labels == predictions)
    accuracy = correct.sum() / np.size(correct)
    return accuracy


def poly_kernel(X, Y, p):
    """Calculate polynomial kernel of degree p as (1 + X.dot(Y.T)) ** p.
    Use to create Gram or similarity matrices.

    Args:
        X (ndarray): first vector or matrix.
        Y (ndarray): second vector or matrix.
        p (int): degree of polynomial kernel.

    Returns:
        K (float): calculated polynomial kernel.
    """
    K = (1 + X.dot(Y.T)) ** p
    return K


def online_perceptron(X_train, y_train, X_val, y_val, max_iter):
    """Trains online perceptron, without random shuffling of training data.
        Calls calc_predictions() and calc_accuracy() from metrics.py.

    Args:
        X_train (ndarray): ndarray of training features.
        y_train (ndarray): ndarray of training labels.
        X_val (ndarray): ndarray of validation features.
        y_val (ndarray): ndarray of validation labels.
        max_iter (int): maximum number of iterations for training.

    Returns:
        results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
        list of validation accuracies for each iteration, and a list of weights from each iteration.
    """

    # Number of features and number of samples.
    n_train = np.size(X_train, axis=0)
    n_features = np.size(X_train, axis=1)

    # Initialize all weights as zero, create list for training and validation accuracy for each iteration.
    weights = np.zeros(n_features, dtype=np.float64)
    weights_list = []
    train_acc_list = []
    val_acc_list = []

    # Run training algorithm.
    print('\nRunning online perceptron...')
    for iteration in range(max_iter):
        print('Current iteration: ' + str(iteration + 1))
        for sample in range(n_train):
            loss = y_train[sample] * (weights.T.dot(X_train[sample]))
            if loss <= 0:
                weights += (y_train[sample] * X_train[sample])

        # Calculate predictions and get accuracy for each iteration, append to lists.
        train_pred = predict(X_train, weights)
        val_pred = predict(X_val, weights)

        train_acc = accuracy(train_pred, y_train)
        train_acc_list.append(train_acc)

        val_acc = accuracy(val_pred, y_val)
        val_acc_list.append(val_acc)

        weights_list.append(weights.tolist())

    results = {'model': 'online',
               'iterations': max_iter,
               'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'weights': weights_list}
    return results


def average_perceptron(X_train, y_train, X_val, y_val, max_iter):
    """Trains average perceptron, without random shuffling of training data.
        Calls calc_predictions() and calc_accuracy() from metrics.py.

    Args:
        X_train (ndarray): ndarray of training features.
        y_train (ndarray): ndarray of training labels.
        X_val (ndarray): ndarray of validation features.
        y_val (ndarray): ndarray of validation labels.
        max_iter (int): maximum number of iterations for training.

    Returns:
        results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
        list of validation accuracies for each iteration, and a list of average weights from each iteration.
    """

    # Number of features and number of samples.
    n_train = np.size(X_train, axis=0)
    n_features = np.size(X_train, axis=1)

    # Initialize all weights as zero, count to 1, create list for training and validation accuracy for each iteration.
    weights = np.zeros(n_features, dtype=np.float64)
    avg_weights = np.zeros(n_features, dtype=np.float64)
    count = 1
    avg_weights_list = []
    train_acc_list = []
    val_acc_list = []

    # Run training algorithm.
    print('\nRunning average perceptron...')
    for iteration in range(max_iter):
        print('Current iteration: ' + str(iteration + 1))
        for sample in range(n_train):
            loss = y_train[sample] * (weights.T.dot(X_train[sample]))
            if loss <= 0:
                weights += (y_train[sample] * X_train[sample])
            avg_weights = (count * avg_weights + weights) / (count + 1)
            count += 1

        # Calculate predictions and get accuracy for each iteration, append to lists.
        train_pred = predict(X_train, avg_weights)
        val_pred = predict(X_val, avg_weights)

        train_acc = accuracy(train_pred, y_train)
        train_acc_list.append(train_acc)

        val_acc = accuracy(val_pred, y_val)
        val_acc_list.append(val_acc)

        avg_weights_list.append(avg_weights.tolist())

    results = {'model': 'average',
               'iterations': max_iter,
               'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'avg_weights': avg_weights_list}
    return results


def kernel_perceptron(X_train, y_train, X_val, y_val, p, max_iter):
    """Trains a polynomial kernelized perceptron, without random shuffling of training data.
        Calls calc_predictions() and calc_accuracy() from metrics.py.

    Args:
        X_train (ndarray): ndarray of training features.
        y_train (ndarray): ndarray of training labels.
        X_val (ndarray): ndarray of validation features.
        y_val (ndarray): ndarray of validation labels.
        p (int): degree of polynomial kernel.
        max_iter (int): maximum number of iterations for training.

    Returns:
        results (dict): Dictionary with number of iterations, list of training accuracies for each iteration,
        list of validation accuracies for each iteration, and a list of weights from each iteration.
    """

    # Number of samples.
    n_train = np.size(X_train, axis=0)

    # Initialize all alphas as zero, create list for training and validation accuracy for each iteration.
    alphas = np.zeros(n_train, dtype=int)
    alphas_list = []
    train_acc_list = []
    val_acc_list = []

    # Compute gram (K) matrix.
    print('\nComputing Gram matrix on training set for p = ' + str(p) + '.')
    K = poly_kernel(X_train, X_train, p)

    # Run training algorithm.
    print('Running polynomial kernel perceptron...')
    for iteration in range(max_iter):
        print('Current iteration: ' + str(iteration + 1))
        for sample in range(n_train):
            u = np.dot(alphas * y_train, K[:, sample])
            if y_train[sample] * u <= 0:
                alphas[sample] += 1

        # Calculate similarity matrix for train and validation sets.
        K_val = poly_kernel(X_train, X_val, p)

        # Calculate predictions and get accuracy for each iteration, append to lists.
        train_pred = predict_kernel(alphas, y_train, K)
        val_pred = predict_kernel(alphas, y_train, K_val)

        train_acc = accuracy(train_pred, y_train)
        train_acc_list.append(train_acc)

        val_acc = accuracy(val_pred, y_val)
        val_acc_list.append(val_acc)

        alphas_list.append(alphas.tolist())

    results = {'model': 'kernel',
               'p': p,
               'iterations': max_iter,
               'train_acc': train_acc_list,
               'val_acc': val_acc_list,
               'alphas': alphas_list}
    return results
