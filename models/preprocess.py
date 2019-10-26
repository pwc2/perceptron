"""
    File name: preprocess.py
    Author: Patrick Cummings
    Date created: 10/24/2019
    Date last modified: 10/26/2019
    Python Version: 3.7

    preprocess() converts labels to {-1, 1} as required, and adds a bias feature set to 1.
    normalize_features() normalizes data with respect to the absolute min and max gray-scale intensity values from
    the training set (we're assuming all pixel features are already on the same scale).
    Cleaned DataFrames saved as both .csv and .pkl files in /data folder.
"""
import pickle
from pathlib import Path

import pandas as pd


def preprocess(csv_file, name):
    """Read in .csv file, set labels {-1. 1} as required for training and validation sets.
    If pre-processing test set, just sets column names for features to match training and validation sets.
    Save pickled DataFrame in /data folder.

    Args:
        csv_file (str): Name of .csv file in /data folder to preprocess.
        name (str): Specify whether 'train', 'validation', or 'test'.

    Returns:
        None
    """
    # Get path to data folder and csv_file.
    data_path = Path(__file__).parent.resolve().joinpath(Path('..', 'data'))
    csv_file = data_path.joinpath(Path(csv_file))

    if name is 'test':
        # Set columns for features in test set to match train and validation
        df = pd.read_csv(csv_file, header=None, names=[x for x in range(1, 785)])
        # Set bias feature to 1 and make sure it's first column in test DataFrame.
        df = df.assign(bias=1).set_index('bias').reset_index()
    else:
        # Set column name to 'label' for labels, change labels to 1 and -1
        df = pd.read_csv(csv_file, header=None)
        df.rename({0: 'label'}, axis=1, inplace=True)
        df.replace({'label': {3: 1, 5: -1}}, inplace=True)
        # Set bias feature to 1 and make sure it's first column in DataFrame after labels.
        df = df.assign(bias=1).set_index('bias').reset_index()
        df = df.set_index('label').reset_index()

    # Grab original filename to save normalized data.
    file_name = str(csv_file.stem) + '_clean.csv'
    df.to_csv(data_path.joinpath(Path(file_name)), index=False)

    # Get path to data folder and serialize and save cleaned DataFrame.
    file_name = str(csv_file.stem) + '_clean.pkl'
    with open(data_path.joinpath(Path(file_name)), 'wb') as fpkl:
        pickle.dump(df, fpkl, pickle.HIGHEST_PROTOCOL)

# def normalize_features(train_pkl, other_pkl, name):
#     """Normalize features to (0, 1) range with respect to training set.
#
#     Args:
#         train_pkl (str): Name of .pkl file in /data with DataFrame of preprocessed training data.
#         other_pkl (str): Name of .pkl file in /data with DataFrame of preprocessed data to normalize.
#         name (str): String specifying if other DataFrame is 'train', 'validation', or 'test' set.
#
#     Returns:
#         None
#     """
#     # Get path to data folder and .csv files.
#     data_path = Path(__file__).parent.resolve().joinpath(Path('..', 'data'))
#     train_path = data_path.joinpath(Path(train_pkl))
#     other_path = data_path.joinpath(Path(other_pkl))
#
#     # Load cleaned DataFrames.
#     with open(train_path, 'rb') as fpkl1:
#         train_df = pickle.load(fpkl1)
#     with open(other_path, 'rb') as fpkl2:
#         other_df = pickle.load(fpkl2)
#
#     # Extract the features to be normalized, and overall max and min gray-scale values from training set
#     # Since I'm assuming all pixel features are already on the same scale
#     other_features = pd.DataFrame(other_df.loc[:, other_df.columns != 'label'])
#     train_max = train_df.values.max()
#     train_min = train_df.values.min()
#
#     # Normalize all features wrt to training set.
#     norm_df = (other_features - train_min) / (train_max - train_min)
#
#     # Fill NaN's created during normalization when max == min.
#     norm_df = norm_df.fillna(0)
#
#     # Reset bias feature to 1 and make sure it's first column in DataFrame.
#     norm_df = norm_df.assign(bias=1).set_index('bias').reset_index()
#
#     # Replace non-normalized columns with normalized columns before saving to .csv.
#     other_df.update(norm_df)
#
#     # Grab original filename to save normalized data.
#     file_name = str(other_path.stem) + '_norm.csv'
#     other_df.to_csv(data_path.joinpath(Path(file_name)), index=False)
