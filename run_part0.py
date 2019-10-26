"""
    File name: run_part0.py
    Author: Patrick Cummings
    Date created: 10/24/2019
    Date last modified: 10/26/2019
    Python Version: 3.7

    Pre-processes data to create cleaned DataFrames, saved in /data folder.
    Cleaned DataFrames saved as .pkl files in /data.
"""

from models import preprocess

# Pre-processing - Part 0.

print('\nInitializing preprocessing...')

# Clean .csv files
preprocess.preprocess('pa2_train.csv', 'train')
preprocess.preprocess('pa2_valid.csv', 'validation')
preprocess.preprocess('pa2_test_no_label.csv', 'test')

# Normalize training, validation, and test sets
# preprocess.normalize_features('pa2_train_clean.pkl', 'pa2_train_clean.pkl', 'train')
# preprocess.normalize_features('pa2_train_clean.pkl', 'pa2_valid_clean.pkl', 'validation')
# preprocess.normalize_features('pa2_train_clean.pkl', 'pa2_test_no_label_clean.pkl', 'test')

print('Completed preprocessing.\n')
