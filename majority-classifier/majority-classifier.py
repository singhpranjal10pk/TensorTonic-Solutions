import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """

    # Find the majority class
    majority_label = np.bincount(y_train).argmax()

    # Predict the majority class for every test sample
    return np.full(len(X_test), majority_label)