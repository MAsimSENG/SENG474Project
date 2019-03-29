#Preprocessor methods on the feature matrix

import numpy as np
from sklearn.decomposition import pca

# x_train is the training data
def train_normalize(x_train):
    x_train_mean = np.mean(x_train, axis=0)
    x_train_std = np.std(x_train, axis=0)

    return x_train_mean, x_train_std